import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_value_function,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.drildice_impl import DrilDICEImpl, DrilDICEModules
from ...torch_utility import TorchMiniBatch

from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
from ..utility import (
    assert_action_space_with_dataset,
    build_scalers_with_transition_picker
)
from typing import Optional, Dict, Callable, Generator, Tuple
from typing_extensions import Self
from ...metrics import EvaluatorProtocol
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...constants import (
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
    LoggingStrategy,
)

from ...dataset import (
    ReplayBufferBase, TransitionMiniBatch
)
from ...base import save_config

__all__ = ["DrilDICEConfig", "DrilDICE"]

@dataclasses.dataclass()
class DrilDICEConfig(LearnableConfig):
    r"""Config of DrilDICE algorithm. (https://openreview.net/pdf?id=lHcvjsQFQq)

    DrilDICE is an offline imitation learning method that 
    aims to mitigate the covariate shift problem caused by the discrepancy between the expert and dataset.
    This implementation is based on the original code (https://github.com/tzs930/drildice).

    The policy is trained as a weighted supervised regression.

    .. math::

        J(\pi) = \mathbb{E}_{s_t, a_t \sim D}
            [ w_f(s_t, a_t) l(\pi(s_t), a_t) ]

    where :math:`l` is a supervised loss function which has several options. 
    Typically, we use mean squared error (MSE):
    .. math::
        l(\pi(s_t), a_t) = || \pi(s_t) - a_t ||^2

    The :math:`w_f(s_t, a_t)` is a worst-case weight which is calculated by OptiDICE trick.
    Its calculation is dependent on f-divergence choice. The available options are Soft-TV and Soft-chi^2 divergence.

    References:
        * `Seo et al., "Mitigating Covariate Shift in Behavioral Cloning via Robust Stationary Distribution Correction."
          NeurIPS 2024 <https://openreview.net/pdf?id=lHcvjsQFQq>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        nu_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        nu_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the nu.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        nu_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the nu.
        nu_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            nu function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        alpha (float): regularization term that favors a given dataset
        f_divergence_type (str): f-divergence type. The available options
            are ``['SoftTV', 'SoftChi2']``.
        
    """

    actor_learning_rate: float = 3e-4
    nu_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    nu_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    nu_encoder_factory: EncoderFactory = make_encoder_field()
    # nu_func_factory: QFunctionFactory = make_q_func_field()

    batch_size: int = 100
    gamma: float = 0.99
    f_divergence_type: str = "SoftTV"
    alpha: float = 1.0
    max_weight: float = 20.0
    n_nus: int = 1
    
    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DrilDICE":
        return DrilDICE(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "drildice"


class DrilDICE(QLearningAlgoBase[DrilDICEImpl, DrilDICEConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # nu_funcs, nu_func_forwarder = create_continuous_q_function(
        #     observation_shape,
        #     action_size,
        #     self._config.nu_encoder_factory,
        #     self._config.nu_func_factory,
        #     n_ensembles=1,
        #     device=self._device,
        #     enable_ddp=self._enable_ddp,
        # )
        nu_func = create_value_function(
            observation_shape,
            self._config.nu_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        nu_optim = self._config.nu_optim_factory.create(
            nu_func.named_modules(), lr=self._config.nu_learning_rate
        )

        modules = DrilDICEModules(
            imitator=policy,
            optim=actor_optim,
            nu_func=nu_func,
            nu_optim=nu_optim,
        )

        self._impl = DrilDICEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            f_divergence_type=self._config.f_divergence_type,
            max_weight=self._config.max_weight,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
    
    def update(self, batch_nu: TransitionMiniBatch, batch_nu0: TransitionMiniBatch, batch_wbc:TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        torch_batch_nu = TorchMiniBatch.from_batch(
            batch=batch_nu,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )

        torch_batch_nu0 = TorchMiniBatch.from_batch(
            batch=batch_nu0,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )

        torch_batch_wbc = TorchMiniBatch.from_batch(
            batch=batch_wbc,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )

        loss = self._impl.update(torch_batch_nu, torch_batch_nu0, torch_batch_wbc, self._grad_step)
        self._grad_step += 1
        return loss
    
    def fitter(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        LOG.info("dataset info", dataset_info=dataset.dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset.dataset_info)

        # initialize scalers
        build_scalers_with_transition_picker(self, dataset)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset.dataset_info.action_size
            observation_shape = (
                dataset.sample_transition().observation_signature.shape
            )
            if len(observation_shape) == 1:
                observation_shape = observation_shape[0]  # type: ignore
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch_nu = dataset.sample_transition_batch(
                            self._config.batch_size
                        )

                        batch_nu0 = dataset.sample_transition_batch(
                            self._config.batch_size
                        )

                        batch_wbc = dataset.sample_transition_batch(
                            self._config.batch_size
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch_nu, batch_nu0, batch_wbc)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                if (
                    logging_strategy == LoggingStrategy.STEPS
                    and total_step % logging_steps == 0
                ):
                    metrics = logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if evaluators:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(self, dataset)
                    logger.add_metric(name, test_score)

            # save metrics
            if logging_strategy == LoggingStrategy.EPOCH:
                metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()


register_learnable(DrilDICEConfig)
