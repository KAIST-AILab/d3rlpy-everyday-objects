import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.drildice_impl import DrilDICEImpl, DrilDICEModules

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
    nu_func_factory: QFunctionFactory = make_q_func_field()

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
        
        nu_funcs, nu_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.nu_encoder_factory,
            self._config.nu_func_factory,
            n_ensembles=1,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        nu_optim = self._config.nu_optim_factory.create(
            nu_funcs.named_modules(), lr=self._config.nu_learning_rate
        )

        modules = DrilDICEModules(
            imitator=policy,
            optim=actor_optim,
            nu_optim=nu_optim,
        )

        self._impl = DrilDICEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            nu_func_forwarder=nu_func_forwarder,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            f_divergence_type=self._config.f_divergence_type,
            max_weight=self._config.max_weight,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(DrilDICEConfig)
