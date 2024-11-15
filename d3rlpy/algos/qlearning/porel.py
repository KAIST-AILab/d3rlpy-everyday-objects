import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import MeanQFunctionFactory
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.porel_impl import PorelImpl, PorelModules

__all__ = ["PorelConfig", "Porel"]


@dataclasses.dataclass()
class PorelConfig(LearnableConfig):
    r"""PorelDICE algorithm.

    PorelDICE is the offline RL algorithm that avoids using single sample estimate
    to solve OptiDICE.

    There are three functions to train in PorelDICE. First, the state-value function
    is trained via relaxed stationary distribution correction estimation. The degree of
    relaxation is decided by epsilon < 0. We also relax initial state distribution to 
    dataset state distribution, which is a common practice in semi-gradient DICE algorithms.

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [(1-gamma) V_\psi (s) + \alpha f_star((Q_\theta (s, a) - V_\psi (s))/\alpha)

    where :math:`f_star (x) = 0.5 * x**2 + x  (x \geq \epsilon - 1)
                            = \epsilon x - 0.5 * \epsilon **2 + \epsilon - 0.5 (otherwise) `.

    The Q-function is trained with the state-value function.

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    Finally, the policy function is trained by using advantage weighted
    regression.

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [\exp(\beta (Q_\theta - V_\psi(s))) \log \pi_\phi(a|s)]

    References:
        * `Kim, Woosung, Donghyeon Ki, and Byung-Jun Lee. "Relaxed Stationary Distribution Correction Estimation for Improved Offline Policy Optimization."
          Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 12. 2024.`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the value function.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        alpha (float): alpha for value function training.
        epsilon (float): epsilon for the degree of relaxation.
        weight_temp (float): Inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): Maximum advantage weight value to clip.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    alpha: float = 0.1
    epsilon : float = - 1.0
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "Porel":
        return Porel(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "porel"


class Porel(QLearningAlgoBase[PorelImpl, PorelConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._config.critic_learning_rate
        )

        modules = PorelModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = PorelImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            alpha = self._config.alpha,
            epsilon=self._config.epsilon,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(PorelConfig)
