import dataclasses

import torch

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    build_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from ....types import Shape, TorchObservation
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseCriticLoss,
    DDPGBaseImpl,
    DDPGBaseModules,
)

__all__ = ["PorelImpl", "PorelModules"]


@dataclasses.dataclass(frozen=True)
class PorelModules(DDPGBaseModules):
    policy: NormalPolicy
    value_func: ValueFunction


@dataclasses.dataclass(frozen=True)
class PorelCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class PorelImpl(DDPGBaseImpl):
    _modules: PorelModules
    _alpha: float
    _epsilon: float
    _weight_temp: float
    _max_weight: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: PorelModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        alpha: float,
        epsilon: float,
        weight_temp: float,
        max_weight: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._alpha = alpha
        self._epsilon = epsilon
        self._weight_temp = weight_temp
        self._max_weight = max_weight

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> PorelCriticLoss:
        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        v_loss = self.compute_value_loss(batch)
        return PorelCriticLoss(
            critic_loss=q_loss + v_loss,
            q_loss=q_loss,
            v_loss=v_loss,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        # compute log probability
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return DDPGBaseActorLoss(-(weight * log_probs).mean())

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        diff = (q_t.detach() - v_t) / self._alpha
        #weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        loss = (1-self._gamma) * v_t + self._alpha * torch.where(diff > self._epsilon-1, 0.5 * diff **2 + diff, self._epsilon * diff - 0.5 * self._epsilon**2 + self._epsilon - 0.5) 
        return (loss).mean()

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()
