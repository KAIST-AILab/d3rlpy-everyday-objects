import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict, Union

import torch
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    NormalPolicy,
    Policy,
    ValueFunction,
    ContinuousQFunction,
    ContinuousEnsembleQFunctionForwarder,
)
from ....torch_utility import Modules, TorchMiniBatch, train_api
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .bc_impl import BCBaseModules, BCBaseImpl
import torch.nn.functional as F

__all__ = ["DrilDICEImpl", "DrilDICEModules"]


@dataclasses.dataclass(frozen=True)
class DrilDICEModules(BCBaseModules):
    imitator: NormalPolicy
    nu_func: ValueFunction
    nu_optim: Optimizer

@dataclasses.dataclass(frozen=True)
class ImitatorLoss:
    imitator_loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class NuLoss:
    nu_loss: torch.Tensor

class DrilDICEImpl(BCBaseImpl):
    _modules: DrilDICEModules
    _alpha: float
    _gamma: float
    _f_divergence_type: str
    _max_weight: float
    _nu_func_forwarder: ContinuousEnsembleQFunctionForwarder

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DrilDICEModules,
        gamma: float,
        alpha: float,
        f_divergence_type: str,
        max_weight: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )

        self._alpha = alpha
        self._gamma = gamma
        self._f_divergence_type = f_divergence_type
        self._max_weight = max_weight
        # self._policy_type = policy_type

        # soft TV-distance
        EPS = 1e-8
        self.f_fn = \
            lambda x: 0.5 * torch.log( 0.5 * (torch.exp(x-1) + torch.exp(1-x)) + EPS)
        self.w_fn =  \
            lambda x: F.relu(0.5 * (torch.log( 1 + 2 * torch.clip(x, -0.5, 0.5) + EPS) - torch.log (1 - 2 * torch.clip(x, -0.5, 0.5) + EPS))  + 1)
            # lambda x: torch.exp(torch.min(x, 0)) if x < 0 else x + 1
        self.f_w_fn = \
            lambda x: self.f_fn(self.w_fn(x))

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).squashed_mu
    
    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")

    @train_api
    def update(self, batch: TorchMiniBatch, batch0: TorchMiniBatch, batch_bc:TorchMiniBatch, grad_step: int) -> Dict[str, float]:
        result_dict1 = self.outer_update(batch, batch0, grad_step)
        result_dict2 = self.inner_update(batch_bc, grad_step)
        result_dict1.update(result_dict2)

        return result_dict1
    
    def outer_update(
        self, batch: TorchMiniBatch, batch0: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        return self.update_nu(batch, batch0)
    
    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        return self.update_imitator(batch)
    
    def update_nu(self, batch: TorchMiniBatch, batch0: TorchMiniBatch) -> Dict[str, float]:
        self._modules.nu_optim.zero_grad()

        loss = self.compute_nu_loss(batch.observations, batch.actions, batch.next_observations, batch.terminals, batch0.observations)

        loss.nu_loss.backward()
        self._modules.nu_optim.step()

        return asdict_as_float(loss)
    
    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions, batch.next_observations, batch.terminals)

        loss.imitator_loss.backward()
        self._modules.optim.step()

        return asdict_as_float(loss)

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor, obs_t_prime: TorchObservation, terminals: torch.Tensor
    ) -> torch.Tensor:
        C_pi_s = ((act_t - self._modules.imitator(obs_t).squashed_mu) ** 2).mean(dim=-1)
        nu_s = self._modules.nu_func(obs_t).reshape(-1)
        nu_s_prime =  self._modules.nu_func(obs_t_prime).reshape(-1)
        terminals = terminals.reshape(-1)

        e_target = C_pi_s + (1 - terminals) * self._gamma * nu_s_prime - nu_s
        w_e = self.w_fn (e_target / self._alpha)
        
        wbc_loss = (w_e.detach() * C_pi_s).mean()
        # bc_loss = F.mse_loss(self._modules.imitator(obs_t).squashed_mu, act_t)
        return ImitatorLoss(imitator_loss=wbc_loss)
    
    def compute_nu_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor, obs_t_prime: TorchObservation, terminals: torch.Tensor,
        obs_t0: TorchObservation
    ) -> torch.Tensor:
        C_pi_s = ((act_t - self._modules.imitator(obs_t).squashed_mu) ** 2).mean(dim=-1)
        nu_s0 = self._modules.nu_func(obs_t0).reshape(-1)
        nu_s_t = self._modules.nu_func(obs_t).reshape(-1)
        nu_s_t_prime= self._modules.nu_func(obs_t_prime).reshape(-1)
        
        terminals  = terminals.reshape(-1)
        e_target = C_pi_s + (1. - terminals) * self._gamma * nu_s_t_prime - nu_s_t
        f_w_nu = self.f_w_fn(e_target / self._alpha)
        w_nu = self.w_fn(e_target / self._alpha)
        
        nu_loss0 = (1-self._gamma) * nu_s0.mean()
        nu_loss1 = -self._alpha * f_w_nu.mean()
        nu_loss2 = (w_nu * e_target).mean()

        nu_loss = nu_loss0 + nu_loss1 + nu_loss2
        
        # nu_loss = F.mse_loss(self._modules.imitator(obs_t).squashed_mu, act_t)
        return NuLoss(nu_loss=nu_loss)

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim

