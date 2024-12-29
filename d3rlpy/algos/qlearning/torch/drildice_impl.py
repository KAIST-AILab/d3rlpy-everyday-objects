import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict, Union

import torch
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    NormalPolicy,
    Policy,
    ContinuousEnsembleQFunctionForwarder,
    compute_deterministic_imitation_loss,
    compute_stochastic_imitation_loss,
)
from ....torch_utility import Modules, TorchMiniBatch
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .bc_impl import BCBaseModules, BCBaseImpl
import torch.nn.functional as F

__all__ = ["DrilDICEImpl", "DrilDICEModules"]


@dataclasses.dataclass(frozen=True)
class DrilDICEModules(BCBaseModules):
    imitator: NormalPolicy
    nu_optim: Optimizer


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
        nu_func_forwarder: ContinuousEnsembleQFunctionForwarder,
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
        self._nu_func_forwarder = nu_func_forwarder
        # self._policy_type = policy_type

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).squashed_mu
    
    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        return self.update_imitator(batch)
    
    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions)

        loss.backward()
        self._modules.optim.step()

        return asdict_as_float(loss)

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        bc_loss = F.mse_loss(self._modules.imitator(obs_t).squashed_mu, act_t)
        # return bc_loss

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim

