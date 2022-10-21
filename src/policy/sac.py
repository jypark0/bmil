from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.exploration import BaseNoise
from tianshou.policy import DiscreteSACPolicy, SACPolicy
from torch.distributions import Categorical

from src.utils import to_np


class SACandBCPolicy(SACPolicy):
    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        env_ratio=0.05,
        demo_ratio=0.1,
        bc_coef=2,
        include_trace_in_bc=False,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau,
            gamma,
            alpha,
            reward_normalization,
            estimation_step,
            exploration_noise,
            deterministic_eval,
            **kwargs,
        )
        self.env_ratio = env_ratio
        self.demo_ratio = demo_ratio
        self.bc_coef = bc_coef
        self.include_trace_in_bc = include_trace_in_bc

    def update(
        self,
        sample_size: int,
        env_buffer: Optional[ReplayBuffer],
        demo_buffer: Optional[ReplayBuffer] = None,
        model_buffer: Optional[ReplayBuffer] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if env_buffer is None:
            return {}

        self.updating = True

        # Env buffer
        env_sample_size = sample_size
        if demo_buffer is not None or model_buffer is not None:
            env_sample_size = int(self.env_ratio * sample_size)

        if env_sample_size > 0:
            env_batch, env_indices = env_buffer.sample(env_sample_size)
            env_batch = self.process_fn(env_batch, env_buffer, env_indices)
        else:
            return {}

        # Demo buffer
        if demo_buffer is not None:
            if model_buffer is not None:
                demo_sample_size = int(
                    min(self.demo_ratio * sample_size, sample_size - env_sample_size)
                )
            else:
                demo_sample_size = sample_size - env_sample_size

        if demo_sample_size > 0:
            demo_batch, demo_indices = demo_buffer.sample(demo_sample_size)
            demo_batch = self.process_fn(demo_batch, demo_buffer, demo_indices)
        else:
            demo_batch = Batch()

        model_sample_size = sample_size - env_sample_size - demo_sample_size
        if model_sample_size > 0:
            model_batch, model_indices = model_buffer.sample(model_sample_size)
            model_batch = self.process_fn(model_batch, model_buffer, model_indices)
        else:
            model_batch = Batch()

        # Combine batches
        batch = Batch.cat([env_batch, demo_batch, model_batch])
        assert len(batch) == sample_size, f"total_size={len(batch)} != {sample_size}"

        # Pass bcbatch separately to compute BC loss
        if self.include_trace_in_bc:
            bc_batch = Batch.cat([demo_batch, model_batch])
        else:
            bc_batch = demo_batch
        result = self.learn(batch, bc_batch, **kwargs)

        # Usually for updating priorities (not used here)
        self.post_process_fn(env_batch, env_buffer, env_indices)
        if not demo_batch.is_empty():
            self.post_process_fn(demo_batch, demo_buffer, demo_indices)
        if not model_batch.is_empty():
            self.post_process_fn(model_batch, model_buffer, model_indices)
        self.updating = False

        return result

    def learn(self, batch: Batch, bc_batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a).flatten()
        current_q2a = self.critic2(batch.obs, a).flatten()
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten()
            - torch.min(current_q1a, current_q2a)
        ).mean()

        # behavior cloning loss
        if not bc_batch.is_empty():
            bc_result = self(bc_batch)
            # inv_var = 1 / bc_result.logits[1]
            bc_loss = self.bc_coef * torch.mean(
                torch.pow(bc_result.act - to_torch_as(bc_batch.act, bc_result.act), 2)
                # * inv_var
            )

            # bc_result = self(bc_batch)
            # bc_loss = -self.bc_coef * bc_result.log_prob.mean()

            loss = actor_loss + bc_loss
        else:
            loss = actor_loss

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss": loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if not bc_batch.is_empty():
            result["loss/bc"] = bc_loss.item()

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
