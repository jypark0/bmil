from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import MLP

from src.utils import to_np


class Normalizer:
    def __init__(self, device, eps=1e-6):
        self.eps = torch.tensor([eps]).to(device)
        self.device = device

    def fit(self, data):
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)

        self.mu = torch.from_numpy(self.mu).to(self.device)
        self.mu.requires_grad_(False)
        self.std = torch.from_numpy(self.std).to(self.device)
        self.std.requires_grad_(False)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = to_torch_as(data, self.mu)

        result = (data - self.mu) / torch.maximum(self.std, self.eps)
        result = torch.clamp(result, -5, 5)

        # result = (data - mu) / np.maximum(std, eps)
        # result = np.clip(result, -5, 5)
        return result

    def inverse_transform(self, data):
        if isinstance(data, np.ndarray):
            data = to_torch_as(data, self.mu)
        return self.mu + self.std * data


class AdroitPerturb:
    def __init__(self, env, mode):
        self.env_id = env.spec.id
        self.dim = int(np.prod(env.observation_space.shape))
        if mode not in ["full"]:
            raise ValueError(f"mode ({mode}) must be 'full'")
        self.mode = mode

        self.pos_idxs = [0, 2]

    def perturb(self, batch, full, ns_coef, stddev):
        with torch.no_grad():
            batch_dim = batch.act.shape[0]

            mask = np.zeros(self.dim)

            if self.mode == "full":
                mask += full

            noises = np.random.randn(batch_dim, self.dim)
            noises = mask * noises

            denmlz_noise = noises * stddev[None, :]

            state_dist = np.linalg.norm(noises, axis=-1) / np.maximum(
                np.linalg.norm(mask), 1e-6
            )

            new_batch = Batch(batch, copy=True)
            if not (isinstance(new_batch.obs, Batch) and new_batch.obs.is_empty()):
                new_batch["obs"] += denmlz_noise
            new_batch["returns"] = new_batch.returns - ns_coef * state_dist

        return new_batch


class MazePerturb:
    def __init__(self, env, mode):
        self.env_id = env.spec.id
        self.dim = int(np.prod(env.observation_space.shape))
        if mode not in ["full", "pos"]:
            raise ValueError(f"mode ({mode}) must be 'full', 'pos'")
        self.mode = mode

        self.pos_idxs = [0, 1]

    def perturb(self, batch, pos, full, ns_coef, stddev):
        with torch.no_grad():
            batch_dim = batch.act.shape[0]

            mask = np.zeros(self.dim)

            if self.mode == "pos":
                mask[self.pos_idxs] += pos
            elif self.mode == "full":
                mask += full

            noises = np.random.randn(batch_dim, self.dim)
            noises = mask * noises

            denmlz_noise = noises * stddev[None, :]

            state_dist = np.linalg.norm(noises, axis=-1) / np.maximum(
                np.linalg.norm(mask), 1e-6
            )

            new_batch = Batch(batch, copy=True)
            if not (isinstance(new_batch.obs, Batch) and new_batch.obs.is_empty()):
                new_batch["obs"] += denmlz_noise
            new_batch["returns"] = new_batch.returns - ns_coef * state_dist

        return new_batch


class FetchPerturb:
    def __init__(self, env, mode):
        self.env_id = env.spec.id
        self.dim = int(np.prod(env.observation_space.shape))
        if mode not in ["full", "arm_gripper"]:
            raise ValueError(f"mode ({mode}) must be 'full', 'arm_gripper'")
        self.mode = mode

        if self.env_id.startswith("Push"):
            self.arm_idxs = [0, 1, 2]
        elif self.env_id.startswith("Pick"):
            self.arm_idxs = [0, 1, 2]
            self.gripper_idxs = [9, 10]

    def perturb(self, batch, arm, gripper, full, ns_coef, stddev):
        with torch.no_grad():
            batch_dim = batch.act.shape[0]

            mask = np.zeros(self.dim)

            if self.mode == "arm_gripper":
                if arm != 0 and hasattr(self, "arm_idxs"):
                    mask[self.arm_idxs] += arm
                if gripper != 0 and hasattr(self, "gripper_idxs"):
                    mask[self.gripper_idxs] += gripper
            elif self.mode == "full":
                mask += full

            noises = np.random.randn(batch_dim, self.dim)
            noises = mask * noises

            denmlz_noise = noises * stddev[None, :]

            state_dist = np.linalg.norm(noises, axis=-1) / np.maximum(
                np.linalg.norm(mask), 1e-6
            )

            new_batch = Batch(batch, copy=True)
            if not (isinstance(new_batch.obs, Batch) and new_batch.obs.is_empty()):
                new_batch["obs"] += denmlz_noise
            new_batch["returns"] = new_batch.returns - ns_coef * state_dist

        return new_batch


class DynamicsModel(torch.nn.Module):
    def __init__(self, preprocess_net, observation_shape, max_grad_norm, device):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(observation_shape))
        input_dim = getattr(preprocess_net, "output_dim")
        self.last = MLP(
            input_dim, self.output_dim, (), device=self.device  # type: ignore
        )
        self.max_grad_norm = max_grad_norm

        self.optim = None

        self.scaler = Normalizer(device)

    def forward(self, obs, act, info={}):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        act = torch.as_tensor(act, device=self.device, dtype=torch.float32)

        inputs = torch.cat([obs, act], dim=1)
        logits, _ = self.preprocess(inputs)
        deltas = self.last(logits)
        deltas = self.scaler.inverse_transform(deltas)

        next_states = obs + deltas

        return next_states

    def _dynamics_optimizer(self, batch: Batch, batch_size: int):
        losses = []
        for minibatch in batch.split(batch_size):
            pred_next = self.forward(minibatch.obs, minibatch.act)
            pred_deltas = self.scaler.transform(
                pred_next - to_torch_as(minibatch.obs, pred_next)
            )

            deltas = to_torch_as(
                minibatch.obs_next - minibatch.obs, pred_deltas
            ).flatten(1)
            deltas = self.scaler.transform(deltas)

            dynamics_loss = torch.sqrt(
                torch.pow(pred_deltas - deltas, 2).mean(-1)
            ).mean()

            self.optim.zero_grad()
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optim.step()
            losses.append(dynamics_loss.item())
        return losses

    def update_model(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs
    ) -> Dict[str, Any]:
        if buffer is None:
            return {}
        batch, _ = buffer.sample(sample_size)
        self.updating = True

        # dynamics
        dynamics_loss = self._dynamics_optimizer(batch, **kwargs)

        self.updating = False
        return {
            "loss/dynamics": dynamics_loss,
        }


class ValueFunction(torch.nn.Module):
    def __init__(
        self,
        value_model,
        optim,
        tau,
        gamma,
        n_interpolations,
        n_perturb,
        ns_coef,
        perturb_type,
        perturb_coef,
        perturb_fn,
        device,
    ):
        super().__init__()
        self.value = value_model
        self.optim = optim
        self.value_old = deepcopy(value_model)
        self.value_old.eval()

        self.scaler = Normalizer(device)

        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma

        # Data augmentation
        self.n_interpolations = n_interpolations

        # Negative sampling
        self.n_perturb = n_perturb
        self.ns_coef = ns_coef
        self.perturb_type = perturb_type
        self.perturb_coef = perturb_coef
        self.perturb_fn = perturb_fn

        # Counter for update_value during RL phase
        self.iters = -1

        self.device = device

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        nmlz_inputs = self.scaler.transform(inputs)
        return self.value(nmlz_inputs)

    def augment_batch(self, batch):
        with torch.no_grad():
            t = np.random.rand(self.n_interpolations)

            aug_obs = self.interpolate(batch.obs, batch.obs_next, t).reshape(
                -1, batch.obs.shape[-1]
            )
            aug_ret = self.interpolate(batch.returns, batch.returns + 1, t).reshape(-1)

            aug_batch = Batch(batch, copy=True)
            for k, v in aug_batch.items():
                if k not in ["obs", "returns"] and not (
                    isinstance(v, Batch) and v.is_empty()
                ):
                    v = np.tile(
                        v[None, ...], [self.n_interpolations, *([1] * len(v.shape))]
                    )
                    aug_batch[k] = v.reshape(np.prod(v.shape[:2]), *v.shape[2:])

            aug_batch.obs = aug_obs
            aug_batch.returns = aug_ret

        return aug_batch

    def interpolate(self, x, y, coef):
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        z = coef * x.T + (1.0 - coef) * y.T
        return z.T.reshape(x.shape[0] * len(coef), *x.shape[1:])

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for tgt_param, src_param in zip(
            self.value_old.parameters(), self.value.parameters()
        ):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1 - self.tau) * tgt_param.data
            )

    def process_fn(self, batch: Batch) -> Batch:
        # Compute bellman returns
        with torch.no_grad():
            nmlz_obs_next = self.scaler.transform(batch.obs_next)
            target_v_torch = self.value_old(nmlz_obs_next).flatten()

        target_v = to_np(target_v_torch)
        target_v = batch.rew + (~batch.done) * self._gamma * target_v
        batch.returns = target_v

        return batch

    def _value_optimizer(
        self, batch: Batch, batch_size: int, negative_sampling=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Doesn't work with prioritized replay buffer due to augmentations and perturbations"""

        env = None
        if self.perturb_coef.get("arm") and self.perturb_coef["arm"] != 0:
            # PickAndPlace
            if self.perturb_coef.get("gripper") and self.perturb_coef["gripper"] != 0:
                env = "pick"
            else:
                env = "push"
        elif self.perturb_coef.get("pos") and self.perturb_coef["pos"] != 0:
            env = "maze"
        else:
            env = "adroit"

        losses = []

        for minibatch in batch.split(batch_size):
            if self.iters == 0:
                break
            self.iters -= 1

            minibatch = self.process_fn(minibatch)
            # Perturb obs
            new_batch = []
            with torch.no_grad():
                if env == "pick":
                    perturb_batch = self.perturb_fn(
                        minibatch,
                        arm=0.0,
                        gripper=self.perturb_coef["gripper"],
                        full=self.perturb_coef["full"],
                        ns_coef=self.ns_coef,
                    )
                    perturb_batch.weights = negative_sampling * torch.ones(
                        len(minibatch), dtype=torch.float32, device=self.device
                    )
                    new_batch.append(perturb_batch)
                    perturb_batch = self.perturb_fn(
                        minibatch,
                        arm=self.perturb_coef["arm"],
                        gripper=0,
                        full=self.perturb_coef["full"],
                        ns_coef=self.ns_coef,
                    )
                    perturb_batch.weights = negative_sampling * torch.ones(
                        len(minibatch), dtype=torch.float32, device=self.device
                    )
                    new_batch.append(perturb_batch)
                elif env == "push":
                    for _ in range(self.n_perturb):
                        perturb_batch = self.perturb_fn(
                            minibatch,
                            arm=self.perturb_coef["arm"],
                            gripper=self.perturb_coef["gripper"],
                            full=self.perturb_coef["full"],
                            ns_coef=self.ns_coef,
                        )
                        perturb_batch.weights = negative_sampling * torch.ones(
                            len(minibatch), dtype=torch.float32, device=self.device
                        )
                        new_batch.append(perturb_batch)
                elif env == "maze":
                    for _ in range(self.n_perturb):
                        perturb_batch = self.perturb_fn(
                            minibatch,
                            pos=self.perturb_coef["pos"],
                            full=self.perturb_coef["full"],
                            ns_coef=self.ns_coef,
                        )
                        perturb_batch.weights = negative_sampling * torch.ones(
                            len(minibatch), dtype=torch.float32, device=self.device
                        )
                        new_batch.append(perturb_batch)
                elif env == "adroit":
                    for _ in range(self.n_perturb):
                        perturb_batch = self.perturb_fn(
                            minibatch,
                            full=self.perturb_coef["full"],
                            ns_coef=self.ns_coef,
                        )
                        perturb_batch.weights = negative_sampling * torch.ones(
                            len(minibatch), dtype=torch.float32, device=self.device
                        )
                        new_batch.append(perturb_batch)
                else:
                    raise ValueError(f"env = {env}")

                # Data augmentation
                aug_batch = self.augment_batch(minibatch)
                aug_batch.weights = torch.ones(
                    self.n_interpolations * len(minibatch),
                    dtype=torch.float32,
                    device=self.device,
                )
                new_batch.append(aug_batch)

                new_batch = Batch.cat(new_batch)

            current_v = self.forward(new_batch.obs).flatten()
            target_v = new_batch.returns
            td = current_v - to_torch_as(target_v, current_v)
            loss = torch.sum(td.pow(2) * new_batch.weights) / torch.sum(
                new_batch.weights
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
            self.sync_weight()
        return losses

    def update_value(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs
    ) -> Dict[str, Any]:
        if buffer is None:
            return {}

        batch, _ = buffer.sample(sample_size)

        self.updating = True

        # value
        value_loss = self._value_optimizer(batch, **kwargs)

        self.updating = False
        return {
            "loss/value": value_loss,
        }


class Imitator(BasePolicy):
    def __init__(
        self,
        value_function,
        dynamics_model,
        bc_policy,
        action_shape,
        k,
        vins_alpha,
        rl_alpha,
        device,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_function = value_function
        self.dynamics_model = dynamics_model

        self.bc_policy = bc_policy

        self.action_shape = action_shape

        # Action noise
        self.k = k
        self.vins_alpha = vins_alpha
        self.rl_alpha = rl_alpha

        self.device = device

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("function shouldn't have been called")

    def sample_actions(self, batch, state, bc_prior):
        with torch.no_grad():

            n = batch.obs.shape[0]

            if bc_prior:
                prior = to_np(self.bc_policy(batch, state).act.unsqueeze(1))
                act = prior + self.vins_alpha * np.random.uniform(
                    low=-1, high=1, size=(n, self.k, self.action_shape)
                )
                # Map actions to [-1, 1]
                act = np.clip(act, -1, 1)

            else:
                act = self.rl_alpha * np.random.uniform(
                    low=-1, high=1, size=(n, self.k, self.action_shape)
                )

        return act

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        bc_prior=True,
    ) -> Batch:
        n = batch.obs.shape[0]

        act = self.sample_actions(batch, state, bc_prior)

        act = act.reshape(-1, *act.shape[2:])
        obs = np.expand_dims(batch.obs, 1)
        obs = np.repeat(obs, self.k, axis=1)

        # Flatten dims [B, k, ...] -> [B * k, ...]
        obs = obs.reshape(-1, *obs.shape[2:])
        with torch.no_grad():
            pred_next = to_np(self.dynamics_model(obs, act))
            values = to_np(self.value_function(pred_next)).squeeze(-1)

        # Reshape to [B, k, ...]
        act = act.reshape(n, self.k, self.action_shape)
        values = values.reshape(n, self.k)

        action_idxs = np.argmax(values, axis=1)
        actions = act[np.arange(n), action_idxs]

        return Batch(act=actions)
