import itertools
from typing import Sequence, Type, Union

import numpy as np
import torch
import torch.distributions as torchdist
import torch.nn.functional as F
from tianshou.data import Batch
from tianshou.utils.net.common import MLP

from src.dynamics.utils import StandardScaler
from src.utils import to_torch

ModuleType = Type[torch.nn.Module]

SIGMA_MIN = -10
SIGMA_MAX = 0.5


# Ref: https://github.com/Xingyu-Lin/mbpo_pytorch.git
# Ref: https://github.com/takuseno/d3rlpy/blob/f884c5ce0a67c16b5a43dfc0a327f653fba5480d/d3rlpy/models/torch/dynamics.py
class ProbabilisticNet(torch.nn.Module):
    def __init__(
        self,
        in_dim: Sequence[int],
        out_dim: Sequence[int],
        hidden_sizes: Sequence[int],
        activation: Union[ModuleType, Sequence[ModuleType]],
        device: Union[str, int, torch.device],
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.preprocess = MLP(
            in_dim,
            hidden_sizes[-1],
            hidden_sizes[:-1],
            activation=activation,
            device=self.device,
        )

        self.mu = MLP(hidden_sizes[-1], out_dim, device=self.device)
        self.logvar = MLP(hidden_sizes[-1], out_dim, device=self.device)

        init_min = torch.empty(1, out_dim, dtype=torch.float32).fill_(SIGMA_MIN)
        init_max = torch.empty(1, out_dim, dtype=torch.float32).fill_(SIGMA_MAX)
        self.min_logvar = torch.nn.Parameter(init_min)
        self.max_logvar = torch.nn.Parameter(init_max)

    def forward(self, x):
        logits = self.preprocess(x)

        # Mu
        mu = self.mu(logits)

        # Logstd
        logvar = self.logvar(logits)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar - F.softplus(logvar - self.min_logvar)

        return mu, logvar


class BackwardModel(torch.nn.Module):
    def __init__(
        self,
        act_net,
        obs_net,
        optim,
        dist_fn,
        device,
        action_space=None,
        observation_space=None,
        bounded_act=False,
        bounded_obs=False,
        lr_scheduler=None,
        update_method="fixed_iter",
        obs_delta=False,
        use_scaler=False,
        improvement_threshold=0.001,
        predict_reward=False,
        zero_reward=False,
        demo_reward=None,
        act_scale=None,
    ):
        super().__init__()

        self.act_net = act_net
        self.obs_net = obs_net
        self.optim = optim
        self.dist_fn = dist_fn
        self.device = device

        if action_space is not None:
            self.act_low = torch.from_numpy(action_space.low).float().to(device)
            self.act_high = torch.from_numpy(action_space.high).float().to(device)
        if observation_space is not None:
            self.obs_low = torch.from_numpy(observation_space.low).float().to(device)
            self.obs_high = torch.from_numpy(observation_space.high).float().to(device)

        self.bounded_act = bounded_act
        self.bounded_obs = bounded_obs
        self.lr_scheduler = lr_scheduler
        self.update_method = update_method
        self.obs_delta = obs_delta
        self.improvement_threshold = improvement_threshold
        self.predict_reward = predict_reward
        self.zero_reward = zero_reward
        self.demo_reward = demo_reward
        self.act_scale = torch.from_numpy(act_scale).float().to(device)

        self.use_scaler = use_scaler
        if use_scaler:
            self.act_scaler = StandardScaler()
            self.obs_scaler = StandardScaler()
        else:
            self.act_scaler = None
            self.obs_scaler = None

    def update(self, **kwargs):
        if self.update_method == "fixed_iter":
            return self.update_iter(**kwargs)
        elif self.update_method == "converge":
            return self.update_converge(**kwargs)
        else:
            raise NotImplementedError

    def update_iter(
        self, demo_buffer, batch_size, n_updates, env_buffer=None, **kwargs
    ):
        demo_batch, _ = demo_buffer.sample(0)
        if env_buffer is not None and len(env_buffer) > 0:
            env_batch, _ = env_buffer.sample(0)
        else:
            env_batch = Batch()
        batch = Batch.cat([demo_batch, env_batch])

        batch.act_input = batch.obs_next.copy()
        batch.act_label = batch.act.copy()
        batch.obs_input = np.hstack([batch.obs_next, batch.act])
        if self.obs_delta:
            if self.predict_reward:
                batch.obs_label = np.hstack(
                    [batch.rew[:, None], batch.obs - batch.obs_next]
                )
            else:
                batch.obs_label = batch.obs - batch.obs_next
        else:
            if self.predict_reward:
                batch.obs_label = np.hstack([batch.rew[:, None], batch.obs])
            else:
                batch.obs_label = batch.obs

        if self.use_scaler:
            self.act_scaler.fit(batch.act_input)
            batch.act_input = self.act_scaler.transform(batch.act_input)
            self.obs_scaler.fit(batch.obs_input)
            batch.obs_input = self.obs_scaler.transform(batch.obs_input)

        result = {
            "update/grad_step": 0,
            "update/epoch": 0,
            "train/loss": [],
            "train/act/mse": [],
            "train/act/var": [],
            "train/act/var_bounds": [],
            "train/obs/mse": [],
            "train/obs/var": [],
            "train/obs/var_bounds": [],
        }
        for _ in range(n_updates):
            for b in batch.split(batch_size, merge_last=True):
                b.to_torch(dtype=torch.float32, device=self.device)

                # Action_dist
                act_mu, act_logvar = self.act_net(b.act_input)
                if self.bounded_act:
                    act_mu = torch.clamp(act_mu, min=self.act_low, max=self.act_high)

                # Calculate mse_loss and var_loss
                inv_var = torch.exp(-act_logvar)
                act_mse_loss = torch.mean(torch.pow(act_mu - b.act, 2) * inv_var)
                act_var_loss = torch.mean(act_logvar)
                act_var_bounds_loss = torch.sum(self.act_net.max_logvar) - torch.sum(
                    self.act_net.min_logvar
                )

                # Obs_dist
                obs_mu, obs_logvar = self.obs_net(b.obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low - b.obs_next,
                                max=self.obs_high - b.obs_next,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu.clone(),
                                min=self.obs_low - b.obs_next,
                                max=self.obs_high - b.obs_next,
                            )
                    else:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low,
                                max=self.obs_high,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu.clone(), min=self.obs_low, max=self.obs_high
                            )

                inv_var = torch.exp(-obs_logvar)
                obs_mse_loss = torch.mean(torch.pow(obs_mu - b.obs_label, 2) * inv_var)
                obs_var_loss = torch.mean(obs_logvar)
                obs_var_bounds_loss = torch.sum(self.obs_net.max_logvar) - torch.sum(
                    self.obs_net.min_logvar
                )

                # Calculate loss
                train_loss = (
                    (act_mse_loss + obs_mse_loss)
                    + (act_var_loss + obs_var_loss)
                    + 1e-2 * (act_var_bounds_loss + obs_var_bounds_loss)
                )

                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                result["update/grad_step"] += 1
                result["train/loss"].append(train_loss.item())
                result["train/act/mse"].append(act_mse_loss.item())
                result["train/act/var"].append(act_var_loss.item())
                result["train/act/var_bounds"].append(act_var_bounds_loss.item())
                result["train/obs/mse"].append(obs_mse_loss.item())
                result["train/obs/var"].append(obs_var_loss.item())
                result["train/obs/var_bounds"].append(obs_var_bounds_loss.item())

            # increment epoch
            result["update/epoch"] += 1

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return result

    def update_converge(
        self,
        demo_buffer,
        batch_size,
        env_buffer=None,
        holdout_ratio=0.2,
        max_epochs_since_update=5,
        **kwargs
    ):
        demo_batch, _ = demo_buffer.sample(0)
        if env_buffer is not None and len(env_buffer) > 0:
            env_batch, _ = env_buffer.sample(0)
        else:
            env_batch = Batch()
        batch = Batch.cat([demo_batch, env_batch])

        batch.act_input = batch.obs_next.copy()
        batch.act_label = batch.act.copy()

        batch.obs_input = np.hstack([batch.obs_next, batch.act])
        if self.obs_delta:
            if self.predict_reward:
                batch.obs_label = np.hstack(
                    [batch.rew[:, None], batch.obs - batch.obs_next]
                )
            else:
                batch.obs_label = batch.obs - batch.obs_next
        else:
            if self.predict_reward:
                batch.obs_label = np.hstack([batch.rew[:, None], batch.obs])
            else:
                batch.obs_label = batch.obs

        train_idx = int(len(batch) * (1 - holdout_ratio))
        # Shuffle train/test
        train_batch, test_batch = tuple(batch.split(train_idx, shuffle=True))

        if self.use_scaler:
            self.act_scaler.fit(train_batch.act_input)
            train_batch.act_input = self.act_scaler.transform(train_batch.act_input)
            test_batch.act_input = self.act_scaler.transform(test_batch.act_input)

            self.obs_scaler.fit(train_batch.obs_input)
            train_batch.obs_input = self.obs_scaler.transform(train_batch.obs_input)
            test_batch.obs_input = self.obs_scaler.transform(test_batch.obs_input)

        result = {
            "update/grad_step": 0,
            "update/epoch": 0,
            "train/loss": [],
            "train/act/mse": [],
            "train/act/var": [],
            "train/act/var_bounds": [],
            "train/obs/mse": [],
            "train/obs/var": [],
            "train/obs/var_bounds": [],
            "test/loss": [],
            "test/act/mse": [],
            "test/obs/mse": [],
        }

        best_test_loss = 1e4
        epochs_since_update = 0

        for _ in itertools.count():
            for b in train_batch.split(batch_size, merge_last=True):
                b.to_torch(dtype=torch.float32, device=self.device)

                # Action_dist
                act_mu, act_logvar = self.act_net(b.act_input)
                if self.bounded_act:
                    act_mu = torch.clamp(act_mu, min=self.act_low, max=self.act_high)

                # Calculate mse_loss and var_loss
                inv_var = torch.exp(-act_logvar)
                act_mse_loss = torch.mean(torch.pow(act_mu - b.act_label, 2) * inv_var)
                act_var_loss = torch.mean(act_logvar)
                act_var_bounds_loss = torch.sum(self.act_net.max_logvar) - torch.sum(
                    self.act_net.min_logvar
                )

                # Obs_dist
                obs_mu, obs_logvar = self.obs_net(b.obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low - b.obs_next,
                                max=self.obs_high - b.obs_next,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu.clone(),
                                min=self.obs_low - b.obs_next,
                                max=self.obs_high - b.obs_next,
                            )
                    else:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low,
                                max=self.obs_high,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu.clone(), min=self.obs_low, max=self.obs_high
                            )

                # Calculate mse_loss and var_loss
                inv_var = torch.exp(-obs_logvar)
                obs_mse_loss = torch.mean(torch.pow(obs_mu - b.obs_label, 2) * inv_var)
                obs_var_loss = torch.mean(obs_logvar)
                obs_var_bounds_loss = torch.sum(self.obs_net.max_logvar) - torch.sum(
                    self.obs_net.min_logvar
                )

                # Calculate loss
                train_loss = (
                    (act_mse_loss + obs_mse_loss)
                    + (act_var_loss + obs_var_loss)
                    + 1e-2 * (act_var_bounds_loss + obs_var_bounds_loss)
                )

                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                result["update/grad_step"] += 1
                result["train/loss"].append(train_loss.item())
                result["train/act/mse"].append(act_mse_loss.item())
                result["train/act/var"].append(act_var_loss.item())
                result["train/act/var_bounds"].append(act_var_bounds_loss.item())
                result["train/obs/mse"].append(obs_mse_loss.item())
                result["train/obs/var"].append(obs_var_loss.item())
                result["train/obs/var_bounds"].append(obs_var_bounds_loss.item())

            # Evaluate model on test_batch
            with torch.no_grad():
                for b in test_batch.split(batch_size, merge_last=True):
                    b.to_torch(dtype=torch.float32, device=self.device)

                    # Action_dist
                    act_mu, act_logvar = self.act_net(b.act_input)
                    if self.bounded_act:
                        act_mu = torch.clamp(
                            act_mu, min=self.act_low, max=self.act_high
                        )

                    # Calculate mse_loss
                    act_mse_loss = torch.mean(torch.pow(act_mu - b.act_label, 2))

                    # Obs_dist
                    obs_mu, obs_logvar = self.obs_net(b.obs_input)
                    if self.bounded_obs:
                        if self.obs_delta:
                            if self.predict_reward:
                                obs_mu[..., 1:] = torch.clamp(
                                    obs_mu.clone()[..., 1:],
                                    min=self.obs_low - b.obs_next,
                                    max=self.obs_high - b.obs_next,
                                )
                            else:
                                obs_mu = torch.clamp(
                                    obs_mu.clone(),
                                    min=self.obs_low - b.obs_next,
                                    max=self.obs_high - b.obs_next,
                                )
                        else:
                            if self.predict_reward:
                                obs_mu[..., 1:] = torch.clamp(
                                    obs_mu.clone()[..., 1:],
                                    min=self.obs_low,
                                    max=self.obs_high,
                                )
                            else:
                                obs_mu = torch.clamp(
                                    obs_mu.clone(), min=self.obs_low, max=self.obs_high
                                )

                    # Calculate mse_loss
                    obs_mse_loss = torch.mean(torch.pow(obs_mu - b.obs_label, 2))

                    # Calculate loss
                    test_loss = act_mse_loss + obs_mse_loss

                    result["test/loss"].append(test_loss.item())
                    result["test/act/mse"].append(act_mse_loss.item())
                    result["test/obs/mse"].append(obs_mse_loss.item())

                # Update best_test_loss and epochs_since_update
                # calculate percent improvement
                improvement = (best_test_loss - test_loss) / best_test_loss
                if improvement > self.improvement_threshold:
                    best_test_loss = test_loss
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1

                # Check if no improvement for max_epochs_since_update
                if epochs_since_update > max_epochs_since_update:
                    break

            result["update/epoch"] += 1

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return result

    def generate_trace(
        self, trace_start, dones, trace_size, samples_per_start, **kwargs
    ):
        # Reshape obs from (n_valuable_states,d) -> (n_valuable_states * samples_per_start, d)
        trace_start = np.tile(trace_start, (samples_per_start, 1))
        dones = np.tile(dones, (samples_per_start))

        if self.use_scaler:
            self.act_scaler.fit(trace_start)
            obs_next = self.act_scaler.transform(trace_start)
            obs_next = to_torch(obs_next, self.device)
        else:
            obs_next = to_torch(trace_start, self.device)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        with torch.no_grad():
            for t in range(trace_size):
                if t == 0:
                    next_observations.append(to_torch(trace_start, self.device))
                else:
                    next_observations.append(obs_next)
                # Action_dist
                act_mu, act_logvar = self.act_net(obs_next)
                if self.bounded_act:
                    act_mu = torch.clamp(act_mu, min=self.act_low, max=self.act_high)

                if kwargs["mode"] == "entropy":
                    act_var = kwargs["scale_coef"] * (-act_logvar) * act_logvar.exp()
                    act_var = torch.clamp(act_var, min=1e-12)
                else:
                    act_var = act_logvar.exp()

                act_dist = self.dist_fn(act_mu, act_var)
                act = act_dist.rsample().float()

                if kwargs["mode"] == "resample":
                    unif_dist = torchdist.Uniform(
                        low=act - kwargs["scale_coef"] * self.act_scale,
                        high=act + kwargs["scale_coef"] * self.act_scale,
                    )
                    act = unif_dist.sample().float()

                actions.append(act)

                # Obs_dist
                if t == 0:
                    obs_input = torch.cat(
                        [to_torch(trace_start, self.device), act], dim=-1
                    )
                else:
                    obs_input = torch.cat([obs_next, act], dim=-1)
                if self.use_scaler:
                    self.obs_scaler.fit(obs_input)
                    obs_input = self.obs_scaler.transform(obs_input)

                obs_mu, obs_logvar = self.obs_net(obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low - obs_next,
                                max=self.obs_high - obs_next,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu,
                                min=self.obs_low - obs_next,
                                max=self.obs_high - obs_next,
                            )
                    else:
                        if self.predict_reward:
                            obs_mu[..., 1:] = torch.clamp(
                                obs_mu.clone()[..., 1:],
                                min=self.obs_low,
                                max=self.obs_high,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu, min=self.obs_low, max=self.obs_high
                            )
                obs_dist = self.dist_fn(obs_mu, obs_logvar.exp())
                out = obs_dist.rsample().float()
                if self.predict_reward:
                    rew, obs_out = out[..., 0], out[..., 1:]
                else:
                    obs_out = out
                    rew = torch.zeros(
                        obs_out.shape[0], dtype=torch.float32, device=self.device
                    )

                if self.zero_reward:
                    # Add demo_reward if on demonstration
                    if t == 0 and self.demo_reward:
                        rew = self.demo_reward * torch.ones_like(rew)
                    else:
                        rew.zero_()
                rewards.append(rew)

                # Done only possible on demonstration, otherwise false
                if t == 0:
                    terminal = to_torch(dones, self.device)
                else:
                    terminal = torch.zeros_like(rew)
                terminals.append(terminal)

                if self.obs_delta:
                    obs = obs_next + obs_out
                else:
                    obs = out

                observations.append(obs)

                obs_next = obs

        # If self.predict_reward=false, rew=0, done=false (except for starts)
        # Ensure dim == 3 (time, start, d)
        traces = Batch(
            obs=torch.stack(observations),
            act=torch.stack(actions),
            rew=torch.stack(rewards),
            done=torch.stack(terminals),
            obs_next=torch.stack(next_observations),
            info={},
        )
        traces.to_numpy()

        return traces
