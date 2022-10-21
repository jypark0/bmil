import itertools
from typing import Type

import numpy as np
import torch
from tianshou.data import Batch

from src.utils import to_torch

ModuleType = Type[torch.nn.Module]

SIGMA_MIN = -10
SIGMA_MAX = 0.5


class ForwardModel(torch.nn.Module):
    def __init__(
        self,
        obs_net,
        optim,
        dist_fn,
        device,
        observation_space=None,
        bounded_obs=False,
        lr_scheduler=None,
        update_method="fixed_iter",
        obs_delta=False,
        improvement_threshold=0.001,
    ):
        super().__init__()

        self.obs_net = obs_net
        self.optim = optim
        self.dist_fn = dist_fn
        self.device = device

        if observation_space is not None:
            self.obs_low = torch.from_numpy(observation_space.low).float().to(device)
            self.obs_high = torch.from_numpy(observation_space.high).float().to(device)

        self.bounded_obs = bounded_obs
        self.lr_scheduler = lr_scheduler
        self.update_method = update_method
        self.obs_delta = obs_delta
        self.improvement_threshold = improvement_threshold

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

        batch.obs_input = np.hstack([batch.obs, batch.act])
        if self.obs_delta:
            batch.obs_label = batch.obs_next - batch.obs
        else:
            batch.obs_label = batch.obs_next

        result = {
            "update/grad_step": 0,
            "update/epoch": 0,
            "train/loss": [],
            "train/obs/mse": [],
            "train/obs/var": [],
            "train/obs/var_bounds": [],
        }
        for _ in range(n_updates):
            for b in batch.split(batch_size, merge_last=True):
                b.to_torch(dtype=torch.float32, device=self.device)

                # Obs_dist
                obs_mu, obs_logvar = self.obs_net(b.obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        obs_mu = torch.clamp(
                            obs_mu.clone(),
                            min=self.obs_low - b.obs,
                            max=self.obs_high - b.obs,
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
                train_loss = obs_mse_loss + obs_var_loss + 1e-2 * obs_var_bounds_loss

                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                result["update/grad_step"] += 1
                result["train/loss"].append(train_loss.item())
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

        batch.obs_input = np.hstack([batch.obs, batch.act])
        if self.obs_delta:
            batch.obs_label = batch.obs_next - batch.obs
        else:
            batch.obs_label = batch.obs_next

        train_idx = int(len(batch) * (1 - holdout_ratio))

        # Shuffle train/test
        train_batch, test_batch = tuple(batch.split(train_idx, shuffle=True))

        result = {
            "update/grad_step": 0,
            "update/epoch": 0,
            "train/loss": [],
            "train/obs/mse": [],
            "train/obs/var": [],
            "train/obs/var_bounds": [],
            "test/loss": [],
            "test/obs/mse": [],
        }

        best_test_loss = 1e4
        epochs_since_update = 0

        for _ in itertools.count():
            for b in train_batch.split(batch_size, merge_last=True):
                b.to_torch(dtype=torch.float32, device=self.device)

                # Obs_dist
                obs_mu, obs_logvar = self.obs_net(b.obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        obs_mu = torch.clamp(
                            obs_mu.clone(),
                            min=self.obs_low - b.obs,
                            max=self.obs_high - b.obs,
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
                train_loss = obs_mse_loss + obs_var_loss + 1e-2 * obs_var_bounds_loss

                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                result["update/grad_step"] += 1
                result["train/loss"].append(train_loss.item())
                result["train/obs/mse"].append(obs_mse_loss.item())
                result["train/obs/var"].append(obs_var_loss.item())
                result["train/obs/var_bounds"].append(obs_var_bounds_loss.item())

            # Evaluate model on test_batch
            with torch.no_grad():
                for b in test_batch.split(batch_size, merge_last=True):
                    b.to_torch(dtype=torch.float32, device=self.device)

                    # Obs_dist
                    obs_mu, obs_logvar = self.obs_net(b.obs_input)
                    if self.bounded_obs:
                        if self.obs_delta:
                            obs_mu = torch.clamp(
                                obs_mu.clone(),
                                min=self.obs_low - b.obs,
                                max=self.obs_high - b.obs,
                            )
                        else:
                            obs_mu = torch.clamp(
                                obs_mu.clone(), min=self.obs_low, max=self.obs_high
                            )

                    # Calculate mse_loss
                    obs_mse_loss = torch.mean(torch.pow(obs_mu - b.obs_label, 2))

                    # Calculate loss
                    test_loss = obs_mse_loss

                    result["test/loss"].append(test_loss.item())
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

    def generate_trace(self, trace_start, policy, trace_size, samples_per_start):
        # Reshape obs from (n_valuable_states,d) -> (n_valuable_states * samples_per_start, d)
        trace_start = np.tile(trace_start, (samples_per_start, 1))

        obs = to_torch(trace_start, self.device)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        with torch.no_grad():
            for t in range(trace_size):
                observations.append(obs)

                act = policy.forward(Batch(obs=obs, info={})).act
                actions.append(act)

                # Obs_dist
                obs_input = torch.cat([obs, act], dim=-1)

                obs_mu, obs_logvar = self.obs_net(obs_input)
                if self.bounded_obs:
                    if self.obs_delta:
                        obs_mu = torch.clamp(
                            obs_mu,
                            min=self.obs_low - obs,
                            max=self.obs_high - obs,
                        )
                    else:
                        obs_mu = torch.clamp(
                            obs_mu, min=self.obs_low, max=self.obs_high
                        )
                obs_dist = self.dist_fn(obs_mu, obs_logvar.exp())
                obs_out = obs_dist.rsample().float()

                if self.obs_delta:
                    obs_next = obs + obs_out
                else:
                    obs_next = obs_out
                next_observations.append(obs_next)

                obs = obs_next

                rew = torch.zeros(obs.shape[0], dtype=torch.float32, device=self.device)
                rewards.append(rew)

                terminal = torch.zeros_like(rew)
                terminals.append(terminal)

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
