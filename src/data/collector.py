import time
import warnings
from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np
import torch
from tianshou.data import Batch, Collector, ReplayBuffer, to_numpy
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from tqdm import tqdm


class TqdmPosCollector(Collector):
    """
    Modified original tianshou collector to use tqdm if wanted,
    and to save the initial positions and success rates
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

        env_id = env.spec[0].id
        # Fetch
        if any(env_id.startswith(prefix) for prefix in ["Push", "Pick"]):
            self.pos_idxs = np.s_[0:3]

        # Maze
        elif any(env_id.startswith(prefix) for prefix in ["Point", "Ant"]):
            self.pos_idxs = np.s_[0:2]
        # Adroit
        elif env_id.startswith("Adroit"):
            # Not used for Adroit environments
            self.pos_idxs = np.array([0, 2])
        else:
            raise NotImplementedError

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        forward_kwargs: dict = {},
        disable_tqdm: bool = True,
    ) -> Dict[str, Any]:
        """Add tqdm to collect() and gather init_pos and success_ratio"""

        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_success = []
        episode_start_indices = []
        episode_start_pos = []

        # Append initial pos
        start_pos = self.data.obs[..., self.pos_idxs]

        if n_step is not None:
            pbar = tqdm(total=n_step, desc="Step", disable=disable_tqdm)
        elif n_episode is not None:
            pbar = tqdm(total=n_episode, desc="Episode", disable=disable_tqdm)

        with pbar:
            while True:
                assert len(self.data) == len(ready_env_ids)
                # restore the state: if the last state is None, it won't store
                last_state = self.data.policy.pop("hidden_state", None)

                # get the next action
                if random:
                    try:
                        act_sample = [
                            self._action_space[i].sample() for i in ready_env_ids
                        ]
                    except TypeError:  # envpool's action space is not for per-env
                        act_sample = [
                            self._action_space.sample() for _ in ready_env_ids
                        ]
                    self.data.update(act=act_sample)
                else:
                    if no_grad:
                        with torch.no_grad():  # faster than retain_grad version
                            # self.data.obs will be used by agent to get result
                            result = self.policy(
                                self.data, last_state, **forward_kwargs
                            )
                    else:
                        result = self.policy(self.data, last_state, **forward_kwargs)
                    # update state / act / policy into self.data
                    policy = result.get("policy", Batch())
                    assert isinstance(policy, Batch)
                    state = result.get("state", None)
                    if state is not None:
                        policy.hidden_state = state  # save state into buffer
                    act = to_numpy(result.act)
                    if self.exploration_noise:
                        act = self.policy.exploration_noise(act, self.data)
                    self.data.update(policy=policy, act=act)

                # get bounded and remapped actions first (not saved into buffer)
                action_remap = self.policy.map_action(self.data.act)
                # step in env
                result = self.env.step(action_remap, ready_env_ids)  # type: ignore
                obs_next, rew, done, info = result

                self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
                if self.preprocess_fn:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            done=self.data.done,
                            info=self.data.info,
                            policy=self.data.policy,
                            env_id=ready_env_ids,
                        )
                    )

                if render:
                    self.env.render()
                    if render > 0 and not np.isclose(render, 0):
                        time.sleep(render)

                # add data into the buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.data, buffer_ids=ready_env_ids
                )

                # collect statistics
                step_count += len(ready_env_ids)

                if n_step is not None:
                    pbar.update(len(ready_env_ids))

                if np.any(done):
                    env_ind_local = np.where(done)[0]
                    env_ind_global = ready_env_ids[env_ind_local]
                    episode_count += len(env_ind_local)
                    episode_lens.append(ep_len[env_ind_local])
                    episode_rews.append(ep_rew[env_ind_local])
                    episode_start_indices.append(ep_idx[env_ind_local])
                    # episode_success.append(
                    #     [np.sum(r) != -50.0 for r in ep_rew[env_ind_local]]
                    # )
                    episode_success.append(
                        [
                            ep_info.get("is_success", 0)
                            for ep_info in info[env_ind_local]
                        ]
                    )
                    episode_start_pos.append(start_pos[env_ind_local])
                    # now we copy obs_next to obs, but since there might be
                    # finished episodes, we have to reset finished envs first.
                    obs_reset = self.env.reset(env_ind_global)
                    if self.preprocess_fn:
                        obs_reset = self.preprocess_fn(
                            obs=obs_reset, env_id=env_ind_global
                        ).get("obs", obs_reset)
                    self.data.obs_next[env_ind_local] = obs_reset
                    start_pos[env_ind_local] = obs_reset[..., self.pos_idxs]
                    for i in env_ind_local:
                        self._reset_state(i)

                    # remove surplus env id from ready_env_ids
                    # to avoid bias in selecting environments
                    if n_episode:
                        surplus_env_num = len(ready_env_ids) - (
                            n_episode - episode_count
                        )
                        if surplus_env_num > 0:
                            mask = np.ones_like(ready_env_ids, dtype=bool)
                            mask[env_ind_local[:surplus_env_num]] = False
                            ready_env_ids = ready_env_ids[mask]
                            self.data = self.data[mask]
                            start_pos = start_pos[mask]
                    if n_episode is not None:
                        pbar.update(len(env_ind_local))

                self.data.obs = self.data.obs_next

                if (n_step and step_count >= n_step) or (
                    n_episode and episode_count >= n_episode
                ):
                    break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(np.concatenate, [episode_rews, episode_lens, episode_start_indices])
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        if episode_count > 0:
            init_pos = np.concatenate(episode_start_pos)
            success = np.concatenate(episode_success)
            success_ratio = np.sum(success) / episode_count
        else:
            init_pos = np.array([])
            success = np.array([])
            success_ratio = 0.0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "init_pos": init_pos,
            "success": success,
            "success_ratio": success_ratio,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }
