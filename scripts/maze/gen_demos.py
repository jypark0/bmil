import argparse
import pickle
from distutils.util import strtobool
from pathlib import Path

import mujoco_maze
import numpy as np
import torch
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tqdm import tqdm

from src.envs.utils import make_env
from src.utils import seed_all, to_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_checkpoint", type=str)
    parser.add_argument("--demonstration_path", type=str)
    parser.add_argument("--env_id", type=str)
    parser.add_argument(
        "--deterministic_eval",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(args=parse_args()):
    # Env
    envs = DummyVectorEnv(
        [
            make_env(
                args.env_id,
                wrappers=[{"gym.wrappers.FlattenObservation": {}}],
                seed=args.seed,
            )
            for _ in range(1)
        ]
    )

    env = envs.workers[0].env
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    # Policy
    device = "cpu"
    hidden_sizes = [256, 256, 256]

    net_a = Net(obs_dim, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        act_dim,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    net_c1 = Net(
        obs_dim, act_dim, hidden_sizes=hidden_sizes, concat=True, device=device
    )
    net_c2 = Net(
        obs_dim, act_dim, hidden_sizes=hidden_sizes, concat=True, device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=0.005,
        gamma=0.99,
        alpha=0,
        estimation_step=1,
        action_space=env.action_space,
        deterministic_eval=args.deterministic_eval,
    )

    policy_ckpt = torch.load(args.policy_checkpoint, map_location=device)
    policy.load_state_dict(policy_ckpt["policy"])
    policy.eval()

    seed_all(args.seed)

    data = Batch(
        obs={},
        act={},
        rew={},
        done={},
        obs_next={},
        info={},
        policy={},
    )
    # Save all episodes to buffer
    buffer = dict(
        obs=[],
        obs_achieved_goal=[],
        obs_desired_goal=[],
        act=[],
        rew=[],
        done=[],
        obs_next=[],
        obs_next_achieved_goal=[],
        obs_next_desired_goal=[],
    )

    # Train
    # Policy forward() expects Batch with batch dimension
    n_ep = 0
    pbar = tqdm(total=args.n_episodes, desc="Episodes")
    while n_ep < args.n_episodes:
        episode = Batch()

        done = False
        data.obs = envs.reset()
        # data.obs = obs[0]
        while not done:
            with torch.no_grad():
                result = policy(data)

            act = to_np(result.act)
            data.update(act=act)

            # Save original action to buffer, use remap only for step()
            action_remap = policy.map_action(act)
            obs_next, rew, done, info = envs.step(action_remap)

            data.update(obs_next=obs_next, rew=rew, done=done, info=info)

            episode = Batch.cat([episode, data])

            data.obs = data.obs_next

        # Print msg if didn't reach goal, don't add to buffer
        pbar.set_postfix({"Ep_length": len(episode)})
        # print(f"Ep_len: {len(episode)}")
        if len(episode) == envs.spec[0].max_episode_steps:
            print("Episode did not reach goal")
            continue

        n_ep += 1
        pbar.update(1)

        # Relabel rewards to be sparse (1000 for goal, 0 otherwise)
        rew_relabel = np.zeros_like(episode.rew)
        rew_relabel[episode.done] = env.task.goal_reward
        episode.rew = rew_relabel

        # Split obs and obs_next
        episode.obs_achieved_goal = episode.obs[..., -2:]
        episode.obs_desired_goal = episode.obs[..., -4:-2]
        episode.obs = episode.obs[..., :-4]

        episode.obs_next_achieved_goal = episode.obs_next[..., -2:]
        episode.obs_next_desired_goal = episode.obs_next[..., -4:-2]
        episode.obs_next = episode.obs_next[..., :-4]

        # Add episode to buffer
        for k, v in episode.items():
            if k in buffer.keys():
                buffer[k].append(v)

    pbar.close()
    envs.close()

    print(
        f"Average Rew: {np.concatenate(buffer['rew']).sum() / len(buffer['rew'])}, "
        f"Average Len: {np.concatenate(buffer['rew']).shape[0] / len(buffer['rew'])}"
    )

    # Buffer is structured as buffer["obs"][str(ep_idx)][step, ...]
    # Save replay buffer to disk.
    Path(args.demonstration_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.demonstration_path, "wb") as f:
        pickle.dump(buffer, f)


if __name__ == "__main__":
    main()
