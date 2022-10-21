import argparse
import glob
import importlib
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


# Based on file rl-baselines3-zoo/utils/record_video.py
def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument(
        "-f", "--folder", help="Log folder", type=str, default="rl-trained-agents"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument(
        "-n", "--n_timesteps", help="number of timesteps", default=1000, type=int
    )
    parser.add_argument(
        "--num_threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--demonstration_path", help="save filename for demonstrations", type=str
    )
    parser.add_argument(
        "--n_episodes", help="number of successful demonstrations", type=int, default=1
    )
    parser.add_argument("--n_envs", help="number of environments", default=1, type=int)
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load_last_checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--norm_reward",
        action="store_true",
        default=False,
        help="Normalize reward if applicable (trained with VecNormalize)",
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument(
        "--reward_log", help="Where to log reward", default="", type=str
    )
    parser.add_argument(
        "--gym_packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env_kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(
            log_path, f"rl_model_{args.load_checkpoint}_steps.zip"
        )
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(
                f"No checkpoint found for {algo} on {env_id}, path: {log_path}"
            )

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        torch.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(
        stats_path, norm_reward=args.norm_reward, test_mode=True
    )

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(
                f, Loader=yaml.UnsafeLoader
            )  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(
        model_path, env=env, custom_objects=custom_objects, **kwargs
    )

    def squeeze_obs(obs):
        squeezed_obs = deepcopy(obs)
        for k, v in squeezed_obs.items():
            squeezed_obs[k] = np.squeeze(v, axis=0)
        return squeezed_obs

    # Train
    buffer = dict(obs=[], act=[], rew=[], done=[], obs_next=[])
    n_ep = 0
    state = None

    pbar = tqdm(total=args.n_episodes, desc="Episodes")
    while n_ep < args.n_episodes:
        episode = dict(obs=[], act=[], rew=[], done=[], obs_next=[])

        done = False
        obs = env.reset()
        while not done:
            episode["obs"].append(squeeze_obs(obs))
            with torch.no_grad():
                action, state = model.predict(
                    obs, state=state, deterministic=args.deterministic
                )

            episode["act"].append(np.squeeze(action, axis=0))

            obs_next, rew, done, _ = env.step(action)

            episode["rew"].append(np.squeeze(rew, axis=0))
            episode["done"].append(np.squeeze(done, axis=0))
            episode["obs_next"].append(squeeze_obs(obs_next))

            obs = obs_next

        # Print msg if didn't reach goal, don't add to buffer
        pbar.set_postfix(
            {
                "Ep_length": len(episode["obs"]),
                "Ep_reward": np.array(episode["rew"]).sum(),
            }
        )
        if episode["rew"][-1] != 0:
            print("Episode did not reach goal")
            continue

        n_ep += 1
        pbar.update(1)

        # Add episode to buffer
        for k, v in episode.items():
            if k in buffer.keys():
                item = np.asarray(v)
                if item.ndim == 1 and k not in ["obs", "obs_next"]:
                    item = item[:, None]
                buffer[k].append(item)
    pbar.close()
    env.close()

    Path(args.demonstration_path).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Average Rew: {np.concatenate(buffer['rew']).sum() / len(buffer['rew'])}, "
        f"Average Len: {np.concatenate(buffer['rew']).shape[0] / len(buffer['rew'])}"
    )

    # Buffer is structured as buffer["obs"][str(ep_idx)][step, ...]
    # Save replay buffer to disk.
    with open(args.demonstration_path, "wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
