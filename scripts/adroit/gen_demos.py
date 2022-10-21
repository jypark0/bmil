import pickle
from pathlib import Path

import click
import d4rl
import gym
import numpy as np
from mjrl.utils.gym_env import GymEnv

DESC = """
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
"""

# MAIN =========================================================
@click.command(help=DESC)
@click.option("--env_name", type=str, help="environment to load", default="relocate-v0")
# @click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option("--num_trajs", type=int, help="Num trajectories", default=1)
@click.option("--demonstration_path", type=str)
@click.option("--policy_checkpoint", type=str)
@click.option(
    "--mode",
    type=str,
    help="exploration or evaluation mode for policy",
    default="evaluation",
)
def main(env_name, mode, num_trajs, policy_checkpoint, demonstration_path, clip=True):
    e = GymEnv(env_name)
    with open(policy_checkpoint, "rb") as f:
        pi = pickle.load(f)
    # render policy
    pol_playback(env_name, pi, demonstration_path, num_trajs, clip=clip)


def extract_params(policy):
    params = policy.trainable_params

    in_shift = policy.model.in_shift.data.numpy()
    in_scale = policy.model.in_scale.data.numpy()
    out_shift = policy.model.out_shift.data.numpy()
    out_scale = policy.model.out_scale.data.numpy()

    fc0w = params[0].data.numpy()
    fc0b = params[1].data.numpy()

    _fc0w = np.dot(fc0w, np.diag(1.0 / in_scale))
    _fc0b = fc0b - np.dot(_fc0w, in_shift)

    assert _fc0w.shape == fc0w.shape
    assert _fc0b.shape == fc0b.shape

    fclw = params[4].data.numpy()
    fclb = params[5].data.numpy()

    _fclw = np.dot(np.diag(out_scale), fclw)
    _fclb = fclb * out_scale + out_shift

    assert _fclw.shape == fclw.shape
    assert _fclb.shape == fclb.shape

    out_dict = {
        "fc0/weight": _fc0w,
        "fc0/bias": _fc0b,
        "fc1/weight": params[2].data.numpy(),
        "fc1/bias": params[3].data.numpy(),
        "last_fc/weight": _fclw,
        "last_fc/bias": _fclb,
        "last_fc_log_std/weight": _fclw,
        "last_fc_log_std/bias": _fclb,
    }
    return out_dict


def pol_playback(env_name, pi, demonstration_path, num_trajs=100, clip=True):
    e = gym.make(env_name, terminate_on_success=True)
    e.reset()

    # Save all episodes to buffer
    buffer = dict(
        obs=[],
        act=[],
        rew=[],
        done=[],
        obs_next=[],
    )

    n_ep = 0

    while n_ep < num_trajs:
        e.reset()
        returns = 0
        obs_ = []
        act_ = []
        rew_ = []
        done_ = []
        obs_next_ = []
        for t in range(e._max_episode_steps):
            obs = e.get_obs()
            obs_.append(obs)

            action, infos = pi.get_action(obs)
            action = pi.get_action(obs)[0]  # eval

            if clip:
                action = np.clip(action, -1, 1)

            act_.append(action)

            obs_next, rew, done, info = e.step(action)
            returns += rew
            rew_.append(rew)
            obs_next_.append(obs_next)

            if t == (e._max_episode_steps - 1):
                done = True
            done_.append(done)

            if done:
                e.reset()
                break

            # e.env.mj_render() # this is much faster
            # e.render()
        print(n_ep, returns, t, info["goal_achieved"])
        if info["goal_achieved"]:
            buffer["obs"].append(np.array(obs_).astype(np.float32))
            buffer["act"].append(np.array(act_).astype(np.float32))
            buffer["rew"].append(np.array(rew_).astype(np.float32))
            buffer["done"].append(np.array(done_).astype(np.bool_))
            buffer["obs_next"].append(np.array(obs_next_).astype(np.float32))
            n_ep += 1
        else:
            print("Episode did not reach goal")

    # write out pickle
    Path(demonstration_path).parent.mkdir(parents=True, exist_ok=True)
    with open(demonstration_path, "wb") as f:
        pickle.dump(buffer, f)

    # write out hdf5 file
    # obs_ = np.array(obs_).astype(np.float32)
    # act_ = np.array(act_).astype(np.float32)
    # rew_ = np.array(rew_).astype(np.float32)
    # done_ = np.array(term_).astype(np.bool_)
    # timeout_ = np.array(timeout_).astype(np.bool_)
    # info_qpos_ = np.array(info_qpos_).astype(np.float32)
    # info_qvel_ = np.array(info_qvel_).astype(np.float32)
    # info_mean_ = np.array(info_mean_).astype(np.float32)
    # info_logstd_ = np.array(info_logstd_).astype(np.float32)
    #
    # if clip:
    #     dataset = h5py.File("%s_expert_clip.hdf5" % env_name, "w")
    # else:
    #     dataset = h5py.File("%s_expert.hdf5" % env_name, "w")
    #
    # # dataset.create_dataset('observations', obs_.shape, dtype='f4')
    # dataset.create_dataset("observations", data=obs_, compression="gzip")
    # dataset.create_dataset("actions", data=act_, compression="gzip")
    # dataset.create_dataset("rewards", data=rew_, compression="gzip")
    # dataset.create_dataset("terminals", data=term_, compression="gzip")
    # dataset.create_dataset("timeouts", data=timeout_, compression="gzip")
    # # dataset.create_dataset('infos/qpos', data=info_qpos_, compression='gzip')
    # # dataset.create_dataset('infos/qvel', data=info_qvel_, compression='gzip')
    # dataset.create_dataset("infos/action_mean", data=info_mean_, compression="gzip")
    # dataset.create_dataset(
    #     "infos/action_log_std", data=info_logstd_, compression="gzip"
    # )
    # for k in info_env_state_:
    #     dataset.create_dataset(
    #         "infos/%s" % k,
    #         data=np.array(info_env_state_[k], dtype=np.float32),
    #         compression="gzip",
    #     )
    #
    # # write metadata
    # policy_params = extract_params(pi)
    # dataset["metadata/algorithm"] = np.string_("DAPG")
    # dataset["metadata/policy/nonlinearity"] = np.string_("tanh")
    # dataset["metadata/policy/output_distribution"] = np.string_("gaussian")
    # for k, v in policy_params.items():
    #     dataset["metadata/policy/" + k] = v


if __name__ == "__main__":
    main()
