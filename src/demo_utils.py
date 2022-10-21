import pickle

import numpy as np


def load_demos(filename, env_id):
    with open(filename, "rb") as f:
        demo = pickle.load(f)

    # Post-processing for Fetch envs
    if any(env_id.startswith(prefix) for prefix in ["Push", "Pick"]):
        demo["obs_achieved_goal"] = []
        demo["obs_desired_goal"] = []
        demo["obs_next_achieved_goal"] = []
        demo["obs_next_desired_goal"] = []
        for e in range(len(demo["obs"])):
            demo["obs_achieved_goal"].append([])
            demo["obs_desired_goal"].append([])
            demo["obs_next_achieved_goal"].append([])
            demo["obs_next_desired_goal"].append([])
            for t in range(len(demo["obs"][e])):
                # Observation is an OrderedDict
                # Split keys and flatten trajectories
                obs = demo["obs"][e][t].copy()
                demo["obs"][e][t] = obs["observation"]
                demo["obs_achieved_goal"][-1].append(obs["achieved_goal"])
                demo["obs_desired_goal"][-1].append(obs["desired_goal"])

                obs_next = demo["obs_next"][e][t].copy()
                demo["obs_next"][e][t] = obs_next["observation"]
                demo["obs_next_achieved_goal"][-1].append(obs_next["achieved_goal"])
                demo["obs_next_desired_goal"][-1].append(obs_next["desired_goal"])

            for key in [
                "obs",
                "obs_achieved_goal",
                "obs_desired_goal",
                "obs_next",
                "obs_next_achieved_goal",
                "obs_next_desired_goal",
            ]:
                demo[key][e] = np.vstack(demo[key][e])

    return demo
