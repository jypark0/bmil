#!/bin/bash

# Convenience script to save demonstrations for both Push and PickAndPlace. Define variables for the path to the expert policy (POLICY_PATH) and the path to save the demonstrations (SAVE_PATH) before execution.

# Push-v2
python save_demos.py --algo tqc --env Push-v2 -f ${POLICY_PATH} --exp-id 0 --load_best --demonstration_path ${SAVE_PATH}/5episodes.pkl --n_episodes 5 --env_kwargs random_gripper:False random_object:False random_goal:False terminate_on_success:True

# PickAndPlace-v2
python save_demos.py --algo tqc --env PickAndPlace-v2 -f ${POLICY_PATH} --exp-id 0 --load_best --demonstration_path ${SAVE_PATH}/10episodes.pkl --n_episodes 10 --env_kwargs random_gripper:False random_object:False random_goal:False terminate_on_success:True
