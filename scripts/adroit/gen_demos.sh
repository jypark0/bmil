#!/bin/bash

NUM_EPISODES=20
ENV_ID="AdroitRelocate-v0"
DEMONSTRATION_PATH=data/demos/adroit
# Define POLICY_CHECKPOINT before using
# POLICY_CHECKPOINT="<POLICY_CHECKPOINT_PATH>"

# Setup
pip install -e .

python -u scripts/adroit/gen_demos.py --env_name "$ENV_ID" --policy_checkpoint "$POLICY_CHECKPOINT" --demonstration_path "${DEMONSTRATION_PATH}/${ENV_ID}/${NUM_EPISODES}episodes.pkl" --num_trajs "$NUM_EPISODES"
