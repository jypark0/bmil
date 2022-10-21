#!/bin/bash

NUM_EPISODES=(20)
ENV_IDS=(
  "GoalPointRegionUMaze-v2"
  "GoalPointWallRoom5x11-v1"
  "GoalPointCorridor7x7-v2"
  "GoalAntRegionUMaze-v2"
  "GoalAntWallRoom5x11-v1"
  "GoalAntCorridor7x7-v2"
)
DEMONSTRATION_PATH=data/demos/maze

# Setup
pip install -e .

CMD=(python -u scripts/maze/gen_demos.py
)

for env in "${ENV_IDS[@]}"; do
  # Define POLICY_CHECKPOINT before using
  # POLICY_CHECKPOINT="<POLICY_CHECKPOINT_PATH>"
  echo "[${env}] POLICY_CHECKPOINT: ${POLICY_CHECKPOINT}"
  for n_ep in "${NUM_EPISODES[@]}"; do

    # Deterministic
    echo "[${env}] N_EP: ${n_ep},  Deterministic"
    "${CMD[@]}" --env_id "$env" --policy_checkpoint "$POLICY_CHECKPOINT" --demonstration_path "${DEMONSTRATION_PATH}/${env}/${n_ep}episodes.pkl" --n_episodes "$n_ep" --deterministic_eval true
  done
done
