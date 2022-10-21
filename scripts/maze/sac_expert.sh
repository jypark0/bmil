#!/bin/bash

DEVICE=cuda
# Possible values=
# GoalPointRegionUMaze-v2
# GoalPointRoom5x11-v1
# GoalPointCorridor7x7-v2
# GoalAntRegionUMaze-v2
# GoalAntRoom5x11-v1
# GoalAntCorridor7x7-v2
ENV=GoalPointRegionUMaze-v2
CKPT=""

# Setup
pip install -e .
wandb login "$WANDB_API_KEY"

CMD=(python -u experiments/sac_expert.py
  +experiment=sac_expert/goalenv
  device="$DEVICE"
  env.id="$ENV"
  policy.checkpoint_path="$CKPT"
  logger.wandb.project='sac_expert'
  logger.wandb.name="${ENV}"
)

echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"
