#!/bin/bash

DEVICE="cuda"
# Possible values=
# pointregionumaze
# point5x11
# point7x7
# antregionumaze
# ant5x11
# ant7x7
TASK="antregionumaze"
# Possible values=bc, bmil, vins
METHOD="bmil"

# Setup
pip install -e .
wandb login "$WANDB_API_KEY"

CMD=(python -u experiments/"${METHOD}".py
  +experiment="${METHOD}"/maze_"${TASK}"
  policy.train.n_epoch=20
  device="$DEVICE"
)

echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"
