#!/bin/bash

DEVICE="cuda"
# Possible values=push, pick
TASK="pick"
# Possible values=bc, bmil, vins
METHOD="vins"

# Setup
pip install -e .
wandb login "$WANDB_API_KEY"

CMD=(python -u experiments/"${METHOD}".py
  +experiment="${METHOD}"/fetch_"${TASK}"
  device="$DEVICE"
)

echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"
