# Robust Imitation of a Few Demonstrations with a Backwards Model

This repository implements code for the paper [Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337).

## Citation

If you find this work useful, please cite the paper as follows:
```
@inproceedings{park2022bmil,
  title={Robust Imitation of a Few Demonstrations with a Backwards Model},
  author={Park, Jung Yeon and Wong, Lawson L.S.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022},
  url={https://arxiv.org/abs/2210.09337}
}
```

## Requirements

- Python 3.7+ (used 3.9)
- MuJoCo 2.1.0 (see https://github.com/openai/mujoco-py for installation instructions)
- See `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

The requirements include additional libraries such as [mujoco-maze](https://github.com/jypark0/mujoco-maze) and [d4rl](https://github.com/Farama-Foundation/D4RL) for the Maze and Adroit environments respectively.
If the packages do not install, clone the repositories directly and run `pip install <path to repo>`.

`mujoco-maze` is forked from https://github.com/kngwyu/mujoco-maze and edited to include the custom environments and their goal-oriented versions. See its `README.md` for more details.

We use [Weights and Biases ](https://wandb.ai/site) to track experiment results and [Hydra](https://hydra.cc/) for configuration management.

## Expert Policies/Demonstrations

### Fetch

For the Fetch environments, we use the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo.git) repository . We use the same hyperparameter settings as in the repo and train HER+TQC until convergence on `gym==0.21.0`.

Some small modifications need to be made to the `rl-baselines3-zoo` repo. Copy scripts `scripts/fetch/gen_demos.py` and `scripts/fetch/gen_demos.sh`  to the top-level folder containing the `rl-baselines3-zoo` code. Place custom environments for `Push-v2` and `PickAndPlace-v2` inside the `rl-baselines3-zoo` repo and modify the hyperparameter config file (see the [rl-baselines3-zoo README](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/README.md) for more details). Then one can train expert policies using the `train.py` script. To generate demonstrations, run `save_demos.sh` inside the `rl-baselines3-zoo` folder.

### Maze

For the Maze environments, we use goal-oriented SAC [[1]](#1), where the goal is given as part of the observation. We use the GoalEnv interface provided in `gym`.

To train the expert policy, run the provided script:

```bash
bash scripts/maze/sac_expert.sh
```

Change the variable `env_id` to the desired environment ID. The prefix 'Goal' is used to indicate that the environment is goal-oriented.
The specific environment IDs are:
- GoalPointRegionUMaze-v2
- GoalPointRoom5x11-v1
- GoalPointCorridor7x7-v2
- GoalAntRegionUMaze-v2
- GoalAntRoom5x11-v1
- GoalAntCorridor7x7-v2

To generate demonstrations, see files `scripts/maze/gen_demos.py` and `scripts/maze/gen_demos.sh`.

### Adroit

For the Adroit environment, we use the pre-trained policy given in DAPG [[2]](#2) [policy_checkpoint](https://github.com/aravindr93/hand_dapg/blob/master/dapg/policies/relocate-v0.pickle). To generate demonstrations, see `scripts/adroit/gen_demos.py` and `scripts/adroit/gen_demos.sh`.

## Experiments

### Fetch

To train and evaluate BMIL, run the following command:
```bash
python experiments/bmil.py +experiments=bmil/fetch_pick
```

A run script is provided in `scripts/fetch/run.sh`. Change the `TASK` and `METHOD` variables accordingly.

### Maze

To train and evaluate BMIL, run the following command:
```bash
python experiments/bmil.py +experiments=bmil/maze_point5x11
```

A run script is provided in `scripts/maze/run.sh`. Change the `TASK` and `METHOD` variables accordingly.

### Adroit

To train and evaluate BMIL, run the following command:
```bash
python experiments/bmil.py +experiments=bmil/adroit_relocate
```

A run script is provided in `scripts/adroit/run.sh`. Change the `METHOD` variable accordingly.


## References

<a id="1">[1]</a>
[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018.

<a id="2">[2]</a>
[Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087), Rajeswaran et al, 2018.
