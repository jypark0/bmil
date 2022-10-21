from functools import partial
from pathlib import Path

import d4rl
import hydra
import mujoco_maze
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Batch

from src.demo_utils import load_demos
from src.envs.utils import make_env
from src.evaluate import evaluate_adroit, evaluate_fetch, evaluate_maze
from src.logger import get_logger
from src.plot.maze import render_demonstrations as render_maze_demos
from src.trainer.utils import evaluate_policy
from src.trainer.vins import VINSTrainer
from src.utils import seed_all, trunc_normal_init, uniform_init, xavier_init


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log = get_logger(__name__)

    # Seed
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Init WandBLogger
    tags = None if cfg.logger.wandb.get("id") else cfg.logger.wandb.tags
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    log.info(f"Instantiating logger <{cfg.logger.wandb._target_}>")
    logger = instantiate(cfg.logger.wandb, wandb_config, tags=tags)

    # Env
    train_envs = instantiate(
        cfg.train.env,
        env_fns=[
            make_env(
                env_id=cfg.env.id,
                wrappers=cfg.env.wrappers,
                seed=cfg.seed + i,
                env_kwargs=cfg.env.init,
            )
            for i in range(cfg.train.n_envs)
        ],
    )
    test_envs = instantiate(
        cfg.test.env,
        env_fns=[
            make_env(
                env_id=cfg.env.id,
                wrappers=cfg.env.wrappers,
                seed=None,
                env_kwargs=cfg.env.init,
            )
            for _ in range(cfg.test.n_envs)
        ],
    )

    # Collect env information
    env = train_envs.workers[0].env
    obs_dim = int(np.prod(env.observation_space.shape))
    if env.action_space.shape:
        act_dim = int(np.prod(env.action_space.shape))
    elif env.action_space.n:
        act_dim = env.action_space.n

    # Add info about obs/act space into wandb.config
    obs_info = {"dim": obs_dim}
    act_info = {"dim": act_dim}
    logger.run.config.env["obs"] = obs_info
    logger.run.config.env["act"] = act_info
    print(logger.run.config)

    # Demonstration
    demonstration = load_demos(cfg.demonstration.path, cfg.env.id)
    if any(cfg.env.id.startswith(prefix) for prefix in ["Goal", "Point", "Ant"]):
        demonstration_img = render_maze_demos(demonstration["obs"], env)
        logger.log_image("imgs/demonstration", demonstration_img)

    # Replay Buffer
    log.info(f"Buffer: instantiating buffers")
    demos = []
    train_buffer = instantiate(cfg.train.buffer)
    for e in range(len(demonstration["obs"])):
        for t in range(len(demonstration["obs"][e])):
            demo_batch = Batch(
                obs=demonstration["obs"][e][t].astype(np.float32),
                act=demonstration["act"][e][t].astype(np.float32),
                rew=demonstration["rew"][e][t].astype(np.float32).squeeze(),
                done=demonstration["done"][e][t].astype(bool).squeeze(),
                obs_next=demonstration["obs_next"][e][t].astype(np.float32),
            )

            demos.append(demo_batch)

            # Add demonstration to buffer
            for _ in range(cfg.demonstration.repeat):
                train_buffer.add(demo_batch)
    demos = Batch.stack(demos)

    # Compute demonstration obs stddev
    stddev = demos.obs.std(axis=0)
    perturb_cls = instantiate(cfg.policy.perturb, env=env)
    perturb_fn = partial(perturb_cls.perturb, stddev=stddev)
    logger.run.config.demonstration["stddev"] = stddev

    # Value
    value_net = instantiate(
        cfg.policy.value_net,
        state_shape=obs_dim,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-12),
        activation=torch.nn.ReLU,
    )
    value_model = instantiate(
        cfg.policy.value_model,
        preprocess_net=value_net,
    ).to(cfg.device)
    value_model.apply(xavier_init)
    value_model.last.apply(partial(uniform_init, a=-3e-3, b=3e-3))
    value_optim = instantiate(
        cfg.policy.value_optimizer, params=value_model.parameters()
    )
    value_function = instantiate(
        cfg.policy.value_function,
        value_model=value_model,
        optim=value_optim,
        perturb_fn=perturb_fn,
    )
    # Fit value normalizer
    value_function.scaler.fit(demos.obs)

    # Dynamics
    dynamics_net = instantiate(
        cfg.policy.dynamics_net,
        state_shape=obs_dim,
        action_shape=act_dim,
        activation=torch.nn.ReLU,
        concat=True,
    )
    dynamics_model = instantiate(
        cfg.policy.dynamics_model,
        preprocess_net=dynamics_net,
        observation_shape=obs_dim,
    ).to(cfg.device)
    dynamics_model.apply(partial(trunc_normal_init, std=1e-2))
    dynamics_optim = instantiate(
        cfg.policy.dynamics_optimizer, params=dynamics_model.parameters()
    )
    dynamics_model.optim = dynamics_optim
    # Fit dynamics normalizer
    dynamics_model.scaler.fit(demos.obs_next - demos.obs)

    bc_net = instantiate(
        cfg.bc_policy.net, state_shape=obs_dim, activation=torch.nn.ReLU
    )
    bc_actor = instantiate(
        cfg.bc_policy.actor, preprocess_net=bc_net, action_shape=act_dim
    ).to(cfg.device)
    bc_actor.apply(xavier_init)
    bc_optim = instantiate(cfg.bc_policy.optimizer, params=bc_actor.parameters())
    bc_policy = instantiate(
        cfg.bc_policy.bc, model=bc_actor, optim=bc_optim, action_space=env.action_space
    )

    if OmegaConf.select(cfg, "bc_policy.path"):
        bc_policy_ckpt = torch.load(cfg.bc_policy.path, map_location=cfg.device)
        bc_policy.load_state_dict(bc_policy_ckpt["policy"])

    # Imitator
    imitator = instantiate(
        cfg.policy.imitator,
        value_function=value_function,
        dynamics_model=dynamics_model,
        bc_policy=bc_policy,
        action_shape=act_dim,
    )

    # Collector
    train_collector = instantiate(
        cfg.train.collector, policy=imitator, env=train_envs, buffer=train_buffer
    )
    bc_test_collector = instantiate(cfg.test.collector, policy=bc_policy, env=test_envs)
    policy_test_collector = instantiate(
        cfg.test.collector, policy=imitator, env=test_envs
    )

    # Checkpoint fn
    def checkpoint_fn(save_path, is_best):
        # Policy
        policy_filename = "policy.pt"
        if is_best:
            policy_filename = "policy-best.pt"

        # Policy
        torch.save(
            {
                "value_function": value_function.state_dict(),
                "value_optim": value_optim.state_dict(),
                "dynamics_model": dynamics_model.state_dict(),
                "dynamics_optim": dynamics_optim.state_dict(),
                "bc_policy": bc_policy.state_dict(),
                "bc_optim": bc_optim.state_dict(),
                "imitator": imitator.state_dict(),
            },
            Path(save_path) / policy_filename,
        )

    # trainer
    timestamp = VINSTrainer(
        value_function=value_function,
        dynamics_model=dynamics_model,
        bc_policy=bc_policy,
        imitator=imitator,
        train_collector=train_collector,
        policy_test_collector=policy_test_collector,
        bc_test_collector=bc_test_collector,
        cfg=cfg,
        logger=logger,
        checkpoint_fn=checkpoint_fn,
    )
    train_envs.close()

    # Evaluate on eval_envs
    log.info("[Test]")
    test_result = evaluate_policy(
        policy=imitator,
        collector=policy_test_collector,
        n_episode=cfg.test.n_ep,
        collect_kwargs={"disable_tqdm": False},
    )
    logger.write({"test/success_ratio": test_result["success_ratio"]}, timestamp)
    log.info(f"[Test] success_ratio: {test_result['success_ratio']:.3f}")
    log.info(
        f"[Test] Reward: {test_result['rew']:.2f} +/- {test_result['rew_std']:.2f}"
    )
    log.info(
        f"[Test] Length: {test_result['len']:.2f} +/- {test_result['len_std']:.2f}"
    )
    test_envs.close()

    # Evaluate on eval_envs
    if any(cfg.env.id.startswith(prefix) for prefix in ["Push", "Pick"]):
        evaluate_fetch(imitator, cfg, env, logger, timestamp)
    elif any(cfg.env.id.startswith(prefix) for prefix in ["Point", "Ant"]):
        evaluate_maze(imitator, cfg, env, logger, timestamp)
    elif cfg.env.id.startswith("Adroit"):
        evaluate_adroit(imitator, cfg, logger, timestamp)

    logger.close()


if __name__ == "__main__":
    main()
    wandb.finish()
