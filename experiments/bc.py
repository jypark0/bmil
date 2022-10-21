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
from src.trainer.bc import BCTrainer
from src.trainer.utils import evaluate_policy
from src.utils import seed_all


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

    # Collect env information
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
    env = test_envs.workers[0].env
    obs_dim = int(np.prod(env.observation_space.shape))
    if env.action_space.shape:
        act_dim = int(np.prod(env.action_space.shape))
    elif env.action_space.n:
        act_dim = env.action_space.n

    # Add info about obs/act space into cfg and wandb.config
    obs_info = {"dim": obs_dim}
    act_info = {"dim": act_dim}
    logger.run.config.env["obs"] = obs_info
    logger.run.config.env["act"] = act_info

    print(logger.run.config)

    # Demonstrations
    demonstration = load_demos(cfg.demonstration.path, cfg.env.id)
    if any(cfg.env.id.startswith(prefix) for prefix in ["Goal", "Point", "Ant"]):
        demonstration_img = render_maze_demos(demonstration["obs"], env)
        logger.log_image("imgs/demonstration", demonstration_img)

    # Replay Buffer
    log.info(f"Buffer: instantiating buffers")
    demo_buffer = instantiate(cfg.demonstration.buffer)
    for e in range(len(demonstration["obs"])):
        for t in range(len(demonstration["obs"][e])):
            demo_batch = Batch(
                obs=demonstration["obs"][e][t].astype(np.float32),
                act=demonstration["act"][e][t].astype(np.float32),
                rew=demonstration["rew"][e][t].astype(np.float32).squeeze(),
                done=demonstration["done"][e][t].astype(bool).squeeze(),
                obs_next=demonstration["obs_next"][e][t].astype(np.float32),
            )

            # Add demonstration to buffer
            for _ in range(cfg.demonstration.repeat):
                demo_buffer.add(demo_batch)

    # Policy
    net = instantiate(
        cfg.policy.net,
        state_shape=obs_dim,
        activation=torch.nn.ReLU,
    )
    actor = instantiate(cfg.policy.actor, preprocess_net=net, action_shape=act_dim).to(
        cfg.device
    )
    optim = instantiate(cfg.policy.optimizer, params=actor.parameters())

    log.info(f"Policy: instantiating policy <{cfg.policy.bc._target_}>")
    policy = instantiate(
        cfg.policy.bc, model=actor, optim=optim, action_space=env.action_space
    )
    logger.watch_model(policy)

    # Checkpoint fn
    def checkpoint_fn(save_path, **kwargs):
        # Policy
        policy_filename = "policy.pt"
        torch.save(
            {"policy": policy.state_dict(), "optim": optim.state_dict()},
            Path(save_path) / policy_filename,
        )

    # trainer
    timestamp = BCTrainer(
        policy=policy,
        demo_buffer=demo_buffer,
        cfg=cfg,
        logger=logger,
        checkpoint_fn=checkpoint_fn,
    )

    # Evaluate on test_envs
    log.info("[Test]")
    policy_test_collector = instantiate(
        cfg.test.collector, policy=policy, env=test_envs
    )
    test_result = evaluate_policy(
        policy=policy,
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
        evaluate_fetch(policy, cfg, env, logger, timestamp)
    elif any(cfg.env.id.startswith(prefix) for prefix in ["Point", "Ant"]):
        evaluate_maze(policy, cfg, env, logger, timestamp)
    elif cfg.env.id.startswith("Adroit"):
        evaluate_adroit(policy, cfg, logger, timestamp)

    logger.close()


if __name__ == "__main__":
    main()
    wandb.finish()
