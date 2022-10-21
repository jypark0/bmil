import pprint
from pathlib import Path

import hydra
import mujoco_maze
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tianshou.policy import SACPolicy

from src.envs.utils import make_env
from src.logger import get_logger
from src.trainer.sac_expert import SACExpertTrainer
from src.utils import seed_all, xavier_init


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
    log.info(f"Instantiating envs")
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
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    # Add info about obs/act space into wandb.config
    obs_info = {"dim": obs_dim}
    act_info = {"dim": act_dim}
    logger.run.config.env["obs"] = obs_info
    logger.run.config.env["act"] = act_info

    # print(OmegaConf.to_yaml(cfg, resolve=True))
    print(logger.run.config)

    # Policy
    log.info(f"Policy: instantiating actor <{cfg.policy.actor._target_}>")
    net_actor = instantiate(
        cfg.policy.net_a, state_shape=obs_dim, activation=torch.nn.ReLU
    )
    actor = instantiate(
        cfg.policy.actor, preprocess_net=net_actor, action_shape=act_dim
    ).to(cfg.device)
    # Apply xavier init (gain=1, bias=0)
    actor.apply(xavier_init)
    actor_optim = instantiate(cfg.policy.actor_optimizer, params=actor.parameters())

    log.info(f"Policy: instantiating critic1 <{cfg.policy.critic1._target_}>")
    net_critic1 = instantiate(
        cfg.policy.net_c1,
        state_shape=obs_dim,
        action_shape=act_dim,
        activation=torch.nn.ReLU,
    )
    critic1 = instantiate(cfg.policy.critic1, preprocess_net=net_critic1).to(cfg.device)
    # Apply xavier init (gain=1, bias=0)
    critic1.apply(xavier_init)
    critic1_optim = instantiate(
        cfg.policy.critic1_optimizer, params=critic1.parameters()
    )

    log.info(f"Policy: instantiating critic2 <{cfg.policy.critic2._target_}>")
    net_critic2 = instantiate(
        cfg.policy.net_c2,
        state_shape=obs_dim,
        action_shape=act_dim,
        activation=torch.nn.ReLU,
    )
    critic2 = instantiate(cfg.policy.critic2, preprocess_net=net_critic2).to(cfg.device)
    # Apply xavier init (gain=1, bias=0)
    critic2.apply(xavier_init)
    critic2_optim = instantiate(
        cfg.policy.critic2_optimizer, params=critic2.parameters()
    )

    alpha = cfg.policy.auto_alpha.alpha
    if cfg.policy.auto_alpha.on:
        target_entropy = -np.prod(env.action_space.shape)
        # Initialize alpha
        init_value = alpha
        log_alpha = torch.log(
            init_value * torch.ones(1, device=cfg.device)
        ).requires_grad_(True)
        alpha_optim = torch.optim.Adam([log_alpha], lr=cfg.policy.auto_alpha.lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    log.info(f"Policy: instantiating policy <{SACPolicy}>")
    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        **cfg.policy.sac,
        alpha=alpha,
        action_space=env.action_space,
    )
    logger.watch_model(policy)

    # Checkpoint fn
    def checkpoint_fn(save_path, is_best):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Policy
        policy_filename = "policy.pt"
        if is_best:
            policy_filename = "policy-best.pt"

        torch.save(
            {
                "policy": policy.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "critic1_optim": critic1_optim.state_dict(),
                "critic2_optim": critic2_optim.state_dict(),
                "log_alpha": log_alpha,
                "alpha_optim": alpha_optim.state_dict(),
            },
            Path(save_path) / policy_filename,
        )

    def restore_fn(save_path, policy):
        policy_ckpt = torch.load(save_path, map_location=cfg.device)
        policy.load_state_dict(policy_ckpt["policy"])
        actor_optim.load_state_dict(policy_ckpt["actor_optim"])
        critic1_optim.load_state_dict(policy_ckpt["critic1_optim"])
        critic2_optim.load_state_dict(policy_ckpt["critic2_optim"])
        log_alpha = policy_ckpt["log_alpha"]
        alpha_optim.load_state_dict(policy_ckpt["alpha_optim"])

        alpha = (target_entropy, log_alpha, alpha_optim)
        policy.alpha = alpha

    # Stop fn
    def stop_fn(stat):
        if stat > env.spec.reward_threshold:
            return True
        else:
            return False

    # Load from checkpoint if given
    if OmegaConf.select(cfg, "policy.checkpoint_path"):
        log.info(f"Loading best policy from specified checkpoint")
        restore_fn(cfg.policy.checkpoint_path, policy)

    # Collector
    log.info(f"Instantiating buffer and collectors")
    train_buffer = instantiate(cfg.train.buffer)
    train_collector = instantiate(
        cfg.train.collector, policy=policy, env=train_envs, buffer=train_buffer
    )
    policy_test_collector = instantiate(
        cfg.test.collector, policy=policy, env=test_envs
    )

    # trainer
    training_result = SACExpertTrainer(
        policy=policy,
        train_collector=train_collector,
        policy_test_collector=policy_test_collector,
        cfg=cfg,
        logger=logger,
        checkpoint_fn=checkpoint_fn,
        stop_fn=stop_fn,
    )
    train_envs.close()
    test_envs.close()

    log.info("[Training] Results")
    pprint.pprint(training_result)

    logger.close()


if __name__ == "__main__":
    main()
