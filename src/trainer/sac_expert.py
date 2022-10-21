import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import tqdm
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Collector
from tianshou.policy import SACPolicy
from tianshou.trainer.utils import gather_info
from tianshou.utils import MovAvg

from src.logger import WandBLogger, get_logger
from src.trainer.utils import evaluate_policy


def SACExpertTrainer(
    policy: SACPolicy,
    train_collector: Collector,
    policy_test_collector: Collector,
    cfg: Union[Namespace, DictConfig],
    logger: Optional[WandBLogger] = None,
    checkpoint_fn: Optional[Callable[[None], None]] = None,
    stop_fn=None,
) -> Dict[str, Union[float, str]]:

    log = get_logger(__name__)
    timestamp = {"epoch": 0, "env_ep": 0, "env_step": 0, "policy_grad_step": 0}

    # Collect random samples in the beginning
    if (
        OmegaConf.select(cfg, "policy.train.start_timesteps")
        and cfg.policy.train.start_timesteps > 0
        and not cfg.policy.checkpoint_path
    ):
        log.info(
            f"Precollecting {cfg.policy.train.start_timesteps} steps from random policy"
        )
        precollect_result = train_collector.collect(
            n_step=cfg.policy.train.start_timesteps, random=True
        )
        timestamp["env_ep"] += precollect_result["n/ep"]
        timestamp["env_step"] += precollect_result["n/st"]

    train_collector.reset_stat()
    if policy_test_collector is not None:
        policy_test_collector.reset_stat()

    env_spec = train_collector.env.spec[0]
    env_id = env_spec.id
    best_result = {
        "epoch": 0,
        "rew": float("-inf"),
        "rew_std": 0,
        "len": float("inf"),
        "len_std": 0,
    }

    # For stop_fn
    reward_stat = MovAvg(size=5)

    last_rew, last_len = 0.0, 0
    save_path = Path(logger.run.dir) / env_id

    start_time = time.time()

    for epoch in range(timestamp["epoch"], cfg.policy.train.n_epoch):
        pbar = tqdm.tqdm(
            total=cfg.policy.train.step_per_epoch,
            desc=f"[{env_id}] Epoch #{epoch}",
            file=sys.stdout,
        )
        policy.train()
        with pbar:
            while pbar.n < pbar.total:
                # Collect policy rollouts
                train_result = train_collector.collect(
                    n_step=cfg.policy.train.step_per_collect,
                )
                logger.log_train_steps("train", train_result, timestamp)

                # Update timestamps
                pbar.update(train_result["n/st"])
                timestamp["env_ep"] += train_result["n/ep"]
                timestamp["env_step"] += train_result["n/st"]

                # Create data
                if train_result["n/ep"] > 0:
                    last_rew = train_result["rews"].mean()
                    last_len = train_result["lens"].mean()
                data = {
                    "env_step": str(timestamp["env_step"]),
                    "rew": f"{last_rew:.2f}",
                    "len": f"{last_len:.2f}",
                }
                pbar.set_postfix(**data)

                # Update policy
                for _ in range(cfg.policy.train.update_per_collect):
                    policy_losses = policy.update(
                        cfg.policy.batch_size, train_collector.buffer
                    )
                    timestamp["policy_grad_step"] += 1

                # Log only last loss
                logger.log_all("policy", policy_losses, timestamp)
        policy.eval()

        # Test
        if (
            cfg.test.epoch_frequency != 0
            and (epoch + 1) % cfg.test.epoch_frequency == 0
        ):
            test_result = evaluate_policy(
                policy=policy,
                collector=policy_test_collector,
                n_episode=cfg.test.n_ep,
            )
            logger.log_all(
                "test",
                {
                    k: v
                    for k, v in test_result.items()
                    if k in ["len", "rew", "rew_std", "len_std"]
                },
                timestamp,
            )
            test_rew, test_rew_std = test_result["rew"], test_result["rews"].std()
            test_len, test_len_std = test_result["len"], test_result["lens"].std()
            # Update best records (use episode length)
            if test_len <= best_result["len"]:
                best_result = {
                    "epoch": timestamp["epoch"],
                    "rew": test_rew,
                    "rew_std": test_rew_std,
                    "len": test_len,
                    "len_std": test_len_std,
                }
                logger.log_all("best", best_result, timestamp)

                if checkpoint_fn:
                    log.info(f"[Epoch {epoch}] Saving best checkpoint")
                    checkpoint_fn(save_path, is_best=True)
            else:
                checkpoint_fn(save_path, is_best=False)

            log.info(
                f"[Epoch {epoch}] Test: rew={test_rew:.2f}, " f"len={test_len:.2f}"
            )
            log.info(
                f"[Epoch {epoch}] Best: rew={best_result['rew']:.2f}, "
                f"len={best_result['len']:.2f} (from epoch {best_result['epoch']})"
            )

            # Check stopping criterion
            reward_stat.add(test_result["rew"])
            if epoch > 4 and stop_fn(reward_stat.get()):
                log.info(
                    f"[Epoch {epoch}] Stopping: Reward average {reward_stat.get()} > {env_spec.reward_threshold}"
                )
                break

        # Increment epoch
        timestamp["epoch"] += 1

    info = gather_info(
        start_time,
        train_collector,
        policy_test_collector,
        best_reward=best_result["rew"],
        best_reward_std=best_result["rew_std"],
    )
    info["len"] = best_result["len"]
    info["len_std"] = best_result["len_std"]

    return info
