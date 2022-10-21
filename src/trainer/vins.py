import sys
from argparse import Namespace
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm
from omegaconf import DictConfig
from tianshou.data import Collector

from src.logger import WandBLogger, get_logger
from src.trainer.utils import evaluate_policy


def VINSTrainer(
    value_function,
    dynamics_model,
    bc_policy,
    imitator,
    train_collector: Collector,
    policy_test_collector: Collector,
    bc_test_collector: Collector,
    cfg: Union[Namespace, DictConfig],
    logger: Optional[WandBLogger] = None,
    checkpoint_fn: Optional[Callable[[None], None]] = None,
) -> Dict[str, Union[float, str]]:

    log = get_logger(__name__)

    timestamp = {
        "bc_epoch": 0,
        "bc_grad_step": 0,
        "value_epoch": 0,
        "policy_grad_step": 0,
        "model_epoch": 0,
        "model_grad_step": 0,
        "rl_epoch": 0,
        "env_ep": 0,
        "env_step": 0,
    }

    train_collector.reset_stat()
    if bc_test_collector is not None:
        bc_test_collector.reset_stat()
    if policy_test_collector is not None:
        policy_test_collector.reset_stat()

    best_result = {
        "bc_epoch": 0,
        "value_epoch": 0,
        "model_epoch": 0,
        "rl_epoch": 0,
        "rew": float("-inf"),
        "rew_std": 0,
        "len": float("inf"),
        "len_std": 0,
        "success_ratio": 0,
    }
    last_rew, last_len = 0.0, 0

    # Value
    log.info("Training Value")
    pbar = tqdm.trange(
        timestamp["value_epoch"],
        cfg.policy.value_train.n_epoch,
        desc=f"Value Epoch",
        file=sys.stdout,  # print tqdm to stdout
    )
    for epoch in pbar:
        value_function.train()
        # Train on entire buffer
        value_result = value_function.update_value(
            0,
            train_collector.buffer,
            batch_size=cfg.policy.value_batch_size,
            negative_sampling=True,
        )
        value_function.eval()
        logger.log_mean("train", value_result, timestamp)

        # Update timestamp
        timestamp["policy_grad_step"] += len(value_result["loss/value"])
        pbar.set_postfix(
            loss=f"{np.mean(value_result['loss/value']):.2f}",
        )

        timestamp["value_epoch"] += 1

    # BC
    if cfg.bc_policy.train.n_epoch > 0:
        log.info("Training BC Policy")
        for epoch in tqdm.trange(
            timestamp["bc_epoch"],
            cfg.bc_policy.train.n_epoch,
            desc="BC Epoch",
            file=sys.stdout,
            disable=cfg.bc_policy.train.step_per_epoch >= 1000,
        ):
            pbar = tqdm.tqdm(
                total=cfg.bc_policy.train.step_per_epoch,
                desc=f"BC Epoch {epoch}",
                file=sys.stdout,
                disable=cfg.bc_policy.train.step_per_epoch < 1000,
            )
            iters = 0
            bc_policy.train()
            with pbar:
                while iters < pbar.total:
                    bc_result = bc_policy.update(
                        cfg.bc_policy.batch_size, train_collector.buffer
                    )

                    # Update timestamps
                    pbar.update(1)
                    iters += 1
                    timestamp["bc_grad_step"] += 1

                    logger.log_mean("bc/train", bc_result, timestamp)
                    pbar.set_postfix(
                        loss=f"{bc_result['loss']:.3f}",
                    )
            bc_policy.eval()

            timestamp["bc_epoch"] += 1

    # Model
    log.info("Training Model")
    pbar = tqdm.trange(
        timestamp["model_epoch"],
        cfg.policy.model_train.n_epoch,
        desc=f"Model Epoch ",
        file=sys.stdout,  # print tqdm to stdout
    )
    for epoch in pbar:
        dynamics_model.train()
        # Train on entire buffer
        model_result = dynamics_model.update_model(
            0,
            train_collector.buffer,
            batch_size=cfg.policy.model_batch_size,
        )
        dynamics_model.eval()
        logger.log_mean("train", model_result, timestamp)

        # Update timestamp
        timestamp["model_grad_step"] += len(model_result["loss/dynamics"])

        pbar.set_postfix(
            loss=f"{np.mean(model_result['loss/dynamics']):.3f}",
        )

        timestamp["model_epoch"] += 1

    # Save policy
    logger.save(timestamp, checkpoint_fn, is_best=False, epoch_key="value_epoch")

    # Test before RL if needed
    if cfg.policy.rl_train.n_epoch > 0:
        test_result = evaluate_policy(
            policy=imitator,
            collector=policy_test_collector,
            n_episode=10,
            collect_kwargs={"forward_kwargs": {"bc_prior": True}},
        )
        logger.log_all(
            "test",
            {
                k: v
                for k, v in test_result.items()
                if k in ["len", "rew", "rew_std", "len_std", "success_ratio"]
            },
            timestamp,
        )
        test_rew, test_rew_std = test_result["rew"], test_result["rews"].std()
        test_len, test_len_std = test_result["len"], test_result["lens"].std()
        best_result = {
            "bc_epoch": timestamp["bc_epoch"],
            "value_epoch": timestamp["value_epoch"],
            "model_epoch": timestamp["model_epoch"],
            "rl_epoch": timestamp["rl_epoch"],
            "rew": test_rew,
            "rew_std": test_rew_std,
            "len": test_len,
            "len_std": test_len_std,
            "success_ratio": test_result["success_ratio"],
        }
        log.info(
            f"Test: rew={test_rew:.2f}, "
            f"len={test_len:.2f}, "
            f"ratio={test_result['success_ratio']:.2f}"
        )

    # RL
    for epoch in range(timestamp["rl_epoch"], cfg.policy.rl_train.n_epoch):
        pbar = tqdm.tqdm(
            total=cfg.policy.rl_train.step_per_epoch,
            desc=f"RL Epoch #{epoch}",
            file=sys.stdout,  # print tqdm to stdout
        )
        with pbar:
            while pbar.n < pbar.total:
                # Collect policy rollouts
                train_result = train_collector.collect(
                    n_step=cfg.policy.rl_train.step_per_collect,
                    disable_tqdm=True,
                    forward_kwargs={"bc_prior": False},
                )
                logger.log_train_steps("train", train_result, timestamp)

                # Update timestamps
                pbar.update(train_result["n/st"])
                timestamp["env_ep"] += train_result["n/ep"]
                timestamp["env_step"] += train_result["n/st"]

                # Create data
                if train_result["n/ep"] > 0:
                    last_rew = train_result["rew"]
                    last_len = train_result["len"]
                data = {
                    "env_step": str(timestamp["env_step"]),
                    "rew": f"{last_rew:.1f}",
                    "len": f"{last_len:.1f}",
                }

                # Update model
                n_samples = cfg.policy.rl_train.step_per_collect
                buffer_size = len(train_collector.buffer)
                n_model_epochs = (
                    n_samples * (cfg.policy.model_train.n_epoch) // buffer_size + 1
                )
                dynamics_model.train()
                for _ in range(n_model_epochs):
                    model_result = dynamics_model.update_model(
                        0,
                        train_collector.buffer,
                        batch_size=cfg.policy.model_batch_size,
                    )
                    timestamp["model_grad_step"] += len(model_result["loss/dynamics"])
                dynamics_model.eval()

                # Log only mean of last loss
                logger.log_mean("rl", model_result, timestamp)

                # Update value
                n_iters = n_samples * 10 // cfg.policy.value_batch_size
                n_value_epochs = (
                    n_iters
                    // ((buffer_size - n_samples) // cfg.policy.value_batch_size)
                    + 1
                )
                value_function.iters = n_iters
                value_function.train()
                for _ in range(n_value_epochs):
                    value_result = value_function.update_value(
                        0,
                        train_collector.buffer,
                        batch_size=cfg.policy.value_batch_size,
                        negative_sampling=False,
                    )
                    timestamp["policy_grad_step"] += len(value_result["loss/value"])
                    if value_function.iters == 0:
                        break
                value_function.iters = -1
                pbar.set_postfix(
                    model=f"{np.mean(model_result['loss/dynamics']):.3f}",
                    value=f"{np.mean(value_result['loss/value']):.2f}",
                    **data,
                )
                value_function.eval()

                # Log only mean of last loss
                logger.log_mean("rl", value_result, timestamp)

        imitator.dynamics_model = dynamics_model
        imitator.value_function = value_function

        # Test
        if (epoch + 1) % 10 == 0:
            test_result = evaluate_policy(
                policy=imitator,
                collector=policy_test_collector,
                n_episode=10,
                collect_kwargs={"forward_kwargs": {"bc_prior": False}},
            )
            logger.log_all(
                "test",
                {
                    k: v
                    for k, v in test_result.items()
                    if k in ["len", "rew", "rew_std", "len_std", "success_ratio"]
                },
                timestamp,
            )
            test_rew, test_rew_std = (
                test_result["rews"].mean(),
                test_result["rews"].std(),
            )
            test_len, test_len_std = (
                test_result["lens"].mean(),
                test_result["lens"].std(),
            )
            # If success_ratio exists, use that. Else, if both rewards are equal, use episode lengths
            if test_result.get("success_ratio") is not None:
                cond = test_result["success_ratio"] >= best_result["success_ratio"]
            elif test_rew == best_result["rew"]:
                cond = test_len <= best_result["len"]
            else:
                cond = test_rew >= best_result["rew"]

            if cond:
                best_result = {
                    "bc_epoch": timestamp["bc_epoch"],
                    "value_epoch": timestamp["value_epoch"],
                    "model_epoch": timestamp["model_epoch"],
                    "rl_epoch": timestamp["rl_epoch"],
                    "rew": test_rew,
                    "rew_std": test_rew_std,
                    "len": test_len,
                    "len_std": test_len_std,
                    "success_ratio": test_result["success_ratio"],
                }
                if checkpoint_fn:
                    log.info(f"[RL Epoch {epoch}] Saving best checkpoint")
                    logger.save(
                        timestamp, checkpoint_fn, is_best=True, epoch_key="rl_epoch"
                    )
                logger.log_all("best", best_result, timestamp)
            else:
                logger.save(
                    timestamp, checkpoint_fn, is_best=False, epoch_key="rl_epoch"
                )
            log.info(
                f"[RL Epoch {epoch}] Test: rew={test_rew:.2f}, "
                f"len={test_len:.2f}, "
                f"ratio={test_result['success_ratio']:.2f}"
            )
            log.info(
                f"[RL Epoch {epoch}] Best: rew={best_result['rew']:.2f}, "
                f"len={best_result['len']:.2f}, "
                f"ratio={best_result['success_ratio']:.2f} (from epoch {best_result['rl_epoch']})"
            )

        timestamp["rl_epoch"] += 1

    return timestamp
