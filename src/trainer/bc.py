import sys
from argparse import Namespace
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import ImitationPolicy

from src.logger import WandBLogger
from src.utils import threshold_linear_fn


def BCTrainer(
    policy: ImitationPolicy,
    demo_buffer: ReplayBuffer,
    cfg: Union[Namespace, DictConfig],
    dynamics_model=None,
    demos: Batch = None,
    logger: Optional[WandBLogger] = None,
    checkpoint_fn: Optional[Callable[[None], None]] = None,
) -> Dict[str, Union[float, str]]:

    timestamp = {
        "epoch": 0,
        "policy_grad_step": 0,
        "dynamics_epoch": 0,
        "dynamics_grad_step": 0,
    }

    # Main training loop
    for epoch in tqdm.trange(
        timestamp["epoch"],
        cfg.policy.train.n_epoch,
        desc="Epoch",
        file=sys.stdout,
        disable=cfg.policy.train.step_per_epoch >= 1000,
    ):
        # Dynamics Model
        model_buffer = None
        if dynamics_model is not None:
            # Update model
            if cfg.dynamics.mode in ["backward", "forward"]:
                dynamics_result = dynamics_model.update(
                    demo_buffer=demo_buffer,
                    batch_size=cfg.dynamics.batch_size,
                    n_updates=cfg.dynamics.n_updates,
                    env_buffer=None,
                    holdout_ratio=0.2,
                    max_epochs_since_update=3,
                )
                logger.log_mean("dynamics", dynamics_result, timestamp)
                timestamp["dynamics_epoch"] += dynamics_result["update/epoch"]
                timestamp["dynamics_grad_step"] += dynamics_result["update/grad_step"]

                logger.log_all(
                    "dynamics",
                    {
                        "update/total_epoch": timestamp["dynamics_epoch"],
                        "update/total_grad_step": timestamp["dynamics_grad_step"],
                    },
                    timestamp,
                )
                logger.write(
                    {"info/dynamics/lr": dynamics_model.optim.param_groups[0]["lr"]},
                    timestamp,
                )

            # Generate traces from valuable states
            # traces: [trace_size, n_traces, d]
            if OmegaConf.select(cfg, "trace.epoch_schedule"):
                trace_size = threshold_linear_fn(
                    timestamp["epoch"],
                    cfg.trace.epoch_schedule,
                    cfg.trace.size_schedule,
                )
            else:
                trace_size = cfg.trace.size

            if cfg.dynamics.mode == "backward":
                traces = dynamics_model.generate_trace(
                    demos.obs,
                    demos.done,
                    trace_size,
                    cfg.trace.samples_per_start,
                    **cfg.trace.noise_method,
                )
            elif cfg.dynamics.mode == "forward":
                traces = dynamics_model.generate_trace(
                    demos.obs,
                    policy,
                    trace_size,
                    cfg.trace.samples_per_start,
                )

            logger.write(
                {"trace/num": traces.obs.shape[1], "trace/size": traces.obs.shape[0]},
                timestamp,
            )

            # Reverse traces and then add traces to buffer in correct order
            if cfg.dynamics.mode == "backward":
                for k, v in traces.items():
                    if isinstance(v, Batch) and v.is_empty():
                        continue
                    v = np.flipud(v)

            model_buffer = ReplayBuffer(size=np.prod(traces.obs.shape[:2]))
            for n in range(traces.obs.shape[1]):
                for t in range(traces.obs.shape[0]):
                    batch = Batch(
                        obs=traces.obs[t, n],
                        act=traces.act[t, n],
                        rew=traces.rew[t, n],
                        done=traces.done[t, n],
                        obs_next=traces.obs_next[t, n],
                    )
                    model_buffer.add(batch)

            # Flatten traces
            # traces: [time, n_traces, d] -> [*, d]
            for k, v in traces.items():
                if isinstance(v, Batch) and v.is_empty():
                    continue
                if v.ndim == 2:
                    traces[k] = v.reshape((-1))
                elif v.ndim == 3:
                    traces[k] = v.reshape((-1, v.shape[-1]))
                else:
                    raise ValueError(f"Incorrect shape for traces, key:{k}")

        # Update policy
        pbar = tqdm.tqdm(
            total=cfg.policy.train.step_per_epoch,
            desc=f"Epoch #{epoch}",
            file=sys.stdout,  # print tqdm to stdout
            disable=cfg.policy.train.step_per_epoch < 1000,
        )
        iters = 0
        policy.train()
        with pbar:
            while iters < pbar.total:
                if model_buffer is None:
                    train_result = policy.update(cfg.policy.batch_size, demo_buffer)
                else:
                    train_result = policy.update(
                        cfg.policy.batch_size, demo_buffer, model_buffer
                    )

                # Update timestamps
                pbar.update(1)
                iters += 1
                timestamp["policy_grad_step"] += 1

                logger.log_mean("train", train_result, timestamp)
                pbar.set_postfix(
                    loss=f"{train_result['loss']:.3f}",
                )
        policy.eval()

        # Save policy
        logger.save(timestamp, checkpoint_fn, is_best=False)

        timestamp["epoch"] += 1

    return timestamp
