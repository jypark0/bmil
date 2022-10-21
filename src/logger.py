import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import wandb


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


class WandBLogger:
    """WandB Logger. Log all episodes results using global_step."""

    def __init__(
        self,
        wandb_config,
        **wandb_kwargs,
    ):
        self.run = wandb.init(config=wandb_config, **wandb_kwargs)

        # Add more keys to wandb.config
        self.run.config.save_path = Path(self.run.dir).resolve()
        self.run.config.id = self.run.id

    def watch_model(self, model):
        self.run.watch(model, log=None, log_graph=True)

    def write(self, result: dict, timestamp: dict = {}) -> None:
        self.run.log({**result, **timestamp})

    def log_train_episodes(
        self, prefix: str, result: dict, timestamp: dict = {}
    ) -> None:
        """Use writer to log all episodes

        :param dict result: a dict containing several episode results
        :param dict timestamp: a dict containing global timestamps of episode
        (can have both global steps and num_episodes)

        """
        if result.get("n/ep", None) > 0:
            # Just use order of given episodes to calculate global step
            cumsum = np.cumsum(result["lens"])

            for i in range(result["n/ep"]):
                # Episode result
                ep_result = {
                    f"{prefix}/{k}": v[i]
                    for k, v in result.items()
                    if k in ["rews", "lens"]
                }
                # Episode timestamp
                t = {**timestamp}
                t["env_ep"] += i
                t["env_step"] += cumsum[i]

                self.write(ep_result, t)

    def log_train_steps(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        if result.get("n/ep", None) > 0:
            # Episode result
            ep_result = {
                f"{prefix}/{k}": v
                for k, v in result.items()
                if k in ["rew", "rew_std", "len", "len_std", "success_ratio"]
            }
            # Episode timestamp
            t = {**timestamp}
            t["env_ep"] += result["n/ep"]
            t["env_step"] += result["n/st"]

            self.write(ep_result, t)

    def log_all(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log all values in result
        result = {f"{prefix}/{k}": v for k, v in result.items()}

        self.write(result, timestamp)

    def log_mean(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log mean values in result
        result = {f"{prefix}/{k}": np.mean(v) for k, v in result.items()}

        self.write(result, timestamp)

    def log_last(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log last value only
        result = {f"{prefix}/{k}": v[-1] for k, v in result.items()}

        self.write(result, timestamp)

    def log_image(self, tag: str, img: np.ndarray, timestamp: dict = {}) -> None:
        """Visualize image"""
        # Tensorboard add_image doesn't sync with wandb
        # self.writer.add_image(tag, values, step, dataformats="HWC")
        self.write({tag: wandb.Image(img)}, timestamp)

    def log_video(self, tag: str, video: np.ndarray, timestamp: dict = {}) -> None:
        """Log numpy array to video"""
        self.write({tag: wandb.Video(video, fps=15, format="mp4")}, timestamp)

    def save(
        self,
        timestamp: dict,
        checkpoint_fn: Callable[[None], None],
        is_best: bool,
        epoch_key: str = "epoch",
        **kwargs,
    ) -> None:
        if checkpoint_fn:
            checkpoint_fn(save_path=Path(self.run.dir), is_best=is_best, **kwargs)

            # Save timestamp
            timestamp_filename = "timestamp.pt"
            if is_best:
                timestamp_filename = "timestamp-best.pt"
                self.write({"info/saved_best": timestamp[epoch_key]}, timestamp)

            torch.save(
                timestamp,
                Path(self.run.dir) / timestamp_filename,
            )

    def close(self):
        self.run.finish()
