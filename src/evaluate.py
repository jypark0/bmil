import numpy as np
import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.envs.utils import make_env
from src.logger import get_logger
from src.plot.fetch import create_2D_plot
from src.plot.maze import heatmap, render_scatter

log = get_logger(__name__)


def evaluate_fetch(policy, cfg, env, logger, timestamp):

    eval_envs = instantiate(
        cfg.eval.env,
        env_fns=[
            make_env(
                env_id=cfg.env.id,
                wrappers=cfg.env.wrappers,
                seed=None,
                env_kwargs=cfg.eval.env_kwargs,
            )
            for _ in range(cfg.eval.n_envs)
        ],
    )

    eval_collector = instantiate(cfg.eval.collector, policy=policy, env=eval_envs)
    policy.eval()
    eval_result = eval_collector.collect(n_episode=cfg.eval.n_ep, disable_tqdm=False)
    eval_envs.close()

    logger.write({"eval/success_ratio": eval_result["success_ratio"]}, timestamp)
    log.info(f"[Evaluation] success_ratio: {eval_result['success_ratio']:.3f}")
    log.info(
        f"[Evaluation] Reward: {eval_result['rew']:.2f} +/- {eval_result['rew_std']:.2f}"
    )
    log.info(
        f"[Evaluation] Length: {eval_result['len']:.2f} +/- {eval_result['len_std']:.2f}"
    )

    start_pos = None
    object_pos = None
    goal_pos = None
    if not cfg.env.init.random_gripper:
        start_pos = env.initial_gripper_xpos
    if env.has_object:
        object_pos = np.hstack([env.initial_object_pos, [env.height_offset]])
    if not cfg.env.init.random_goal:
        goal_pos = env.initial_goal_pos

    img = create_2D_plot(
        eval_result["init_pos"],
        eval_result["success"],
        start_pos=start_pos,
        object_pos=object_pos,
        goal_pos=goal_pos,
    )
    logger.log_image("eval/init_pos", img)


def evaluate_maze(policy, cfg, env, logger, timestamp):
    valid_rowcol = env.valid_rowcol()
    env_init = OmegaConf.to_container(cfg.eval.env_kwargs, resolve=True)

    st = env.maze_structure
    success_rate_rc = np.zeros((len(st), len(st[0])))
    success_init_pos = []

    policy.eval()

    for rc in tqdm.tqdm(valid_rowcol, desc="RowCol"):

        eval_envs = instantiate(
            cfg.eval.env,
            env_fns=[
                make_env(
                    env_id=cfg.env.id,
                    wrappers=cfg.env.wrappers,
                    seed=None,
                    env_kwargs={**env_init, "init_rowcol": rc},
                )
                for _ in range(cfg.eval.n_envs)
            ],
        )
        eval_collector = instantiate(cfg.eval.collector, policy=policy, env=eval_envs)
        eval_result = eval_collector.collect(n_episode=cfg.eval.n_ep)
        eval_envs.close()

        success_rate_rc[rc[0], rc[1]] = eval_result["success_ratio"]

        # Filter out only start_positions from successful episodes
        init_pos = eval_result["init_pos"][eval_result["success"]]
        success_init_pos.append(init_pos)

    success_ratio = success_rate_rc[tuple(np.array(valid_rowcol).T)].mean()
    success_init_pos = np.concatenate(success_init_pos)

    logger.write({"eval/success_ratio": success_ratio}, timestamp)
    log.info(f"[Evaluation] success_ratio: {success_ratio:.3f}")

    # Render heatmap of success ratios
    robust_img = heatmap(
        env, success_rate_rc, imshow_kwargs={"vmin": 0, "vmax": 1}, set_text=False
    )
    logger.log_image("eval/heatmap", robust_img)

    # Render successful initial positions
    robust_pos_img = render_scatter(success_init_pos, env)
    logger.log_image("eval/successful_init_pos", robust_pos_img)


def evaluate_adroit(policy, cfg, logger, timestamp):
    eval_envs = instantiate(
        cfg.eval.env,
        env_fns=[
            make_env(
                env_id=cfg.env.id,
                wrappers=cfg.env.wrappers,
                seed=None,
                env_kwargs=cfg.eval.env_kwargs,
            )
            for _ in range(cfg.eval.n_envs)
        ],
    )

    eval_collector = instantiate(cfg.eval.collector, policy=policy, env=eval_envs)
    policy.eval()
    eval_result = eval_collector.collect(n_episode=cfg.eval.n_ep, disable_tqdm=False)
    eval_envs.close()

    logger.write({"eval/success_ratio": eval_result["success_ratio"]}, timestamp)
    log.info(f"[Evaluation] success_ratio: {eval_result['success_ratio']:.3f}")
    log.info(
        f"[Evaluation] Reward: {eval_result['rew']:.2f} +/- {eval_result['rew_std']:.2f}"
    )
    log.info(
        f"[Evaluation] Length: {eval_result['len']:.2f} +/- {eval_result['len_std']:.2f}"
    )
