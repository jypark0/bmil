from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from src.plot.utils import colorbar, fig_to_img


def get_map(env):
    # Get env image
    img, tile_size, shift = env.get_top_down_map()
    fig, ax = plt.subplots(constrained_layout=True, dpi=150)
    ax.set_axis_off()
    ax.imshow(img, interpolation="none", resample=False)

    return fig, ax, tile_size, shift


def heatmap(env, grid: np.ndarray, imshow_kwargs={}, set_text=False) -> np.ndarray:
    structure = env.maze_structure

    # Create heatmap
    fig, ax = plt.subplots(constrained_layout=True, dpi=150)
    ax.set_axis_off()
    im = ax.imshow(grid, cmap="viridis", **imshow_kwargs)
    colorbar(im, grid.shape[0] / grid.shape[1])

    # Loop over values and create text annotations.
    if set_text:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if structure[r][c].is_wall_or_chasm():
                    continue
                ax.text(
                    c,
                    r,
                    f"{grid[r][c]:.2f}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                )
    ax.invert_yaxis()

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    return img_rgb


def render_demonstrations(observations, env):
    # Obs: list of numpy arrays [ep_idx][time, d]
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.arange(len(observations)) % cmap.N)

    # Flip sign of observations for y coordinate to match Mujoco camera
    observations = deepcopy(observations)
    for i in range(len(observations)):
        observations[i][..., 1] = -observations[i][..., 1]

    fig, ax, tile_size, shift = get_map(env)

    for i in range(len(observations)):
        cur_pos = tile_size * observations[i][:-1, :2] + shift
        next_pos = tile_size * observations[i][1:, :2] + shift
        ax.quiver(
            cur_pos[:, 0],
            cur_pos[:, 1],
            (next_pos[:, 0] - cur_pos[:, 0]),
            (next_pos[:, 1] - cur_pos[:, 1]),
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors[i],
            width=0.003,
            headaxislength=2,
            headlength=2,
            alpha=1,
        )

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    return img_rgb


def render_scatter(pos, env):
    """Don't use blitting. Recreate figure every time."""
    fig, ax, tile_size, shift = get_map(env)
    cmap = plt.get_cmap("tab10")
    color = cmap(0)

    pos = deepcopy(pos)

    if isinstance(pos, list):
        pos = np.concatenate(pos)

    # Flip sign of observations for y coordinate to match Mujoco camera
    pos[..., 1] = -pos[..., 1]
    pos = tile_size * pos + shift

    ax.scatter(pos[:, 0], pos[:, 1], color=color, s=1, alpha=0.2)
    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    return img_rgb
