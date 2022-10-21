import matplotlib.pyplot as plt

from src.plot.utils import fig_to_img


def create_2D_plot(init_pos, success, start_pos=None, object_pos=None, goal_pos=None):
    xlim = [1.15, 1.55]
    ylim = [0.55, 0.95]

    fig = plt.figure(constrained_layout=True, dpi=150)
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")

    # Successful points
    success_pos = init_pos[success.astype(bool)]
    fail_pos = init_pos[~success.astype(bool)]

    ax.scatter(
        success_pos[..., 0], success_pos[..., 1], color="tab:green", s=2, alpha=0.5
    )
    ax.scatter(fail_pos[..., 0], fail_pos[..., 1], color="#bcbcbc", s=2, alpha=0.5)

    if start_pos is not None:
        ax.scatter(*start_pos[:2], color="tab:red", s=100, alpha=0.5)
    if object_pos is not None:
        ax.scatter(*object_pos[:2], color="tab:orange", s=100, alpha=0.5)
    if goal_pos is not None:
        ax.scatter(*goal_pos[:2], color="tab:blue", s=100, alpha=0.5)

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)

    return img_rgb
