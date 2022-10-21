import matplotlib.pyplot as plt
import numpy as np


def colorbar(mappable, im_ratio):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    cbar = fig.colorbar(mappable, fraction=0.05 * im_ratio)
    plt.sca(last_axes)
    return cbar


def fig_to_img(fig):
    img_rgb = np.reshape(
        np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8"),
        newshape=fig.canvas.get_width_height()[::-1] + (3,),
    )
    plt.close(fig)
    return img_rgb
