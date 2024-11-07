from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm


def gaussian_figures(means, stds, fig_name="output", names=None, start=0.3, stop=0.9, title="Plot", *args):
    """
    This function is for plotting gaussians in a picture having their means and standard deviation
    :param means: means vector to be plotted
    :param stds: standard deviation vector to be plotted
    :param fig_name: name of the figure without the extension but comprehensive of the path
    :param names: names of the means and stds
    :param start: value where x axis will start
    :param stop: value where x axis will end
    :param title: title of the plot
    :param args: other args added to plt.plot if not differently set
    :return:
    """
    if names is None:
        names = []
    plt.figure()
    plt.title(title)
    x_axis = np.arange(start, stop, 0.001)
    for mean, sd in zip(means, stds):
        if sd == 0:
            sd = 0.00001
        plt.plot(x_axis, norm.pdf(x_axis, mean, sd), scaley=True, *args)
    plt.legend([names[i] + " " + str(round(means[i], 2)) for i in range(len(names))])
    plt.yticks([])
    plt.savefig(fig_name + ".png")

