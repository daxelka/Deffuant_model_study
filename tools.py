import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def uniform_opinion(n_nodes, limits=(0.0, 1.0)):
    rng = np.random.default_rng()
    start, end = limits
    opinion = rng.uniform(start, end, (n_nodes,))
    return opinion


def show_distribution(opinions):
    bins = [0.01 * n for n in range(100)]
    plt.hist(opinions, bins=bins, density=True)
    plt.title("Histogram of opinions")
    plt.show()


def density_plot(vector, x_limits=tuple(), y_limits=tuple(), title="", x_label=""):
    sns.set_style("white")
    sns.histplot(data=vector, stat="density", color='gray', alpha=0.15)
    sns.kdeplot(data=vector)
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    plt.show()




