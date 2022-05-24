import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
import scipy.stats as stats
import scipy.optimize
from scipy import interpolate
import seaborn as sns



def random_opinion(n_nodes):
    rng = np.random.default_rng()
    opinion_distribution = rng.random((n_nodes,))
    return opinion_distribution


def uniform_opinion(n_nodes, limits=(0.0, 1.0)):
    rng = np.random.default_rng()
    start, end = limits
    opinion = rng.uniform(start, end, (n_nodes,))
    return opinion


def inverse_transform_sampling(pdf, N=1000000, x_limits=(0, 1)):
    """ Sampling of a random variable from a user specified probability density function by inverse transform sampling.

    Args:
        pdf:
            probability density function. Must take a single argument, x, in form of np.array, and return np.array.
        N:
            number of samples. integer
        x_limits:
            boundaries of truncated pdf. tuple (x_lower, x_upper)

    Returns: N random variables sampled from given truncated pdf. np.array of size N

    """
    x_lower, x_upper = x_limits
    x = np.linspace(x_lower, x_upper, int(N))
    y = pdf(x)  # probability density function, pdf
    cdf_y = np.cumsum(y)  # cumulative distribution function
    cdf_y = cdf_y / cdf_y.max()  # normalization of cdf
    inverse_cdf = interpolate.interp1d(cdf_y, x, fill_value="extrapolate")  # this is a function

    # Generates samples from given pdf
    uniform_samples = random(int(N))
    required_samples = inverse_cdf(uniform_samples)
    return required_samples



def normal_opinion(n_nodes, mu, sigma, lower_bound, upper_bound):
    mu_to_use, sigma_to_use = mu, sigma
    # mu_to_use, sigma_to_use = corrector(mu, sigma, lower_bound, upper_bound)
    opinion = trunc_samples(mu=mu_to_use, sigma=sigma_to_use, lower=lower_bound, upper=upper_bound, num_samples=n_nodes)
    return opinion

def show_distribution(opinions):
    bins = [0.01 * n for n in range(100)]
    plt.hist(opinions, bins=bins, density=True)
    plt.title("Histogram of opinions")
    plt.show()


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, color = 'steelblue'):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    # patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
    #                  edgecolor='C0', fill=False, linewidth=1)
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor= color, fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def density_plot(vector, x_limits=tuple(), y_limits=tuple(), title="", x_label=""):
    sns.set_style("white")
    # sns.displot(data=vector, kind="kde")
    # sns.displot(data=vector, kde=True)
    # sns.histplot(data=vector, stat="density", fill=False, color ='gray')
    sns.histplot(data=vector, stat="density", color='gray', alpha=0.15)
    # sns.kdeplot(data=vector, color="b")
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


def truncated_mean_std(mu, sigma, lower, upper):
    # N.B. lower/upper are the actual values, not Z-scaled
    alpha = (lower - mu)/sigma
    beta = (upper - mu)/sigma
    d_pdf = (stats.norm.pdf(alpha) - stats.norm.pdf(beta))
    wd_pdf = (alpha * stats.norm.pdf(alpha) - beta * stats.norm.pdf(beta))
    d_cdf = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
    mu_trunc = mu + sigma * (d_pdf / d_cdf)
    var_trunc = sigma**2 * (1 + wd_pdf / d_cdf - (d_pdf/d_cdf)**2)
    std_trunc = var_trunc**0.5
    return mu_trunc, std_trunc

def trunc_samples(mu, sigma, lower, upper, num_samples=1000):
    n = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    samples = n.rvs(num_samples)
    return samples

def corrector(mu, sigma, lower, upper):
    target = np.array([mu, sigma])
    result = scipy.optimize.minimize(
        lambda x: ((target - truncated_mean_std(x[0], x[1], lower, upper))**2).sum(),
        x0=[mu, sigma])
    return result.x

