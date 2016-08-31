import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
from scipy.stats.mstats import zscore


def multiblock_scatter_plot(X, Y, mode='whiskers', color='blue', ax=None):
    """ Draw scatter-plot of multi-block points.

    Parameters
    ----------
    X, Y : array_like
        (N x K) N X- and Y-coordinates of K blocks.
    mode : str
        Drawing mode of multi-block points_
        * `"whiskers"` - Draw point of mean value and
          thin lines to other points.
        * `"bezier"` - Draw star-like shape connecting multi-block points.
    color_by : str, Callable
        Either string specifying color or a function which returns
        a acceptable argument `color`-argument for `matplotlib.pyplot.scatter`
        given `X` and `Y`
    ax : matplotlib.pyplot.Axes, optional
        If provided plot is drawn at `ax`.

    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    if not X.shape == Y.shape:
        raise ValueError('X and Y does not match')

    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure

    if isinstance(color, str):
        data_color = [color for _ in X]
    elif callable(color):
        data_color = color(X, Y)
    else:
        raise ValueError('color must be string or callable.')

    for i, (x_row, y_row) in enumerate(zip(X, Y)):
        if mode == 'whiskers':
            plot_whiskers(ax, x_row, y_row)
        elif mode == 'bezier':
            plot_bezier(ax, x_row, y_row, color=data_color[i])

    x_means = X.mean(axis=1)
    y_means = Y.mean(axis=1)

    if mode == 'whiskers':
        ax.scatter(x_means, y_means, c=data_color, s=100)
    else:
        ax.scatter(x_means, y_means, c='none', s=0)

    return f, ax


def color_by_variance_zscore(X, Y, cmap=None, cutoff_z=3, over_color='red'):
    """ Assigns color according to Z-score of variances.

    Parameters
    ----------
    X, Y : array_like
        (N x K) N X- and Y-coordinates of K blocks.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use, default `viridis`
    cutoff_z : float
        Observations with variance Z-score above `cutoff_z` are drawn
        with `over_color`
    over_color : string, tuple, array_like
        Color string or rgb(a)-values.

    Returns
    -------
    np.ndarray
    """
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis')
    var = variance_sum_of_squares(X, Y)
    color_values = abs(zscore(var))
    norm = colors.Normalize(0, cutoff_z)
    cmap.set_over(over_color)
    color_map = plt.cm.ScalarMappable(norm, cmap)

    return color_map.to_rgba(color_values)


def color_by_normalized_variance(X, Y, cmap=None):
    """ Assigns color according to variance sum-of-squares
    normalized according to the observation with maximum variance.

    Parameters
    ----------
    X, Y : array_like
        (N x K) N X- and Y-coordinates of K blocks.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use, default `viridis`

    Returns
    -------
    np.ndarray
    """
    if cmap is None:
        cmap = plt.cm.get_cmap('viridis')
    var = variance_sum_of_squares(X, Y)
    color_values = var / var.max()
    norm = colors.Normalize(0, 1)
    color_map = plt.cm.ScalarMappable(norm, cmap)

    return color_map.to_rgba(color_values)


def variance_sum_of_squares(X, Y):
    """ Calculate the variance sum of squares of
    of observations with coordinates `X` and `Y`

    Parameters
    ----------
    X, Y : array_like
        (N x K) N X- and Y-coordinates of K blocks.

    Returns
    -------
    np.ndarray[float]
    """
    x_means = X.mean(axis=1)
    y_means = Y.mean(axis=1)
    var = ((X - np.atleast_2d(x_means).T) ** 2 + (
    Y - np.atleast_2d(y_means).T) ** 2).sum(1)
    return var


def plot_whiskers(ax, x, y):
    """ Draws lines at `ax` from the given points' mean-value
    to the points given in `x` and `y`

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis-object to draw at.
    x, y : np.ndarray[float]
        Sets of X- and Y-coordinates.
    """
    x_mean = x.mean()
    y_mean = y.mean()
    for x, y in zip(x, y):
        ax.plot([x_mean, x], [y_mean, y], color=np.array([0, 0, 0, .5]))


def plot_bezier(ax, x, y, color=None):
    """ Draws star-like shapes at `ax` around the given
    points' mean-value to the points given in `x` and `y`

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis-object to draw at.
    x, y : np.ndarray[float]
        Sets of X- and Y-coordinates.
    color : str, tuple, np.ndarray, None
        Face-color of star-shape.
    """
    XY = np.column_stack((x, y))
    x_mean = x.mean()
    y_mean = y.mean()
    mean = np.array([x_mean, y_mean])

    corners = find_polygon_with_maximum_area(XY)
    vertices = [corners[0]]
    codes = [Path.MOVETO]

    for corner in np.roll(corners, -1, axis=0):
        vertices.extend([mean, corner])
        codes.extend([Path.CURVE3, Path.CURVE3])

    path = Path(vertices, codes)

    patch = patches.PathPatch(path, facecolor=color, edgecolor='black')
    ax.add_patch(patch)


def find_polygon_with_maximum_area(xy):
    """ Given 2D-array find permutation which maximizes
    polygon area.

    Parameters
    ----------
    xy : array_like
        N x 2-array of coordinates.

    Returns
    -------
    array_like
        N x 2-array.
    """
    permutations = [np.vstack(perm) for perm in itertools.permutations(xy)]
    areas = list()

    for perm in permutations:
        x = perm[:, 0]
        y = perm[:, 1]

        # Calculat are using Shoelace-formula (google it).
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        areas.append(area)

    i = np.argmax(areas)
    return permutations[i]


if __name__ == '__main__':
    import time
    n = 128
    noise = 2
    np.random.seed(42)
    x = np.random.randn(n) * 10
    y = np.random.randn(n) * 10

    X = np.array([x.copy(), x.copy(), x.copy()]).T
    Y = np.array([y.copy(), y.copy(), y.copy()]).T

    X += np.random.randn(n, 3) * noise
    Y += np.random.randn(n, 3) * noise

    t = time.time()
    f, ax = multiblock_scatter_plot(X, Y, color_by='ss_z')
    t1 = time.time()
    f, ax = multiblock_scatter_plot(X, Y, mode='bezier', color_by='ss_z')
    t2 = time.time()

    print(t1 - t)
    print(t2 - t1)

    plt.show()

    plt.show()