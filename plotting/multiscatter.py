import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
from scipy.stats.mstats import zscore


def multiblock_scatter_plot(X, Y, mode='whiskers', color='blue', ax=None,
                            x_label=None, y_label=None, title=None,
                            label_fn=None):
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
    x_label, y_label : str, optional
        Axis labels.
    title : str, optional
        Plot title.
    label_fn : Callable, optional
        Function which given axis will label it.

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

    ax.axhline(0, color='k', zorder=-1)
    ax.axvline(0, color='k', zorder=-1)

    # Draw points of each block according to mode.
    for i, (x_row, y_row) in enumerate(zip(X, Y)):
        if mode == 'whiskers':
            plot_whiskers(ax, x_row, y_row)
        elif mode == 'bezier':
            plot_bezier(ax, x_row, y_row, color=data_color[i])

    x_means = X.mean(axis=1)
    y_means = Y.mean(axis=1)

    if mode == 'whiskers':
        # Draw mean-value points.
        ax.scatter(x_means, y_means, c=data_color, s=100)
    else:
        # This sets the xlim and ylim to proper values.
        ax.scatter(x_means, y_means, c='none', s=0)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if label_fn is not None:
        if not callable(label_fn):
            raise ValueError('label_fn must be callable')
        label_fn(ax)

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
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

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
        ax.plot([x_mean, x], [y_mean, y], color=np.array([0, 0, 0, .5]),
                zorder=-.5)


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

        # Area calculation using Shoelace-formula (google it).
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        areas.append(area)

    i = np.argmax(areas)
    return permutations[i]


def smart_annotate(ax, data, labels, sizes=None):
    """ Label axis "smartly" by applying force-based layout
    to avoid label-point overlap.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis to label.
    data : np.ndarray
        (m x 2), data points.
    labels : list
        List of strings for points to labelled, points
        with labels `None` will not be labelled.
    sizes : list[float]
        List of point sizes in plot.

    Returns
    -------
    list[matplotlib.text.Annotation]
    """
    m, n = data.shape
    A = np.zeros((m * 2, m * 2))
    pos = np.zeros((m * 2, 2), dtype='float32')

    # Set-up adjacency matrix.
    for i, (x, y) in enumerate(data):
        pos[i*2: i*2 + 2, 0] = x
        pos[i*2: i*2 + 2, 1] = y
        if labels[i] is not None:
            A[i * 2, i * 2 + 1] = A[i * 2 + 1, i * 2] = 1

    # Calculate positions using a force-based spring algorithm.
    fixed = list(range(0, m * 2, 2))
    positions = force_layout(A, pos=pos, fixed=fixed)
    label_positions = positions[list(range(1, m * 2 + 1, 2))]

    if sizes is None:
        sizes = [20] * m

    # Span in X and Y.
    x_max, x_min = ax.get_xlim()
    y_max, y_min = ax.get_ylim()

    annotations = list()
    for i, (xy, lab_xy) in enumerate(zip(data, label_positions)):
        if labels[i] is None:
            continue
        xy = np.array(xy)
        lab_xy = np.array(lab_xy)

        # Normalize distance to data-point.
        diff = (xy - lab_xy) / np.linalg.norm(xy - lab_xy)
        span = np.linalg.norm(np.array([x_max - x_min, y_max - y_min]))
        if np.isnan(diff).any():
            diff = 0
        else:
            diff *= .02 * span

        # Adjust for point size.
        shift = ax.transData.inverted().transform((sizes[i], 0))
        shift /= np.linalg.norm(shift)
        shift *= .015 * span

        # Annotate.
        text = ax.annotate(labels[i], xy=xy, xycoords='data',
                           xytext=xy - (diff - shift), textcoords='data')
        annotations.append(text)

    return annotations


def force_layout(A, k=None, pos=None, fixed=None, n_iter=50):
    """ Use Fruchterman-Reingold layout algorithm to layout nodes in `A`.

    Adopted from NetworkX `spring_layout`.

    :param A: Adjacency matrix.
    :param k: Optimal node-node distance.
    :param pos: Initioal positions.
    :param fixed: Indexes of nodes with fixed position.
    :param n_iter: Number of iterations.
    :return: Position array (n-nodes x 2)
    """

    nnodes, _ = A.shape

    if pos is None:
        # Random initial positions
        pos = np.asarray(np.random.random((nnodes, 2)))
    else:
        # Make sure positions are of same type as matrix.
        pos = pos.astype(A.dtype)

    # Optimal distance between nodes.
    if k is None:
        try:
            k = np.sqrt(1.0 / nnodes)
        except ZeroDivisionError:
            return

    # Initial "temperature"  is about .1 of domain area (=1x1)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1

    # Linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(n_iter + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)

    for iteration in range(n_iter):

        for i in range(pos.shape[1]):
            delta[:, :, i] = pos[:, i, None] - pos[:, i]

        # Distances. (Enforce min-distance of 0.01)
        distance = np.sqrt((delta ** 2).sum(axis=-1))
        distance = np.where(distance < 0.01, 0.01, distance)

        # Forces.
        disp_arr = (delta.T * (k * k / distance ** 2 - A.astype('float32') * distance / k)).T
        displacement = disp_arr.sum(axis=1)

        # Calculate position shift.
        length = np.sqrt((displacement ** 2).sum(axis=1))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.transpose(np.transpose(displacement)*t/length)

        # Update position of un-fixed nodes and temperature.
        if fixed is not None:
            delta_pos[fixed] = 0.0
        pos += delta_pos
        t -= dt

    return pos


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import sys
    import functools

    parser = argparse.ArgumentParser('multiblock scatter plot')
    parser.add_argument('X', help='X-coordinates, observations as '
                                  'rows and blocks as columns')
    parser.add_argument('Y', help='X-coordinates, observations as '
                                  'rows and blocks as columns')

    parser.add_argument(
        '-c', '--color_by', choices=['default', 'normalized_ss', 'ss_zscore'],
        default='default',
        help=('How to color data points.\n'
              '* default : Default coloring\n'
              '* normalized_ss : Coloring according to normalized '
              'variance sum of squares\n'
              '* ss_zscore : Coloring according to Z-transformed '
              'variance sum of squares.')
    )
    parser.add_argument('--z_cutoff', default=3, type=float,
                        help=('If --color_by is set to "ss_zscore", points with'
                              ' z-score over cutoff is colored according to '
                              '--over_color. Used to determine labelled '
                              'points if "--label_above_zcutoff" is used.'))
    parser.add_argument('--over_color', default='red',
                        help='Color to use for points with Z-score '
                             'above --z_cutoff. See matplotlib documentation '
                             'for valid color-strings.')
    parser.add_argument('--xlabel', default='x', help='Axis X-label')
    parser.add_argument('--ylabel', default='y', help='Axis Y-label')
    parser.add_argument('--title', default='', help='Plot title.')
    parser.add_argument('--colormap', default='viridis',
                        help='Colormap to use for plot. See matplotlib '
                             'documentation for valid colormap-strings.')
    parser.add_argument('--grid', default=False, action='store_true',
                        help='If set, plot will be drawn with grid.')
    parser.add_argument('--style', choices=plt.style.available,
                        default='classic', help='Matplotlib style to use.')
    parser.add_argument('--point_style', choices=('whiskers', 'bezier'),
                        default='whiskers', help='Drawing style of points.')
    parser.add_argument('-o', '--output', default='multiblockscatter.png',
                        help='Output filename.')
    parser.add_argument('--read_excel_index', action='store_true', default=False,
                        help='If set, treat first column of Excel spreadsheet '
                             'as index column.')
    parser.add_argument('--label_above_zcutoff', action='store_true',
                        default=False,
                        help='Label points with variance sum of squares '
                             'above "z_cutoff".')
    parser.add_argument('--label_top_n', type=int, default=0,
                        help='Label points with top n variance sum of squares. '
                             'Overrides "--label_above_zcutoff"')

    args = parser.parse_args()

    plt.style.use(args.style)

    data = []
    for path in (args.X, args.Y):
        fformat = path.split('.')[-1]
        if fformat == 'csv':
            df = pd.DataFrame.from_csv(path, index_col=0)
        elif fformat in ('xlsx', 'xls'):
            if args.read_excel_index:
                df = pd.read_excel(path, index_col=0)
            else:
                df = pd.read_excel(path)
        else:
            sys.exit('Invalid file-format: {}'.format(fformat))
        data.append(df)

    X_df, Y_df = data
    X = X_df.values
    Y = Y_df.values

    # ------------------ Prepare coloring function ---------------
    if args.color_by == 'default':
        color = 'blue'
    elif args.color_by == 'normalized_ss':
        color = functools.partial(color_by_normalized_variance,
                                  cmap=args.colormap)
    else:
        color = functools.partial(color_by_variance_zscore,
                                  cmap=args.colormap,
                                  cutoff_z=args.z_cutoff,
                                  over_color=args.over_color)

    # ------------------ Prepare label function -----------------
    if args.label_top_n or args.label_above_zcutoff:
        var = pd.Series(variance_sum_of_squares(X, Y), index=X_df.index)
        xy = np.column_stack((X.mean(axis=1), Y.mean(axis=1)))
        labels = np.array(var.index)
        if args.label_top_n:
            n = args.label_top_n
            top_n = pd.Series(var, index=X_df.index).nlargest(n).index
            labels[np.where(~np.in1d(labels, top_n))] = None
        else:
            labels[zscore(var) < args.z_cutoff] = None

        label_fn = functools.partial(smart_annotate, data=xy, labels=labels)
    else:
        label_fn = lambda ax: None

    # ------------------- Plot ----------------------------------
    f, ax = plt.subplots(dpi=600)
    multiblock_scatter_plot(X, Y, mode=args.point_style, color=color, ax=ax,
                            x_label=args.xlabel, y_label=args.ylabel,
                            title=args.title, label_fn=label_fn)

    if args.grid:
        ax.grid(args.grid)

    f.savefig(args.output, dpi=600)