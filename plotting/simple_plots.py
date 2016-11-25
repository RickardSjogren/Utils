import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import patches
from scipy import stats


def series_bar_chart(series, n=30, reverse=False, **kwargs):
    """ Plot bar chart of top (or bottom) `n` observations of series.

    Parameters
    ----------
    series : pandas.Series
        Series with data.
    n : int
        Number of top (or bottom) observations to plot.
    reverse : bool
        If True, plot bottom `n` observations.
    **kwargs
        Keyword arguments passed to `matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    f, ax = plt.subplots(**kwargs)
    series.sort_values(ascending=reverse)[:n].plot.bar(ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0], xlim[1] + 1])
    ylim = ax.get_ylim()

    d = (ylim[1] - ylim[0]) * .05
    xd = (xlim[1] - xlim[0]) * .01
    ax.plot((n - d, n + d), (-d, d), color=(.85, .85, .85), clip_on=False,
            zorder=10)
    ax.plot((n + xd - d, n + xd + d), (-d, d), color=(.85, .85, .85),
            clip_on=False, zorder=10)
    ax.set_ylim(ylim)
    ax.grid(False)

    return f, ax


def hotelling_axis(series, alpha):
    """ Calculate confidence cut-off from origin for hotellings T2-
    ellipsis for 2D-score plot.

    Parameters
    ----------
    series : pandas.Series
        One-dimensional data.
    alpha : float
        Confidence cutoff.

    Returns
    -------
    float
    """
    s = series.var()
    N = len(series)
    f = stats.f.ppf(alpha, 2, N - 2)
    return np.sqrt(s * f * 2 * (N ** 2 - 1) / (N * (N - 2)))


def score_plot(df, columns, ax=None, alpha=.95, *args, **kwargs):
    """ Scatter-plot with Hotelling's T2-confidence ellipsis.

    Parameters
    ----------
    df : pandas DataFrame
        Data to plot.
    columns : list[str]
        Column labels.
    ax : matplotlib.pyplot.Axes, optional
        Axis to plot on.
    *args
        Positional arguments passed to :py:`pandas.DataFrame.plot.scatter`
    **kwargs
        Keyword arguments passed to :py:`pandas.DataFrame.plot.scatter`

    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    ax = df.plot.scatter(columns[0], columns[1],
                         *args, ax=ax, zorder=2, **kwargs)

    ellipse = patches.Ellipse(
        (0, 0),
        2 * hotelling_axis(df[columns[0]], alpha),
        2 * hotelling_axis(df[columns[1]], alpha),
        facecolor='none'
    )
    ax.add_patch(ellipse)

    xlim = max(map(abs, ax.get_xlim()))
    ylim = max(map(abs, ax.get_ylim()))
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.axvline(0, c=(.2, .2, .2), zorder=1)
    ax.axhline(0, c=(.2, .2, .2), zorder=1)
    return ax


def make_legend(color_dict, ax=None, *args, **kwargs):
    """ Create legend given label-color map.

    Parameters
    ----------
    color_dict : dict[str, color]
        Label-color mapping.
    ax : matplotlib.pyplot.Axes, optional
        Axis to plot on.
    *args
        Positional arguments passed :py:`matplotlib.pyplot.legend`.
    **kwargs
        Keyword arguments passed :py:`matplotlib.pyplot.legend`.
    Returns
    -------
    matplotlib.legend.Legend
    """
    handles = list()

    for label, color in color_dict.items():
        p = lines.Line2D([0], [0], color=(0, 0, 0, 0), label=label,
                         markeredgewidth=plt.rcParams['patch.linewidth'],
                         marker='o', markerfacecolor=color,
                         markeredgecolor=(0, 0, 0))
        handles.append(p)

    if ax is None:
        legend = plt.legend(handles=handles, *args, **kwargs)
    else:
        legend = plt.legend(handles=handles, *args, **kwargs)

    return legend