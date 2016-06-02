from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Span
from bokeh.io import gridplot
import pandas as pd
import itertools

BREWER12plus12 = [
    '#a6cee3',
    '#1f78b4',
    '#b2df8a',
    '#33a02c',
    '#fb9a99',
    '#e31a1c',
    '#fdbf6f',
    '#ff7f00',
    '#cab2d6',
    '#6a3d9a',
    '#ffff99',
    '#b15928',
    '#8dd3c7',
    '#ffffb3',
    '#bebada',
    '#fb8072',
    '#80b1d3',
    '#fdb462',
    '#b3de69',
    '#fccde5',
    '#d9d9d9',
    '#bc80bd',
    '#ccebc5',
    '#ffed6f'
]

def scatter_with_hover(df, x, y,
                       fig=None, cols=None, name=None, marker='x',
                       fig_width=500, fig_height=500, **kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic
    tooltips showing columns from `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted
    x : str
        Name of the column to use for the x-axis values
    y : str
        Name of the column to use for the y-axis values
    fig : bokeh.plotting.Figure, optional
        Figure on which to plot (if not given then a new figure will be created)
    cols : list of str
        Columns to show in the hover tooltip (default is to show all)
    name : str
        Bokeh series name to give to the scattered data
    marker : str
        Name of marker to use for scatter plot
    **kwargs
        Any further arguments to be passed to fig.scatter

    Returns
    -------
    bokeh.plotting.Figure
        Figure (the same as given, or the newly created figure)

    Example
    -------
    fig = scatter_with_hover(df, 'A', 'B')
    show(fig)
    fig = scatter_with_hover(df, 'A', 'B', cols=['C', 'D', 'E'], marker='x', color='red')
    show(fig)

    Author
    ------
    Robin Wilson <robin@rtwilson.com>
    with thanks to Max Albert for original code example
    """

    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        fig = figure(width=fig_width, height=fig_height, tools=['box_zoom', 'reset'])

    # We're getting data from the given dataframe
    source = ColumnDataSource(data=df)

    # We need a name so that we can restrict hover tools to just this
    # particular 'series' on the plot. You can specify it (in case it
    # needs to be something specific for other reasons), otherwise
    # we just use 'main'
    if name is None:
        name = 'main'

    # Actually do the scatter plot - the easy bit
    # (other keyword arguments will be passed to this function)
    fig.scatter(df[x], df[y], source=source, name=name, marker=marker, **kwargs)

    # Now we create the hover tool, and make sure it is only active with
    # the series we plotted in the previous line
    hover = HoverTool(names=[name])

    if cols is None:
        # Display *all* columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in df.columns]
    else:
        # Display just the given columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in cols]

    hover.tooltips.append(('index', '$index'))

    # Finally add/enable the tool
    fig.add_tools(hover)

    return fig


def plot_score_loading(scores, loadings, R2, n_pcs=2,
                       obs_labels=None, var_labels=None,
                       obs_class_labels=None, var_class_labels=None,
                       height=500, width=500):
    """ For each component combination of `n_pcs` first component
    plot score and loading pairs.

    Parameters
    ----------
    scores : array_like
        Array with all model scores as columns.
    loadings : array_like
        Array with all model loadings as columns.
    R2 : array_like
        Array with R2-values for each model component.
    n_pcs : int
        Number of components to use.
    obs_labels : list[str], optional
        Observation labels.
    var_labels : list[str], optional
        Variable labels.
    obs_class_labels : list[str], optional
        Observation classification.
    var_class_labels : list[str], optional
        Variable classification.
    height : int
        Height of each subplot, default 500.
    width : int
        Width of each subplot, default 500.

    Returns
    -------
    bokeh.plotting.figure
    """
    cols = ['C{}'.format(i + 1) for i in range(scores.shape[1])]

    obs_labels = obs_labels or list(map(str, range(len(scores))))
    var_labels = var_labels or list(map(str, range(len(loadings))))

    score_df = pd.DataFrame(scores, index=obs_labels, columns=cols)
    loading_df = pd.DataFrame(loadings, index=var_labels, columns=cols)

    score_df['labels'] = obs_labels
    loading_df['labels'] = var_labels

    if obs_class_labels is not None:
        score_df['class'] = obs_class_labels
    if var_class_labels is not None:
        loading_df['class'] = var_class_labels

    score_color_map = make_color_mapping(obs_class_labels, BREWER12plus12)
    loading_color_map = make_color_mapping(var_class_labels,
                                           BREWER12plus12[len(score_color_map):])

    score_source = ColumnDataSource(data=score_df)
    loading_source = ColumnDataSource(data=loading_df)

    plot_pairs = list()
    TOOLS = "box_select,lasso_select,pan,help,box_zoom,wheel_zoom,reset"

    for i, j in itertools.combinations_with_replacement(range(n_pcs), 2):
        if i == j:
            continue
        PC1 = 'C{}'.format(i + 1)
        PC2 = 'C{}'.format(j + 1)

        # Score plot.
        score_title = 't[{}] vs t[{}]'.format(i + 1, j + 1)
        t1_lab = 't[{}] ({:.2f} %)'.format(i + 1, R2[i])
        t2_lab = 't[{}] ({:.2f} %)'.format(j + 1, R2[j])

        score_plot = scatter_plot(PC1, PC2, score_source, obs_class_labels,
                                  score_color_map, tools=TOOLS, height=height,
                                  title=score_title, x_axis_label=t1_lab,
                                  y_axis_label=t2_lab, width=width)

        # Loading plot.
        p1_lab = 'p[{}]'.format(i + 1)
        p2_lab = 'p[{}]'.format(j + 1)
        loading_title = '{} vs {}'.format(p1_lab, p2_lab)

        loading_plot = scatter_plot(PC1, PC2, loading_source, var_class_labels,
                                    loading_color_map, tools=TOOLS,
                                    height=height, title=loading_title,
                                    x_axis_label=p1_lab, y_axis_label=p2_lab,
                                    width=width)

        plot_pairs.append([score_plot, loading_plot])

    fig = gridplot(plot_pairs)
    return fig


def scatter_plot(x, y, source, class_labels=None, cmap=None, **kwargs):
    """ Make class colored scatter-plot.

    Parameters
    ----------
    x : str
        X-column.
    y : str
        Y-column.
    source : bokeh.plotting.ColumnDataSource
        Data to plot.
    class_labels : list, optional
        Class labels.
    cmap : dict, optional
        Color mapping of class labels, needid if `class_labels`
        is provided.
    **kwargs
        Keyword arguments passed to `bokeh.plotting.figure`.
    Returns
    -------
    bokeh.plotting.figure.Figure
    """
    plot = figure(**kwargs)
    tooltips = {'x, y': '@{}, @{}'.format(x, y),
                'i': '@labels'}

    hover = HoverTool(tooltips=tooltips)
    if class_labels:
        colors = list()
        for c in set(class_labels):
            plot.circle([], [], color=cmap[c], legend=c)
        for c in class_labels:
            colors.append(cmap[c])
        hover.tooltips.append(('Class', '@class'))
    else:
        colors = '#3182bd'

    plot.circle(x, y, source=source, color=colors)
    plot.add_tools(hover)

    hline = Span(location=0, dimension='width')
    vline = Span(location=0, dimension='height')

    plot.renderers.extend([hline, vline])

    return plot


def make_color_mapping(class_labels, palette):
    """ Make class-to-color mapping.

    Parameters
    ----------
    class_labels : list[str]
        Sequence of labels.
    palette : list
        Sequence of colors.

    Returns
    -------
    dict
    """
    color_map = dict()
    if class_labels is not None:
        classes = pd.Series(class_labels)
        for i, c in enumerate(classes.unique()):
            color_map[c] = palette[i]

    return color_map


if __name__ == '__main__':
    import numpy as np
    scores = np.random.randn(10, 3)
    loadings = np.random.randn(20, 3)
    R2 = np.array([31.9012398021, 12.09821390213, 1.9082139021])
    obs_labels = list('abcdefghij')
    var_labels = list('ABCDEFGHIJKLMNOPQRST')
    obs_class_labels = list('aaaabbbbcc')
    var_class_labels = list('AAAAAAAAAABBBBBCCDDD')
    plot_score_loading(scores, loadings, R2)