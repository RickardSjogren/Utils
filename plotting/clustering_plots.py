import warnings
import scipy.cluster.hierarchy as scipy_hc
import scipy.spatial.distance as scipy_dist
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import numpy as np


def clustered_heatmap(data, cluster_kwargs=None, *args, **kwargs):
    """ Cluster data using hierarchical clustering and plot in a
    heatmap with dendrograms.

    Parameters
    ----------
    data : array_like
        Input data to plot.
    cluster_kwargs : dict, optional
        Keyword arguments passed to :py:`do_clustering`
    *args
        Positional arguments passed to :py:`make_clustered_heatmap`.
    **kwargs
        Keyword arguments passed to :py:`make_clustered_heatmap`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure-instance with plot.
    axes : numpy.ndarray
        Array with axes on which plot is drawn.
    row_link : numpy.ndarray
        Scipy linkage matrix for rows.
    col_link : numpy.ndarray
        Scipy linkage matrix for columns.
    dist : numpy.ndarray
        Distance matrix from clustering.
    """
    # Cluster and draw dendrograms.
    row_link, col_link = hierarchical_clustering(
        data, **(cluster_kwargs or dict()))

    f, axes = make_clustered_heatmap(data, col_link, row_link, *args, **kwargs)

    return f, axes, row_link, col_link


def make_clustered_heatmap(data, col_link, row_link, row_labels=None,
                           col_labels=None, row_colors=None, col_colors=None,
                           colorbar_label=None, row_color_kwargs=None,
                           col_color_kwargs=None):
    """ Given data and linkage, plot heatmap and dendrograms.

    Parameters
    ----------
    data : array_like
        Data array to plot.
    col_link : array_like
        Scipy linkage matrix for columns.
    row_link : array_like
        Scipy linkage matrix for rows.
    row_labels : list[str], optional
        Row labels to use.
    col_labels : list[str], optional
        Column labels to use.
    row_colors : list, optional
        One dimensional array of colors to use for rows.
    col_colors : list, optional
        One dimensional array of colors to use for columns.
    colorbar_label : str, optional
        Title bar of colorbar.
    row_color_kwargs : dict, optional
        Keyword arguments of row color-bar passed to
        :py:`plot_color_sequence`
    col_color_kwargs : dict, optional
        Keyword arguments of column color-bar passed
        to :py:`plot_color_sequence`

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Figure-instance with plot.
    axes : numpy.ndarray
        Array with axes on which plot is drawn.
    """
    fig = plt.figure()

    has_row_colors = row_colors is not None
    has_col_colors = col_colors is not None

    # Prepare axes.
    main_ax, left_ax, top_ax, cb_ax, gs = setup_clustermap_axes(fig,
                                                                has_col_colors,
                                                                has_row_colors)

    # Draw dendrograms.
    row_dendrogram = scipy_hc.dendrogram(
        row_link, orientation='left', link_color_func=lambda x: 'black',
        ax=left_ax, color_threshold=0)
    col_dendrogram = scipy_hc.dendrogram(
        col_link, orientation='top', link_color_func=lambda x: 'black',
        ax=top_ax, color_threshold=0)

    # Reorder data matrix.
    col_ind = col_dendrogram['leaves']
    row_ind = row_dendrogram['leaves']
    data = data[:, col_ind]
    data = data[row_ind, :]

    # Plot row or column colors if they are provided.
    if has_row_colors:
        row_colors = np.array(row_colors)[row_ind]
        row_col_ax = fig.add_subplot(gs[3 + int(has_col_colors):, 3])
        plot_color_sequence(row_colors, 'vertical', row_col_ax,
                            **(row_color_kwargs or dict()))
    else:
        row_col_ax = None

    if has_col_colors:
        col_colors = np.array(col_colors)[col_ind]
        col_col_ax = fig.add_subplot(gs[3, 3 + int(has_row_colors):])
        plot_color_sequence(col_colors, 'horizontal', col_col_ax,
                            **(col_color_kwargs or dict()))
    else:
        col_col_ax = None

    label_clustermap_axes(row_labels, col_labels, row_ind, col_ind, main_ax)

    # Pick color-map and -normalization from data amplitude.
    cb_ticks, cmap, norm = pick_colormap(data)

    # Draw heatmap and colorbar.
    img = main_ax.pcolormesh(data, cmap=plt.get_cmap(cmap), norm=norm)
    cb = plt.colorbar(img, cb_ax, orientation='horizontal')
    cb.set_ticks(cb_ticks)
    if colorbar_label is not None:
        cb_ax.set_title(colorbar_label)

    return fig, np.array([main_ax, left_ax, top_ax, cb_ax,
                          row_col_ax, col_col_ax])


def plot_network_vectors(data, linkage, pcolormesh_kwargs=None,
                         threshold=None, clust_cmap=None):
    """ Plot dendrogram with rearranged data-vectors underneath.

    Format of data is `{name: 1D-array}`

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Name-data pairs.
    linkage : array_like
        Scipy linkage matrix
    pcolormesh_kwargs : dict[str, dict], optional
        Keys are keys of `data` and values are keyword-arguments passed to
        `matplotlib.pyplot.pcolormesh`
    threshold : float, option
        If provided, clusters will be assigned using `'distance'`-criterion
        of :py:`scipy.cluster.hierarchy.fcluster` using `treshold` as cutoff.
    clust_cmap : matplotlib.colors.Colormap
        Colormap to draw colors from for assigned clusters according to `threshold`

    Returns
    -------
    matplotlib.Figure
    numpy.ndarray[matplotlib.Axes]
    dict
    np.ndarray
    dict
    """
    if clust_cmap is None:
        brewer_colors = [
            '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
            '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
        ]
        clust_cmap = colors.ListedColormap(brewer_colors)

    f = plt.figure()

    has_class_bar = threshold is not None
    n_subplots = len(data) + 1 + int(has_class_bar)
    height_ratios = [2] + ([.5] if has_class_bar else []) + [1] * len(data)
    gs = gridspec.GridSpec(n_subplots, 1, height_ratios=height_ratios)

    axes = np.array([f.add_subplot(gs[i]) for i in range(n_subplots)])

    axes[0].set_xticks([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    for ax in axes[1:]:
        ax.set_xticks([])
        ax.set_yticks([])

    dendrogram = scipy_hc.dendrogram(linkage, orientation='top', ax=axes[0],
                                     link_color_func=lambda x: 'black',
                                     color_threshold=0)

    ind = dendrogram['leaves']
    pcolormesh_kwargs = pcolormesh_kwargs or dict()

    if threshold is not None:
        clusters = scipy_hc.fcluster(linkage, threshold, 'distance')
        n_clust = len(set(clusters))
        n_colors = len(clust_cmap.colors)
        if n_clust > n_colors:
            axes[1].axis('off')
            cluster_coloring = None
            warnings.warn(
                'Too many clusters ({}), clusters not drawn'.format(n_clust))
        else:
            axes[1].pcolormesh(np.atleast_2d(clusters[ind]), cmap=clust_cmap,
                               vmin=0, vmax=n_colors)
            cluster_coloring = {clust: clust_cmap(clust) for clust in set(clusters)}
    else:
        clusters = None
        cluster_coloring = None

    for ax, (name, data) in zip(axes[1 + int(has_class_bar):], data.items()):
        try:
            kwargs = pcolormesh_kwargs[name]
        except KeyError:
            kwargs = dict()

        ax.pcolormesh(np.atleast_2d(data[ind]), **kwargs)
        ax.set_ylabel(name)

    f.subplots_adjust(hspace=0, wspace=0)

    return f, axes, dendrogram, clusters, cluster_coloring


def label_clustermap_axes(row_labels, col_labels, row_ind, col_ind, main_ax):
    """ Label clustermap axes.

    Parameters
    ----------
    row_labels : list[str] | None
        Row labels.
    col_labels : list[str] | None
        Column labels.
    row_ind : numpy.ndarray
        1D index-array for rows.
    col_ind : numpy.ndarray
        1D-index-array for columns.
    main_ax : matplotlib.Axes
        Axes containing heatmap.
    """
    if row_labels is not None:
        row_labels = np.array(row_labels)[row_ind]
        main_ax.yaxis.tick_right()
        main_ax.set_yticklabels(row_labels)
    else:
        main_ax.set_yticks([])
    if col_labels is not None:
        col_labels = np.array(col_labels)[col_ind]
        main_ax.set_xticklabels(col_labels)
    else:
        main_ax.set_xticks([])


def setup_clustermap_axes(fig, has_col_colors=False, has_row_colors=False):
    """ Instantiate and arrange subplot axes for clustermap.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure containing axes.
    has_col_colors : bool
        If True, place will be left for column color bar.
    has_row_colors : bool
        If True, place will be left for row color bar.

    Returns
    -------
    main_ax : matplotlib.Axes
        Axes for heatmap.
    left_ax : matplotlib.Axes
        Axes for left dendrogram.
    top_ax : matplotlib.Axes
        Axes for top dendrogram.
    cb_axis : matplotlib.Axes
        Axis for drawing heatmap colorbar.
    gs : matplotlib.gridspec.Gridspec
        Gridspec instance used for arranging subplots.
    """
    row_col_size = int(has_row_colors)
    col_col_size = int(has_col_colors)
    height = 16 + col_col_size
    width = 16 + row_col_size

    gs = gridspec.GridSpec(height, width)
    gs.update(wspace=0.1, hspace=0.1)

    main_ax = fig.add_subplot(gs[3 + col_col_size:, 3 + row_col_size:])
    left_ax = fig.add_subplot(gs[3 + col_col_size:, :3])
    top_ax = fig.add_subplot(gs[:3, 3 + row_col_size:])
    cb_ax = fig.add_subplot(gs[1, :3])

    left_ax.axis('off')
    top_ax.axis('off')
    main_ax.grid(False)

    return main_ax, left_ax, top_ax, cb_ax, gs


def pick_colormap(data):
    """ Determine colormap to use from input data.

    If all-negative or all-positive values, pick `viridis`,
    otherwise `BrBg`

    Parameters
    ----------
    data : array_like
        Input data.

    Returns
    -------
    cb_ticks : list
        Colobar axis tick
    cmap : matplotlib.colors.Colormap
        Picked colormap.
    norm : matplotlib.colors.Normalize
        `Normalize`-instance constructed from data.
    """
    amp_max = max(abs(data).min(), abs(data).max())
    if data.min() < 0 and data.max() > 0:
        cmap = 'bwr'
        norm = colors.Normalize(-amp_max, amp_max)
        cb_ticks = [norm.vmin, 0, norm.vmax]
    else:
        cmap = 'viridis'
        if data.min() >= 0:
            norm = matplotlib.colors.Normalize(0, amp_max)
            cb_ticks = [0, norm.vmax / 2, norm.vmax]
        else:
            norm = matplotlib.colors.Normalize(-amp_max, 0)
            cb_ticks = [norm.vmin, norm.vmin / 2, 0]

    return cb_ticks, cmap, norm


def hierarchical_clustering(data, distance='correlation', method='ward'):
    """ Perform hierarchical clustering on distance matrix.

    Parameters
    ----------
    data : array_like
        Data matrix to cluster, precompupted distances.
    distance : 'str'
        Distance metric to use. Passed as `metric`-key word to
        `scipy.spatial.distance.pdist` if not equal to `'precomputed'`
    method : str
        Linkage method, passed to `scipy.cluster.hierarchy.linkage`,
        default method is `"ward"`.
    metric : str
        Distance method, passed to `scipy.cluster.hierarchy.linkage`,
        defaults to Euclidean distance..

    Returns
    -------
    row_linkage : numpy.ndarray
        Row linkage matrix.
    col_linkage : numpy.ndarray
        Column linkage matrix.
    row_dist : numpy.ndarray
        Distance matrix
    """
    symmetric = False
    if distance == 'precomputed':
        try:
            symmetric = np.allclose(data, data.T)
        except ValueError:
            symmetric = False

        if not symmetric:
            raise ValueError('precomputed distance not symmetric')

        row_dist = col_dist = data.copy()
    else:
        try:
            row_dist = scipy_dist.squareform(
                scipy_dist.pdist(data, metric=distance))
            col_dist = scipy_dist.squareform(
                scipy_dist.pdist(data.T, metric=distance))
        except ValueError:
            raise

    row_linkage = scipy_hc.linkage(row_dist, method=method)

    if symmetric:
        col_linkage = row_linkage
    else:
        col_linkage = scipy_hc.linkage(col_dist, method=method)

    return row_linkage, col_linkage


def plot_color_sequence(color_seq, orientation, ax=None,
                        pcolormesh_kwargs=None):
    """

    Parameters
    ----------
    color_seq : array_like
        Array of valid matplotlib-colors or sequence of data values
        which will be mapped using `matplotlib.pyplot.pcolormesh`
    orientation : {'horizontal', 'vertical'}
        Orientation to plot on.
    ax : matplotlib.pyplot.Axes, optional
        Axis instance to plot on.
    pcolormesh_kwargs : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.pcolormesh`

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        Resulting color-mesh.
    """
    kwargs = dict()
    if ax is None:
        f, ax = plt.subplots()

    ax.set_xticks([])
    ax.set_yticks([])

    if color_seq.ndim == 1 and color_seq.dtype == float:
        color_array = color_seq
        amp_max = max(abs(color_array.min()), abs(color_array.max()))
        if color_array.min() < 0 and color_array.max() > 0:
            kwargs['cmap'] = 'BrBG'
            kwargs['vmin'] = -amp_max
            kwargs['vmax'] = amp_max
        else:
            kwargs['cmap'] = 'summer'
            if color_array.min() < 0:
                kwargs['vmin'] = -amp_max
                kwargs['vmax'] = 0
            else:
                kwargs['vmin'] = 0
                kwargs['vmax'] = amp_max
    else:
        unique_colors = np.vstack({tuple(element) for element in color_seq})
        color_array = np.empty((len(color_seq),), dtype=int)

        for color in unique_colors:
            where_color = np.where(color_seq == color)
            indexes = np.unique(where_color[0])
            color_array.flat[indexes] = np.where(unique_colors == color)[0]

        if unique_colors.dtype == np.dtype('<U1'):
            unique_colors = unique_colors.flatten()

        kwargs.update({
            'cmap': colors.ListedColormap(unique_colors),
            'vmin': 0,
            'vmax': len(unique_colors)
        })

    # Make sure array is oriented properly.
    if orientation == 'horizontal':
        color_array = np.atleast_2d(color_array)
    elif orientation == 'vertical':
        color_array = np.atleast_2d(color_array).T
    else:
        raise ValueError(orientation)

    kwargs.update(pcolormesh_kwargs or dict())
    mesh = ax.pcolormesh(color_array, **kwargs)

    return mesh