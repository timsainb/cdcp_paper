import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
from tqdm.autonotebook import tqdm
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import collections as mc
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import gridspec
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree
from matplotlib import lines
import matplotlib.colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def scatter_spikes(
    z,
    spikes,
    z_samples=None,
    column_size=10,
    pal_color="hls",
    matshow_kwargs={"cmap": plt.cm.Greys},
    scatter_kwargs={"alpha": 0.5, "s": 1},
    line_kwargs={"lw": 1, "ls": "dashed", "alpha": 1},
    color_points=False,
    figsize=(10, 10),
    range_pad=0.1,
    x_range=None,
    y_range=None,
    enlarge_points=0,
    draw_lines=True,
    n_subset=-1,
    ax=None,
    show_scatter=True,
    border_line_width=1,
    range_pct=0.99,
    plot_spike_line=False,
    img_origin="lower",
):
    """"""
    n_columns = column_size * 4 - 4
    pal = sns.color_palette(pal_color, n_colors=n_columns)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(column_size, column_size)

    if x_range is None and y_range is None:
        xmin, xmax = np.sort(np.vstack(z)[:, 0])[
            np.array([int(len(z) * (1 - range_pct)), int(len(z) * range_pct)])
        ]
        ymin, ymax = np.sort(np.vstack(z)[:, 1])[
            np.array([int(len(z) * (1 - range_pct)), int(len(z) * range_pct)])
        ]
        # xmin, ymin = np.min(z, axis=0)
        # xmax, ymax = np.max(z, axis=0)
        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        xmin, xmax = x_range
        ymin, ymax = y_range

    x_block = (xmax - xmin) / column_size
    y_block = (ymax - ymin) / column_size

    # ignore segments outside of range
    z = np.array(z)
    mask = np.array(
        [(z[:, 0] > xmin) & (z[:, 1] > ymin) & (z[:, 0] < xmax) & (z[:, 1] < ymax)]
    )[0]

    if "labels" in scatter_kwargs:
        scatter_kwargs["labels"] = np.array(scatter_kwargs["labels"])[mask]
    if z_samples is None:
        spikes = np.array(spikes)[mask]
    z = z[mask]

    # prepare the main axis
    main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])
    # main_ax.scatter(z[:, 0], z[:, 1], **scatter_kwargs)
    if show_scatter:
        scatter_projections(projection=z, ax=main_ax, fig=fig, **scatter_kwargs)

    # loop through example columns
    axs = {}
    for column in range(n_columns):
        # get example column location
        if column < column_size:
            row = 0
            col = column

        elif (column >= column_size) & (column < (column_size * 2) - 1):
            row = column - column_size + 1
            col = column_size - 1

        elif (column >= ((column_size * 2) - 1)) & (column < (column_size * 3 - 2)):
            row = column_size - 1
            col = column_size - 3 - (column - column_size * 2)
        elif column >= column_size * 3 - 3:
            row = n_columns - column
            col = 0

        axs[column] = {"ax": fig.add_subplot(gs[row, col]), "col": col, "row": row}

        # sample a point in z based upon the row and column
        xpos = xmin + x_block * col + x_block / 2
        ypos = ymax - y_block * row - y_block / 2
        # main_ax.text(x=xpos, y=ypos, s=column, color=pal[column])

        axs[column]["xpos"] = xpos
        axs[column]["ypos"] = ypos

    main_ax.set_xlim([xmin, xmax])
    main_ax.set_ylim([ymin, ymax])

    # create a voronoi diagram over the x and y pos points
    points = [[axs[i]["xpos"], axs[i]["ypos"]] for i in axs.keys()]

    voronoi_kdtree = cKDTree(points)
    vor = Voronoi(points)

    # plot voronoi
    # voronoi_plot_2d(vor, ax = main_ax);

    # find where each point lies in the voronoi diagram
    if z_samples is None:
        z = z[:n_subset]
        point_dist, point_regions = voronoi_kdtree.query(list(z))
    else:
        point_dist, point_regions = voronoi_kdtree.query(list(z_samples))

    lines_list = []

    # loop through regions and select a point
    for key in axs.keys():
        # sample a point in (or near) voronoi region
        nearest_points = np.argsort(np.abs(point_regions - key))
        possible_points = np.where(point_regions == point_regions[nearest_points][0])[0]
        chosen_point = np.random.choice(a=possible_points, size=1)[0]
        point_regions[chosen_point] = 1e4

        def plot_chans(chans, ax):
            std_chan = np.std(chans)
            for ci, chan in enumerate(chans):
                ax.plot(chan + 3 * (std_chan * ci))  # , color="k")
            if plot_spike_line:
                ax.axvline(30, color="k", ls="dashed", alpha=0.5)

        plot_chans(spikes[chosen_point], axs[key]["ax"])

        axs[key]["ax"].set_xticks([])
        axs[key]["ax"].set_yticks([])
        if color_points:
            plt.setp(axs[key]["ax"].spines.values(), color=pal[key])

        for i in axs[key]["ax"].spines.values():
            i.set_linewidth(border_line_width)

        # draw a line between point and image
        if draw_lines:
            mytrans = (
                axs[key]["ax"].transAxes + axs[key]["ax"].figure.transFigure.inverted()
            )

            line_end_pos = [0.5, 0.5]

            if axs[key]["row"] == 0:
                line_end_pos[1] = 0
            if axs[key]["row"] == column_size - 1:
                line_end_pos[1] = 1

            if axs[key]["col"] == 0:
                line_end_pos[0] = 1
            if axs[key]["col"] == column_size - 1:
                line_end_pos[0] = 0

            infig_position = mytrans.transform(line_end_pos)
            if z_samples is None:
                xpos, ypos = main_ax.transLimits.transform(
                    (z[chosen_point, 0], z[chosen_point, 1])
                )
            else:
                xpos, ypos = main_ax.transLimits.transform(
                    (z_samples[chosen_point, 0], z_samples[chosen_point, 1])
                )
            mytrans2 = main_ax.transAxes + main_ax.figure.transFigure.inverted()
            infig_position_start = mytrans2.transform([xpos, ypos])

            lines_list.append(
                lines.Line2D(
                    [infig_position_start[0], infig_position[0]],
                    [infig_position_start[1], infig_position[1]],
                    transform=fig.transFigure,
                    **line_kwargs,
                )
            )
    if draw_lines:
        for l in lines_list:
            fig.lines.append(l)

    gs.update(wspace=0, hspace=0)
    # gs.update(wspace=0.5, hspace=0.5)

    fig = plt.gcf()

    if ax is not None:
        buf = io.BytesIO()
        plt.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        ax.imshow(im)
        plt.close(fig)

    return fig, axs, main_ax, [xmin, xmax, ymin, ymax]


def scatter_images(
    z,
    spikes,
    matshow_cmap='viridis',
    labels=None,
    z_samples=None,
    column_size=10,
    pal_color="hls",
    matshow_kwargs={"cmap": plt.cm.Greys},
    scatter_kwargs={"alpha": 0.5, "s": 1},
    line_kwargs={"lw": 1, "ls": "dashed", "alpha": 1},
    color_points=False,
    figsize=(10, 10),
    range_pad=0.1,
    x_range=None,
    y_range=None,
    enlarge_points=0,
    draw_lines=True,
    n_subset=-1,
    ax=None,
    show_scatter=True,
    border_line_width=1,
    range_pct=0.99,
    plot_spike_line=False,
    img_origin="lower",
):
    """"""
    n_columns = column_size * 4 - 4
    pal = sns.color_palette(pal_color, n_colors=n_columns)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(column_size, column_size)

    if x_range is None and y_range is None:
        xmin, xmax = np.sort(np.vstack(z)[:, 0])[
            np.array([int(len(z) * (1 - range_pct)), int(len(z) * range_pct)])
        ]
        ymin, ymax = np.sort(np.vstack(z)[:, 1])[
            np.array([int(len(z) * (1 - range_pct)), int(len(z) * range_pct)])
        ]
        # xmin, ymin = np.min(z, axis=0)
        # xmax, ymax = np.max(z, axis=0)
        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        xmin, xmax = x_range
        ymin, ymax = y_range

    x_block = (xmax - xmin) / column_size
    y_block = (ymax - ymin) / column_size

    # ignore segments outside of range
    z = np.array(z)
    mask = np.array(
        [(z[:, 0] > xmin) & (z[:, 1] > ymin) & (z[:, 0] < xmax) & (z[:, 1] < ymax)]
    )[0]

    if labels is not None:
        labels = np.array(labels)[mask]
    if z_samples is None:
        spikes = np.array(spikes)[mask]
    z = z[mask]

    # prepare the main axis
    main_ax = fig.add_subplot(gs[1 : column_size - 1, 1 : column_size - 1])
    # main_ax.scatter(z[:, 0], z[:, 1], **scatter_kwargs)
    if show_scatter:
        scatter_projections(projection=z, ax=main_ax, fig=fig, labels=labels, **scatter_kwargs)

    # loop through example columns
    axs = {}
    for column in range(n_columns):
        # get example column location
        if column < column_size:
            row = 0
            col = column

        elif (column >= column_size) & (column < (column_size * 2) - 1):
            row = column - column_size + 1
            col = column_size - 1

        elif (column >= ((column_size * 2) - 1)) & (column < (column_size * 3 - 2)):
            row = column_size - 1
            col = column_size - 3 - (column - column_size * 2)
        elif column >= column_size * 3 - 3:
            row = n_columns - column
            col = 0

        axs[column] = {"ax": fig.add_subplot(gs[row, col]), "col": col, "row": row}

        # sample a point in z based upon the row and column
        xpos = xmin + x_block * col + x_block / 2
        ypos = ymax - y_block * row - y_block / 2
        # main_ax.text(x=xpos, y=ypos, s=column, color=pal[column])

        axs[column]["xpos"] = xpos
        axs[column]["ypos"] = ypos

    main_ax.set_xlim([xmin, xmax])
    main_ax.set_ylim([ymin, ymax])

    # create a voronoi diagram over the x and y pos points
    points = [[axs[i]["xpos"], axs[i]["ypos"]] for i in axs.keys()]

    voronoi_kdtree = cKDTree(points)
    vor = Voronoi(points)

    # plot voronoi
    # voronoi_plot_2d(vor, ax = main_ax);

    # find where each point lies in the voronoi diagram
    if z_samples is None:
        z = z[:n_subset]
        point_dist, point_regions = voronoi_kdtree.query(list(z))
    else:
        point_dist, point_regions = voronoi_kdtree.query(list(z_samples))

    lines_list = []

    # loop through regions and select a point
    for key in axs.keys():
        # sample a point in (or near) voronoi region
        nearest_points = np.argsort(np.abs(point_regions - key))
        possible_points = np.where(point_regions == point_regions[nearest_points][0])[0]
        chosen_point = np.random.choice(a=possible_points, size=1)[0]
        point_regions[chosen_point] = 1e4

        axs[key]["ax"].matshow(spikes[chosen_point], aspect='auto', cmap=matshow_cmap)

        axs[key]["ax"].set_xticks([])
        axs[key]["ax"].set_yticks([])
        if color_points:
            plt.setp(axs[key]["ax"].spines.values(), color=pal[key])

        for i in axs[key]["ax"].spines.values():
            i.set_linewidth(border_line_width)

        # draw a line between point and image
        if draw_lines:
            mytrans = (
                axs[key]["ax"].transAxes + axs[key]["ax"].figure.transFigure.inverted()
            )

            line_end_pos = [0.5, 0.5]

            if axs[key]["row"] == 0:
                line_end_pos[1] = 0
            if axs[key]["row"] == column_size - 1:
                line_end_pos[1] = 1

            if axs[key]["col"] == 0:
                line_end_pos[0] = 1
            if axs[key]["col"] == column_size - 1:
                line_end_pos[0] = 0

            infig_position = mytrans.transform(line_end_pos)
            if z_samples is None:
                xpos, ypos = main_ax.transLimits.transform(
                    (z[chosen_point, 0], z[chosen_point, 1])
                )
            else:
                xpos, ypos = main_ax.transLimits.transform(
                    (z_samples[chosen_point, 0], z_samples[chosen_point, 1])
                )
            mytrans2 = main_ax.transAxes + main_ax.figure.transFigure.inverted()
            infig_position_start = mytrans2.transform([xpos, ypos])

            lines_list.append(
                lines.Line2D(
                    [infig_position_start[0], infig_position[0]],
                    [infig_position_start[1], infig_position[1]],
                    transform=fig.transFigure,
                    **line_kwargs,
                )
            )
    if draw_lines:
        for l in lines_list:
            fig.lines.append(l)

    gs.update(wspace=0, hspace=0)
    # gs.update(wspace=0.5, hspace=0.5)

    fig = plt.gcf()

    if ax is not None:
        buf = io.BytesIO()
        plt.savefig(buf, dpi=300, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        im = Image.open(buf)
        ax.imshow(im)
        plt.close(fig)

    return fig, axs, main_ax, [xmin, xmax, ymin, ymax]


def scatter_projections(
    projection,
    labels=None,
    ax=None,
    figsize=(10, 10),
    alpha=0.1,
    s=1,
    color="k",
    color_palette="tab20",
    categorical_labels=True,
    show_legend=True,
    tick_pos="bottom",
    tick_size=16,
    cbar_orientation="vertical",
    log_x=False,
    log_y=False,
    grey_unlabelled=True,
    fig=None,
    colornorm=False,
    rasterized=True,
    equalize_axes=True,
    print_lab_dict=False,  # prints color scheme
):
    """creates a scatterplot of syllables using some projection"""

    # color labels
    if labels is not None:
        if categorical_labels:
            if (color_palette == "tab20") & (len(np.unique(labels)) < 20):
                pal = sns.color_palette(color_palette, n_colors=20)
                pal = np.array(pal)[
                    np.linspace(0, 19, len(np.unique(labels))).astype("int")
                ]
                # print(pal)
            else:
                pal = sns.color_palette(color_palette, n_colors=len(np.unique(labels)))
            lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labels))}
            if grey_unlabelled:
                if -1 in lab_dict.keys():
                    lab_dict[-1] = [0.95, 0.95, 0.95, 1.0]
                if print_lab_dict:
                    print(lab_dict)
            colors = np.array([lab_dict[i] for i in labels])
    else:
        colors = color

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

        # plot
    if colornorm:
        norm = norm = matplotlib.colors.LogNorm()
    else:
        norm = None
    if categorical_labels or labels is None:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            color=colors,
            norm=norm,
        )

    else:
        cmin = np.quantile(labels, 0.01)
        cmax = np.quantile(labels, 0.99)
        sct = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            vmin=cmin,
            vmax=cmax,
            cmap=plt.get_cmap(color_palette),
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            c=labels,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if labels is not None:
        if categorical_labels == True:
            legend_elements = [
                Line2D([0], [0], marker="o", color=value, label=key)
                for key, value in lab_dict.items()
            ]
        if show_legend:
            if not categorical_labels:
                if cbar_orientation == "horizontal":
                    axins1 = inset_axes(
                        ax,
                        width="50%",  # width = 50% of parent_bbox width
                        height="5%",  # height : 5%
                        loc="upper left",
                    )
                    # cbar = fig.colorbar(sct, cax=axins1, orientation=cbar_orientation

                else:
                    axins1 = inset_axes(
                        ax,
                        width="5%",  # width = 50% of parent_bbox width
                        height="50%",  # height : 5%
                        loc="lower right",
                    )
                cbar = fig.colorbar(sct, cax=axins1, orientation=cbar_orientation)
                cbar.ax.tick_params(labelsize=tick_size)
                axins1.xaxis.set_ticks_position(tick_pos)
            else:
                ax.legend(handles=legend_elements)
    if equalize_axes:
        ax.axis("equal")
    return ax