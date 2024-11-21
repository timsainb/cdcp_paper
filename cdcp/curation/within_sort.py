import numpy
from scipy.stats import pearsonr
import spikeinterface.widgets as sw
import spikeinterface.toolkit as st
import spikeextractors as se
from IPython.display import display
import matplotlib
from umap import UMAP
from cuml import UMAP as cumlUMAP
import matplotlib.pyplot as plt


def get_putatively_overlapping_units(
    unit, geom, sort_units, sort, max_max_channel_distance, labels_recording
):

    # get max channel
    max_channel = sort.get_unit_property(unit, "max_channel")

    # get channels that overlap with this one
    max_channel_position = geom[max_channel]
    potentially_overlapping_channels = [
        unit_n
        for unit_n, i in enumerate(geom)
        if np.linalg.norm(max_channel_position - i) < max_max_channel_distance
    ]

    # get units that overlap with this one
    potentially_overlapping_units = [
        unit_i
        for unit_i in sort_units
        if (
            sort.get_unit_property(unit_i, "max_channel")
            in potentially_overlapping_channels
        )
        & (unit_i > unit)
    ]
    potentially_overlapping_units = [
        i for i in potentially_overlapping_units if i in np.unique(labels_recording)
    ]
    return np.array(potentially_overlapping_units)


def plot_unit_comparison(
    sort,
    unit,
    comparison_unit,
    row,
    percentile=95,
    pad_prop=0.25,
    resolution=600,
    pal="Set1",
):

    # get umap projection for comparison
    pc_features_unit = sort.get_unit_spike_features(unit, "pc_features")
    pc_features_comparison_unit = sort.get_unit_spike_features(
        comparison_unit, "pc_features"
    )

    pc_features_unit = np.reshape(
        pc_features_unit,
        (len(pc_features_unit), np.product(np.shape(pc_features_unit)[1:])),
    )
    pc_features_comparison_unit = np.reshape(
        pc_features_comparison_unit,
        (
            len(pc_features_comparison_unit),
            np.product(np.shape(pc_features_comparison_unit)[1:]),
        ),
    )

    # model = UMAP(verbose=True)

    # pc_umap = model.fit_transform(np.vstack([pc_features_unit, pc_features_comparison_unit]))
    # pc_umap = np.random.rand(len(pc_features_unit)+len(pc_features_comparison_unit),2)
    model = cumlUMAP()  # verbose=True)
    pc_umap = model.fit_transform(
        np.vstack([pc_features_unit, pc_features_comparison_unit])
    )

    ## create an image of the cross correlogram and export as an image array
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    w_cch = sw.plot_crosscorrelograms(
        sort, unit_ids=[unit, comparison_unit], bin_size=0.1, window=5, ax=ax2
    )
    ax2.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax2.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    with tempfile.NamedTemporaryFile(mode="wb") as png:
        plt.savefig(png.name, bbox_inches="tight", pad_inches=0, transparent=True)
        cc_img_array = plt.imread(png.name + ".png", format=".png")
    plt.close()

    fix, ax = plt.subplots(figsize=(4, 1.5))
    w_ach = sw.plot_autocorrelograms(
        sort, bin_size=1, window=10, unit_ids=[unit, comparison_unit], ax=ax
    )
    with tempfile.NamedTemporaryFile(mode="wb") as png:
        plt.savefig(png.name, bbox_inches="tight", pad_inches=0, transparent=True)
        ac_img_array = plt.imread(png.name + ".png", format=".png")
    plt.close()

    fix, ax = plt.subplots(figsize=(4, 1.5))
    w_ach = sw.plot_isi_distribution(
        sort, bins=10, window=1, unit_ids=[unit, comparison_unit], ax=ax
    )
    with tempfile.NamedTemporaryFile(mode="wb") as png:
        plt.savefig(png.name, bbox_inches="tight", pad_inches=0, transparent=True)
        isi_img_array = plt.imread(png.name + ".png", format=".png")
    plt.close()

    cat_masked = np.array(
        [i if i in [unit, comparison_unit] else 1e4 for i in row.spike_labels]
    )

    # make big plot
    fig = plt.figure(figsize=(11 * 3, 4 * 3), constrained_layout=True)
    gs = fig.add_gridspec(4, 11)

    ax1 = ax = fig.add_subplot(gs[:2, :2])
    x, y = row.spike_projections.astype("float32").T
    scatter_datashader(
        x,
        y,
        cat_masked,
        ax=ax1,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )

    ax2 = ax = fig.add_subplot(gs[:2, 2:4])
    x, y = (
        np.vstack(
            [
                row.spike_projections[unit_mask],
                row.spike_projections[comparison_unit_mask],
            ]
        )
        .astype("float32")
        .T
    )
    scatter_datashader(
        x,
        y,
        cat,
        ax=ax2,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Units")
    ax.axis("off")
    [xmin, xmax] = ax2.get_xlim()
    [ymin, ymax] = ax2.get_ylim()
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor=(0, 0, 0, 0.5),
        facecolor="none",
    )
    ax1.add_patch(rect)

    ax35 = ax = fig.add_subplot(gs[1, 4])
    x, y = (
        np.vstack(
            [
                row.spike_projections[unit_mask],
                row.spike_projections[comparison_unit_mask],
            ]
        )
        .astype("float32")
        .T
    )
    cat_mc = np.concatenate(
        [row.max_channel[unit_mask], row.max_channel[comparison_unit_mask]]
    ).astype("int")
    scatter_datashader(
        x,
        y,
        cat=cat_mc,
        cat_is_categorical=True,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
        pal=pal,
    )
    ax.set_title("Max channel")
    ax.axis("off")

    ax3 = ax = fig.add_subplot(gs[0, 4])
    x, y = (
        np.vstack(
            [
                row.spike_projections[unit_mask],
                row.spike_projections[comparison_unit_mask],
            ]
        )
        .astype("float32")
        .T
    )
    cat_w = np.concatenate(
        [
            row.spike_width_samples[unit_mask],
            row.spike_width_samples[comparison_unit_mask],
        ]
    ).astype("float32")
    scatter_datashader(
        x,
        y,
        cat=cat_w,
        cat_is_categorical=False,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Spike width")
    ax.axis("off")

    ax4 = ax = fig.add_subplot(gs[0, 5])
    x, y = (
        np.vstack(
            [
                row.spike_projections[unit_mask],
                row.spike_projections[comparison_unit_mask],
            ]
        )
        .astype("float32")
        .T
    )
    amp = row.spike_amplitudes
    amp_min, amp_max = np.percentile(row.spike_amplitudes, [5, 95])
    amp[amp > amp_max] = amp_max
    amp[amp < amp_min] = amp_min
    cat_w = np.concatenate([amp[unit_mask], amp[comparison_unit_mask]]).astype(
        "float32"
    )
    scatter_datashader(
        x,
        y,
        cat=cat_w,
        cat_is_categorical=False,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Spike Amplitude")
    ax.axis("off")

    ax5 = ax = fig.add_subplot(gs[1, 5])
    x, y = (
        np.vstack(
            [
                row.spike_projections[unit_mask],
                row.spike_projections[comparison_unit_mask],
            ]
        )
        .astype("float32")
        .T
    )
    amp_diff = row.top_channels_amp_diff
    amp_diff_min, amp_diff_max = np.percentile(amp_diff, [5, 95])
    amp_diff[amp_diff > amp_diff_max] = amp_diff_max
    amp_diff[amp_diff < amp_diff_min] = amp_diff_min
    cat_w = np.concatenate(
        [amp_diff[unit_mask], amp_diff[comparison_unit_mask]]
    ).astype("float32")
    scatter_datashader(
        x,
        y,
        cat=cat_w,
        cat_is_categorical=False,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Amp diff")
    ax.axis("off")

    ax6 = ax = fig.add_subplot(gs[0:2, 6:8])
    x = np.concatenate(
        [row.spike_times[unit_mask], row.spike_times[comparison_unit_mask]]
    ).astype("float32")

    #
    y = np.concatenate(
        [
            sort.get_unit_spike_features(unit, "amplitudes"),
            sort.get_unit_spike_features(comparison_unit, "amplitudes"),
        ]
    ).flatten()
    # y = np.concatenate(
    #    [row.spike_amplitudes[unit_mask], row.spike_amplitudes[comparison_unit_mask]]
    # ).astype("float32")
    scatter_datashader(
        x,
        y,
        cat,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Amplitude x Time")

    ax7 = ax = fig.add_subplot(gs[0, 8:10])
    x = np.concatenate(
        [row.spike_times[unit_mask], row.spike_times[comparison_unit_mask]]
    ).astype("float32")
    y = np.concatenate(
        [
            row.spike_width_samples[unit_mask],
            row.spike_width_samples[comparison_unit_mask],
        ]
    ).astype("float32")
    scatter_datashader(
        x,
        y,
        cat,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Spike width x Time")

    ax8 = ax = fig.add_subplot(gs[1, 8:10])
    amp_diff = row.top_channels_amp_diff
    amp_diff_min, amp_diff_max = np.percentile(amp_diff, [5, 95])
    amp_diff[amp_diff > amp_diff_max] = amp_diff_max
    x = np.concatenate(
        [row.spike_times[unit_mask], row.spike_times[comparison_unit_mask]]
    ).astype("float32")
    y = np.concatenate([amp_diff[unit_mask], amp_diff[comparison_unit_mask]]).astype(
        "float32"
    )
    scatter_datashader(
        x,
        y,
        cat,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("Amp diff. x Time")

    ax8 = ax = fig.add_subplot(gs[0:2, 10])
    ax.set_title("Spike geometric mean")
    x = np.concatenate(
        [row.est_spike_x[unit_mask], row.est_spike_x[comparison_unit_mask]]
    ).astype("float32")
    y = np.concatenate(
        [row.est_spike_y[unit_mask], row.est_spike_y[comparison_unit_mask]]
    ).astype("float32")
    nanmask = (np.isnan(x) == False) & (np.isnan(y) == False)
    scatter_datashader(
        x[nanmask],
        y[nanmask],
        cat[nanmask],
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )

    ax81 = ax = fig.add_subplot(gs[2:4, 0:2])
    x, y = row.spike_projections.astype("float32").T
    c = np.concatenate(
        [np.zeros(len(pc_features_unit)), np.ones(len(pc_features_comparison_unit))]
    )
    scatter_datashader(
        pc_umap[:, 0],
        pc_umap[:, 1],
        c,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("PCA UMAP X Label")

    ax82 = ax = fig.add_subplot(gs[2:4, 2:4])
    x, y = row.spike_projections.astype("float32").T
    c = np.concatenate(
        [sort.get_unit_spike_train(unit), sort.get_unit_spike_train(comparison_unit)]
    )
    scatter_datashader(
        pc_umap[:, 0],
        pc_umap[:, 1],
        c,
        cat_is_categorical=False,
        ax=ax,
        percentile=percentile,
        pad_prop=pad_prop,
        resolution=resolution,
    )
    ax.set_title("PCA UMAP X Time")

    ax9 = ax = fig.add_subplot(gs[2, 4:6])
    ax.plot(unit_template.T)
    ax.set_title(unit)

    ax10 = ax = fig.add_subplot(gs[3, 4:6])
    ax.plot(comparison_template.T)
    ax.set_title(comparison_unit)
    ax10.get_shared_x_axes().join(ax10, ax9)

    ax11 = ax = fig.add_subplot(gs[2:4, 6:9])
    ax.imshow(cc_img_array)
    ax.set_title("Crosscorrelograms")
    ax.axis("off")

    ax12 = ax = fig.add_subplot(gs[2, 9:11])
    ax.imshow(ac_img_array)
    ax.set_title("Autocorr.")
    ax.axis("off")

    ax13 = ax = fig.add_subplot(gs[3, 9:11])
    ax.imshow(isi_img_array)
    ax.set_title("ISI.")
    ax.axis("off")
