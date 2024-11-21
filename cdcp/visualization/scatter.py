import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datashader as ds
import colorcet as cc
import tempfile
import datashader.transfer_functions as transf
import seaborn as sns


def scatter_datashader(
    x,
    y,
    cat=None,
    cat_is_categorical=True,
    ax=None,
    percentile=98,
    pad_prop=0.25,
    resolution=600,
    pal="Set1",
    non_categorical_colormap=None,
):

    if non_categorical_colormap is None:
        # non_categorical_colormap = cc.coolwarm[::-1]
        non_categorical_colormap = list(sns.color_palette("viridis").as_hex())
    # compute range
    x_range = np.percentile(x, [100 - percentile, percentile])
    y_range = np.percentile(y, [100 - percentile, percentile])
    padding_x = (x_range[1] - x_range[0]) * pad_prop
    padding_y = (y_range[1] - y_range[0]) * pad_prop
    x_range = [x_range[0] - padding_x, x_range[1] + padding_x]
    y_range = [y_range[0] - padding_y, y_range[1] + padding_y]

    if cat is None:
        ds_df = pd.DataFrame({"x": x, "y": y})
    else:
        if cat_is_categorical:
            ds_df = pd.DataFrame({"x": x, "y": y, "cat": cat})
            ds_df["cat"] = ds_df["cat"].astype("category")
        else:
            ds_df = pd.DataFrame({"x": x, "y": y, "cat": cat})
            cat_range = np.percentile(cat, [100 - percentile, percentile])
            cat[cat < cat_range[0]] = cat_range[0]
            cat[cat > cat_range[1]] = cat_range[1]

    canvas = ds.Canvas(
        x_range=x_range, y_range=y_range, plot_width=resolution, plot_height=resolution
    )

    if cat is None:
        agg = canvas.points(ds_df, "x", "y", agg=ds.count())
        img = transf.shade(agg)  # , color_key=list(cc.glasbey_dark))
    else:
        if cat_is_categorical:
            agg = canvas.points(ds_df, "x", "y", agg=ds.count_cat("cat"))
            color_key = {
                i: "#%02x%02x%02x" % tuple((np.array(j) * 255).astype("int"))
                for i, j in zip(
                    np.unique(cat), sns.color_palette(pal, len(np.unique(cat)))
                )
            }
            img = transf.shade(agg, color_key=color_key)
        else:
            agg = canvas.points(ds_df, "x", "y", agg=ds.mean("cat"))
            img = transf.shade(agg, cmap=non_categorical_colormap, how="linear")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    with tempfile.NamedTemporaryFile(mode="wb") as png:
        ds.utils.export_image(
            img=img, filename=png.name, fmt=".png", background="white"
        )
        img_array = plt.imread(png.name + ".png", format=".png")
        im = ax.imshow(
            img_array,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            aspect="auto",
        )
    # ax.axis("equal")
