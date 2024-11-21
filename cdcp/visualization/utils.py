import numpy as np
import seaborn as sns


def get_cat_colors(cat, palette="tab10", randomize=True):
    unique_cat = np.unique(cat)
    pal = sns.color_palette(palette, len(unique_cat))
    if randomize:
        pal = np.random.permutation(pal)
    pal_dict = {cat: pal[i] for i, cat in enumerate(unique_cat)}
    colors = [pal_dict[i] for i in cat]
    return colors, pal, pal_dict


def get_scatter_limits(x, y, _range=95, padding=0.2):
    d = (100 - _range) / 2
    x_min, x_max = np.percentile(x, (d, 100 - d))
    y_min, y_max = np.percentile(y, (d, 100 - d))
    y_pad = (y_max - y_min) * padding
    x_pad = (x_max - x_min) * padding
    y_min -= y_pad
    x_min -= x_pad
    y_max += y_pad
    x_max += y_pad
    xlim = (x_min, x_max)
    ylim = (y_min, y_max)
    return xlim, ylim
