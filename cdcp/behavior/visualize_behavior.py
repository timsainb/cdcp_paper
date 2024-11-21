import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cdcp.visualization.colors import colors


def plot_summary_data(
    bird,
    bird_summary_data,
    bins,
    ax=None,
    error=True,
    lines=True,
    scatter=True,
    scatter_size=10,
    error_size=3,
    legend=True,
    title=True,
):
    bird_summary_data = bird_summary_data[bird_summary_data.cueing == True]

    if ax == None:
        _, ax = plt.subplots(nrows=1, figsize=(16, 4))
    if title:
        ax.set_title(bird)
    bins_dict = {i: b for i, b in enumerate(bins)}
    if len(bird_summary_data) == 0:
        return
    for cue_id in np.unique(bird_summary_data.cue_id):
        binsi = bins
        cue_df = bird_summary_data[bird_summary_data.cue_id == cue_id]
        if len(cue_df) == 0:
            print(len(cue_df))
            continue
        cue_str = cue_df.cue_prob["mean"].values[0]
        cue_dir = cue_df.cue_direction.iloc[0]
        if len(cue_df) < len(bins):
            binsi = [bins_dict[i] for i in cue_df.pos_bin]
        mean = cue_df.response_bool["mean"].values
        std = cue_df.response_bool["std"].values
        sem = std / np.sqrt(cue_df.response_bool["len"].values)
        if lines:
            ax.plot(
                binsi, mean, label=cue_id, color=colors["cue"][cue_dir][cue_str], lw=3
            )
        if error:
            ax.errorbar(
                binsi,
                mean,
                yerr=sem,
                xerr=None,
                color=colors["cue"][cue_dir][cue_str],
                lw=error_size,
                ls="none",
            )
        if scatter:
            ax.scatter(
                binsi,
                cue_df.response_bool["mean"].values,
                color=colors["cue"][cue_dir][cue_str],
                s=scatter_size,
                label=cue_id,
            )
    if legend:
        ax.legend()
    # ax.axvline(62.5, color="k", ls="dashed")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 127])


def moving_average(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_stages(data, n_avg=5000):
    _, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    ax = axs.flatten()[0]
    for cue in np.unique(data.cue_class.values):
        if cue == "N":
            continue
        (data.cue_class == cue).rolling(n_avg).mean().plot(
            ax=ax, label=cue, alpha=0.75, color=colors["cue_type"][cue], lw=3
        )
    ax.legend()
    ax.set_title("Cues")

    ax = axs.flatten()[1]
    for cue_prob in np.unique(np.unique(data.cue_prob.values)):
        if cue == "N":
            continue
        (data.cue_prob == cue_prob).rolling(n_avg).mean().plot(
            ax=ax, label=cue_prob, alpha=cue_prob, color="k", lw=3
        )
    ax.legend()
    ax.set_title("Cue probability")

    ax = axs.flatten()[2]
    for left_stim in np.unique(data.left_stim.values):
        (data.left_stim == left_stim).rolling(n_avg).mean().plot(
            ax=ax, label=left_stim, alpha=0.75, lw=3
        )
    ax.legend()
    ax.set_title("Stims")

    ax = axs.flatten()[3]
    ax.plot(
        data.interpolation_point.isin([0, 127]).rolling(n_avg).mean(), lw=3, color="k"
    )
    ax.set_title("Binary stims")

    ax = axs.flatten()[4]
    ax.plot(data["correct"].fillna(0).rolling(n_avg).mean().fillna(0), label="accuracy")
    ax.plot(
        (data["response"] == "L").fillna(0).rolling(n_avg).mean().fillna(0),
        label="%  Left",
    )
    ax.legend()
    ax.set_ylim([0, 1])
    ax.set_title("Performance")

    ax = axs.flatten()[5]
    ax.plot((data["correct"] == True).cumsum(skipna=True))
    ax.set_title("Number of trials")

    ax = axs.flatten()[6]
    data[data.type_ == "normal"][["cue_id", "interpolation_point"]][-10000:].pivot(
        columns="cue_id", values="interpolation_point"
    ).plot.hist(stacked=True, bins=32, ax=ax)
    ax.set_title("stimuli frequency")

    ax = axs.flatten()[7]
    data["left_class"] = np.array(data["class_"] == "L").astype("int")
    data[data.type_ == "normal"][["cue_id", "left_class"]][-10000:].pivot(
        columns="cue_id", values="left_class"
    ).plot.hist(stacked=True, bins=2, ax=ax)
    ax.set_title("stimuli frequency")

    plt.tight_layout()
    plt.show()
