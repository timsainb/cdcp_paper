import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from scipy.stats import ttest_ind_from_stats


import copy
from cdcp.visualization.colors import colors as cdcp_colors
from cdcp.spiketrain_analysis.spiketrain_utils import (
    bin_interp_points,
    get_average_response_vector,
)


def visualize_spikerate_by_cue_and_class(
    stim_class_by_stim_class,
    cued_spikes_by_cue,
    cued_spikes_by_interp_point,
    cued_spikes_by_stim_class,
):
    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    ax = axs[0]
    labels = [row.stim_class for idx, row in stim_class_by_stim_class.iterrows()]
    ticks = [1, 2]
    colors = [
        cdcp_colors["reinforce"][i]["dark"]
        for i in stim_class_by_stim_class.stim_class.values
    ]
    ax.bar(
        ticks,
        stim_class_by_stim_class["mean"].values,
        color=colors,
        linewidth=5,
        yerr=stim_class_by_stim_class["sem"].values,
    )
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=labels)
    ax.set_title("spikerate by class")

    ax = axs[1]
    labels = [row.cue for idx, row in cued_spikes_by_cue.iterrows()]
    ticks = [1, 2, 3]
    colors = [cdcp_colors["cue"][i[1]]["light"] for i in cued_spikes_by_cue.cue.values]
    ax.bar(
        ticks,
        cued_spikes_by_cue["mean"].values,
        color=colors,
        linewidth=5,
        yerr=cued_spikes_by_cue["sem"].values,
    )
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=labels)
    ax.set_title("spikerate by cue")

    ax = axs[2]
    edgecolors = [
        cdcp_colors["reinforce"][i]["dark"]
        for i in cued_spikes_by_stim_class.stim_class.values
    ]
    labels = [
        row.cue + "->" + row.stim_class
        for idx, row in cued_spikes_by_stim_class.iterrows()
    ]
    ticks = [0, 1, 2.5, 3.5, 5, 6]
    colors = [
        cdcp_colors["cue"][i[1]]["light"] for i in cued_spikes_by_stim_class.cue.values
    ]
    ax.bar(
        ticks,
        cued_spikes_by_stim_class["mean"].values,
        edgecolor=edgecolors,
        color=colors,
        linewidth=5,
        yerr=cued_spikes_by_stim_class["sem"].values,
    )
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=labels)
    ax.set_title("spikerate by cue x class")

    ax = axs[3]
    for cue in ["NC", "CL", "CR"]:
        mask = cued_spikes_by_interp_point.cue == cue
        x = cued_spikes_by_interp_point[mask].interp_point_binned.values
        y = cued_spikes_by_interp_point[mask]["mean"].values
        sd = cued_spikes_by_interp_point[mask]["std"].values
        n = cued_spikes_by_interp_point[mask]["count"].values
        sem = sd / np.sqrt(n)
        ax.plot(x, y, label=cue, color=cdcp_colors["cue"][cue[1]]["light"], lw=3)
        ax.fill_between(
            x, y - sem, y + sem, alpha=0.1, color=cdcp_colors["cue"][cue[1]]["light"]
        )
    ax.legend()
    ax.set_title("spikerate by cue x interpolation point")
    plt.show()


def get_z_scored_spikerate(row, spikerate_stats_by_interp_point):
    if (row.interp, row.interp_point_binned) in spikerate_stats_by_interp_point.index:
        mean_val = spikerate_stats_by_interp_point.loc[
            (row.interp, row.interp_point_binned)
        ]["mean"]
        std_val = spikerate_stats_by_interp_point.loc[
            (row.interp, row.interp_point_binned)
        ]["std"]
        z_score = (row.n_spikes - mean_val) / std_val
    else:
        z_score = np.nan
    return z_score


def z_score(x):
    return (x - np.mean(x)) / np.std(x)


def compute_spike_counts_by_cue_and_interp_class(
    trial_aligned_spikes, n_interp_point_bins=16, flip_bins=True, plot=False
):

    # subset only active trials for analyses
    trial_aligned_spikes = copy.deepcopy(
        trial_aligned_spikes[trial_aligned_spikes.passive == False]
    )
    trial_aligned_spikes = copy.deepcopy(
        trial_aligned_spikes[trial_aligned_spikes.response.isin(["left", "right"])]
    )

    if len(trial_aligned_spikes) < 100:
        print("too few trials: skipping")
        return

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point_binned"] = bin_interp_points(
        trial_aligned_spikes["interp_point"].values.astype(int),
        n_interp_point_bins,
        flip_bins=flip_bins,
    )

    # count spikes of only categorical stim for each trial
    trial_aligned_spikes["n_spikes"] = [
        np.sum((row.spike_times > 0) & (row.spike_times < 1))
        if row.cue == "NC"
        else np.sum((row.spike_times > 1) & (row.spike_times < 2))
        for idx, row in trial_aligned_spikes.iterrows()
    ]

    # stim class
    if flip_bins:
        trial_aligned_spikes["stim_class"] = "R"
        trial_aligned_spikes.loc[
            trial_aligned_spikes.interp_point > 63, "stim_class"
        ] = "L"
    else:
        trial_aligned_spikes["stim_class"] = "L"
        trial_aligned_spikes.loc[
            trial_aligned_spikes.interp_point > 63, "stim_class"
        ] = "R"

    # get descriptive stats of spikerates for each interpolation and interp point
    spikerate_stats_by_interp_point = (
        trial_aligned_spikes[["interp_point_binned", "interp", "n_spikes"]]
        .groupby(["interp", "interp_point_binned"])
        .describe()["n_spikes"][["count", "mean", "std"]]
    )

    # z score all spikerates using descriptive stats
    z_scored_spikerate = np.array(
        [
            get_z_scored_spikerate(row, spikerate_stats_by_interp_point)
            for idx, row in tqdm(
                trial_aligned_spikes.iterrows(),
                total=len(trial_aligned_spikes),
                desc="zscore",
                leave=False,
            )
        ]
    )

    # add spikerates to dataset
    trial_aligned_spikes = copy.deepcopy(
        trial_aligned_spikes[pd.isnull(z_scored_spikerate) == False]
    )
    trial_aligned_spikes["z_scored_spikerate"] = z_scored_spikerate[
        pd.isnull(z_scored_spikerate) == False
    ]

    if len(trial_aligned_spikes) < 100:
        print("too few trials after z score: skipping")
        return

    # get overall z scored spikerate (not controlling for anything)
    trial_aligned_spikes["z_scored_overall_spikerate"] = z_score(
        trial_aligned_spikes["n_spikes"].values
    )

    # spike rate by stim class
    stim_class_by_stim_class = (
        trial_aligned_spikes[["z_scored_overall_spikerate", "stim_class"]]
        .groupby("stim_class")
        .describe()["z_scored_overall_spikerate"][["count", "mean", "std"]]
        .reset_index()
    )

    # spike rate by stim class and interp x cue
    cued_spikes_by_cue_list = []
    cued_spikes_by_interp_point_list = []
    cued_spikes_by_stim_class_list = []
    for cue_list, identifier in tqdm(
        [
            (["CL0", "CL1"], "CL"),
            (["CR0", "CR1"], "CR"),
            (["NC"], "NC"),  # "CN",
            # (["CN"], "CN"), # "CN",
        ],
        desc="cue",
        leave=False,
    ):
        subset_spikes = trial_aligned_spikes[trial_aligned_spikes.cue.isin(cue_list)]
        if len(subset_spikes) == 0:
            continue
        # cue only (0th order)
        cued_spikes_by_cue = subset_spikes.describe()["z_scored_spikerate"][
            ["count", "mean", "std"]
        ]
        cued_spikes_by_cue["cue"] = identifier
        cued_spikes_by_cue_list.append(cued_spikes_by_cue)

        # cue by interp class (1st order)
        cued_spikes_by_interp_point = (
            subset_spikes[["interp", "interp_point_binned", "z_scored_spikerate"]]
            .groupby("interp_point_binned")
            .describe()["z_scored_spikerate"][["count", "mean", "std"]]
        )
        cued_spikes_by_interp_point["cue"] = identifier
        cued_spikes_by_interp_point_list.append(cued_spikes_by_interp_point)

        # cue by interp point (also 1st order...)
        cued_spikes_by_stim_class = (
            subset_spikes[["interp", "stim_class", "z_scored_spikerate"]]
            .groupby("stim_class")
            .describe()["z_scored_spikerate"][["count", "mean", "std"]]
        )
        cued_spikes_by_stim_class["cue"] = identifier
        cued_spikes_by_stim_class_list.append(cued_spikes_by_stim_class)

    cued_spikes_by_cue = pd.concat(
        [pd.DataFrame(i).T for i in cued_spikes_by_cue_list]
    ).reset_index()
    cued_spikes_by_stim_class = pd.concat(cued_spikes_by_stim_class_list).reset_index()
    cued_spikes_by_interp_point = pd.concat(
        cued_spikes_by_interp_point_list
    ).reset_index()

    cued_spikes_by_stim_class["sem"] = cued_spikes_by_stim_class["std"].values.astype(
        "float"
    ) / np.sqrt(cued_spikes_by_stim_class["count"]).values.astype("float")
    cued_spikes_by_interp_point["sem"] = cued_spikes_by_interp_point[
        "std"
    ].values.astype("float") / np.sqrt(
        cued_spikes_by_interp_point["count"]
    ).values.astype(
        "float"
    )
    cued_spikes_by_cue["sem"] = cued_spikes_by_cue["std"].values.astype(
        "float"
    ) / np.sqrt(cued_spikes_by_cue["count"].values.astype("float"))

    stim_class_by_stim_class["sem"] = stim_class_by_stim_class["std"].values.astype(
        "float"
    ) / np.sqrt(stim_class_by_stim_class["count"].values.astype("float"))

    if plot:
        visualize_spikerate_by_cue_and_class(
            stim_class_by_stim_class,
            cued_spikes_by_cue,
            cued_spikes_by_interp_point,
            cued_spikes_by_stim_class,
        )

    return compute_spikerate_stats(
        stim_class_by_stim_class,
        cued_spikes_by_cue,
        cued_spikes_by_interp_point,
        cued_spikes_by_stim_class,
    )


def ttest_from_row(row1, row2):
    return ttest_ind_from_stats(
        mean1=row1["mean"],
        std1=row1["std"],
        nobs1=row1["count"],
        mean2=row2["mean"],
        std2=row2["std"],
        nobs2=row2["count"],
    )


def compute_spikerate_stats(
    stim_class_by_stim_class,
    cued_spikes_by_cue,
    cued_spikes_by_interp_point,
    cued_spikes_by_stim_class,
):
    # try:
    all_dict = {}

    # spike rate by cue x stim class
    for idx, row in cued_spikes_by_stim_class.iterrows():
        all_dict["spike_rate_z_{}_{}".format(row.cue, row.stim_class)] = row["mean"]

    # spike_rate for cue by interp point
    for cue in cued_spikes_by_interp_point.cue.unique():
        mask = cued_spikes_by_interp_point.cue == cue
        all_dict["spike_rate_ip_{}".format(cue)] = cued_spikes_by_interp_point[
            mask
        ].interp_point_binned.values
        all_dict["spike_rate_mean_{}".format(cue)] = cued_spikes_by_interp_point[mask][
            "mean"
        ].values

    # does neuron respond more to one stim or the other
    mask1 = stim_class_by_stim_class.stim_class == "L"
    mask2 = stim_class_by_stim_class.stim_class == "R"
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0):
        row1 = stim_class_by_stim_class[mask1].iloc[0]
        row2 = stim_class_by_stim_class[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["stim_preference_t"] = t
        all_dict["stim_preference_p"] = p
        all_dict["spike_rate_z_L"] = row1["mean"]
        all_dict["spike_rate_z_R"] = row2["mean"]

    # does neuron respond more to one cue or the other
    mask1 = cued_spikes_by_cue.cue == "CL"
    mask2 = cued_spikes_by_cue.cue == "CR"
    mask3 = cued_spikes_by_cue.cue == "NC"
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0) & (np.sum(mask3) > 0):
        row1 = cued_spikes_by_cue[mask1].iloc[0]
        row2 = cued_spikes_by_cue[mask2].iloc[0]
        row3 = cued_spikes_by_cue[mask3].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["cue_preference_t"] = t
        all_dict["cue_preference_p"] = p
        all_dict["spike_rate_z_CL"] = row1["mean"]
        all_dict["spike_rate_z_CR"] = row2["mean"]
        all_dict["spike_rate_z_NC"] = row3["mean"]

    # does neuron cued left have a preference for one side or the other
    mask1 = (cued_spikes_by_stim_class.cue == "CL") & (
        cued_spikes_by_stim_class.stim_class == "L"
    )
    mask2 = (cued_spikes_by_stim_class.cue == "CL") & (
        cued_spikes_by_stim_class.stim_class == "R"
    )
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0):
        row1 = cued_spikes_by_stim_class[mask1].iloc[0]
        row2 = cued_spikes_by_stim_class[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["left_cue_stim_preference_t"] = t
        all_dict["left_cue_stim_preference_p"] = p

    # does neuron cued right have a preference for one side or the other
    mask1 = (cued_spikes_by_stim_class.cue == "CR") & (
        cued_spikes_by_stim_class.stim_class == "L"
    )
    mask2 = (cued_spikes_by_stim_class.cue == "CR") & (
        cued_spikes_by_stim_class.stim_class == "R"
    )
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0):
        row1 = cued_spikes_by_stim_class[mask1].iloc[0]
        row2 = cued_spikes_by_stim_class[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["right_cue_stim_preference_t"] = t
        all_dict["right_cue_stim_preference_p"] = p

    # does a neuron responding to a left stimuli have a preference for one cue vs the other
    mask1 = (cued_spikes_by_stim_class.cue == "CR") & (
        cued_spikes_by_stim_class.stim_class == "L"
    )
    mask2 = (cued_spikes_by_stim_class.cue == "CL") & (
        cued_spikes_by_stim_class.stim_class == "L"
    )
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0):
        row1 = cued_spikes_by_stim_class[mask1].iloc[0]
        row2 = cued_spikes_by_stim_class[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["left_stim_cue_preference_t"] = t
        all_dict["left_stim_cue_preference_p"] = p

    # does a neuron responding to a right stimuli have a preference for one cue vs the other
    mask1 = (cued_spikes_by_stim_class.cue == "CR") & (
        cued_spikes_by_stim_class.stim_class == "R"
    )
    mask2 = (cued_spikes_by_stim_class.cue == "CL") & (
        cued_spikes_by_stim_class.stim_class == "R"
    )
    if (np.sum(mask1) > 0) & (np.sum(mask2) > 0):
        row1 = cued_spikes_by_stim_class[mask1].iloc[0]
        row2 = cued_spikes_by_stim_class[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        all_dict["right_stim_cue_preference_t"] = t
        all_dict["right_stim_cue_preference_p"] = p

    # difference between spike rates for cues by cue type
    t_list = []
    p_list = []
    ip_list = []
    ips = cued_spikes_by_interp_point.interp_point_binned.unique()
    for ip in ips:
        mask1 = (cued_spikes_by_interp_point.cue == "CL") & (
            cued_spikes_by_interp_point.interp_point_binned == ip
        )
        if np.sum(mask1) == 0:
            continue
        row1 = cued_spikes_by_interp_point[mask1].iloc[0]
        mask2 = (cued_spikes_by_interp_point.cue == "CR") & (
            cued_spikes_by_interp_point.interp_point_binned == ip
        )
        if np.sum(mask2) == 0:
            continue
        row2 = cued_spikes_by_interp_point[mask2].iloc[0]
        t, p = ttest_from_row(row1, row2)
        t_list.append(t)
        p_list.append(p)
        ip_list.append(ip)
    all_dict["cued_spike_rate_by_interp_t"] = np.array(t_list)
    all_dict["cued_spike_rate_by_interp_p"] = np.array(p_list)
    all_dict["cued_spike_rate_by_interp_ip"] = np.array(ip_list)
    # except IndexError:
    #    print("Not enough data, skipping")
    #    return

    return pd.Series(all_dict)
