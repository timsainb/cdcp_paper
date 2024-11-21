import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm

pd.set_option("display.max_columns", 500)
from joblib import Parallel, delayed
from functools import reduce
import scipy.stats
from cdcp.spiketrain_analysis.spiketrain_utils import (
    bin_interp_points,
    get_average_response_vector,
    create_dense_similarity_matrix,
    get_spike_train_vector,
)
from cdcp.spiketrain_analysis.neurometric import (
    get_interp_points_dists_from_similarity_matrix,
)
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic
import copy
from cdcp.spiketrain_analysis.spiketrain_utils import get_unit_spike_trains, corr2_coeff


from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)


def get_similarity_matrix(x, y, metric="correlation"):
    if metric == "correlation":
        sm = corr2_coeff(x, y)
    elif metric == "manhattan":
        sm = 1 / (1 + euclidean_distances(x, y))
    elif metric == "euclidean":
        sm = 1 / (1 + manhattan_distances(x, y))
    elif metric == "cosine":
        sm = cosine_similarity(x, y)
    return sm


def get_mean_similarity(similarity_dict, cue="CL", n_interp_point_bins=16):
    mean_similarity_matrix = np.zeros((n_interp_point_bins, n_interp_point_bins))
    n_similarity_matrix = np.zeros((n_interp_point_bins, n_interp_point_bins))
    mean_similarity_matrix[:] = np.nan
    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            if ip1 in similarity_dict[cue]:
                if ip2 in similarity_dict[cue][ip1]:
                    mean_similarity_matrix[ip1, ip2] = mean_similarity_matrix[
                        ip2, ip1
                    ] = np.nanmean(similarity_dict[cue][ip1][ip2])
                    n_similarity_matrix[ip1, ip2] = n_similarity_matrix[
                        ip2, ip1
                    ] = np.sum(np.isnan(similarity_dict[cue][ip1][ip2]) == False)

    return mean_similarity_matrix, n_similarity_matrix


def get_cued_similarity_difference(
    similarity_dict, n_interp_point_bins, cue_A="CL", cue_B="CR", equal_sizes=False
):
    cue_cue_distance_mat_stat = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_stat[:] = np.nan

    cue_cue_distance_mat_p = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_p[:] = np.nan

    cue_cue_distance_mat_d = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_d[:] = np.nan

    cue_cue_count_A = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_count_A[:] = np.nan

    cue_cue_count_B = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_count_B[:] = np.nan

    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            sim_A = np.array(similarity_dict[cue_A][ip1][ip2])
            sim_B = np.array(similarity_dict[cue_B][ip1][ip2])
            sim_A = sim_A[np.isnan(sim_A) == False]
            sim_B = sim_B[np.isnan(sim_B) == False]

            if (len(sim_A) > 1) and (len(sim_B) > 1):
                if equal_sizes:
                    if len(sim_A) > len(sim_B):
                        subset = np.random.choice(
                            len(sim_A), size=len(sim_B), replace=False
                        )
                        sim_A = sim_A[subset]
                    if len(sim_B) > len(sim_A):
                        subset = np.random.choice(
                            len(sim_B), size=len(sim_A), replace=False
                        )
                        sim_B = sim_B[subset]

                if np.all(sim_A == sim_A[0]) & np.all(np.abs(sim_B - sim_A[0]) < 1e-10):
                    # all similarity values are the same
                    # stat = 0
                    # p = 0.5
                    continue
                else:
                    try:
                        stat, p = scipy.stats.mannwhitneyu(
                            sim_A.astype("float32"),
                            sim_B.astype("float32"),
                            # method="asymptotic",
                        )
                    except ValueError as e:
                        print(e)
                        stat = np.nan
                        p = np.nan

                cue_cue_distance_mat_stat[ip1, ip2] = stat
                cue_cue_distance_mat_stat[ip2, ip1] = stat

                cue_cue_distance_mat_p[ip1, ip2] = p
                cue_cue_distance_mat_p[ip2, ip1] = p

                cue_cue_distance_mat_d[ip2, ip1] = np.nanmean(sim_A) - np.nanmean(sim_B)
                cue_cue_distance_mat_d[ip1, ip2] = np.nanmean(sim_A) - np.nanmean(sim_B)

                # count the number
                cue_cue_count_A[ip2, ip1] = np.sum(np.isnan(sim_A) == False)
                cue_cue_count_A[ip1, ip2] = np.sum(np.isnan(sim_A) == False)
                cue_cue_count_B[ip2, ip1] = np.sum(np.isnan(sim_B) == False)
                cue_cue_count_B[ip1, ip2] = np.sum(np.isnan(sim_B) == False)

    return (
        cue_cue_distance_mat_stat,
        cue_cue_distance_mat_p,
        cue_cue_distance_mat_d,
        cue_cue_count_A,
        cue_cue_count_B,
    )


def compute_cued_sm_differences(
    trial_aligned_spikes,
    n_time_bins,
    n_interp_point_bins,
    passive=False,
    flip_bins=True,
    equal_sizes=False,
    exclude_non_responses=False,
    all_cues=[
        (["CL0", "CL1"], "CL"),
        (["CR0", "CR1"], "CR"),
        (["CR1"], "CR1"),
        (["CL1"], "CL1"),
        (["CR0"], "CR0"),
        (["CL0"], "CL0"),
        (["NC"], "NC"),
        (["CN"], "CN"),
    ],
    similarity_metrics=["cosine", "euclidean"],  # , "manhattan" # "correlation",
):

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point_binned"] = bin_interp_points(
        trial_aligned_spikes["interp_point"].values.astype(int),
        n_interp_point_bins,
        flip_bins=flip_bins,
    )
    # exclude passive trials
    if passive:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.passive == True
        ]
    else:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.passive == False
        ]

    if exclude_non_responses:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.response.isin(["left", "right"])
        ]

    all_dict = {}
    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        # subset
        interp_trial_aligned_spikes = trial_aligned_spikes[
            (trial_aligned_spikes.interp == interpolation)
        ]

        # create shuffled cue condition
        interp_trial_aligned_spikes["cue_shuffled"] = interp_trial_aligned_spikes["cue"]
        for ip_bin in np.unique(interp_trial_aligned_spikes.interp_point_binned.values):
            m = (
                (interp_trial_aligned_spikes.interp_point_binned == ip_bin)
                & (interp_trial_aligned_spikes.cue.isin(["CL0", "CL1", "CR0", "CR1"]))
            ).values
            permuted_cues = list(
                np.random.permutation(interp_trial_aligned_spikes.iloc[m]["cue"].values)
            )
            interp_trial_aligned_spikes.loc[m, "cue_shuffled"] = permuted_cues

        similarity_metric_dict = create_similarity_metric_dict(
            interp_trial_aligned_spikes,
            n_interp_point_bins,
            all_cues,
            cue_col="cue",
            similarity_metrics=similarity_metrics,
        )

        similarity_metric_dict_shuffled = create_similarity_metric_dict(
            interp_trial_aligned_spikes,
            n_interp_point_bins,
            all_cues,
            cue_col="cue_shuffled",
            similarity_metrics=similarity_metrics,
        )

        for (sm_dict, sm_identifier) in [
            (similarity_metric_dict, ""),
            (similarity_metric_dict_shuffled, "_shuffled"),
        ]:

            # save the average similarity matrices
            for similarity_metric in similarity_metrics:
                for cue_list, cue_identifier in all_cues:
                    mean_similarity_matrix, n_similarity_matrix = get_mean_similarity(
                        sm_dict[similarity_metric], cue=cue_identifier
                    )
                    all_dict[
                        "{}_sm{}_{}_{}".format(
                            cue_identifier,
                            sm_identifier,
                            similarity_metric,
                            interpolation,
                        )
                    ] = mean_similarity_matrix
                    all_dict[
                        "{}_sm_n{}_{}_{}".format(
                            cue_identifier,
                            sm_identifier,
                            similarity_metric,
                            interpolation,
                        )
                    ] = n_similarity_matrix

            # save the distances between similarity matrices (in mean difference, stat, and p val)
            # between CL and CR, CL and CN, CR and CN
            for similarity_metric in similarity_metrics:
                for cue1, cue2 in [["CL", "CR"], ["CL", "NC"], ["CR", "NC"]]:
                    stat, p, d, count_a, count_b = get_cued_similarity_difference(
                        sm_dict[similarity_metric],
                        n_interp_point_bins,
                        cue_A=cue1,
                        cue_B=cue2,
                        equal_sizes=equal_sizes,
                    )
                    all_dict[
                        "{}_{}_sm_stat{}_{}_{}".format(
                            cue1, cue2, sm_identifier, similarity_metric, interpolation
                        )
                    ] = stat
                    all_dict[
                        "{}_{}_sm_p{}_{}_{}".format(
                            cue1, cue2, sm_identifier, similarity_metric, interpolation
                        )
                    ] = p
                    all_dict[
                        "{}_{}_sm_d{}_{}_{}".format(
                            cue1, cue2, sm_identifier, similarity_metric, interpolation
                        )
                    ] = d
                    all_dict[
                        "{}_{}_count_a{}_{}_{}".format(
                            cue1, cue2, sm_identifier, similarity_metric, interpolation
                        )
                    ] = count_a
                    all_dict[
                        "{}_{}_count_b{}_{}_{}".format(
                            cue1, cue2, sm_identifier, similarity_metric, interpolation
                        )
                    ] = count_b
    return pd.Series(all_dict)


def create_similarity_metric_dict(
    interp_trial_aligned_spikes,
    n_interp_point_bins,
    all_cues,
    similarity_metrics,
    cue_col="cue",
):
    similarity_metric_dict = {
        metric: {
            cue: {i: {j: [] for j in range(i + 1)} for i in range(n_interp_point_bins)}
            for cue in [c[-1] for c in all_cues]
        }
        for metric in similarity_metrics
    }
    for cue_list, cue_identifier in tqdm(
        all_cues,
        desc="cue",
        leave=False,
    ):
        cue_interp_trial_aligned_spikes = interp_trial_aligned_spikes[
            interp_trial_aligned_spikes[cue_col].isin(cue_list)
        ]
        nex = len(cue_interp_trial_aligned_spikes)

        if nex < 2:
            continue

        unique_interpolation_points = np.unique(
            cue_interp_trial_aligned_spikes.interp_point_binned.values.astype(int)
        )

        for ip1 in unique_interpolation_points:
            for ip2 in unique_interpolation_points[unique_interpolation_points <= ip1]:
                # get mask for spikes
                ip1_spikes_mask = (
                    cue_interp_trial_aligned_spikes.interp_point_binned == ip1
                )
                ip2_spikes_mask = (
                    cue_interp_trial_aligned_spikes.interp_point_binned == ip2
                )

                if (np.sum(ip1_spikes_mask) == 0) or (np.sum(ip2_spikes_mask) == 0):
                    continue

                # grab spiketrains
                ip1_spikes = np.stack(
                    cue_interp_trial_aligned_spikes[
                        ip1_spikes_mask
                    ].spike_vectors.values
                )
                ip2_spikes = np.stack(
                    cue_interp_trial_aligned_spikes[
                        ip2_spikes_mask
                    ].spike_vectors.values
                )

                # similarity
                for similarity_metric in similarity_metrics:
                    similarity_matrix = get_similarity_matrix(
                        ip1_spikes, ip2_spikes, metric=similarity_metric
                    )
                    if ip1 == ip2:
                        similarity_vals = similarity_matrix[
                            np.tril_indices(len(similarity_matrix), k=-1)
                        ]
                    else:
                        similarity_vals = similarity_matrix.flatten()
                    similarity_metric_dict[similarity_metric][cue_identifier][ip1][
                        ip2
                    ] = similarity_vals

    return similarity_metric_dict