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
    mean_similarity_matrix[:] = np.nan
    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            if ip1 in similarity_dict[cue]:
                if ip2 in similarity_dict[cue][ip1]:
                    mean_similarity_matrix[ip1, ip2] = mean_similarity_matrix[
                        ip2, ip1
                    ] = np.nanmean(similarity_dict[cue][ip1][ip2])

    return mean_similarity_matrix


def get_cued_similarity_difference(
    similarity_dict, n_interp_point_bins, cue_A="CL", cue_B="CR", equal_sizes=False
):
    cue_cue_distance_mat_stat = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_stat[:] = np.nan

    cue_cue_distance_mat_p = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_p[:] = np.nan

    cue_cue_distance_mat_d = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_d[:] = np.nan

    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            sim_A = np.array(similarity_dict[cue_A][ip1][ip2])
            sim_B = np.array(similarity_dict[cue_B][ip1][ip2])
            sim_A = sim_A[np.isnan(sim_A) == False]
            sim_B = sim_B[np.isnan(sim_B) == False]

            if (len(sim_A) > 0) and (len(sim_B) > 0):
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

                if np.all(sim_A == sim_A[0]) & np.all(sim_B == sim_A[0]):
                    stat = 0
                    p = 0.5
                else:
                    stat, p = scipy.stats.mannwhitneyu(sim_A, sim_B)

                cue_cue_distance_mat_stat[ip1, ip2] = stat
                cue_cue_distance_mat_stat[ip2, ip1] = stat

                cue_cue_distance_mat_p[ip1, ip2] = p
                cue_cue_distance_mat_p[ip2, ip1] = p

                cue_cue_distance_mat_d[ip2, ip1] = np.nanmean(sim_A) - np.nanmean(sim_B)
                cue_cue_distance_mat_d[ip1, ip2] = np.nanmean(sim_A) - np.nanmean(sim_B)

    return cue_cue_distance_mat_stat, cue_cue_distance_mat_p, cue_cue_distance_mat_d


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
    similarity_metrics=["cosine", "euclidean"], # , "manhattan" # "correlation", 
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
            trial_aligned_spikes.response.isin(['left', 'right'])
        ]

    all_dict = {}
    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        skip_interp = False

        # subset
        interp_trial_aligned_spikes = trial_aligned_spikes[
            (trial_aligned_spikes.interp == interpolation)
        ]

        # create shuffled cue condition
        interp_trial_aligned_spikes['cue_shuffled'] = None
        for ip_bin in np.unique(interp_trial_aligned_spikes.interp_point_binned.values):
            m = (
                (interp_trial_aligned_spikes.interp_point_binned == ip_bin)
                & (interp_trial_aligned_spikes.cue.isin(["CL0", "CL1", "CR0", "CR1"]))
            ).values
            permuted_cues = list(np.random.permutation(
                    interp_trial_aligned_spikes.iloc[m]["cue"].values
                ))
            interp_trial_aligned_spikes.loc[m, "cue_shuffled"] = permuted_cues 


        similarity_metric_dict = {
            metric: {
                cue: {
                    i: {j: [] for j in range(i + 1)} for i in range(n_interp_point_bins)
                }
                for cue in ["CL", "CR", "NC"]
            }
            for metric in similarity_metrics
        }
        for cue_list, cue_identifier in tqdm(
            all_cues,
            desc="cue",
            leave=False,
        ):
            cue_interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.cue.isin(cue_list)
            ]
            nex = len(cue_interp_trial_aligned_spikes)

            if nex < 2:
                skip_interp = True
                break

            unique_interpolation_points = np.unique(
                cue_interp_trial_aligned_spikes.interp_point_binned.values.astype(int)
            )

            for ip1 in unique_interpolation_points:
                for ip2 in unique_interpolation_points[
                    unique_interpolation_points <= ip1
                ]:
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

            if skip_interp:
                continue

        # save the average similarity matrices
        for similarity_metric in similarity_metrics:
            for cue_list, cue_identifier in all_cues:
                all_dict[
                    "{}_sm_{}_{}".format(
                        cue_identifier, similarity_metric, interpolation
                    )
                ] = get_mean_similarity(
                    similarity_metric_dict[similarity_metric], cue=cue_identifier
                )

        # save the distances between similarity matrices (in mean difference, stat, and p val)
        # between CL and CR, CL and CN, CR and CN
        for similarity_metric in similarity_metrics:
            for cue1, cue2 in [["CL", "CR"], ["CL", "NC"], ["CR", "NC"]]:
                stat, p, d = get_cued_similarity_difference(
                    similarity_metric_dict[similarity_metric],
                    n_interp_point_bins,
                    cue_A=cue1,
                    cue_B=cue2,
                    equal_sizes=equal_sizes,
                )
                all_dict[
                    "{}_{}_sm_stat_{}_{}".format(
                        cue1, cue2, similarity_metric, interpolation
                    )
                ] = stat
                all_dict[
                    "{}_{}_sm_p_{}_{}".format(
                        cue1, cue2, similarity_metric, interpolation
                    )
                ] = p
                all_dict[
                    "{}_{}_sm_d_{}_{}".format(
                        cue1, cue2, similarity_metric, interpolation
                    )
                ] = d

    return pd.Series(all_dict)
