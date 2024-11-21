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


def z_score(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)


def get_mean_similarity(similarity_dict, n_interp_point_bins=16):
    mean_similarity_matrix = np.zeros((n_interp_point_bins, n_interp_point_bins))
    n_similarity_matrix = np.zeros((n_interp_point_bins, n_interp_point_bins))
    mean_similarity_matrix[:] = np.nan
    for ip1 in range(n_interp_point_bins):
        for ip2 in range(n_interp_point_bins):
            if ip1 in similarity_dict:
                if ip2 in similarity_dict[ip1]:
                    mean_similarity_matrix[ip1, ip2] = np.nanmean(
                        similarity_dict[ip1][ip2]
                    )
                    n_similarity_matrix[ip1, ip2] = np.sum(
                        np.isnan(similarity_dict[ip1][ip2]) == False
                    )

    return mean_similarity_matrix, n_similarity_matrix


def compute_cued_sm_differences_between_cue(
    trial_aligned_spikes,
    n_time_bins,
    n_interp_point_bins,
    passive=False,
    flip_bins=True,
    equal_sizes=False,
    exclude_non_responses=False,
    all_cues=["CL1", "CL0", "NC", "CN", "CR0", "CR1"],
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
        # for each cue combination
        for ci1, cue1 in enumerate(all_cues):
            for ci2, cue2 in enumerate(all_cues):  # [: ci1 + 1]):

                # for each similarity dictionary
                for (sm_identifier, cue_col) in [
                    ("", "cue"),
                    ("_shuffled", "cue_shuffled"),
                ]:
                    # create a dictionary of similarity values
                    sm_dict = {
                        metric: {
                            i: {j: [] for j in range(n_interp_point_bins)}
                            for i in range(n_interp_point_bins)
                        }
                        for metric in similarity_metrics
                    }
                    cue1_interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                        interp_trial_aligned_spikes[cue_col] == cue1
                    ]
                    cue2_interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                        interp_trial_aligned_spikes[cue_col] == cue2
                    ]
                    if (len(cue1_interp_trial_aligned_spikes) == 0) | (
                        len(cue2_interp_trial_aligned_spikes) == 0
                    ):
                        continue
                    if (cue1 == cue2) & (len(cue1_interp_trial_aligned_spikes) == 1):
                        continue

                    # for each interpolation point combination, get similarity and populate dictionary
                    for ip1 in np.arange(n_interp_point_bins):
                        for ip2 in np.arange(n_interp_point_bins):
                            # get mask for spikes
                            ip1_spikes_mask = (
                                cue1_interp_trial_aligned_spikes.interp_point_binned
                                == ip1
                            )
                            ip2_spikes_mask = (
                                cue2_interp_trial_aligned_spikes.interp_point_binned
                                == ip2
                            )
                            if (np.sum(ip1_spikes_mask) == 0) or (
                                np.sum(ip2_spikes_mask) == 0
                            ):
                                continue

                            # grab spiketrai ns
                            ip1_spikes = np.stack(
                                cue1_interp_trial_aligned_spikes[
                                    ip1_spikes_mask.values
                                ].spike_vectors.values
                            )
                            ip2_spikes = np.stack(
                                cue2_interp_trial_aligned_spikes[
                                    ip2_spikes_mask.values
                                ].spike_vectors.values
                            )

                            # similarity
                            for similarity_metric in similarity_metrics:
                                similarity_matrix = get_similarity_matrix(
                                    ip1_spikes, ip2_spikes, metric=similarity_metric
                                )
                                if (ip1 == ip2) & (cue1 == cue2):
                                    similarity_vals = similarity_matrix[
                                        np.tril_indices(len(similarity_matrix), k=-1)
                                    ]
                                else:
                                    similarity_vals = similarity_matrix.flatten()
                                sm_dict[similarity_metric][ip1][ip2] = similarity_vals

                    # save the average similarity matrices
                    for similarity_metric in similarity_metrics:
                        (
                            mean_similarity_matrix,
                            n_similarity_matrix,
                        ) = get_mean_similarity(sm_dict[similarity_metric])
                        all_dict[
                            "{}_{}_sm{}_{}_{}".format(
                                cue1,
                                cue2,
                                sm_identifier,
                                similarity_metric,
                                interpolation,
                            )
                        ] = mean_similarity_matrix
                        all_dict[
                            "{}_{}_sm_n{}_{}_{}".format(
                                cue1,
                                cue2,
                                sm_identifier,
                                similarity_metric,
                                interpolation,
                            )
                        ] = n_similarity_matrix

    return pd.Series(all_dict)
