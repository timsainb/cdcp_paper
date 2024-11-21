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
    create_dense_response_vector,
    get_spike_train_vector,
)
from cdcp.spiketrain_analysis.neurometric import (
    get_interp_points_dists_from_similarity_matrix,
)
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic
import copy
from cdcp.spiketrain_analysis.spiketrain_utils import get_unit_spike_trains

interpolations = [
    "AE",
    "AF",
    "AG",
    "AH",
    "BE",
    "BF",
    "BG",
    "BH",
    "CE",
    "CF",
    "CG",
    "CH",
    "DE",
    "DF",
    "DG",
    "DH",
]


def z_score(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)


def get_cued_rv(
    spike_vector_dict,
    n_interp_point_bins,
    n_time_bins,
    cue_A="CL",
    cue_B="CR",
    equal_sizes=False,
):
    spike_vector_difference_mat_d = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector_difference_mat_d[:] = np.nan
    spike_vector_difference_mat_stat = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector_difference_mat_stat[:] = np.nan
    spike_vector_difference_mat_p = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector_difference_mat_p[:] = np.nan

    for ip1 in range(n_interp_point_bins):
        rv_A = np.array(spike_vector_dict[cue_A][ip1])
        rv_B = np.array(spike_vector_dict[cue_B][ip1])

        if (len(rv_A) == 0) or (len(rv_B) == 0):
            continue

        for time_bin in np.arange(n_time_bins):
            r_A = rv_A[:, time_bin]
            r_B = rv_B[:, time_bin]
            r_A = r_A[np.isnan(r_A) == False]
            r_B = r_B[np.isnan(r_B) == False]

            if (len(r_A) > 0) and (len(r_B) > 0):

                if equal_sizes:
                    if len(r_A) > len(r_B):
                        subset = np.random.choice(
                            len(r_A), size=len(r_B), replace=False
                        )
                        r_A = r_A[subset]
                    if len(r_B) > len(r_A):
                        subset = np.random.choice(
                            len(r_B), size=len(r_A), replace=False
                        )
                        r_B = r_B[subset]

                # if all values are the same, nan
                spike_vector_difference_mat_d[ip1, time_bin] = np.nanmean(
                    r_A
                ) - np.nanmean(r_B)
                #
                if np.all(r_A == r_A[0]) & np.all(r_B == r_A[0]):
                    stat = 0
                    p = 0.5
                else:
                    stat, p = scipy.stats.mannwhitneyu(r_A, r_B)

                spike_vector_difference_mat_stat[ip1, time_bin] = stat
                spike_vector_difference_mat_p[ip1, time_bin] = p

    return (
        spike_vector_difference_mat_stat,
        spike_vector_difference_mat_p,
        spike_vector_difference_mat_d,
    )


def get_mean_spike_vector(
    spike_vector_dict,
    cue="CL",
    n_interp_point_bins=16,
    n_time_bins=25,
    use_median=False,
):
    spike_vector = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector[:] = np.nan
    for ip in range(n_interp_point_bins):
        if ip in spike_vector_dict[cue]:
            if use_median:
                m_ = np.nanmedian(spike_vector_dict[cue][ip], axis=0)
            else:
                m_ = np.nanmean(spike_vector_dict[cue][ip], axis=0)
            spike_vector[ip] = m_
    return spike_vector


def compute_cued_average_rv_differences(
    trial_aligned_spikes,
    n_time_bins,
    n_interp_point_bins,
    equal_sizes=False,
    passive=True,
    flip_bins=True,
):

    # ensure that this is an interpolation trial, and not some other stimulus
    trial_aligned_spikes = trial_aligned_spikes[
        trial_aligned_spikes.interp.isin(interpolations)
    ]

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point_binned"] = bin_interp_points(
        trial_aligned_spikes["interp_point"].values.astype(int),
        n_interp_point_bins,
        flip_bins=flip_bins,
    )

    # for each cue, for each interpolation, get response vectors
    # .  for each interpolation point, get

    all_dict = {}

    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        # subset
        interp_trial_aligned_spikes = trial_aligned_spikes[
            (trial_aligned_spikes.interp == interpolation)
        ]

        # get average response vector to not cued trials
        average_rv, interp_points_this_unit = get_average_response_vector(
            interp_trial_aligned_spikes[interp_trial_aligned_spikes.cue == "NC"],
            spike_trains_col="spike_vectors",
        )
        if len(average_rv) == 0:
            dense_average_rv = np.nan
        else:
            dense_average_rv = create_dense_response_vector(
                average_rv, interp_points_this_unit, n_interp_bins=n_interp_point_bins
            )

        if passive:
            # exclude passive trials
            interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.passive
            ]
        else:
            interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.passive == False
            ]

        skip_interp = False

        spike_vector_dict = {
            cue: {i: [] for i in range(n_interp_point_bins)}
            for cue in ["CL", "CR", "NC"]
        }
        cue_count = {cue: np.zeros(n_interp_point_bins) for cue in ["CL", "CR", "NC"]}

        for cue_list, cue_identifier in [
            (["CL0", "CL1"], "CL"),
            (["CR0", "CR1"], "CR"),
            (["NC"], "NC"),
        ]:
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
                # get mask for spikes
                ip1_spikes_mask = (
                    cue_interp_trial_aligned_spikes.interp_point_binned == ip1
                )

                # save number of times cued
                cue_count[cue_identifier][ip1] = np.sum(ip1_spikes_mask)

                if np.sum(ip1_spikes_mask) == 0:
                    continue

                # grab spiketrains
                ip1_spikes = np.stack(
                    cue_interp_trial_aligned_spikes[
                        ip1_spikes_mask
                    ].spike_vectors.values
                )

                spike_vector_dict[cue_identifier][ip1] = ip1_spikes

            if skip_interp:
                continue

        CL_CR_rv_stat, CL_CR_rv_p, CL_CR_rv_d = get_cued_rv(
            spike_vector_dict,
            n_interp_point_bins,
            n_time_bins,
            cue_A="CL",
            cue_B="CR",
            equal_sizes=equal_sizes,
        )
        CL_NC_rv_stat, CL_NC_rv_p, CL_NC_rv_d = get_cued_rv(
            spike_vector_dict,
            n_interp_point_bins,
            n_time_bins,
            cue_A="CL",
            cue_B="NC",
            equal_sizes=equal_sizes,
        )
        CR_NC_rv_stat, CR_NC_rv_p, CR_NC_rv_d = get_cued_rv(
            spike_vector_dict,
            n_interp_point_bins,
            n_time_bins,
            cue_A="CR",
            cue_B="NC",
            equal_sizes=equal_sizes,
        )

        # get average spike vectors
        spike_vector_CL = get_mean_spike_vector(
            spike_vector_dict,
            cue="CL",
            n_interp_point_bins=n_interp_point_bins,
            n_time_bins=n_time_bins,
        )
        spike_vector_CR = get_mean_spike_vector(
            spike_vector_dict,
            cue="CR",
            n_interp_point_bins=n_interp_point_bins,
            n_time_bins=n_time_bins,
        )
        spike_vector_NC = get_mean_spike_vector(
            spike_vector_dict,
            cue="NC",
            n_interp_point_bins=n_interp_point_bins,
            n_time_bins=n_time_bins,
        )

        # n cued
        all_dict["CL_count_{}".format(interpolation)] = cue_count["CL"]
        all_dict["CL_count_total_{}".format(interpolation)] = np.sum(cue_count["CL"])
        all_dict["CR_count_{}".format(interpolation)] = cue_count["CR"]
        all_dict["CR_count_total_{}".format(interpolation)] = np.sum(cue_count["CR"])
        all_dict["NC_count_{}".format(interpolation)] = cue_count["NC"]
        all_dict["NC_count_total_{}".format(interpolation)] = np.sum(cue_count["NC"])

        # response vectors
        all_dict["CL_rv_avg_{}".format(interpolation)] = spike_vector_CL
        all_dict["CR_rv_avg_{}".format(interpolation)] = spike_vector_CR
        all_dict["NC_rv_avg_{}".format(interpolation)] = spike_vector_NC
        all_dict["NC_rv_avg_passive_active_{}".format(interpolation)] = dense_average_rv

        # response vector differences
        all_dict["CL_CR_rv_stat_{}".format(interpolation)] = CL_CR_rv_stat
        all_dict["CL_CR_rv_p_{}".format(interpolation)] = CL_CR_rv_p
        all_dict["CL_CR_rv_d_{}".format(interpolation)] = CL_CR_rv_d
        all_dict["CL_NC_rv_stat_{}".format(interpolation)] = CL_NC_rv_stat
        all_dict["CL_NC_rv_p_{}".format(interpolation)] = CL_NC_rv_p
        all_dict["CL_NC_rv_d_{}".format(interpolation)] = CL_NC_rv_d
        all_dict["CR_NC_rv_stat_{}".format(interpolation)] = CR_NC_rv_stat
        all_dict["CR_NC_rv_p_{}".format(interpolation)] = CR_NC_rv_p
        all_dict["CR_NC_rv_d_{}".format(interpolation)] = CR_NC_rv_d
    return pd.Series(all_dict)
