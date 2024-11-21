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
from scipy.stats import ttest_ind
from cdcp.spiketrain_analysis.spiketrain_utils import get_unit_spike_trains, corr2_coeff

#%matplotlib inline


def z_score(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)


def get_cued_distance_mat(corr_coef_dict, n_interp_point_bins, cue_A="CL", cue_B="CR"):
    cue_cue_distance_mat_t = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_t[:] = np.nan

    cue_cue_distance_mat_d = np.zeros((n_interp_point_bins, n_interp_point_bins))
    cue_cue_distance_mat_d[:] = np.nan

    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            sim_A = np.array(corr_coef_dict[cue_A][ip1][ip2])
            sim_B = np.array(corr_coef_dict[cue_B][ip1][ip2])
            sim_A = sim_A[np.isnan(sim_A) == False]
            sim_B = sim_B[np.isnan(sim_B) == False]

            if (len(sim_A) > 0) and (len(sim_B) > 0):
                t, p = scipy.stats.ttest_ind(sim_A, sim_B)

                cue_cue_distance_mat_t[ip1, ip2] = t
                cue_cue_distance_mat_t[ip2, ip1] = t

                cue_cue_distance_mat_d[ip2, ip1] = np.nanmean(sim_A) - np.nanmean(sim_B)
                cue_cue_distance_mat_d[ip1, ip2] = np.nanmean(sim_A) - np.nanmean(sim_B)

    return cue_cue_distance_mat_t, cue_cue_distance_mat_d


def get_cued_distance_vec(
    spike_vector_dict, n_interp_point_bins, n_time_bins, cue_A="CL", cue_B="CR"
):
    spike_vector_difference_mat_d = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector_difference_mat_d[:] = np.nan
    spike_vector_difference_mat_t = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector_difference_mat_t[:] = np.nan

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
                t, p = scipy.stats.ttest_ind(r_A, r_B)

                spike_vector_difference_mat_t[ip1, time_bin] = t

                spike_vector_difference_mat_d[ip1, time_bin] = np.nanmean(
                    r_A
                ) - np.nanmean(r_B)

    return spike_vector_difference_mat_t, spike_vector_difference_mat_d


def get_mean_spike_vector(
    spike_vector_dict, cue="CL", n_interp_point_bins=16, n_time_bins=25
):
    spike_vector = np.zeros((n_interp_point_bins, n_time_bins))
    spike_vector[:] = np.nan
    for ip in range(n_interp_point_bins):
        if ip in spike_vector_dict[cue]:
            spike_vector[ip] = np.nanmean(spike_vector_dict[cue][ip], axis=0)
    return spike_vector


def get_mean_correlation(corr_coef_dict, cue="CL", n_interp_point_bins=16):
    corr_coef = np.zeros((n_interp_point_bins, n_interp_point_bins))
    corr_coef[:] = np.nan
    for ip1 in range(n_interp_point_bins):
        for ip2 in range(ip1 + 1):
            if ip1 in corr_coef_dict[cue]:
                if ip2 in corr_coef_dict[cue][ip1]:
                    corr_coef[ip1, ip2] = corr_coef[ip2, ip1] = np.nanmean(
                        corr_coef_dict[cue][ip1][ip2]
                    )

    return corr_coef


def compute_cued_average_sm_rv_differences(
    trial_aligned_spikes,
    n_interp_point_bins=16,
    n_time_bins=25,
    include_passive=True,
    flip_bins=True,
):

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point_binned"] = bin_interp_points(
        trial_aligned_spikes["interp_point"].values.astype(int),
        n_interp_point_bins,
        flip_bins=flip_bins,
    )

    if include_passive == False:
        # exclude passive trials
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.passive == False
        ]

    # get vectors
    spike_vectors_binned = [
        get_spike_train_vector(row, nbins=n_time_bins, return_gauss=False)
        for idx, row in tqdm(
            trial_aligned_spikes.iterrows(),
            total=len(trial_aligned_spikes),
            desc="spike vectors",
            leave=False,
        )
    ]
    spike_vectors_binned = z_score(spike_vectors_binned)
    trial_aligned_spikes["spike_vectors_binned"] = list(spike_vectors_binned)

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
        skip_interp = False

        spike_vector_dict = {
            cue: {i: [] for i in range(n_interp_point_bins)}
            for cue in ["CL", "CR", "NC"]
        }
        corr_coef_dict = {
            cue: {i: {j: [] for j in range(i + 1)} for i in range(n_interp_point_bins)}
            for cue in ["CL", "CR", "NC"]
        }

        for cue_list, cue_identifier in tqdm(
            [
                (["CL0", "CL1"], "CL"),
                (["CR0", "CR1"], "CR"),
                (["NC"], "NC"),
            ],
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
                    if ip1 == ip2:
                        if np.sum(ip1_spikes_mask) < 2:
                            continue

                    # grab spiketrains
                    ip1_spikes = np.stack(
                        cue_interp_trial_aligned_spikes[
                            ip1_spikes_mask
                        ].spike_vectors_binned.values
                    )
                    ip2_spikes = np.stack(
                        cue_interp_trial_aligned_spikes[
                            ip2_spikes_mask
                        ].spike_vectors_binned.values
                    )
                    spike_vector_dict[cue_identifier][ip1] = ip1_spikes

                    # compute correlations
                    corr_coefs = corr2_coeff(ip1_spikes, ip2_spikes)
                    if ip1 == ip2:
                        corr_coefs = corr_coefs[np.tril_indices(len(corr_coefs), k=-1)]
                    else:
                        corr_coefs = corr_coefs.flatten()
                    corr_coef_dict[cue_identifier][ip1][ip2] = corr_coefs

            if skip_interp:
                continue

        CL_CR_dist_mat_t, CL_CR_dist_mat_d = get_cued_distance_mat(
            corr_coef_dict, n_interp_point_bins, cue_A="CL", cue_B="CR"
        )

        CL_NC_dist_mat_t, CL_NC_dist_mat_d = get_cued_distance_mat(
            corr_coef_dict, n_interp_point_bins, cue_A="CL", cue_B="NC"
        )
        CR_NC_dist_mat_t, CR_NC_dist_mat_d = get_cued_distance_mat(
            corr_coef_dict, n_interp_point_bins, cue_A="CR", cue_B="NC"
        )

        CL_CR_rv_t, CL_CR_rv_d = get_cued_distance_vec(
            spike_vector_dict, n_interp_point_bins, n_time_bins, cue_A="CL", cue_B="CR"
        )
        CL_NC_rv_t, CL_NC_rv_d = get_cued_distance_vec(
            spike_vector_dict, n_interp_point_bins, n_time_bins, cue_A="CL", cue_B="NC"
        )
        CR_NC_rv_t, CR_NC_rv_d = get_cued_distance_vec(
            spike_vector_dict, n_interp_point_bins, n_time_bins, cue_A="CR", cue_B="NC"
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

        # get average correlation coefficients
        CC_CL = get_mean_correlation(
            corr_coef_dict, cue="CL", n_interp_point_bins=n_interp_point_bins
        )
        CC_CR = get_mean_correlation(
            corr_coef_dict, cue="CR", n_interp_point_bins=n_interp_point_bins
        )
        CC_NC = get_mean_correlation(
            corr_coef_dict, cue="NC", n_interp_point_bins=n_interp_point_bins
        )

        # distance matrices
        all_dict["CL_rv_avg_{}".format(interpolation)] = spike_vector_CL
        all_dict["CR_rv_avg_{}".format(interpolation)] = spike_vector_CR
        all_dict["NC_rv_avg_{}".format(interpolation)] = spike_vector_NC

        # response vectors
        all_dict["CL_sm_avg_{}".format(interpolation)] = CC_CL
        all_dict["CR_sm_avg_{}".format(interpolation)] = CC_CR
        all_dict["NC_sm_avg_{}".format(interpolation)] = CC_NC

        # distance matrix differences
        all_dict["CL_CR_dist_mat_t_{}".format(interpolation)] = CL_CR_dist_mat_t
        all_dict["CL_CR_dist_mat_d_{}".format(interpolation)] = CL_CR_dist_mat_d
        all_dict["CL_NC_dist_mat_t_{}".format(interpolation)] = CL_NC_dist_mat_t
        all_dict["CL_NC_dist_mat_d_{}".format(interpolation)] = CL_NC_dist_mat_d
        all_dict["CR_NC_dist_mat_t_{}".format(interpolation)] = CR_NC_dist_mat_t
        all_dict["CR_NC_dist_mat_d_{}".format(interpolation)] = CR_NC_dist_mat_d

        # response vector differences
        all_dict["CL_CR_rv_t_{}".format(interpolation)] = CL_CR_rv_t
        all_dict["CL_CR_rv_d_{}".format(interpolation)] = CL_CR_rv_d
        all_dict["CL_NC_rv_t_{}".format(interpolation)] = CL_NC_rv_t
        all_dict["CL_NC_rv_d_{}".format(interpolation)] = CL_NC_rv_d
        all_dict["CR_NC_rv_t_{}".format(interpolation)] = CR_NC_rv_t
        all_dict["CR_NC_rv_d_{}".format(interpolation)] = CR_NC_rv_d
    return pd.Series(all_dict)
