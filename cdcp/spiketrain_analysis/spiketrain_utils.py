import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from scipy.ndimage import gaussian_filter1d

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)


def get_unit_spike_trains(
    unit_recording_ids,
    spikesorting_folder,
    sorter,
    unit,
    trial_aligned_spikes_folder="trial_aligned_spikes",
):
    trial_aligned_spikes_list = []
    for unit, recording_id in tqdm(
        unit_recording_ids, desc="unit spike trains", leave=False
    ):
        trial_aligned_spikes_loc = (
            spikesorting_folder
            / trial_aligned_spikes_folder
            / sorter
            / recording_id
            / "{}.pickle".format(unit)
        )
        if trial_aligned_spikes_loc.exists():
            trial_aligned_spikes = pd.read_pickle(trial_aligned_spikes_loc)
            trial_aligned_spikes["recording_id"] = recording_id

            trial_aligned_spikes["stim"] = [
                i[:-4] if i[-4:].lower() == ".wav" else i
                for i in trial_aligned_spikes.stim.values
            ]

            mask = [
                (i.split("_")[-1].isnumeric() and len(i.split("_")[-1]) == 3)
                for i in trial_aligned_spikes.stim.values
            ]

            # get cue info
            trial_aligned_spikes["cue"] = [
                i.split("_")[0] if mask else np.nan
                for i, m in zip(trial_aligned_spikes.stim.values, mask)
            ]
            trial_aligned_spikes["interp"] = [
                i.split("_")[1] if mask else np.nan
                for i, m in zip(trial_aligned_spikes.stim.values, mask)
            ]

            # for i, m in zip(trial_aligned_spikes.stim.values, mask):
            #    if m:
            #        int(i.split("_")[2])
            trial_aligned_spikes["interp_point"] = [
                np.nan if m == False else int(i.split("_")[2])
                for i, m in zip(trial_aligned_spikes.stim.values, mask)
            ]
            trial_aligned_spikes_list.append(trial_aligned_spikes)
        else:
            0
            # print("{} does not have trial aligned spikes yet".format(recording_id))
            # print('\t', trial_aligned_spikes_loc)
    if len(trial_aligned_spikes_list) < 1:
        return None
    else:
        return pd.concat(trial_aligned_spikes_list)


def get_spike_train_vector(
    row,
    nbins=100,
    gaussian_sigma_ms=5,
    return_gauss=False,
    no_cue=True,
    padding_s=0.1,
    mode="constant",
):
    """
    From a list of rows, create a gaussian smoothed spike impulse vector

    Parameters
    ----------
    row : [type]
        [description]
    gaussian_sigma : int, optional
        [description], by default 5
    nbins : int, optional
        [description], by default 100
    gaussian_sigma_ms : int, optional
        [description], by default 5
    return_gauss : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    if len(row.spike_times) == 0:
        if no_cue:
            return np.zeros(nbins)
        else:
            return np.zeros(nbins * 2)

    bin_ms = 1000 / nbins
    gaussian_sigma = gaussian_sigma_ms / bin_ms
    padding_bins = int(padding_s * nbins)
    if padding_bins - padding_s * nbins != 0:
        raise ValueError("nbins incompatible with padding")
    if no_cue:
        if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
            bins = np.linspace(
                1 - padding_s, 2 + padding_s, nbins + 1 + padding_bins * 2
            )
        else:
            bins = np.linspace(
                0 - padding_s, 1 + padding_s, nbins + 1 + padding_bins * 2
            )
    else:
        if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
            bins = np.linspace(
                0 - padding_s, 2 + padding_s, nbins * 2 + 1 + padding_bins * 2
            )
        else:
            bins = np.linspace(
                -1 - padding_s, 1 + padding_s, nbins * 2 + 1 + padding_bins * 2
            )

    hist = np.histogram(row.spike_times, bins, density=False)[0]
    if padding_bins > 0:
        hist = hist[padding_bins:-padding_bins]

    if return_gauss:
        gauss_convolved_psth = gaussian_filter1d(
            hist.astype("float"), gaussian_sigma, mode=mode
        )
        return gauss_convolved_psth
    else:
        return hist


def get_spike_train_vector_old(
    row,
    nbins=100,
    gaussian_sigma_ms=5,
    return_gauss=False,
    no_cue=True,
    mode="constant",
):
    """
    From a list of rows, create a gaussian smoothed spike impulse vector

    Parameters
    ----------
    row : [type]
        [description]
    gaussian_sigma : int, optional
        [description], by default 5
    nbins : int, optional
        [description], by default 100
    gaussian_sigma_ms : int, optional
        [description], by default 5
    return_gauss : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    bin_ms = 1000 / nbins
    gaussian_sigma = gaussian_sigma_ms / bin_ms

    if no_cue:
        if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
            bins = np.linspace(1, 2, nbins + 1)
        else:
            bins = np.linspace(0, 1, nbins + 1)
    else:
        if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
            bins = np.linspace(0, 2, nbins * 2 + 1)
        else:
            bins = np.linspace(-1, 1, nbins * 2 + 1)

    hist = np.histogram(row.spike_times, bins, density=False)[0]

    if return_gauss:
        gauss_convolved_psth = gaussian_filter1d(
            hist.astype("float"), gaussian_sigma, mode=mode
        )
        return gauss_convolved_psth
    else:
        return hist


def bin_interp_points(interp_points, n_bins=16, flip_bins=True):
    """
    Equally divide interp points into bins

    Parameters
    ----------
    interp_points : [type]
        [description]
    n_bins : int, optional
        [description], by default 16

    Returns
    -------
    [type]
        [description]
    """
    if n_bins == 128:
        return interp_points

    # test to ensure that this n_bins equally divdes the data
    ips = np.arange(128)
    bins = np.arange(0, 127, int(128 / n_bins))
    test_bins = np.digitize(ips, bins) - 1
    if flip_bins:
        test_bins = n_bins - 1 - test_bins
    unique_bins, counts = np.unique(np.digitize(ips, bins), return_counts=True)
    # ensure that bins are equally divided
    assert all(x == counts[0] for x in counts)
    binned = np.digitize(interp_points, bins) - 1
    if flip_bins:
        binned = n_bins - 1 - binned
    return binned


def get_average_response_vector(
    trial_aligned_spikes,
    spike_trains_col="spike_trains",
    interp_point_column="interp_point_binned",
):

    interp_points_this_unit = np.unique(
        trial_aligned_spikes[interp_point_column].values
    )

    # get mean response vector for interpolation points
    avg_response_vectors = np.array(
        [
            np.mean(
                trial_aligned_spikes[
                    trial_aligned_spikes[interp_point_column].values == interp_point
                ][spike_trains_col],
                axis=0,
            )
            for interp_point in interp_points_this_unit
        ]
    )
    return avg_response_vectors, interp_points_this_unit


def create_dense_response_vector(
    mean_response_vectors, interp_points_this_unit, n_interp_bins=128
):
    rv = np.zeros((n_interp_bins, mean_response_vectors.shape[1]))
    rv[:] = np.nan
    for i, ip in enumerate(interp_points_this_unit):
        rv[ip] = mean_response_vectors[i]
    return rv


def get_similarity_matrix(x, y=None, metric="correlation"):
    if y == None:
        y = x
    if metric == "correlation":
        sm = corr2_coeff(x, y)
    elif metric == "manhattan":
        sm = 1 / (1 + euclidean_distances(x, y))
    elif metric == "euclidean":
        sm = 1 / (1 + manhattan_distances(x, y))
    elif metric == "cosine":
        sm = cosine_similarity(x, y)
    return sm


def create_dense_similarity_matrix(
    mean_response_vectors,
    interp_points_this_unit,
    n_interp_bins=128,
    similarity_metric="correlation",
):
    """
    Create a dense simlarity matrix from response vectors and interp_points

    Parameters
    ----------
    mean_response_vectors : [type]
        [description]
    interp_points_this_unit : [type]
        [description]
    n_interp_bins : int, optional
        [description], by default 128

    Returns
    -------
    [type]
        [description]
    """
    similarity_matrix = np.zeros((n_interp_bins, n_interp_bins))
    similarity_matrix[:] = np.nan
    sm = get_similarity_matrix(mean_response_vectors, metric=similarity_metric)
    # sm = np.corrcoef(mean_response_vectors)
    for i, ip in enumerate(interp_points_this_unit):
        similarity_matrix[ip, interp_points_this_unit] = sm[i]
    return similarity_matrix


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
