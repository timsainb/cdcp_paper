import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic
from cdcp.spiketrain_analysis.unit_statistics import (
    get_unit_spike_trains,
    get_spike_train_vector,
)
import pandas as pd
import matplotlib.pyplot as plt


def compute_averaged_response_similarities(
    unit_to_analyze,
    spikesorting_folder,
    interpolation=None,
    passive=None,
    cue=None,
    sorter="kilosort2_5",
    gaussian_sigma_ms=10,
    return_gauss=True,
    n_spiketrain_bins=100,
):
    """"""

    trial_aligned_spikes = get_unit_spike_trains(
        unit_to_analyze.sort_units, spikesorting_folder, sorter, unit_to_analyze
    )

    # subset the conditions
    if interpolation is not None:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interpolation
        ]
    if cue is not None:
        trial_aligned_spikes = trial_aligned_spikes[trial_aligned_spikes.cue == cue]

    if passive is not None:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.passive == passive
        ]

    if len(trial_aligned_spikes) < 2:
        return

    # get spike train as a vector
    trial_aligned_spikes["spike_trains"] = [
        get_spike_train_vector(
            row,
            nbins=n_spiketrain_bins,
            gaussian_sigma_ms=gaussian_sigma_ms,
            return_gauss=return_gauss,
        )
        for i, row in tqdm(
            trial_aligned_spikes.iterrows(),
            leave=False,
            desc="spike train",
            total=len(trial_aligned_spikes),
        )
    ]

    # get all available interpolation points
    interp_points_this_unit = np.unique(trial_aligned_spikes.interp_point)

    if len(interp_points_this_unit) < 2:
        return

    ##########
    # TODO an alternative to the similarity matrix of the averaged response vectors would
    # be to take the full similarity matrix of all of the trials, and compute the average
    # similarity (then the diagonal of the matrix would not be 1)
    ##########
    # get mean response vector for interpolation points
    averaged_response_vectors = np.array(
        [
            np.mean(
                trial_aligned_spikes[
                    trial_aligned_spikes.interp_point == interp_point
                ].spike_trains,
                axis=0,
            )
            for interp_point in interp_points_this_unit
        ]
    )
    # create similarity matrix
    # similarity_matrix = cosine_similarity(averaged_response_vectors)
    similarity_matrix = np.corrcoef(averaged_response_vectors + 1e-10)
    return similarity_matrix, interp_points_this_unit, averaged_response_vectors


def compute_population_neurometric_from_distance_matrix(
    merged_units,
    interpolations,
    spikesorting_folder,
    cue=None,
    passive=None,
    identifier="",
    n_time_samples=99,
    verbose=0,
):
    # compute similarity matrix & avg response vectors
    response_vectors_all = np.zeros((128, n_time_samples))
    full_interp_count = np.zeros(128)

    similarity_all_list = []
    similarity_count_all = []

    for interpolation in tqdm(interpolations, desc="interpolation", leave=False):

        # compute average response for each unit
        avg_response_vectors = Parallel(n_jobs=-1, verbose=verbose, prefer="processes")(
            delayed(compute_averaged_response_similarities)(
                unit_to_analyze,
                spikesorting_folder,
                interpolation=interpolation,
                passive=passive,
                cue=cue,
            )
            for idx, unit_to_analyze in tqdm(
                merged_units.iterrows(),
                total=len(merged_units),
                desc="unit response",
                leave=False,
            )
        )

        # get averaged response vector
        avg_response_vectors = [i for i in avg_response_vectors if i is not None]

        similarity_matrix_list = []
        # populate a similarity matrix
        full_similarity_count = np.zeros((128, 128))
        # add similarity to similarity matrix
        for (
            similarity_matrix,
            interp_points_this_unit,
            averaged_response_vectors,
        ) in tqdm(avg_response_vectors, desc="compute unit similarity", leave=False):

            # if there are nans, remove them from calculation
            if np.any(np.isnan(similarity_matrix)):
                mask = np.isnan(np.sum(np.tril(similarity_matrix), axis=1)) == False
                averaged_response_vectors = averaged_response_vectors[mask]
                similarity_matrix = similarity_matrix[mask]
                similarity_matrix = similarity_matrix[:, mask]
                interp_points_this_unit = interp_points_this_unit[mask]

            full_similarity_matrix = np.zeros((128, 128))
            full_similarity_matrix[:] = np.nan

            if len(interp_points_this_unit) < 2:
                continue
            for idx, i in enumerate(interp_points_this_unit):
                full_similarity_matrix[interp_points_this_unit, i] = similarity_matrix[
                    idx
                ]
                full_similarity_count[interp_points_this_unit, i] += 1
            response_vectors_all[interp_points_this_unit] += averaged_response_vectors
            full_interp_count[interp_points_this_unit] += 1

            similarity_matrix_list.append(full_similarity_matrix)

            # plt.matshow(full_similarity_matrix)
            # plt.show()

        # skip if there's not enough data for this interpolation
        if len(similarity_matrix_list) < 1:
            continue

        # for this interpolation, get the average similarity matrix across units, and save it
        avg_similarity_matrix = np.nanmean(similarity_matrix_list, axis=0)
        similarity_all_list.append(avg_similarity_matrix)
        similarity_count_all.append(full_similarity_count)

        # plotting
        plot = True
        if plot:
            lim = np.tril(avg_similarity_matrix)
            np.fill_diagonal(lim, 0)
            print(interpolation)
            plt.matshow(avg_similarity_matrix, vmax=np.nanmax(lim))
            plt.show()

    # for visualization, get averaged response vector
    response_vectors_all = (response_vectors_all.T / full_interp_count).T

    # create a similarity matrix averaged across trials/interpolations (weighted by number of presentations)
    averaged_simiarity_matrix = np.nansum(
        [sm * sc for sm, sc in zip(similarity_all_list, similarity_count_all)], axis=0
    ) / np.sum(similarity_count_all, axis=0)
    np.fill_diagonal(averaged_simiarity_matrix, 0)

    # get distance function
    interp_points = []
    dists = []
    for ri, interp_point in enumerate(np.arange(128)):
        interp_points.append(interp_point)
        mask = np.arange(128) > 63
        mask_left = (np.arange(128) > 63) & (np.arange(128) != interp_point)
        mask_right = (np.arange(128) <= 63) & (np.arange(128) != interp_point)
        similarity_left = averaged_simiarity_matrix[ri][mask_left]
        similarity_right = averaged_simiarity_matrix[ri][mask_right]
        # ignore nans, which correspond to relationships we don't have access to
        dists.append(np.nanmean(similarity_left) - np.nanmean(similarity_right))

    # fit model
    (
        (_min, _max, _inflection, _slope),
        results_logistic,
        y_model,
        r_squared,
    ) = fit_FourParameterLogistic(
        interp_points, dists, _min_bounds=[-1, 1], _max_bounds=[-1, 1]
    )

    # save to dictionary
    all_dict = {}
    all_dict["neurometric_distances_min_{}".format(identifier)] = _min
    all_dict["neurometric_distances_max_{}".format(identifier)] = _max
    all_dict["neurometric_distances_inflection_{}".format(identifier)] = _inflection
    all_dict["neurometric_distances_slope_{}".format(identifier)] = _slope
    all_dict["neurometric_distances_r2_{}".format(identifier)] = r_squared

    # plot
    plot = True
    if plot:
        print(interpolation, (_min, _max, _inflection, _slope))
        interp_range = np.linspace(0, 127, 1000)
        y_interp = FourParameterLogistic(
            {"_min": _min, "_max": _max, "inflection": _inflection, "slope": _slope},
            interp_range,
        )
        fig, axs = plt.subplots(ncols=3, figsize=(10, 3))
        axs[0].matshow(averaged_simiarity_matrix)
        axs[1].scatter(interp_points, dists, alpha=0.5, s=5)
        axs[1].plot(interp_range, y_interp, color="red")
        # fill diagonals to make it easier to see
        axs[2].matshow(response_vectors_all, aspect="auto", cmap=plt.cm.Greys)
        plt.show()

        # plot, averaged over all interpolations, where available
        test = np.nanmean(similarity_all_list, axis=0)
        np.fill_diagonal(test, 0)
        plt.matshow(test)
        plt.matshow(averaged_simiarity_matrix)

    return (
        pd.Series(all_dict),
        interp_points,
        dists,
        avg_similarity_matrix,
        response_vectors_all,
    )
