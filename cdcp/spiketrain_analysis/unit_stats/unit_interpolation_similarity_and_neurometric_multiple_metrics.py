import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm


pd.set_option("display.max_columns", 500)
from cdcp.spiketrain_analysis.spiketrain_utils import (
    bin_interp_points,
    get_average_response_vector,
    create_dense_similarity_matrix,
    get_similarity_matrix
)
from cdcp.spiketrain_analysis.neurometric import (
    get_interp_points_dists_from_similarity_matrix,
)
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic


def compute_interpolation_sm_and_neurometric_multiple_metrics(
    trial_aligned_spikes, n_interp_point_bins=16, plot=False,
    flip_bins=True,
    similarity_metrics=["correlation", "cosine", "euclidean", "manhattan"],
):
    # ensure that transcribed interp points aren't some other stimulus I
    # tried playing back
    trial_aligned_spikes = trial_aligned_spikes[
        (trial_aligned_spikes["interp_point"].values.astype("int") >= 0)
        & (trial_aligned_spikes["interp_point"].values.astype("int") <= 127)
    ]

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point_binned"] = bin_interp_points(
        trial_aligned_spikes["interp_point"].values.astype("int"),
        n_interp_point_bins,
        flip_bins=flip_bins,
    )

    all_dict = {}
    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        # subset
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interpolation
        ]
        # skip inf there's too little data
        if len(interp_trial_aligned_spikes) < 10:
            continue

        # get average response vectors
        avg_response_vectors, interp_points_this_unit = get_average_response_vector(
            interp_trial_aligned_spikes
        )

        for smi, similarity_metric in enumerate(similarity_metrics):

            # compute similarity matrix
            sm = create_dense_similarity_matrix(
                avg_response_vectors,
                interp_points_this_unit,
                n_interp_bins=n_interp_point_bins,
                similarity_metric = similarity_metric
            )

            if len(interp_points_this_unit) != n_interp_point_bins:
                rv = np.zeros((n_interp_point_bins, avg_response_vectors.shape[1]))
                rv[:] = np.nan
                for i, vec in zip(interp_points_this_unit, avg_response_vectors):
                    rv[i] = vec
            else:
                rv = avg_response_vectors

            # get logistic points
            interp_points, dists = get_interp_points_dists_from_similarity_matrix(
                np.arange(n_interp_point_bins), sm, n_interp_point_bins=n_interp_point_bins
            )
            interp_points = np.array(interp_points)[np.isnan(dists) == False]
            dists = np.array(dists)[np.isnan(dists) == False]

            if len(dists) < 8:
                continue

            # fit logistic
            (
                (_min, _max, _inflection, _slope),
                results_logistic,
                y_model,
                r_squared,
            ) = fit_FourParameterLogistic(
                interp_points,
                dists,
                _inflection=(n_interp_point_bins / 2) - 1,
                _inflection_bounds=[
                    0 + int(n_interp_point_bins / 8),
                    n_interp_point_bins - 1 - int(n_interp_point_bins / 8),
                ],
                _min_bounds=[-1, 2],
                _max_bounds=[-1, 2],
            )

            # save results
            all_dict["sm_{}_{}".format(similarity_metric, interpolation)] = sm.astype("float32")
            if smi == 0:
                all_dict["rv_{}".format(interpolation)] = rv.astype("float32")
                all_dict["ips_{}".format(interpolation)] = interp_points_this_unit
            all_dict["nm_min_{}_{}".format(similarity_metric, interpolation)] = _min
            all_dict["nm_max_{}_{}".format(similarity_metric, interpolation)] = _max
            all_dict["nm_r2_{}_{}".format(similarity_metric, interpolation)] = r_squared
            all_dict["nm_inflection_{}_{}".format(similarity_metric, interpolation)] = _inflection
            all_dict["nm_slope_{}_{}".format(similarity_metric, interpolation)] = _slope
            all_dict["nm_range_{}_{}".format(similarity_metric, interpolation)] = _max - _min
            all_dict["nm_scaled_slope_{}_{}".format(similarity_metric, interpolation)] = _slope / (_max - _min)

            # plot

            if plot:
                print((_min, _max, _inflection, _slope))
                fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
                axs[0].matshow(avg_response_vectors, aspect="auto")
                axs[1].matshow(sm)
                axs[2].scatter(interp_points, dists)
                interp_range = np.linspace(0, n_interp_point_bins - 1, 1000)
                y_interp = FourParameterLogistic(
                    {
                        "_min": _min,
                        "_max": _max,
                        "inflection": _inflection,
                        "slope": _slope,
                    },
                    interp_range,
                )
                axs[2].plot(interp_range, y_interp, color="red")
                axs[2].set_ylim([-1, 2])
                plt.show()

    # do the same with the average interpolation
    # get mean similarity matrix and response vectors
    for smi, similarity_metric in enumerate(similarity_metrics):
        sm_list = [
            all_dict["sm_{}_{}".format(similarity_metric, interpolation)]
            for interpolation in tqdm(
                trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
            )
            if "sm_{}_{}".format(similarity_metric, interpolation) in all_dict.keys()
        ]

        if len(sm_list) == 0:
            return pd.Series({})

        sm_list = np.stack(sm_list)

        rv_list = np.stack(
            [
                all_dict["rv_{}".format(interpolation)]
                for interpolation in tqdm(
                    trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
                )
                if "rv_{}".format(interpolation) in all_dict.keys()
            ]
        )
        avg_response_vectors = np.nanmean(rv_list, axis=0)
        sm = np.nanmean(sm_list, axis=0)

        interpolation = "ALL"
        # get logistic points
        interp_points, dists = get_interp_points_dists_from_similarity_matrix(
            np.arange(n_interp_point_bins), sm, n_interp_point_bins=n_interp_point_bins
        )
        # fit logistic
        (
            (_min, _max, _inflection, _slope),
            results_logistic,
            y_model,
            r_squared,
        ) = fit_FourParameterLogistic(
            interp_points,
            dists,
            _inflection=(n_interp_point_bins / 2) - 1,
            _inflection_bounds=[
                0 + int(n_interp_point_bins / 8),
                n_interp_point_bins - 1 - int(n_interp_point_bins / 8),
            ],
            _min_bounds=[-1, 2],
            _max_bounds=[-1, 2],
        )

        # save results
        all_dict["sm_{}_{}".format(similarity_metric, interpolation)] = sm.astype("float32")
        if smi == 0:
            all_dict["rv_{}".format(interpolation)] = avg_response_vectors.astype("float32")
        all_dict["nm_min_{}_{}".format(similarity_metric, interpolation)] = _min
        all_dict["nm_max_{}_{}".format(similarity_metric, interpolation)] = _max
        all_dict["nm_inflection_{}_{}".format(similarity_metric, interpolation)] = _inflection
        all_dict["nm_slope_{}_{}".format(similarity_metric, interpolation)] = _slope
        all_dict["nm_range_{}_{}".format(similarity_metric, interpolation)] = _max - _min
        all_dict["nm_scaled_slope_{}_{}".format(similarity_metric, interpolation)] = _slope / (_max - _min)
        all_dict["nm_r2_{}_{}".format(similarity_metric, interpolation)] = r_squared

        # plot
        if plot:
            print((_min, _max, _inflection, _slope))
            fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
            axs[0].matshow(avg_response_vectors, aspect="auto")
            axs[1].matshow(sm)
            axs[2].scatter(interp_points, dists)
            interp_range = np.linspace(0, n_interp_point_bins - 1, 1000)
            y_interp = FourParameterLogistic(
                {"_min": _min, "_max": _max, "inflection": _inflection, "slope": _slope},
                interp_range,
            )
            axs[2].plot(interp_range, y_interp, color="red")
            axs[2].set_ylim([-1, 2])
            plt.show()

    return pd.Series(all_dict)
