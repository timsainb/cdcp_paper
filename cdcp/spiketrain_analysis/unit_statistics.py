import warnings
from pynndescent import NNDescent
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    RidgeCV,
    LogisticRegressionCV,
)
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeClassifierCV as RidgeClassifierCV_sklearn
from elephant import spike_train_synchrony
import neo
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu as mannwhitneyu_scipy
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.utils.extmath import softmax


class RidgeClassifierCV(RidgeClassifierCV_sklearn):
    # add a predict_proba function
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)


default_regression = RidgeClassifierCV  # LogisticRegressionCV  # RidgeClassifierCV


def mannwhitneyu(x, y, **kwargs):
    try:
        return mannwhitneyu_scipy(x, y, **kwargs)
    except ValueError:
        from unittest import mock

        obj = mock.Mock()
        obj.pvalue = np.nan
        obj.statistic = np.nan
        return obj


def mean_std_with_window(x, y, window_range=np.arange(128), window_size=10):
    """
    Get a smooth mean and standard deviation over a window

    Parameters
    ----------
    x : [type]
        [description]
    y : [type]
        [description]
    window_range : [type], optional
        [description], by default np.arange(128)
    window_size : int, optional
        [description], by default 10

    Returns
    -------
    [type]
        [description]
    """
    mean_signal_with_window = [
        np.mean(
            x[
                (y > (interp_point - window_size / 2))
                & (y < (interp_point + window_size / 2))
            ]
        )
        for interp_point in window_range
    ]
    std_signal_with_window = [
        np.std(
            x[
                (y > (interp_point - window_size / 2))
                & (y < (interp_point + window_size / 2))
            ]
        )
        for interp_point in window_range
    ]
    return mean_signal_with_window, std_signal_with_window


def get_unit_spike_trains(
    unit_recording_ids,
    spikesorting_folder,
    sorter,
    unit,
    save_folder="trial_aligned_spikes",
):
    trial_aligned_spikes_list = []
    for unit, recording_id in tqdm(
        unit_recording_ids, desc="unit spike trains", leave=False
    ):
        trial_aligned_spikes_loc = (
            spikesorting_folder
            / save_folder
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
            # print("\t", trial_aligned_spikes_loc)
    if len(trial_aligned_spikes_list) < 1:
        #breakme
        return None
    else:
        return pd.concat(trial_aligned_spikes_list)


def get_spike_train_vector(
    row, gaussian_sigma=5, nbins=100, gaussian_sigma_ms=5, return_gauss=False
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

    if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
        bins = np.linspace(1, 2, nbins + 1)
    else:
        bins = np.linspace(0, 1, nbins + 1)

    hist = np.histogram(row.spike_times, bins, density=False)[0]

    if return_gauss:
        gauss_convolved_psth = gaussian_filter1d(
            hist.astype("float"), gaussian_sigma, mode="constant"
        )
        return gauss_convolved_psth
    else:
        return hist


def estimate_synchrony(trial_aligned_spikes, nex=100, min_spike_trains=5):
    most_frequent_stim = trial_aligned_spikes.stim.value_counts()[:nex].index
    synchrony = []
    for stim in tqdm(most_frequent_stim, leave=False, desc="synchrony"):
        stim_lengths = trial_aligned_spikes[
            trial_aligned_spikes.stim == stim
        ].stim_length

        # ensure no bad trials
        bad_trials = np.abs(stim_lengths - np.median(stim_lengths)) > 0.1
        spike_trains = trial_aligned_spikes[trial_aligned_spikes.stim == stim][
            bad_trials == False
        ].spike_times.values

        spike_trains = [i for i in spike_trains if len(i) > 2]
        if len(spike_trains) < min_spike_trains:
            continue
        stim_length = np.max(stim_lengths)
        # print([len(i) for i in spike_trains])
        sts = [
            neo.SpikeTrain(
                np.array(i)[np.array(i) <= stim_length],
                units="sec",
                t_stop=stim_length + 0.05,
            )
            for i in spike_trains
        ]
        try:
            synchrony.append(spike_train_synchrony.spike_contrast(sts))
        except ValueError as e:
            print(e)
            continue
    if len(synchrony) == 0:
        return
    return pd.Series(
        {"synchrony_mean": np.mean(synchrony), "synchrony_std": np.std(synchrony)}
    )


def get_unit_predictability(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    prediction_model=default_regression,
    prediction_model_kwargs={},
    cross_val_fold=5,
):
    predictability_dict = {}
    class_prediction_from_spiketrain_scores_all = []
    interp_prediction_from_spiketrain_scores_all = []
    interp_n = []
    for interp in tqdm(
        trial_aligned_spikes.interp.unique(), desc="predictability_interp", leave=False
    ):
        interp_trials = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ].sort_values(by="interp_point")
        if len(interp_trials) < min_classifier_exemplars:
            continue
        interp_n.append(len(interp_trials))
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)

        stim_class = (interp_trials.interp_point.values > 63).astype("int")
        stim_interp_point = interp_trials.interp_point.values

        try:
            class_prediction_from_spiketrain_scores = (
                prediction_model(**prediction_model_kwargs)
                .fit(spike_trains, stim_class)
                .score(spike_trains, stim_class)
            )
        except:
            class_prediction_from_spiketrain_scores = np.nan

        predictability_dict[
            "class_predictability_{}".format(interp)
        ] = class_prediction_from_spiketrain_scores

        class_prediction_from_spiketrain_scores_all.append(
            class_prediction_from_spiketrain_scores
        )
        try:
            interp_point_from_spiketrain_scores = (
                RidgeCV()
                .fit(spike_trains, stim_interp_point)
                .score(spike_trains, stim_interp_point)
            )
        except:
            interp_point_from_spiketrain_scores = np.nan

        predictability_dict["interp_predictability_{}".format(interp)] = np.mean(
            interp_point_from_spiketrain_scores
        )
        interp_prediction_from_spiketrain_scores_all.append(
            interp_point_from_spiketrain_scores
        )
    if len(class_prediction_from_spiketrain_scores_all) == 0:
        return
    if np.sum(interp_n) == 0:
        return
    interp_n = np.array(interp_n) / np.sum(interp_n)
    predictability_dict["class_predictability".format(interp)] = np.sum(
        [i * j for i, j in zip(interp_n, class_prediction_from_spiketrain_scores_all)]
    )

    predictability_dict["interp_predictability".format(interp)] = np.sum(
        [i * j for i, j in zip(interp_n, interp_prediction_from_spiketrain_scores_all)]
    )
    return pd.Series(predictability_dict)


def get_silhouette(trial_aligned_spikes, min_exemplars=100):
    sil_dict = {}
    all_sil = []
    n_interp = []
    for interp in tqdm(
        trial_aligned_spikes.interp.unique(), desc="silhouette_score", leave=False
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        if len(interp_trials) < min_exemplars:
            continue
        n_interp.append(len(interp_trials))
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        stim_class = (interp_trials.interp_point.values > 63).astype("int")
        if len(np.unique(stim_class)) < 2:
            continue
        sil_score = silhouette_score(spike_trains, stim_class)
        # sil_score, _ = silhouette_score_block(spike_trains, stim_class)
        all_sil.append(sil_score)
        sil_dict["sil_{}".format(interp)] = sil_score
    n_interp = np.array(n_interp) / np.sum(n_interp)
    sil_dict["sil"] = np.sum([i * j for i, j in zip(n_interp, all_sil)])
    return pd.Series(sil_dict)


def get_interp_prediction_error_relative_to_average_correct_vs_incorrect(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=True,
    window_size_smooth=10,
):
    """Class prediction errors in correct vs incorrect trials"""
    incorrect_correct_dict = {}
    all_z_score_error_correct = []
    all_z_score_error_incorrect = []

    all_dict = {
        "interp_point": [],
        "interpolation": [],
        "error": [],
        "z_error_relative_avg": [],
        "correct_incorrect": [],
    }

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="correct vs incorrect error",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        if len(interp_trials) < min_classifier_exemplars:
            continue

        # subset trial types
        passive_trials = (interp_trials.correct.isnull()).values
        correct_trials = (interp_trials.correct == True).values
        incorrect_trials = (interp_trials.correct == False).values

        # not enough trials
        if (
            np.sum(incorrect_trials) < 1
            or np.sum(correct_trials) < 1
            or np.sum(passive_trials) < 1
        ):
            continue

        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        true_interp = interp_trials.interp_point.values

        # train linear regression to predict interp from spikes on passive trials
        reg = RidgeCV().fit(spike_trains[passive_trials], true_interp[passive_trials])
        # predict on all data
        predicted_interp = reg.predict(spike_trains)
        # get prediction error
        interp_prediction_error = predicted_interp - true_interp
        # get the mean and std error for each timestep, because
        # .  error tends to be greater in some points
        # .   so we normalize by points
        (
            mean_prediction_error_with_window,
            std_prediction_error_with_window,
        ) = mean_std_with_window(
            interp_prediction_error,
            true_interp,
            window_range=np.arange(128),
            window_size=window_size_smooth,
        )
        # z score data
        interp_prediction_error_z_scored = np.array(
            [
                (i - mean_prediction_error_with_window[interp_point])
                / std_prediction_error_with_window[interp_point]
                for interp_point, i in zip(true_interp, interp_prediction_error)
            ]
        )
        # break out by correct vs incorrect
        prediction_error_correct = np.abs(
            interp_prediction_error_z_scored[correct_trials]
        )
        all_z_score_error_correct.append(prediction_error_correct)
        prediction_error_incorrect = np.abs(
            interp_prediction_error_z_scored[incorrect_trials]
        )
        all_z_score_error_incorrect.append(prediction_error_incorrect)

        # statistically test difference
        mannwhitneyu_test = mannwhitneyu(
            prediction_error_incorrect,
            prediction_error_correct,
            alternative="greater",
        )
        pvalue = mannwhitneyu_test.pvalue
        u_statistic = mannwhitneyu_test.statistic
        incorrect_correct_dict[
            "incorrect_correct_u_stat_interp_point_{}".format(interp)
        ] = u_statistic
        incorrect_correct_dict[
            "incorrect_correct_p_interp_point_{}".format(interp)
        ] = pvalue
        incorrect_correct_dict[
            "interp_mean_incorrect_error_rel_avg_{}".format(interp)
        ] = np.mean(prediction_error_incorrect)
        incorrect_correct_dict[
            "interp_mean_correct_error_rel_avg_{}".format(interp)
        ] = np.mean(prediction_error_correct)

        all_dict["interp_point"].append(
            np.concatenate([true_interp[correct_trials], true_interp[incorrect_trials]])
        )
        all_dict["interpolation"].append(
            np.repeat(interp, np.sum(correct_trials) + np.sum(incorrect_trials))
        )
        all_dict["error"].append(
            np.concatenate(
                [
                    interp_prediction_error[correct_trials],
                    interp_prediction_error[incorrect_trials],
                ]
            )
        )
        all_dict["z_error_relative_avg"].append(
            np.concatenate([prediction_error_correct, prediction_error_incorrect])
        )
        all_dict["correct_incorrect"].append(
            np.repeat(
                [1, 0], [len(prediction_error_correct), len(prediction_error_incorrect)]
            )
        )

    # cannot compute stats of there are no correct or incorrect trials
    if len(all_z_score_error_correct) == 0:
        return None, None
    if len(all_z_score_error_incorrect) == 0:
        return None, None

    all_z_score_error_correct = np.concatenate(all_z_score_error_correct)
    all_z_score_error_incorrect = np.concatenate(all_z_score_error_incorrect)

    incorrect_correct_dict["interp_mean_incorrect_error_rel_avg"] = np.mean(
        all_z_score_error_incorrect
    )
    incorrect_correct_dict["interp_mean_correct_error_rel_avg"] = np.mean(
        all_z_score_error_correct
    )

    # TODO save error, z scored error, interp point, interp as csv
    if len(all_dict["interp_point"]) == 0:
        return None, None
    all_dict["interp_point"] = list(np.concatenate(all_dict["interp_point"]))
    all_dict["interpolation"] = list(np.concatenate(all_dict["interpolation"]))
    all_dict["error"] = list(np.concatenate(all_dict["error"]))
    all_dict["z_error_relative_avg"] = list(
        np.concatenate(all_dict["z_error_relative_avg"])
    )
    all_dict["correct_incorrect"] = list(np.concatenate(all_dict["correct_incorrect"]))

    mannwhitneyu_test = mannwhitneyu(
        all_z_score_error_incorrect,
        all_z_score_error_correct,
        alternative="greater",
    )
    p_value = mannwhitneyu_test.pvalue
    u_statistic = mannwhitneyu_test.statistic
    incorrect_correct_dict["interp_incorrect_correct_u_stat"] = u_statistic
    incorrect_correct_dict["interp_incorrect_correct_p_value"] = p_value

    all_dict_pd = pd.DataFrame(all_dict)
    # all_dict_pd_descriptive = all_dict_pd.groupby(
    #    ["correct_incorrect", "interpolation", "interp_point"]
    # ).agg([np.mean, len])

    return pd.Series(incorrect_correct_dict), all_dict_pd


def get_categorical_prediction_error_relative_to_average_correct_vs_incorrect(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=True,
    prediction_model=default_regression,
    prediction_model_kwargs={},
    window_size_smooth=10,
):
    """Class prediction errors in correct vs incorrect trials"""
    incorrect_correct_dict = {}

    difference_from_mean_error_correct_all = []
    difference_from_mean_error_incorrect_all = []
    all_dict = {
        "interp_point": [],
        "interpolation": [],
        "error": [],
        "error_relative_avg": [],
        "correct_incorrect": [],
    }
    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="correct vs incorrect error",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        if len(interp_trials) < min_classifier_exemplars:
            continue

        # subset trial types
        passive_trials = (interp_trials.correct.isnull()).values
        correct_trials = (interp_trials.correct == True).values
        incorrect_trials = (interp_trials.correct == False).values

        # not enough trials
        if (
            np.sum(incorrect_trials) < 1
            or np.sum(correct_trials) < 1
            or np.sum(passive_trials) < 1
        ):
            continue

        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        true_interp = interp_trials.interp_point.values
        stim_class = (interp_trials.interp_point.values > 63).astype("int")

        if np.sum(passive_trials) < min_classifier_exemplars:
            continue

        reg = prediction_model(**prediction_model_kwargs).fit(
            spike_trains[passive_trials], stim_class[passive_trials]
        )

        if use_prob:
            predicted_class = reg.predict_proba(spike_trains)[:, 1]
        else:
            predicted_class = reg.predict(spike_trains)

        # get the prediction error
        prediction_error = np.abs(stim_class - predicted_class)

        # get the average prediction error by inpterpolation point
        mean_error_with_window, std_error_with_window = mean_std_with_window(
            prediction_error, true_interp, np.arange(128), window_size_smooth
        )

        # get the error relative to average for this interpolation point
        difference_from_mean_error = prediction_error - np.array(
            [mean_error_with_window[i] for i in true_interp]
        )
        # break error out by correct vs incorrect
        difference_from_mean_error_correct = difference_from_mean_error[correct_trials]
        difference_from_mean_error_incorrect = difference_from_mean_error[
            incorrect_trials
        ]
        mean_correct_error = np.mean(difference_from_mean_error_correct)
        mean_incorrect_error = np.mean(difference_from_mean_error_incorrect)
        difference_from_mean_error_correct_all.append(
            difference_from_mean_error_correct
        )
        difference_from_mean_error_incorrect_all.append(
            difference_from_mean_error_incorrect
        )
        incorrect_correct_dict[
            "mean_correct_error_rel_avg_{}".format(interp)
        ] = mean_correct_error
        incorrect_correct_dict[
            "mean_incorrect_error_rel_avg_{}".format(interp)
        ] = mean_incorrect_error

        # test whether average corrected error is greater in correct or incorrect trials
        mannwhitneyu_test = mannwhitneyu(
            difference_from_mean_error_incorrect,
            difference_from_mean_error_correct,
            alternative="greater",
        )
        p_value = mannwhitneyu_test.pvalue
        u_statistic = mannwhitneyu_test.statistic
        incorrect_correct_dict[
            "incorrect_correct_u_stat_{}".format(interp)
        ] = u_statistic
        incorrect_correct_dict["incorrect_correct_p_value_{}".format(interp)] = p_value

        all_dict["interp_point"].append(
            np.concatenate([true_interp[correct_trials], true_interp[incorrect_trials]])
        )
        all_dict["interpolation"].append(
            np.repeat(interp, np.sum(correct_trials) + np.sum(incorrect_trials))
        )
        all_dict["error"].append(
            np.concatenate(
                [prediction_error[correct_trials], prediction_error[incorrect_trials]]
            )
        )
        all_dict["error_relative_avg"].append(
            np.concatenate(
                [
                    difference_from_mean_error_correct,
                    difference_from_mean_error_incorrect,
                ]
            )
        )
        all_dict["correct_incorrect"].append(
            np.repeat(
                [1, 0],
                [
                    len(difference_from_mean_error_correct),
                    len(difference_from_mean_error_incorrect),
                ],
            )
        )

    # cannot compute stats of there are no correct or incorrect trials
    if len(difference_from_mean_error_correct_all) == 0:
        return None, None
    if len(difference_from_mean_error_incorrect_all) == 0:
        return None, None

    difference_from_mean_error_correct_all = np.concatenate(
        difference_from_mean_error_correct_all
    )
    difference_from_mean_error_incorrect_all = np.concatenate(
        difference_from_mean_error_incorrect_all
    )
    mean_correct_error = np.mean(difference_from_mean_error_correct_all)
    mean_incorrect_error = np.mean(difference_from_mean_error_incorrect_all)
    incorrect_correct_dict["mean_correct_error_rel_avg"] = mean_correct_error
    incorrect_correct_dict["mean_incorrect_error_rel_avg"] = mean_incorrect_error

    if len(all_dict["interp_point"]) == 0:
        return None, None
    all_dict["interp_point"] = list(np.concatenate(all_dict["interp_point"]))
    all_dict["interpolation"] = list(np.concatenate(all_dict["interpolation"]))
    all_dict["error"] = list(np.concatenate(all_dict["error"]))
    all_dict["error_relative_avg"] = list(
        np.concatenate(all_dict["error_relative_avg"])
    )
    all_dict["correct_incorrect"] = list(np.concatenate(all_dict["correct_incorrect"]))

    # test whether average corrected error is greater in correct or incorrect trials
    mannwhitneyu_test = mannwhitneyu(
        difference_from_mean_error_incorrect_all,
        difference_from_mean_error_correct_all,
        alternative="greater",
    )
    p_value = mannwhitneyu_test.pvalue
    u_statistic = mannwhitneyu_test.statistic
    incorrect_correct_dict["incorrect_correct_u_stat"] = u_statistic
    incorrect_correct_dict["incorrect_correct_p_value"] = p_value

    # save mean and len rather than individual datapoints
    all_dict_pd = pd.DataFrame(all_dict)
    # all_dict_pd_descriptive = all_dict_pd.groupby(
    #    ["correct_incorrect", "interpolation", "interp_point"]
    # ).agg([np.mean, len])

    return pd.Series(incorrect_correct_dict), all_dict_pd


def get_stimulus_category_decoding_relative_to_cue_shift(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=False,
    prediction_model=default_regression,
    prediction_model_kwargs={},
    window_size_smooth=10,
):
    cue_prob_dict = {
        "CR1": 0.125,
        "CR0": 0.25,
        "CL0": 0.75,
        "CL1": 0.875,
        "NC": 0.5,
        "CN": 0.5,
    }
    all_dict = {
        "interp_point": [],
        "interpolation": [],
        "error": [],
        "difference_from_mean_error": [],
        "cue": [],
        "cue_probability": [],
    }
    cue_shift_dict = {}

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="correct vs incorrect error",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        true_interp = interp_trials.interp_point.values
        true_class = np.array(true_interp > 63).astype("int")

        # get only cued trials (which can be passive)
        uncued_mask = (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values == False
        )
        cued_trials_mask = np.array(interp_trials.response.isnull() == False) & (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values
        )

        if len(interp_trials) < min_classifier_exemplars:
            continue

        # not enough trials
        if np.sum(cued_trials_mask) < 5:
            continue

        # subset trial types
        passive_trials = (interp_trials.correct.isnull()).values
        correct_trials = (interp_trials.correct == True).values
        incorrect_trials = (interp_trials.correct == False).values

        # train regression on only uncued trials
        reg = prediction_model(**prediction_model_kwargs).fit(
            spike_trains[uncued_mask], true_class[uncued_mask]
        )

        # predict class probability
        if use_prob:
            predicted_class = reg.predict_proba(spike_trains)[:, 1]
        else:
            predicted_class = reg.predict(spike_trains)

        # get the average prediction error by inpterpolation point
        mean_prediction_with_window, std_prediction_with_window = mean_std_with_window(
            predicted_class, true_interp, np.arange(128), window_size_smooth
        )

        # get only cued trials
        cues = interp_trials[cued_trials_mask].cue

        # get cue probabilities
        cue_prob = [cue_prob_dict[i] for i in cues]

        # get the error
        error = true_class[cued_trials_mask] - predicted_class[cued_trials_mask]

        # get the difference between the average prediction and the actual prediction
        difference_from_mean_error = (
            np.array(
                [
                    mean_prediction_with_window[i]
                    for i in true_interp[np.array(cued_trials_mask)]
                ]
            )
            - predicted_class[cued_trials_mask]
        )

        # compute the correlation between the cue probability, and the difference in error from the mean error
        #   if the difference is below zero, the cue is causing an overestimation toward category 0. If the difference
        #   if above zero, the cue is causing an overestimation toward category 1.
        if len(cue_prob) >= 2:
            r2_category_prediction, p_category_prediction = pearsonr(
                cue_prob, difference_from_mean_error
            )
        else:
            r2_category_prediction = None
            p_category_prediction = None

        cue_shift_dict[
            "r2_cue_prob_prediction_error_rel_avg_{}".format(interp)
        ] = r2_category_prediction
        cue_shift_dict[
            "p_cue_prob_prediction_error_rel_avg_{}".format(interp)
        ] = p_category_prediction

        all_dict["interp_point"].append(true_interp[cued_trials_mask])
        all_dict["interpolation"].append(np.repeat(interp, np.sum(cued_trials_mask)))
        all_dict["error"].append(error)
        all_dict["difference_from_mean_error"].append(difference_from_mean_error)
        all_dict["cue"].append(cues)
        all_dict["cue_probability"].append(cue_prob)

    if len(all_dict["cue_probability"]) == 0:
        return None, None

    if len(all_dict["interp_point"]) == 0:
        return None, None
    all_dict["interp_point"] = list(np.concatenate(all_dict["interp_point"]))
    all_dict["interpolation"] = list(np.concatenate(all_dict["interpolation"]))
    all_dict["error"] = list(np.concatenate(all_dict["error"]))
    all_dict["difference_from_mean_error"] = list(
        np.concatenate(all_dict["difference_from_mean_error"])
    )
    all_dict["cue"] = list(np.concatenate(all_dict["cue"]))
    all_dict["cue_probability"] = list(np.concatenate(all_dict["cue_probability"]))

    # get the correlation between the shift in prediction error and cue probaility
    r2_category_prediction, p_category_prediction = pearsonr(
        all_dict["cue_probability"], all_dict["difference_from_mean_error"]
    )
    cue_shift_dict["r2_cue_prob_prediction_error_rel_avg"] = r2_category_prediction
    cue_shift_dict["p_cue_prob_prediction_error_rel_avg"] = p_category_prediction
    return pd.Series(cue_shift_dict), pd.DataFrame(all_dict)


def get_interp_decoding_shift_relative_to_cue_shift(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=True,
    nearest_neighbors_interp_point_prediction=False,
    window_size_smooth=10,
):
    cue_prob_dict = {
        "CR1": 0.125,
        "CR0": 0.25,
        "CL0": 0.75,
        "CL1": 0.875,
        "NC": 0.5,
        "CN": 0.5,
    }
    all_dict = {
        "interp_point": [],
        "interpolation": [],
        "interp_prediction_error": [],
        "interp_prediction_error_z_scored": [],
        "cue": [],
        "cue_probability": [],
    }
    cue_shift_dict = {}

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="correct vs incorrect error",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        true_interp = interp_trials.interp_point.values
        true_class = np.array(true_interp > 63).astype("int")

        # get only cued trials (which can be passive)
        uncued_mask = (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values == False
        )
        cued_trials_mask = np.array(interp_trials.response.isnull() == False) & (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values
        )

        if len(interp_trials) < min_classifier_exemplars:
            continue

        # not enough trials
        if np.sum(cued_trials_mask) < 5:
            continue

        # subset trial types
        passive_trials = (interp_trials.correct.isnull()).values
        correct_trials = (interp_trials.correct == True).values
        incorrect_trials = (interp_trials.correct == False).values

        # predict the interpolation point for each datapoint
        if nearest_neighbors_interp_point_prediction:
            predicted_interp = interp_point_nn_prediction(
                interp_trials, spike_trains, true_interp, n_neighbors=n_neighbors_cue
            )
        else:
            predicted_interp = interp_point_lm_prediction(
                interp_trials,
                spike_trains,
                true_interp,
            )
        interp_prediction_error = predicted_interp - true_interp

        # get the average prediction
        (
            mean_prediction_error_with_window,
            std_prediction_error_with_window,
        ) = mean_std_with_window(
            interp_prediction_error,
            true_interp,
            window_range=np.arange(128),
            window_size=window_size_smooth,
        )

        # z score prediction shift, relative to interp and interp point
        # higher values indicate prediction was higher than true interp point
        interp_prediction_error_z_scored = np.array(
            [
                (i - mean_prediction_error_with_window[interp_point])
                / std_prediction_error_with_window[interp_point]
                for interp_point, i in zip(true_interp, interp_prediction_error)
            ]
        )

        # subset trials with a response
        cued_trials_mask = np.array(interp_trials.response.isnull() == False)
        cued_trials = interp_trials[cued_trials_mask]

        # get only cued trials
        cues = interp_trials[cued_trials_mask].cue

        # get cue probabilities
        cue_prob = np.array([cue_prob_dict[i] for i in cues])

        # if there are NaNs, there's too little data to estimate
        if np.any(np.isnan(interp_prediction_error_z_scored[cued_trials_mask])):
            continue

        if len(cue_prob) > 2:
            # perform correlation
            # positive correlation indicates that predictions are moved toward higher values (interp_points)
            # when cued with higher probabilties (CL), and lower values (interp points) when cued with
            # lower probabilities (CR)
            r2_interp_prediction, p_interp_prediction = pearsonr(
                cue_prob, interp_prediction_error_z_scored[cued_trials_mask]
            )

        cue_shift_dict[
            "r2_cue_prob_interp_shift_rel_avg_{}".format(interp)
        ] = r2_interp_prediction
        cue_shift_dict[
            "p_cue_prob_interp_shift_rel_avg_{}".format(interp)
        ] = p_interp_prediction

        all_dict["interp_point"].append(true_interp[cued_trials_mask])
        all_dict["interpolation"].append(np.repeat(interp, np.sum(cued_trials_mask)))
        all_dict["interp_prediction_error"].append(
            interp_prediction_error[cued_trials_mask]
        )
        all_dict["interp_prediction_error_z_scored"].append(
            interp_prediction_error_z_scored[cued_trials_mask]
        )
        all_dict["cue"].append(cues)
        all_dict["cue_probability"].append(cue_prob)

    if len(all_dict["cue_probability"]) == 0:
        return None, None
    if len(all_dict["interp_point"]) == 0:
        return None, None
    all_dict["interp_point"] = list(np.concatenate(all_dict["interp_point"]))
    all_dict["interpolation"] = list(np.concatenate(all_dict["interpolation"]))
    all_dict["interp_prediction_error"] = list(
        np.concatenate(all_dict["interp_prediction_error"])
    )
    all_dict["interp_prediction_error_z_scored"] = list(
        np.concatenate(all_dict["interp_prediction_error_z_scored"])
    )
    all_dict["cue"] = list(np.concatenate(all_dict["cue"]))
    all_dict["cue_probability"] = list(np.concatenate(all_dict["cue_probability"]))

    # get the correlation between the shift in prediction error and cue probaility
    r2_category_prediction, p_category_prediction = pearsonr(
        all_dict["cue_probability"], all_dict["interp_prediction_error_z_scored"]
    )
    cue_shift_dict["r2_cue_prob_interp_shift_rel_avg"] = r2_category_prediction
    cue_shift_dict["p_cue_prob_interp_shift_rel_avg"] = p_category_prediction
    return pd.Series(cue_shift_dict), pd.DataFrame(all_dict)


def interp_point_lm_prediction(
    interp_trials,
    spike_trains,
    stim_interp_point,
):
    uncued_mask = interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]) == False
    reg = RidgeCV().fit(spike_trains[uncued_mask], stim_interp_point[uncued_mask])
    predicted_interp = reg.predict(spike_trains)
    return predicted_interp


def interp_point_nn_prediction(
    interp_trials, spike_trains, stim_interp_point, n_neighbors=25
):
    uncued_mask = interp_trials.cue.isin(["CR1", "CR0", "CL0", "CL1"]) == False
    nn_index = NNDescent(spike_trains[uncued_mask], n_neighbors=n_neighbors + 1)
    uncued_nns = nn_index.query(spike_trains[uncued_mask], k=n_neighbors + 1)[0][:, 1:]
    cued_nns, _ = nn_index.query(spike_trains[uncued_mask == False], k=n_neighbors)
    nn_all = np.zeros((len(spike_trains), n_neighbors)).astype(int)
    nn_all[uncued_mask] = uncued_nns
    nn_all[uncued_mask == False] = cued_nns
    nearest_neighbor_interp_point = stim_interp_point[uncued_mask][
        nn_all.flatten()
    ].reshape(nn_all.shape)
    predicted_interp = np.mean(nearest_neighbor_interp_point, axis=1)
    return predicted_interp


def fit_logistic_to_cues(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=False,
    prediction_model=default_regression,
    prediction_model_kwargs={},
):
    """For each cue, fit a psychometric function to a regression trained on the data"""

    cue_prob_dict = {
        "CR1": 0.125,
        "CR0": 0.25,
        "CL0": 0.75,
        "CL1": 0.875,
        "NC": 0.5,
        "CN": 0.5,
    }
    all_dict = {
        "interp_point": [],
        "interpolation": [],
        "cue": [],
        "cue_probability": [],
        "prediction": [],
    }
    cue_shift_dict = {}

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="correct vs incorrect error",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        spike_trains = spike_trains / np.max(spike_trains)
        true_interp = interp_trials.interp_point.values
        true_class = np.array(true_interp > 63).astype("int")

        # get only cued trials (which can be passive)
        uncued_mask = (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values == False
        )
        cued_trials_mask = np.array(interp_trials.response.isnull() == False) & (
            interp_trials.cue.isin(["CR1", "CR0", "CN", "CL0", "CL1"]).values
        )

        if len(interp_trials) < min_classifier_exemplars:
            continue

        # not enough trials
        if np.sum(cued_trials_mask) < 5:
            continue

        # subset trial types
        passive_trials = (interp_trials.correct.isnull()).values
        correct_trials = (interp_trials.correct == True).values
        incorrect_trials = (interp_trials.correct == False).values

        # train regression on only uncued trials
        reg = prediction_model(**prediction_model_kwargs).fit(
            spike_trains[uncued_mask], true_class[uncued_mask]
        )

        # predict class probability
        if use_prob:
            predicted_class = reg.predict_proba(spike_trains)[:, 1]
        else:
            predicted_class = reg.predict(spike_trains)

        # get only cued trials
        cues = interp_trials[cued_trials_mask].cue

        # get cue probabilities
        cue_prob = [cue_prob_dict[i] for i in cues]

        all_dict["interp_point"].append(true_interp[cued_trials_mask])
        all_dict["interpolation"].append(np.repeat(interp, np.sum(cued_trials_mask)))
        all_dict["cue"].append(cues)
        all_dict["cue_probability"].append(cue_prob)
        all_dict["prediction"].append(predicted_class[cued_trials_mask])
    if len(all_dict["interp_point"]) == 0:
        return None, None
    all_dict["interp_point"] = list(np.concatenate(all_dict["interp_point"]))
    all_dict["interpolation"] = list(np.concatenate(all_dict["interpolation"]))
    all_dict["cue"] = list(np.concatenate(all_dict["cue"]))
    all_dict["cue_probability"] = list(np.concatenate(all_dict["cue_probability"]))
    all_dict["prediction"] = list(np.concatenate(all_dict["prediction"]))
    all_df = pd.DataFrame(all_dict)

    if len(all_dict["cue_probability"]) == 0:
        return None, None

    # fit a model for this cue
    for cue in tqdm(all_df.cue.unique(), leave=False, desc="neurometric cue"):
        cue_mask = all_df.cue == cue
        if np.sum(cue_mask) < 50:
            continue
        (
            (_min, _max, _inflection, _slope),
            results_logistic,
            y_model,
            r_squared,
        ) = fit_FourParameterLogistic(
            all_df[cue_mask].interp_point, all_df[cue_mask].prediction
        )

        cue_shift_dict["neurometric_min_{}".format(cue)] = _min
        cue_shift_dict["neurometric_max_{}".format(cue)] = _max
        cue_shift_dict["neurometric_inflection_{}".format(cue)] = _inflection
        cue_shift_dict["neurometric_slope_{}".format(cue)] = _slope
        cue_shift_dict["neurometric_r2_{}".format(cue)] = r_squared

    return pd.Series(cue_shift_dict), all_df


def compute_spike_triggered_average_strf(
    trial_aligned_spikes,
    interp_df,
    n_freqs,
    strf_pre_spike=1.0,
    strf_post_spike=1.0,
    strf_rate=16,
    min_random_spikes=10,
):
    # subset only trials in my dataset
    trial_aligned_spikes = trial_aligned_spikes[
        trial_aligned_spikes.interp.isnull() == False
    ]
    trial_aligned_spikes = trial_aligned_spikes[
        trial_aligned_spikes.cue.isin(["CN", "NC", "CL0", "CL1", "CR0", "CR1"])
    ]

    pre_pad = int(strf_pre_spike * strf_rate)
    post_pad = int(strf_post_spike * strf_rate)

    mean_freqs = []
    num_spikes = []
    num_random_spikes = []
    mean_active = []
    mean_random = []

    # get the total # of spikes, in order to create baseline rf
    total_spikes = 0
    for idx, row in tqdm(
        trial_aligned_spikes.iterrows(),
        total=len(trial_aligned_spikes),
        leave=False,
    ):
        spike_times = row.spike_times
        if row.cue[0] == "C":
            spike_times = spike_times[spike_times > 1]
            spike_times -= 1
        if len(spike_times) < 1:
            continue

        total_spikes += len(spike_times)

    if int(total_spikes) < 100:
        print("not enough spikes")
        return pd.Series({"sta_receptive_field": None})
    # if int(total_spikes / len(trial_aligned_spikes)) < 1:
    #    print("not enough spikes")
    #    return pd.Series({"sta_receptive_field": None})

    for idx, row in tqdm(
        trial_aligned_spikes.iterrows(),
        total=len(trial_aligned_spikes),
        leave=False,
        desc="strf",
    ):
        if np.isnan(row.interp_point):
            continue
        if type(row.interp) != str:
            continue
        spike_times = row.spike_times

        if row.cue[0] == "C":
            spike_times = spike_times[spike_times > 1]
            spike_times -= 1
        else:
            if row.stim_length > 1.25:
                spike_times = spike_times[spike_times > 1]
                spike_times -= 1

        if row.stim_length > 2.1:
            continue
        # make sure no spikes are occuring after the stimulus
        # TODO: why is this occuring in some stimuli??
        #  possible a bug in alignment for these stimuli
        spike_times = spike_times[spike_times < 1]

        # raise ValueError("Expected stimuli to be 1 or 2 seconds long")
        # get matched randomly distributed response

        matched_random_spikes = np.random.random(
            size=np.max(
                [min_random_spikes, int(total_spikes / len(trial_aligned_spikes))]
            )
        )

        num_random_spikes.append(len(matched_random_spikes))
        num_spikes.append(len(spike_times))

        if interp_df is None:
            interp_df = pd.read_pickle(
                DATA_DIR / "interp_df_unit_statistics_specs_only.pickle"
            )

        # get spectrogram
        spec = interp_df.specs_small[
            np.where(
                (interp_df.interp == row.interp) & (interp_df.pt == row.interp_point)
            )[0][0]
        ]
        mean_freqs.append(np.nanmean(spec, axis=1))

        spec = np.concatenate(
            [np.zeros((n_freqs, pre_pad)), spec, np.zeros((n_freqs, post_pad))], axis=1
        )

        if len(spike_times) < 1:
            mean_active.append(np.zeros((n_freqs, pre_pad + post_pad)))
        else:
            # get spike response
            all_spike_bins = []
            for st in spike_times:
                sub_spec = spec[
                    :,
                    int(st * strf_rate) : int(st * strf_rate) + pre_pad + post_pad,
                ]
                if sub_spec.shape[1] != 32:
                    raise ValueError("Spectrogram is not correct length")
                all_spike_bins.append(sub_spec)

                if np.any(np.isnan(sub_spec)):
                    raise ValueError("NaNs in sub_spec")

            mean_active.append(np.nanmean(all_spike_bins, axis=0))
            if np.any(np.isnan(all_spike_bins)):
                print("nans")
                return pd.Series({"sta_receptive_field": None})
        # get spike response
        all_spike_bins = []
        for st in matched_random_spikes:
            all_spike_bins.append(
                spec[:, int(st * strf_rate) : int(st * strf_rate) + pre_pad + post_pad]
            )
        mean_random.append(np.nanmean(all_spike_bins, axis=0))

        # if np.any(np.isnan(all_spike_bins)):
        #    breakme

    mean_response = np.sum(
        ((np.array(num_spikes) / np.sum(num_spikes)) * np.array(mean_active).T).T,
        axis=0,
    )

    mean_random_response = np.sum(
        (
            (np.array(num_random_spikes) / np.sum(num_random_spikes))
            * np.array(mean_random).T
        ).T,
        axis=0,
    )

    normed_response = mean_response - mean_random_response

    if np.any(np.isnan(normed_response)):
        raise ValueError("NaNs in sub_spec")

    return pd.Series(
        {"sta_receptive_field": normed_response, "num_spikes": total_spikes}
    )


def get_dist(i, j, metric="cosine"):  # cosine
    if metric == "euclidean":
        return euclidean(i, j)
    elif metric == "cosine":
        # handle cosine distance between two spike trains with no spikes
        if (sum(i) == 0) or (sum(j) == 0):
            if (sum(i) == 0) and (sum(j) == 0):
                # if no spiking for both, distance is 1
                return 0
            else:
                # if only one spikes, distance is 0
                return 1
        else:
            return cosine(i, j)


def compute_distance_varying_similarity(
    trial_aligned_spikes, distance_varying_similarity_n_samp=100000
):
    """
    This is a metric for whether interpolation change is related to neural response
    correlation between z-scored instantaneous spike rate and distance in interpolation
    for each interpolation, sample spiketrains (relative to prop of total)
        compute correlation between each pair of spike trains
    """
    all_dists_spike = []
    all_dists_interp = []
    all_dict = {}
    # for each interpolation, grab some spike trains to correlate
    for interp in trial_aligned_spikes.interp.unique():
        trial_aligned_spikes_interp_subset = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ].sort_values(by="interp_point")
        n_to_sample = int(
            (len(trial_aligned_spikes_interp_subset) / len(trial_aligned_spikes))
            * distance_varying_similarity_n_samp
        )

        if n_to_sample < 2:
            continue

        # subset samples
        mask1 = np.random.randint(
            len(trial_aligned_spikes_interp_subset), size=n_to_sample
        )
        mask2 = np.random.randint(
            len(trial_aligned_spikes_interp_subset), size=n_to_sample
        )

        # get distances
        trains1 = np.vstack(
            trial_aligned_spikes_interp_subset.spike_trains.values[mask1]
        )
        trains2 = np.vstack(
            trial_aligned_spikes_interp_subset.spike_trains.values[mask2]
        )

        # log scale
        trains1 = np.log(1 + trains1)
        trains2 = np.log(1 + trains2)

        dists = [get_dist(i, j) for i, j in zip(trains1, trains2)]

        interp_point1 = trial_aligned_spikes_interp_subset.interp_point.values[mask1]
        interp_point2 = trial_aligned_spikes_interp_subset.interp_point.values[mask2]

        dists_interp = np.abs(interp_point1 - interp_point2)

        all_dists_spike.append(dists)
        all_dists_interp.append(dists_interp)

        r, p = pearsonr(dists, dists_interp)

        all_dict["distance_varying_similarity_{}_r".format(interp)] = r
        all_dict["distance_varying_similarity_{}_p".format(interp)] = p

    all_dists_spike = np.concatenate(all_dists_spike)
    all_dists_interp = np.concatenate(all_dists_interp)
    r, p = pearsonr(all_dists_spike, all_dists_interp)
    all_dict["distance_varying_similarity_r"] = r
    all_dict["distance_varying_similarity_p"] = p

    return pd.Series(all_dict)


def compute_logistic_neurometric(
    trial_aligned_spikes,
    min_classifier_exemplars=100,
    use_prob=False,
    prediction_model=default_regression,
    prediction_model_kwargs={},
):
    """For each cue, fit a psychometric function to a regression trained on the data"""

    def z_score(x):
        return (x - np.mean(x)) / np.std(x)

    all_dict = {}

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="logistic_neurometric",
        leave=False,
    ):
        interp_trials = trial_aligned_spikes[trial_aligned_spikes.interp == interp]
        spike_trains = np.vstack(interp_trials.spike_trains)
        if np.sum(spike_trains) == 0:
            continue
        true_interp = interp_trials.interp_point.values
        y = true_class = np.array(true_interp > 63).astype("int")

        X = spike_trains = z_score(spike_trains)

        if len(X) < min_classifier_exemplars:
            continue

        regr = prediction_model()
        regr.fit(X, y)
        regr.score(X, y)
        predicted_y = regr.predict_proba(np.vstack(X))[:, 1]

        # fit model
        (
            (_min, _max, _inflection, _slope),
            results_logistic,
            y_model,
            r_squared,
        ) = fit_FourParameterLogistic(true_interp, predicted_y)

        all_dict["neurometric_regression_min_{}".format(interp)] = _min
        all_dict["neurometric_regression_max_{}".format(interp)] = _max
        all_dict["neurometric_regression_inflection_{}".format(interp)] = _inflection
        all_dict["neurometric_regression_slope_{}".format(interp)] = _slope
        all_dict["neurometric_regression_r2_{}".format(interp)] = r_squared

    return pd.Series(all_dict)


def get_spike_half_width(unit_to_analyze, unit_features):
    spike_half_widths = []
    spike_full_widths = []
    for sort_unit, sort in unit_to_analyze.sort_units:
        unit_row = unit_features[
            (unit_features.unit == sort_unit) & (unit_features.recording_id == sort)
        ].iloc[0]
        best_channel_template = unit_row.template[:, unit_row.best_channel_0]
        peak = np.argmax(np.abs(best_channel_template))
        sign_change = np.where(
            np.sign(best_channel_template[:-1]) != np.sign(best_channel_template[1:])
        )[0]
        sign_changes_after_peak = sign_change[sign_change > peak]
        sign_changes_before_peak = sign_change[sign_change < peak]
        if len(sign_changes_after_peak) > 0:
            spike_half_widths.append(sign_changes_after_peak[0] - peak)
            if len(sign_changes_before_peak) > 0:
                spike_full_widths.append(
                    sign_changes_after_peak[0] - sign_changes_before_peak[-1]
                )
    if len(spike_half_widths) == 0:
        return np.nan, np.nan
    return np.median(spike_half_widths), np.median(spike_full_widths)


def compute_spike_width(
    unit,
    unit_features,
    unit_to_analyze,
):
    spike_half_width, spike_full_width = get_spike_half_width(
        unit_to_analyze, unit_features
    )

    return pd.Series({"half_width": spike_half_width, "full_width": spike_full_width})


def get_n_spikes(trial_aligned_spikes):
    n_spikes = np.zeros(len(trial_aligned_spikes))
    for ii, (idx, row) in enumerate(trial_aligned_spikes.iterrows()):
        if row.cue == "NC":
            n_spikes[ii] = len(row.spike_times)
        if row.cue in ["CN", "CL0", "CL1", "CR0", "CR1"]:
            n_spikes[ii] = np.sum(row.spike_times > 1)
        else:
            n_spikes[ii] = len(row.spike_times)
    return n_spikes


def compute_interpolation_selectivity(trial_aligned_spikes):
    """The extent to which a unit prefers one interpolation over others"""
    all_dict = {}

    n_spikes_all = get_n_spikes(trial_aligned_spikes)

    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="interpolation selectivity",
        leave=False,
    ):
        n_spikes_interp = n_spikes_all[trial_aligned_spikes.interp == interp]

        n_spikes_not_interp = n_spikes_all[trial_aligned_spikes.interp != interp]

        t, p = ttest_ind(n_spikes_interp, n_spikes_not_interp)

        all_dict["interpolation_selectivity_{}_t".format(interp)] = t
        all_dict["interpolation_selectivity_{}_p".format(interp)] = p
    return pd.Series(all_dict)


def compute_interpolation_specific_category_selectivity(trial_aligned_spikes):
    """The extent to which a unit prefers one interpolation over others"""
    all_dict = {}

    for interp in trial_aligned_spikes.interp.unique():
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]
        cat_1_spikes = get_n_spikes(
            interp_trial_aligned_spikes[interp_trial_aligned_spikes.interp_point <= 63]
        )
        cat_2_spikes = get_n_spikes(
            interp_trial_aligned_spikes[interp_trial_aligned_spikes.interp_point > 63]
        )

        t, p = ttest_ind(cat_1_spikes, cat_2_spikes)

        all_dict["interpolation_specific_category_selectivity_{}_t".format(interp)] = t
        all_dict["interpolation_specific_category_selectivity_{}_p".format(interp)] = p
    return pd.Series(all_dict)


def compute_general_category_selectivity(trial_aligned_spikes):
    """The extent to which a unit prefers one interpolation over others"""
    all_dict = {}

    cat_1_spikes = get_n_spikes(
        trial_aligned_spikes[trial_aligned_spikes.interp_point <= 63]
    )
    cat_2_spikes = get_n_spikes(
        trial_aligned_spikes[trial_aligned_spikes.interp_point > 63]
    )

    t, p = ttest_ind(cat_1_spikes, cat_2_spikes)

    all_dict["general_category_selectivity_t"] = t
    all_dict["general_category_selectivity_p"] = p
    return pd.Series(all_dict)


def compute_smoothness_of_response_profile(
    trial_aligned_spikes,
    interpolation_distance_range=5,
    max_neighboring_rows=10,
    max_exemplars=2000,
):

    all_dict = {}
    for interp in tqdm(
        trial_aligned_spikes.interp.unique(),
        desc="smoothness of response profile",
        leave=False,
    ):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]
        if len(interp_trial_aligned_spikes) < 10:
            continue

        if len(interp_trial_aligned_spikes) > max_exemplars:
            mask = np.random.randint(
                len(interp_trial_aligned_spikes), size=max_exemplars
            )
            interp_trial_aligned_spikes = interp_trial_aligned_spikes.iloc[mask]

        avg_dists = np.zeros(len(interp_trial_aligned_spikes))
        interp_points = np.zeros(len(interp_trial_aligned_spikes))
        for ii, (idx, row) in tqdm(
            enumerate(interp_trial_aligned_spikes.iterrows()),
            total=len(interp_trial_aligned_spikes),
            leave=False,
            desc="smooth_response_profile",
        ):
            neighboring_rows = interp_trial_aligned_spikes[
                (
                    np.abs(
                        interp_trial_aligned_spikes.interp_point.values
                        - row.interp_point
                    )
                    <= interpolation_distance_range
                )
                & (
                    np.abs(
                        interp_trial_aligned_spikes.interp_point.values
                        - row.interp_point
                    )
                    > 0
                )
            ]
            if len(neighboring_rows) > max_neighboring_rows:
                mask = np.random.randint(
                    len(neighboring_rows), size=max_neighboring_rows
                )
                neighboring_rows = neighboring_rows.iloc[mask]
            if len(neighboring_rows) < 1:
                continue
            distances = np.sqrt(
                np.sum(
                    (
                        np.log(1 + row.spike_trains)
                        - np.log(1 + np.vstack(neighboring_rows.spike_trains.values))
                    )
                    ** 2,
                    axis=1,
                )
            )
            avg_dist = np.mean(distances)
            # avg_dist = np.mean([cosine(row.spike_trains, row2.spike_trains) for idx, row2 in neighboring_rows.iterrows()])
            avg_dists[ii] = avg_dist
            interp_points[ii] = row.interp_point
        avg_dists = np.array(avg_dists)
        interp_points = np.array(interp_points)

        # get linearity of
        reg = LinearRegression()
        reg.fit(np.expand_dims(interp_points, 1), avg_dists)
        y_pred = reg.predict(np.expand_dims(interp_points, 1))
        score = mean_absolute_percentage_error(avg_dists, y_pred)

        all_dict["smoothness_score_{}".format(interp)] = score

    return pd.Series(all_dict)


def neurometric_from_distance_matrix(
    trial_aligned_spikes,
    identifier="",
    passive=None,
    cue=None,
    interp=None,
    max_trains=10000,
):
    """
    Create a neurometric function derived from a distance matrix between points
    """

    def z_score(x):
        return (x - np.mean(x)) / np.std(x)

    all_dict = {}

    if interp is not None:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]

    if cue is not None:
        trial_aligned_spikes = trial_aligned_spikes[trial_aligned_spikes.cue == cue]

    if passive is not None:
        trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.passive == passive
        ]

    similarity_matrix_list = []
    interp_points_this_unit_list = []
    mean_response_vector_list = []
    interp_points = []
    dists = []
    for interpolation in trial_aligned_spikes.interp.unique():
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interpolation
        ]
        # subset X if it is too large (because distance matrix will blow up)
        if len(interp_trial_aligned_spikes) > max_trains:
            mask = np.random.randint(len(interp_trial_aligned_spikes), size=max_trains)
            interp_trial_aligned_spikes = interp_trial_aligned_spikes.iloc[mask]

        if len(interp_trial_aligned_spikes) < 10:
            continue

        # get all available interpolation points
        interp_points_this_unit = np.unique(interp_trial_aligned_spikes.interp_point)

        # get mean response vector for interpolation points
        response_vectors = np.array(
            [
                np.mean(
                    interp_trial_aligned_spikes[
                        interp_trial_aligned_spikes.interp_point == interp_point
                    ].spike_trains,
                    axis=0,
                )
                for interp_point in interp_points_this_unit
            ]
        )

        if len(response_vectors) < 2:
            continue

        # create similarity matrix
        similarity_matrix = np.corrcoef(response_vectors + 1e-10)
        # similarity_matrix = cosine_similarity(response_vectors)

        # remove any nans in computed r2
        mask = np.isnan(np.sum(np.tril(similarity_matrix), axis=1)) == False
        response_vectors = response_vectors[mask]
        similarity_matrix = similarity_matrix[mask]
        similarity_matrix = similarity_matrix[:, mask]
        interp_points_this_unit = interp_points_this_unit[mask]

        # compute similarity within vs between category

        for ri, interp_point in enumerate(interp_points_this_unit):
            mask = interp_points_this_unit > 63

            # skip if there isn't anything to compare
            if np.sum(mask) < 1:
                continue
            if np.sum(mask == False) < 1:
                continue

            interp_points.append(interp_point)

            dists.append(
                np.mean(similarity_matrix[ri][mask])
                - np.mean(similarity_matrix[ri][mask == False])
            )

        if np.any(np.isnan(dists)):
            raise ValueError("NaNs in dists")

        similarity_matrix_list.append(similarity_matrix)
        interp_points_this_unit_list.append(interp_points_this_unit)
        mean_response_vector_list.append(response_vectors)

    # skip if not enough data
    if (len(interp_points) < 10) or (len(dists) < 10):
        return pd.Series()

    # get the mean response vector and similarity matrix
    interp_points_this_unit = np.unique(np.concatenate(interp_points_this_unit_list))
    msm_list = []
    mrv_list = []
    for sm, iptu, rv in zip(
        similarity_matrix_list, interp_points_this_unit_list, mean_response_vector_list
    ):
        if len(rv.shape) < 2:
            continue

        iptu = iptu.astype(int)
        sm_expanded = np.zeros((128, 128))
        sm_expanded[:] = np.nan
        rv_expanded = np.zeros((128, rv.shape[1]))
        rv_expanded[:] = np.nan
        rv_expanded[iptu, :] = rv
        mrv_list.append(rv_expanded)
        for ipi, ip in enumerate(iptu):
            sm_expanded[ip, iptu] = sm[ipi]
        msm_list.append(sm_expanded)

    mean_similarity_matrix = np.nanmean(msm_list, axis=0)
    mean_response_vector = np.nanmean(mrv_list, axis=0)

    # fit model
    (
        (_min, _max, _inflection, _slope),
        results_logistic,
        y_model,
        r_squared,
    ) = fit_FourParameterLogistic(
        interp_points, dists, _min_bounds=[-1, 1], _max_bounds=[-1, 1]
    )

    all_dict["neurometric_distances_min_{}".format(identifier)] = _min
    all_dict["neurometric_distances_max_{}".format(identifier)] = _max
    all_dict["neurometric_distances_inflection_{}".format(identifier)] = _inflection
    all_dict["neurometric_distances_slope_{}".format(identifier)] = _slope
    all_dict["neurometric_distances_r2_{}".format(identifier)] = r_squared
    all_dict[
        "neurometric_distances_interp_points_this_unit_{}".format(identifier)
    ] = interp_points_this_unit
    all_dict[
        "neurometric_distances_similarity_matrix_{}".format(identifier)
    ] = mean_similarity_matrix
    all_dict[
        "neurometric_distances_response_vector_{}".format(identifier)
    ] = mean_response_vector

    interp_range = np.linspace(0, 127, 1000)
    y_interp = FourParameterLogistic(
        {"_min": _min, "_max": _max, "inflection": _inflection, "slope": _slope},
        interp_range,
    )

    """print(identifier)
    fig, axs = plt.subplots(ncols=3, figsize=(10, 3))
    axs[0].matshow(similarity_matrix)
    axs[1].scatter(interp_points, dists, alpha=0.5, s=5)
    axs[1].plot(interp_range, y_interp, color="red")
    axs[2].matshow(response_vectors, aspect="auto", cmap=plt.cm.Greys)
    plt.show()"""

    return pd.Series(all_dict)


def compute_neurometric_from_distance_matrix(trial_aligned_spikes, max_trains=10000):
    """
    Create a neurometric function derived from a distance matrix between points
    """

    def z_score(x):
        return (x - np.mean(x)) / np.std(x)

    all_neurometric = []
    sm_list = []
    for interp in trial_aligned_spikes.interp.unique():
        neurometric_results = neurometric_from_distance_matrix(
            trial_aligned_spikes,
            passive=None,
            cue=None,
            interp=interp,
            identifier=interp,
            max_trains=max_trains,
        )
        all_neurometric.append(neurometric_results)
        sm_name = "neurometric_distances_similarity_matrix_{}".format(interp)
        if sm_name in neurometric_results:
            sm_list.append(neurometric_results[sm_name])

    if len(sm_list) > 0:
        # create averaged sim matrix
        overall_similarity_matrix = np.nanmean(np.stack(sm_list), axis=0)
        points, dists = get_interp_points_dists_from_similarity_matrix(
            np.arange(128), overall_similarity_matrix
        )
        # fit averaged sm
        (
            (_min, _max, _inflection, _slope),
            results_logistic,
            y_model,
            r_squared,
        ) = fit_FourParameterLogistic(points, dists)
        all_dict = {}
        all_dict["neurometric_distances_min"] = _min
        all_dict["neurometric_distances_max"] = _max
        all_dict["neurometric_distances_inflection"] = _inflection
        all_dict["neurometric_distances_slope"] = _slope
        all_dict["neurometric_distances_r2"] = r_squared
        all_neurometric.append(pd.Series(all_dict))

    """
    # amount of data for each cue will bias this result
    for cue in trial_aligned_spikes.cue.unique():
        neurometric_results = neurometric_from_distance_matrix(
            trial_aligned_spikes,
            passive=None,
            cue=cue,
            interp=None,
            identifier=cue,
            max_trains=max_trains,
        )

        all_neurometric.append(neurometric_results)"""

    for passive in trial_aligned_spikes.passive.unique():
        identifier = "passive" if passive else "active"
        neurometric_results = neurometric_from_distance_matrix(
            trial_aligned_spikes,
            passive=passive,
            cue=None,
            interp=None,
            identifier=identifier,
            max_trains=max_trains,
        )

        all_neurometric.append(neurometric_results)

    return pd.concat(all_neurometric)


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def compute_neurometric_cue_shift_from_distance_matrix(
    trial_aligned_spikes, max_trains=10000, n_response_samples=100
):

    all_dict = {}
    # for each cue condition, get the average correlation coefficient between each spike train, and the average response vectors
    for cue_list, identifier in [
        (["CL0", "CL1"], "CL"),
        (["CR0", "CR1"], "CR"),
        (["CL0", "CL1", "CN", "NC", "CR0", "CR1"], "ALL"),
    ]:

        similarity_matrix_list = []
        avg_response_list = []
        for interpolation in tqdm(
            trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
        ):
            interp_trial_aligned_spikes = trial_aligned_spikes[
                trial_aligned_spikes.interp == interpolation
            ]
            # subset X if it is too large (because distance matrix will blow up)
            if len(interp_trial_aligned_spikes) > max_trains:
                mask = np.random.randint(
                    len(interp_trial_aligned_spikes), size=max_trains
                )
                interp_trial_aligned_spikes = interp_trial_aligned_spikes.iloc[mask]

            if len(interp_trial_aligned_spikes) < 10:
                continue

            # get average response generally
            interp_points_this_unit = np.unique(
                interp_trial_aligned_spikes.interp_point
            )

            # get mean response vector for interpolation points
            avg_response_vectors = np.array(
                [
                    np.mean(
                        interp_trial_aligned_spikes[
                            interp_trial_aligned_spikes.interp_point == interp_point
                        ].spike_trains,
                        axis=0,
                    )
                    for interp_point in interp_points_this_unit
                ]
            )

            #
            cue_interp_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.cue.isin(cue_list)
            ]

            similarity_matrix_n = np.zeros((128, 128))
            similarity_matrix = np.zeros((128, 128))
            # similarity_matrix[:] = np.nan

            avg_response = np.zeros((128, n_response_samples))
            avg_response[:] = np.nan
            interp_points_this_unit = interp_points_this_unit.astype(int)
            for interp_point in interp_points_this_unit:
                if (
                    np.sum(cue_interp_trial_aligned_spikes.interp_point == interp_point)
                    == 0
                ):
                    continue
                interp_point_spike_trains = np.vstack(
                    cue_interp_trial_aligned_spikes[
                        cue_interp_trial_aligned_spikes.interp_point == interp_point
                    ].spike_trains
                )

                similarity = corr2_coeff(
                    avg_response_vectors, interp_point_spike_trains
                )
                if len(interp_point_spike_trains) > 1:
                    similarity = np.nanmean(similarity, axis=1)
                similarity_matrix[
                    interp_points_this_unit, interp_point
                ] += similarity.flatten()
                similarity_matrix_n[interp_points_this_unit, interp_point] += 1

                avg_response[interp_point] = np.mean(interp_point_spike_trains, axis=0)

            similarity_matrix[similarity_matrix_n == 0] = np.nan

            simiarity_matrix_avg = np.nan_to_num(similarity_matrix) + np.nan_to_num(
                similarity_matrix.T
            )
            simiarity_matrix_avg[
                (np.isnan(similarity_matrix) == False)
                & (np.isnan(similarity_matrix.T) == False)
            ] /= 2
            simiarity_matrix_avg[
                np.isnan(similarity_matrix) & np.isnan(similarity_matrix.T)
            ] = np.nan

            # plt.matshow(simiarity_matrix_avg)
            # plt.show()

            similarity_matrix_list.append(simiarity_matrix_avg)
            avg_response_list.append(avg_response)

        # get averaged simlarity matrix over interpolations
        averaged_simiarity_matrix = np.nanmean(similarity_matrix_list, axis=0)

        # just for viz
        averaged_response = np.nanmean(avg_response_list, axis=0)

        """# get distance function
        interp_points = []
        dists = []
        for ri, interp_point in enumerate(np.arange(128)):

            mask = np.arange(128) > 63
            mask_left = (np.arange(128) > 63) & (np.arange(128) != interp_point)
            mask_right = (np.arange(128) <= 63) & (np.arange(128) != interp_point)
            similarity_left = averaged_simiarity_matrix[ri][mask_left]
            similarity_right = averaged_simiarity_matrix[ri][mask_right]
            # ignore nans, which correspond to relationships we don't have access to
            d = np.nanmean(similarity_left) - np.nanmean(similarity_right)
            if np.isnan(d):
                # the entire row is nan
                continue
                # raise ValueError("NaNs in dists")
            dists.append(d)
            interp_points.append(interp_point)"""

        interp_points, dists = get_interp_points_dists_from_similarity_matrix(
            np.arange(128), averaged_simiarity_matrix
        )
        mask = pd.isnull(interp_points) | pd.isnull(dists)
        interp_points = np.array(interp_points)[mask == False]
        dists = np.array(dists)[mask == False]

        if np.any(np.isnan(dists)):
            raise ValueError("NaNs in dists")

        if (len(interp_points) < 10) or (len(dists) < 10):
            print("too few interp_points")
            continue

        # fit model
        (
            (_min, _max, _inflection, _slope),
            results_logistic,
            y_model,
            r_squared,
        ) = fit_FourParameterLogistic(
            interp_points,
            dists,  # _min_bounds=[-1, 1], _max_bounds=[-1, 1]
        )

        all_dict["neurometric_distances_cs_min_{}".format(identifier)] = _min
        all_dict["neurometric_distances_cs_max_{}".format(identifier)] = _max
        all_dict[
            "neurometric_distances_cs_inflection_{}".format(identifier)
        ] = _inflection
        all_dict["neurometric_distances_cs_slope_{}".format(identifier)] = _slope
        all_dict["neurometric_distances_cs_r2_{}".format(identifier)] = r_squared
        all_dict[
            "neurometric_distances_cs_similarity_matrix_{}".format(identifier)
        ] = averaged_simiarity_matrix
        all_dict[
            "neurometric_distances_cs_response_vector_{}".format(identifier)
        ] = averaged_response

        interp_range = np.linspace(0, 127, 1000)
        y_interp = FourParameterLogistic(
            {"_min": _min, "_max": _max, "inflection": _inflection, "slope": _slope},
            interp_range,
        )

        if False:
            print(identifier)
            fig, axs = plt.subplots(ncols=3, figsize=(10, 3))
            axs[0].matshow(averaged_simiarity_matrix)
            axs[1].scatter(interp_points, dists, alpha=0.5, s=5)
            axs[1].plot(interp_range, y_interp, color="red")
            axs[2].matshow(averaged_response, aspect="auto", cmap=plt.cm.Greys)
            plt.show()

    try:
        all_dict["neurometric_distances_cs_mean_r2"] = (
            np.mean(
                [
                    all_dict["neurometric_distances_cs_r2_CL"]
                    + all_dict["neurometric_distances_cs_r2_CR"]
                ]
            )
            / 2
        )
        all_dict["neurometric_distances_cs_cue_shift"] = (
            np.mean(
                [
                    all_dict["neurometric_distances_cs_inflection_CL"]
                    - all_dict["neurometric_distances_cs_inflection_CR"]
                ]
            )
            / 2
        )
    except:
        print("Could not take difference because neurometric was not computed")

    return pd.Series(all_dict)


def compute_unit_temporal_similarity(trial_aligned_spikes):
    # get spike time
    trial_aligned_spikes["trial_time"] = [
        datetime.datetime.strptime("_".join(i.split("_")[2:4])[3:], "%Y-%m-%d_%H-%M-%S")
        + datetime.timedelta(seconds=j / 30000)
        for i, j in zip(
            trial_aligned_spikes.recording_id.values,
            trial_aligned_spikes.frame_begin.values,
        )
    ]

    # sort by trial time
    trial_aligned_spikes = trial_aligned_spikes.sort_values(by="trial_time")

    time_differences = []
    similarities = []
    interps = trial_aligned_spikes.interp.unique()
    for interp in tqdm(interps, total=len(interps)):

        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]

        for interp_point in range(128):
            pt_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.interp_point == interp_point
            ]
            if len(pt_trial_aligned_spikes) < 2:
                continue
            similarity_matrix = np.corrcoef(
                1e-10 + np.vstack(pt_trial_aligned_spikes.spike_trains.values)
            )

            for idx, row in enumerate(similarity_matrix):
                minutes_seperate = np.abs(
                    pt_trial_aligned_spikes.trial_time.values[idx]
                    - pt_trial_aligned_spikes.trial_time.values[idx:]
                ).astype("timedelta64[m]")
                stim_similarity = similarity_matrix[idx][idx:]
                time_differences.append(minutes_seperate)
                similarities.append(stim_similarity)
    time_differences = np.concatenate(time_differences).astype("int")
    similarities = np.concatenate(similarities)

    block_length_minutes = 60
    time_block = []
    similarity_block = []
    beginning_time = 0
    for end_time in np.arange(0, np.max(time_differences), block_length_minutes)[1:]:
        mask = (time_differences > beginning_time) & (time_differences <= end_time)
        similarity_block.append(np.mean(similarities[mask]))
        time_block.append(end_time)
        beginning_time = end_time

    all_dict = {}
    all_dict["similarity_block"] = similarity_block
    return pd.Series(all_dict)


def mean_simlarity_of_similarity_matrix(similarity_matrix):
    return (np.sum(np.tril(similarity_matrix)) - len(similarity_matrix)) / (
        ((len(similarity_matrix) ** 2) / 2) - (len(similarity_matrix) / 2)
    )


def compute_unit_passive_active_similarity(trial_aligned_spikes):
    #
    interps = trial_aligned_spikes.interp.unique()

    mean_similarity_passive_passive_list = []
    mean_similarity_passive_active_list = []
    mean_similarity_active_active_list = []

    for interp in tqdm(interps, total=len(interps)):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]
        for interp_point in interp_trial_aligned_spikes.interp_point.unique():
            stim_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.interp_point == interp_point
            ]
            if len(stim_trial_aligned_spikes) == 0:
                continue

            response_vectors = np.vstack(stim_trial_aligned_spikes.spike_trains.values)

            if (np.sum(stim_trial_aligned_spikes.passive) > 0) and (
                np.sum(stim_trial_aligned_spikes.passive == False) > 0
            ):

                similarity_passive_active = corr2_coeff(
                    response_vectors[stim_trial_aligned_spikes.passive.values == False],
                    response_vectors[stim_trial_aligned_spikes.passive.values],
                )
                mean_similarity_passive_active = np.mean(similarity_passive_active)
                mean_similarity_passive_active_list.append(
                    mean_similarity_passive_active
                )

            if sum(stim_trial_aligned_spikes.passive) > 1:
                similarity_matrix_passive_passive = np.corrcoef(
                    response_vectors[stim_trial_aligned_spikes.passive.values]
                )
                mean_similarity_passive_passive = mean_simlarity_of_similarity_matrix(
                    similarity_matrix_passive_passive
                )
                mean_similarity_passive_passive_list.append(
                    mean_similarity_passive_passive
                )
                mean_similarity_passive_passive_list.append(
                    mean_similarity_passive_passive
                )

            if sum(stim_trial_aligned_spikes.passive == False) > 1:
                similarity_matrix_active_active = np.corrcoef(
                    response_vectors[stim_trial_aligned_spikes.passive.values == False]
                )
                mean_similarity_active_active = mean_simlarity_of_similarity_matrix(
                    similarity_matrix_active_active
                )
                mean_similarity_active_active_list.append(mean_similarity_active_active)

    if len(mean_similarity_passive_passive_list) == 0:
        return pd.Series()
    all_dict = {}
    all_dict["mean_simliarity_passive_passive"] = np.nanmean(
        mean_similarity_passive_passive_list
    )
    all_dict["mean_simliarity_passive_active"] = np.nanmean(
        mean_similarity_passive_active_list
    )
    all_dict["mean_simliarity_active_active"] = np.nanmean(
        mean_similarity_active_active_list
    )
    return pd.Series(all_dict)


def compute_unit_cued_uncued_similarity(trial_aligned_spikes):
    """compare whether similarity is greater within uncued trials, vs between cued and uncued"""
    interps = trial_aligned_spikes.interp.unique()

    mean_similarity_uncued_cued_list = []
    mean_similarity_uncued_list = []

    # subset only active trials so we are not computing differences between active and passive
    trial_aligned_spikes = trial_aligned_spikes[trial_aligned_spikes.passive == False]
    if len(trial_aligned_spikes) < 2:
        return

    for interp in tqdm(interps, total=len(interps)):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]

        if len(interp_trial_aligned_spikes) < 2:
            continue
        for interp_point in interp_trial_aligned_spikes.interp_point.unique():
            stim_trial_aligned_spikes = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.interp_point == interp_point
            ]

            cued_trials = stim_trial_aligned_spikes.cue.isin(
                ["CR1", "CR0", "CL1", "CL0"]
            ).values

            # get average response vector
            response_vectors = np.vstack(stim_trial_aligned_spikes.spike_trains.values)

            if np.sum(cued_trials) > 0:
                if np.sum(cued_trials == False) > 0:
                    similarity_cued_uncued = corr2_coeff(
                        response_vectors[cued_trials == False],
                        response_vectors[cued_trials],
                    )
                    mean_similarity_uncued_cued_list.append(
                        np.nanmean(similarity_cued_uncued)
                    )

            if np.sum(cued_trials == False) > 1:
                similarity_matrix_uncued = np.corrcoef(
                    response_vectors[cued_trials == False]
                )
                mean_similarity_uncued = mean_simlarity_of_similarity_matrix(
                    similarity_matrix_uncued
                )

                mean_similarity_uncued_list.append(mean_similarity_uncued)

    all_dict = {}
    # print(mean_similarity_uncued_cued_list, mean_similarity_uncued_list)
    all_dict["mean_similarity_uncued_cued"] = np.nanmean(
        mean_similarity_uncued_cued_list
    )
    all_dict["mean_similarity_uncued"] = np.nanmean(mean_similarity_uncued_list)

    all_dict["similarity_difference_cued_versus_uncued"] = np.nanmean(
        mean_similarity_uncued_list
    ) - np.nanmean(mean_similarity_uncued_cued_list)

    return pd.Series(all_dict)


def compute_cue_shift_by_interpolation_point(
    trial_aligned_spikes, d=10, cues=["CL0", "CL1", "CR0", "CR1"]
):
    """ """
    interps = trial_aligned_spikes.interp.unique()

    n_point_exemplars_cue = {cue: np.zeros(128) for cue in cues}
    sum_interpolation_shift_cue = {cue: np.zeros(128) for cue in cues}

    cued_avg_similarity_dict_list = []
    for interp in tqdm(interps, total=len(interps), desc="interps", leave=False):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interp
        ]
        # get average response generally
        interp_points_this_unit = np.unique(interp_trial_aligned_spikes.interp_point)

        # skip if this unit has not seen at least the full interpolation
        if len(interp_points_this_unit) < 128:
            continue

        # get mean response vector for interpolation points
        avg_response_vectors = np.array(
            [
                np.mean(
                    interp_trial_aligned_spikes[
                        interp_trial_aligned_spikes.interp_point == interp_point
                    ].spike_trains,
                    axis=0,
                )
                for interp_point in interp_points_this_unit
            ]
        )

        # for each cue, get similarity
        cued_avg_similarity_dict = {}
        for ci, cue in enumerate(cues):
            cued_trials = interp_trial_aligned_spikes[
                interp_trial_aligned_spikes.cue == cue
            ]
            cued_trials = cued_trials[cued_trials.passive == False]

            for interp_point in cued_trials.interp_point.unique():
                interp_point = int(interp_point)
                if interp_point < d:
                    continue
                if interp_point > 128 - d - 1:
                    continue
                cued_spike_trains = np.stack(
                    cued_trials[
                        cued_trials.interp_point == interp_point
                    ].spike_trains.values
                )
                if len(cued_spike_trains) > 0:
                    similarity = corr2_coeff(cued_spike_trains, avg_response_vectors)
                    _left = similarity[:, interp_point - d : interp_point]
                    _right = similarity[:, interp_point + 1 : (interp_point + d + 1)]
                    sum_shift = np.nansum(_left - _right)
                    n_point_exemplars_cue[cue][interp_point] += len(similarity)
                    sum_interpolation_shift_cue[cue][interp_point] += sum_shift

    all_dict = {}
    for cue in cues:
        all_dict["cue_shift_array_{}".format(cue)] = (
            sum_interpolation_shift_cue[cue] / n_point_exemplars_cue[cue]
        )
        all_dict["mean_cue_shift_{}".format(cue)] = np.nanmean(
            sum_interpolation_shift_cue[cue] / n_point_exemplars_cue[cue]
        )
    return pd.Series(all_dict)


def compute_interpolation_neurometric_from_distance_matrix(
    trial_aligned_spikes, max_trains=10000
):
    """
    Create a neurometric function derived from a distance matrix between points
    """

    def z_score(x):
        return (x - np.mean(x)) / np.std(x)

    all_neurometric = []

    for interp in trial_aligned_spikes.interp.unique():
        neurometric_results = neurometric_from_distance_matrix(
            trial_aligned_spikes,
            passive=None,
            cue=None,
            interp=interp,
            identifier=interp,
            max_trains=max_trains,
        )

        all_neurometric.append(neurometric_results)

    return pd.concat(all_neurometric)


def get_interp_points_dists_from_similarity_matrix(
    interp_points_this_unit, similarity_matrix
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # with np.errstate(invalid="ignore"):
        interp_points = []
        dists = []
        for ri, interp_point in enumerate(interp_points_this_unit):
            mask = interp_points_this_unit > 63

            # skip if there isn't anything to compare
            if np.sum(mask) < 1:
                continue
            if np.sum(mask == False) < 1:
                continue

            interp_points.append(interp_point)

            a = np.nanmean(similarity_matrix[ri][mask])
            b = np.nanmean(similarity_matrix[ri][mask == False])
            dist = a / (a + b)

            dists.append(dist)
        return interp_points, dists


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def get_average_response_vector(trial_aligned_spikes):
    # get average response generally
    interp_points_this_unit = np.unique(trial_aligned_spikes.interp_point)

    # get mean response vector for interpolation points
    avg_response_vectors = np.array(
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
    return avg_response_vectors, interp_points_this_unit


def create_avg_similarity_matrix(
    cue_interp_trial_aligned_spikes,
    interp_points_this_unit,
    avg_response_vectors,
    n_response_samples=100,
):

    # create an empty similarity matrix
    similarity_matrix_n = np.zeros((128, 128))
    similarity_matrix = np.zeros((128, 128))
    # create an empty averag response
    avg_response = np.zeros((128, n_response_samples))
    avg_response[:] = np.nan

    # for each interpolation point
    for interp_point in cue_interp_trial_aligned_spikes.interp_point.unique():

        # get trials of this interpolation point
        interp_point_spike_trains = np.vstack(
            cue_interp_trial_aligned_spikes[
                cue_interp_trial_aligned_spikes.interp_point == interp_point
            ].spike_trains
        )
        # get similarity to each of the average response vectors
        similarity = corr2_coeff(avg_response_vectors, interp_point_spike_trains)

        # average over this interpolation point
        if len(interp_point_spike_trains) > 1:
            similarity = np.mean(similarity, axis=1)

        # add to similarity matrix
        similarity_matrix[interp_points_this_unit, interp_point] += similarity.flatten()
        # keep track of how many
        similarity_matrix_n[interp_points_this_unit, interp_point] += 1

        # save avg response
        avg_response[interp_point] = np.mean(interp_point_spike_trains, axis=0)

    # when similarity matrix values don't exist, set them as nans
    similarity_matrix[similarity_matrix_n == 0] = np.nan

    # make matrix symetrical
    simiarity_matrix_avg = np.nan_to_num(similarity_matrix) + np.nan_to_num(
        similarity_matrix.T
    )
    simiarity_matrix_avg[
        (np.isnan(similarity_matrix) == False)
        & (np.isnan(similarity_matrix.T) == False)
    ] /= 2
    simiarity_matrix_avg[
        np.isnan(similarity_matrix) & np.isnan(similarity_matrix.T)
    ] = np.nan
    return simiarity_matrix_avg, avg_response


def compute_similarity_matrix_by_cue_old(trial_aligned_spikes):
    # ensure interp point is an integer
    trial_aligned_spikes["interp_point"] = trial_aligned_spikes[
        "interp_point"
    ].values.astype(int)

    ### Get average response vector for each interpolation
    interp_avg_response_vectors = {}
    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interpolation
        ]
        # skip inf there's too little data
        if len(interp_trial_aligned_spikes) < 10:
            continue

        interp_avg_response_vectors[interpolation] = {}
        avg_response_vectors, interp_points_this_unit = get_average_response_vector(
            interp_trial_aligned_spikes
        )

        interp_avg_response_vectors[interpolation][
            "avg_response_vectors"
        ] = avg_response_vectors
        interp_avg_response_vectors[interpolation][
            "interp_points_this_unit"
        ] = interp_points_this_unit

    all_dict = {}
    for cue_list, identifier in tqdm(
        [
            (["CL0", "CL1"], "CL"),
            (["CR0", "CR1"], "CR"),
            (["CL0", "CL1", "CN", "NC", "CR0", "CR1"], "ALL"),
        ],
        desc="cue",
    ):
        similarity_matrix_list = []
        avg_response_list = []
        for interpolation in tqdm(
            trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
        ):

            # subset cued trials
            cue_interp_trial_aligned_spikes = trial_aligned_spikes[
                (trial_aligned_spikes.cue.isin(cue_list))
                & (trial_aligned_spikes.interp == interpolation)
            ]
            if len(cue_interp_trial_aligned_spikes) < 10:
                continue

            avg_response_vectors = interp_avg_response_vectors[interpolation][
                "avg_response_vectors"
            ]
            interp_points_this_unit = interp_avg_response_vectors[interpolation][
                "interp_points_this_unit"
            ]
            # breakme

            # compute similarity matrix (relative to average response)
            simiarity_matrix_avg, avg_response = create_avg_similarity_matrix(
                cue_interp_trial_aligned_spikes,
                interp_points_this_unit,
                avg_response_vectors,
            )
            similarity_matrix_list.append(simiarity_matrix_avg)
            avg_response_list.append(avg_response)

        # get averaged simlarity matrix over interpolations
        averaged_simiarity_matrix = np.nanmean(similarity_matrix_list, axis=0)

        # just for viz
        averaged_response = np.nanmean(avg_response_list, axis=0)

        all_dict[
            "averaged_simiarity_matrix_{}".format(identifier)
        ] = averaged_simiarity_matrix
        all_dict["averaged_response_{}".format(identifier)] = averaged_response
    return pd.Series(all_dict)


def create_dense_similarity_matrix(mean_response_vectors, interp_points_this_unit):
    similarity_matrix = np.zeros((128, 128))
    similarity_matrix[:] = np.nan
    sm = np.corrcoef(mean_response_vectors)
    for i, ip in enumerate(interp_points_this_unit):
        similarity_matrix[ip, interp_points_this_unit] = sm[i]
    return similarity_matrix


def compute_similarity_matrix_by_cue(trial_aligned_spikes, n_response_samples=100):

    # ensure interp point is an integer
    trial_aligned_spikes["interp_point"] = trial_aligned_spikes[
        "interp_point"
    ].values.astype(int)

    trial_aligned_spikes = trial_aligned_spikes[
        trial_aligned_spikes["interp_point"].values >= 0
    ]
    trial_aligned_spikes = trial_aligned_spikes[
        trial_aligned_spikes["interp_point"].values <= 128
    ]

    ### Get average response vector for each interpolation
    interp_avg_response_vectors = {}
    for interpolation in tqdm(
        trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
    ):
        interp_trial_aligned_spikes = trial_aligned_spikes[
            trial_aligned_spikes.interp == interpolation
        ]
        # skip inf there's too little data
        if len(interp_trial_aligned_spikes) < 10:
            continue

        interp_avg_response_vectors[interpolation] = {}
        avg_response_vectors, interp_points_this_unit = get_average_response_vector(
            interp_trial_aligned_spikes
        )

        similarity_matrix = create_dense_similarity_matrix(
            avg_response_vectors, interp_points_this_unit
        )

        interp_avg_response_vectors[interpolation][
            "avg_response_vectors"
        ] = avg_response_vectors
        interp_avg_response_vectors[interpolation][
            "interp_points_this_unit"
        ] = interp_points_this_unit
        interp_avg_response_vectors[interpolation][
            "similarity_matrix"
        ] = similarity_matrix

    all_dict = {}
    for cue_list, identifier in tqdm(
        [
            (["CL0", "CL1"], "CL"),
            (["CR0", "CR1"], "CR"),
            (["NC", "CN"], "NC"),
            (["CL0", "CL1", "CN", "NC", "CR0", "CR1"], "ALL"),
        ],
        desc="cue",
        leave=False,
    ):
        similarity_matrix_list = []
        avg_response_list = []
        similarity_matrix_n_list = []
        for interpolation in tqdm(
            trial_aligned_spikes.interp.unique(), desc="interpolation", leave=False
        ):

            # subset cued trials
            cue_interp_trial_aligned_spikes = trial_aligned_spikes[
                (trial_aligned_spikes.cue.isin(cue_list))
                & (trial_aligned_spikes.interp == interpolation)
            ]
            if len(cue_interp_trial_aligned_spikes) < 10:
                continue

            avg_response_vectors = interp_avg_response_vectors[interpolation][
                "avg_response_vectors"
            ]
            interp_points_this_unit = interp_avg_response_vectors[interpolation][
                "interp_points_this_unit"
            ]

            # create an empty similarity matrix
            similarity_matrix_n = np.zeros((128, 128))
            similarity_matrix = np.zeros((128, 128))
            # create an empty averag response
            avg_response = np.zeros((128, n_response_samples))
            avg_response[:] = np.nan

            # for each interpolation point
            for interp_point in cue_interp_trial_aligned_spikes.interp_point.unique():

                # get trials of this interpolation point
                interp_point_spike_trains = np.vstack(
                    cue_interp_trial_aligned_spikes[
                        cue_interp_trial_aligned_spikes.interp_point == interp_point
                    ].spike_trains
                )

                # get similarity to each of the average response vectors
                similarity = corr2_coeff(
                    avg_response_vectors, interp_point_spike_trains
                )

                # average over this interpolation point
                if len(interp_point_spike_trains) > 1:
                    similarity = np.mean(similarity, axis=1)

                # get similarity relative to average response similarity
                avg_interp_similarity = interp_avg_response_vectors[interpolation][
                    "similarity_matrix"
                ][interp_point]
                similarity_relative_to_mean = (
                    similarity.flatten()
                    - avg_interp_similarity[interp_points_this_unit]
                )

                # add to similarity matrix
                similarity_matrix[
                    interp_points_this_unit, interp_point
                ] = similarity_relative_to_mean.flatten()
                # keep track of how many
                similarity_matrix_n[interp_points_this_unit, interp_point] = 1

                # save avg response
                avg_response[interp_point] = np.mean(interp_point_spike_trains, axis=0)

            # when similarity matrix values don't exist, set them as nans
            similarity_matrix[similarity_matrix_n == 0] = np.nan

            # make matrix symetrical
            simiarity_matrix_avg = np.nan_to_num(similarity_matrix) + np.nan_to_num(
                similarity_matrix.T
            )
            simiarity_matrix_avg[
                (np.isnan(similarity_matrix) == False)
                & (np.isnan(similarity_matrix.T) == False)
            ] /= 2
            simiarity_matrix_avg[
                np.isnan(similarity_matrix) & np.isnan(similarity_matrix.T)
            ] = np.nan

            similarity_matrix_list.append(simiarity_matrix_avg)
            avg_response_list.append(avg_response)
            similarity_matrix_n_list.append(similarity_matrix_n)

        averaged_simiarity_matrix = np.nanmean(similarity_matrix_list[:-1], axis=0)

        # just for viz
        averaged_response = np.nanmean(avg_response_list, axis=0)

        all_dict[
            "averaged_simiarity_matrix_{}".format(identifier)
        ] = averaged_simiarity_matrix
        all_dict["averaged_response_{}".format(identifier)] = averaged_response

    return pd.Series(all_dict)
