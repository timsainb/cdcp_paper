import spikeinterface as si
import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
from joblib import Parallel, delayed
import spikeinterface as si
import numpy as np
from tqdm.autonotebook import tqdm
from pathlib2 import Path


def get_spikes_for_stimulus(unit_spikes, stimulus, trials, fs, padding_s=0.5):
    """Gets trial aligned spikes
    """

    padding_frames = int(padding_s * fs)
    stimulus_trials = trials[trials.stim == stimulus]
    stim_length = np.median(
        (stimulus_trials.frame_end.values - stimulus_trials.frame_begin.values) / fs
    )
    trial_spike_list = []
    for idx, row in stimulus_trials.iterrows():
        trial_spikes = (
            unit_spikes[
                (unit_spikes > (row.frame_begin - padding_frames))
                & (unit_spikes < (row.frame_end + padding_frames))
            ]
            - float(row.frame_begin)
        ) / float(fs)
        # print(trial_spikes)
        trial_spike_list.append(trial_spikes)
    # print(stimulus_trials.trial_id.values)
    return (
        trial_spike_list,
        stimulus_trials.trial_id.values,
        stimulus_trials.frame_begin.values,
        stimulus_trials.response.values,
        stimulus_trials.correct.values,
        stimulus_trials.reward.values,
        stimulus_trials.punish.values,
        stim_length,
        stimulus_trials.passive.values,
    )


def make_spike_unit_df(unit, unit_spikes, stimulus, trials, fs, stim, padding_s=0.5):
    """ Creates a trial-aligned spike dataframe
    """

    (spikes, trial_ids, frame_begin, response, correct, reward, punish, stim_length, passive) = get_spikes_for_stimulus(
        unit_spikes, stimulus, trials, fs, padding_s=padding_s
    )
    n = len(spikes)
    spike_unit_df = pd.DataFrame(
        {
            "stim": np.repeat(stim, n),
            "trial_id": trial_ids,
            "frame_begin": frame_begin,
            "correct": correct,
            "response": response,
            "punish": punish,
            "reward": reward,
            "stim_length": np.repeat(stim_length, n),
            "unit": np.repeat(unit, n),
            "spike_times": spikes,
            "passive": passive,
            "n_spikes": [len(i) for i in spikes],
        }
    )
    return spike_unit_df


def make_unit_trial_aligned_spike_dataframes(
    spikesorting_folder,
    sorter,
    recording_id,
    unit,
    unit_spikes,
    all_events,
    fs,
    save_folder = "trial_aligned_spikes",
    min_spikes_per_unit=1000,
    overwrite_previous=False,
    padding_s=0.5,
):
    trial_aligned_spikes_loc = (
        spikesorting_folder
        / save_folder
        / sorter
        / recording_id
        / "{}.pickle".format(unit)
    )
    print(trial_aligned_spikes_loc)
    if trial_aligned_spikes_loc.exists():
        if not overwrite_previous:
            return
    if len(unit_spikes) > min_spikes_per_unit:
        unit_trial_aligned_spike_df = pd.concat(
            [
                make_spike_unit_df(
                    unit=unit,
                    unit_spikes=unit_spikes,
                    stimulus=stimulus,
                    trials=all_events,
                    fs=fs,
                    stim=Path(stimulus[5:]).stem,
                    padding_s=padding_s,
                )
                for stimulus in tqdm(
                    all_events.stim.unique(), desc="make_spike_unit_df", leave=False
                )
            ]
        )
        # save trial aligned spikes dataframe
        ensure_dir(trial_aligned_spikes_loc)
        # print(
        #    unit_trial_aligned_spike_df[
        #        unit_trial_aligned_spike_df.passive == False
        #    ].trial_id.unique()
        # )
        # print(trial_aligned_spikes_loc)
        unit_trial_aligned_spike_df.to_pickle(trial_aligned_spikes_loc)
