from scipy.signal.signaltools import resample_poly
import spikeinterface as si
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from cdcp.paths import ensure_dir
from cdcp.spikesorting2.binary import ADC_CHANNELS
from scipy import signal


def find_first_peak(data, thresh_factor=0.3):
    """Find the first peak in a sine wave
    Originally by Zeke Arneodo
    """
    data = data - np.mean(data)
    thresh = np.max(data) * thresh_factor
    a = data[1:-1] - data[2:]
    b = data[1:-1] - data[:-2]
    c = data[1:-1]
    try:
        max_pos = np.where((a >= 0) & (b > 0) & (c > thresh))[0] + 1
    except:
        print("could not find peak")
        return 0
    if len(max_pos) == 0:
        print("could not find peak")
        return 0
    return int(max_pos[0])


def align_binary_timepoints_with_sine(
    start_frame,
    end_frame,
    signal_pad_frames,
    sine_channel,
):
    """
    Given a sine channel (memmap) and an estimated start and end frame, grab
    the true start and end frame based upon the sine wave

    Parameters
    ----------
    start_frame : [type]
        [description]
    end_frame : [type]
        [description]
    signal_pad_frames : [type]
        [description]
    sine_channel : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # find start and end of window from binary edges (accounting for edges due to padding)
    window_start = int(start_frame + signal_pad_frames)
    window_end = int(end_frame - signal_pad_frames)

    # grab window
    window = sine_channel[int(window_start) : int(window_end)]

    # if the window is 0, this isn't a real playback
    if len(window) == 0:
        return
    if np.max(window) == 0:
        return

    # grab peaks
    fp = find_first_peak(window) + window_start
    lp = len(window) - find_first_peak(window[::-1]) + window_start
    return fp, lp


def get_audio_event_timestamps(recording, acute_recording=False):
    # these channels are just different based on the magpi vs other software
    if acute_recording:
        channel_name = "Left Peck"
    else:
        channel_name = "TXD"
    timestamps_end = recording["FPGA_events"][
        (recording["FPGA_events"]["channel_ID"] == channel_name)
        & (np.sign(recording["FPGA_events"]["channel_states"].values) == -1)
    ].timestamps.values

    timestamps_start = recording["FPGA_events"][
        (recording["FPGA_events"]["channel_ID"] == channel_name)
        & (np.sign(recording["FPGA_events"]["channel_states"].values) == 1)
    ].timestamps.values
    if (len(timestamps_start) == 0) or (len(timestamps_end) == 0):
        return [], []
    # ensure that we didn't startor stop recording in the middle of a trial
    if timestamps_start[-1] > timestamps_end[-1]:
        timestamps_start = timestamps_start[:-1]
    if timestamps_start[0] > timestamps_end[0]:
        timestamps_end = timestamps_end[1:]
    return timestamps_start, timestamps_end


def get_audio_events(
    recording,
    recording_row,
    spikesorting_folder,
    signal_pad_s=0.15,
    stim_search_padding_s=0.75,
    ADC_sine_index=1,
    replace_current_pickle=False,
    acute_recording=False,
    verbose=True,
):
    # check if recording has already been aligned
    recording_save_loc = (
        spikesorting_folder
        / "aligned_recordings"
        / "{}.pickle".format(recording_row.recording_id)
    )
    # this is already done outside of the loop...
    # if not replace_current_pickle:
    #    if recording_save_loc.is_file():
    #        print("File already exists: {}".format(recording_save_loc))
    #        return
    if len(recording["ADC_data_channels"]) == 0:
        print("No ADC channels in recording")
        return

    # get recording info
    dat_file = recording["dat_file"]
    sample_rate = recording["info"]["continuous"][0]["sample_rate"]
    num_channels = recording["info"]["continuous"][0]["num_channels"]
    # load recording data into spikeinterface
    recording_si = si.core.BinaryRecordingExtractor(
        dat_file, sampling_frequency=sample_rate, num_chan=num_channels, dtype="int16"
    )
    # grab the sine channel
    sine_channel_num = recording["ADC_data_channels"][ADC_sine_index]
    sine_channel = recording_si.get_traces()[:, sine_channel_num]

    # get timestamps from TXD state
    timestamps_start, timestamps_end = get_audio_event_timestamps(
        recording, acute_recording=acute_recording
    )

    print("# audio events: {}".format(len(timestamps_start)))
    if len(timestamps_start) == 0:
        print("No audio events in this dataset. Skipping")
        # return
    else:
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.set_title("Audio segment lengths")
            ax.hist((timestamps_end - timestamps_start) / sample_rate)
            plt.show()
        pbar = tqdm(total=4)
        pbar.set_description("Loading timestamps")
        # get timestamp array, which corresponds to frames
        timestamps = np.load(dat_file.parent / "timestamps.npy")
        timestamps_to_frames = pd.DataFrame(
            np.arange(len(timestamps)), index=timestamps
        )[0]
        pbar.update(1)
        pbar.set_description("converting timestamps to frames")
        try:
            recording["FPGA_events"]["frames"] = [
                timestamps_to_frames[i]
                for i in recording["FPGA_events"]["timestamps"].values
            ]
            # recording["FPGA_events"]["frames"] = timestamps_to_frames[
            #    recording["FPGA_events"]["timestamps"].values
            # ].values
            # ensure that the # of timestamps matches the recording
            num_frames = recording_si.get_num_frames(segment_index=0)
            if num_frames != len(timestamps):
                print("WARNING: timestamps does not match # frames in recording")
                if np.abs(num_frames - len(timestamps)) > (sample_rate * 5):
                    print(
                        "ERROR: difference {} seconds. Skipping.".format(
                            np.abs(num_frames - len(timestamps)) / sample_rate
                        )
                    )
                    # save recording
                    ensure_dir(recording_save_loc)
                    with open(recording_save_loc, "wb") as output:
                        pickle.dump(recording, output, pickle.HIGHEST_PROTOCOL)
                    return
                    # raise ValueError

            # use TXD state timestamps to get frame #s
            audio_starts = np.array(
                [
                    timestamps_to_frames[i]
                    for i in tqdm(timestamps_start, leave=False, desc="audio starts")
                ]
            )
            audio_stops = np.array(
                [
                    timestamps_to_frames[i]
                    for i in tqdm(timestamps_end, leave=False, desc="audio stops")
                ]
            )
        except Exception as e:
            print(
                "Error aligning timestamps to frames. Switching to manual start/end times \
                from binary signal.",
                e,
            )
            audio_starts, audio_stops, audio_lengths = find_edges_from_binary_data(
                recording,
                recording_si,
                sample_rate=sample_rate,
            )
            if verbose:
                fig, ax = plt.subplots(figsize=(15, 3))
                ax.plot(audio_lengths)
                ax.set_title("audio lengths from binary")
                ax.set_ylim([0, 5])
                plt.show()

        if verbose:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.set_title("Sample sine wave")
            ax.plot(
                recording_si.get_traces(
                    start_frame=audio_starts[0], end_frame=audio_stops[0]
                )[:, sine_channel_num]
            )
            plt.show()
        sine_channel_num, sine_channel_num_changed = ensure_sine_channel(
            sine_channel_num,
            audio_starts,
            audio_stops,
            recording["ADC_data_channels"],
            recording_si,
            sample_rate,
        )
        if sine_channel_num_changed:
            print("sine channel changed")
            sine_channel = recording_si.get_traces()[:, sine_channel_num]
            if verbose:
                print("New sine channel found")
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.set_title("Sample sine wave")
                ax.plot(
                    recording_si.get_traces(
                        start_frame=audio_starts[0], end_frame=audio_stops[0]
                    )[:, sine_channel_num]
                )
                plt.show()

        pbar.update(1)
        pbar.set_description("aligning TXD start and stop with binary")
        # align TDX start and stop times with binary
        signal_pad_frames = int(signal_pad_s * sample_rate)
        aligned_start_stops = [
            align_binary_timepoints_with_sine(
                start_frame, end_frame, signal_pad_frames, sine_channel
            )
            for start_frame, end_frame in zip(
                audio_starts, tqdm(audio_stops, desc="aligning binary with sine")
            )
        ]
        aligned_start_stops = np.vstack(
            [i for i in aligned_start_stops if i is not None]
        )
        n_corrupted_segments = len(audio_starts) - len(aligned_start_stops)
        print("# corrupted segments: {}".format(n_corrupted_segments))

        # add audio event info to the recording
        start_frames = aligned_start_stops[:, 0]
        end_frames = aligned_start_stops[:, 1]
        try:
            start_timestamps = timestamps[start_frames]
            end_timestamps = timestamps[end_frames]
        except:
            len_timestamps = len(timestamps)
            start_timestamps = [
                timestamps[i] if i < len_timestamps else np.nan for i in start_frames
            ]
            end_timestamps = [
                timestamps[i] if i < len_timestamps else np.nan for i in end_frames
            ]

        print(
            "median length",
            np.median(
                np.array(end_timestamps).astype("int")
                - np.array(start_timestamps).astype("int")
            )
            / 30000,
        )

        recording["audio_events"] = pd.DataFrame(
            {
                "timestamp_begin": np.array(start_timestamps).astype("int"),
                "timestamp_end": np.array(end_timestamps).astype("int"),
                "frame_begin": np.array(start_frames).astype("int"),
                "frame_end": np.array(end_frames).astype("int"),
            }
        )

        pbar.update(1)
        pbar.set_description("merging stimuli with names")
        # merge with stimuli names
        stim_search_padding_frames = stim_search_padding_s * sample_rate
        stim_mask = ["stim" in i for i in recording["network_events"].text]
        audio_network_events = recording["network_events"][stim_mask]

        pbar.update(1)
        pbar.set_description("merging audio events from sine with timestamps")
        if len(audio_network_events) > 0:
            #  merge audio events with timestamps for stimuli
            audio_events, failed_merges = merge_audio_events(
                recording["audio_events"],
                audio_network_events,
                end_marker="timestamp_end",
                beginning_marker="timestamp_begin",
                marker_offset=0,
                stim_search_padding_frames=stim_search_padding_frames,
            )
            print("failed to merge {} events".format(failed_merges))

            print(
                "saving {} audio events: {}".format(
                    len(audio_events), datetime.datetime.now()
                )
            )
        else:
            print("No audio network events")
        pbar.close()
    # save recording
    ensure_dir(recording_save_loc)
    with open(recording_save_loc, "wb") as output:
        pickle.dump(recording, output, pickle.HIGHEST_PROTOCOL)


def merge_audio_events(
    audio_events,
    audio_network_events,
    stim_search_padding_frames,
    end_marker="timestamp_end",
    beginning_marker="timestamp_begin",
    marker_offset=0,
):
    """merge audio events defined by sine wav, with labels from zmq"""
    audio_events["stim"] = None
    audio_events["metadata"] = None
    audio_events["channels"] = None
    audio_events["timestamp_stim"] = None
    for idx, row in tqdm(
        audio_events.iterrows(),
        desc="matching audio",
        total=len(audio_events),
        leave=False,
    ):

        starts_before = audio_network_events.timestamps.values > (
            row[beginning_marker] - stim_search_padding_frames + marker_offset
        )
        ends_after = (
            audio_network_events.timestamps.values < row[end_marker] + marker_offset
        )

        if np.any(starts_before & ends_after):
            matching_row = audio_network_events.iloc[
                np.where(starts_before & ends_after)[0][0]
            ]
            audio_events.loc[idx, "channel"] = matching_row.channels
            audio_events.loc[idx, "stim"] = matching_row.text
            audio_events.loc[idx, "metadata"] = matching_row.metadata
            audio_events.loc[idx, "timestamp_stim"] = matching_row.timestamps
    failed_merges = np.sum(audio_events.timestamp_stim.isnull())
    return audio_events, failed_merges


def merge_ephys_and_behavior_events(
    recording_row, spikesorting_folder, behavior_df, re_merge=False
):
    """
    Merges trial info from ephys data, with behavior data

    Parameters
    ----------
    recording_row : [type]
        [description]
    spikesorting_folder : [type]
        [description]
    """
    # save outputted events information
    trial_events_loc = (
        spikesorting_folder
        / "trial_events"
        / "{}_{}.pickle".format(recording_row.recording_id, "trial_events")
    )
    passive_events_loc = (
        spikesorting_folder
        / "passive_events"
        / "{}_{}.pickle".format(recording_row.recording_id, "passive_events")
    )
    network_events_loc = (
        spikesorting_folder
        / "network_events"
        / "{}_{}.pickle".format(recording_row.recording_id, "network_events")
    )
    FPGA_events_loc = (
        spikesorting_folder
        / "FPGA_events"
        / "{}_{}.pickle".format(recording_row.recording_id, "FPGA_events")
    )

    # skip if already completed
    if trial_events_loc.is_file() and not re_merge:
        print("Merged events already exists: {}".format(recording_row.recording_id))
        return

    # load ephys recording events
    recording_pickle_loc = (
        spikesorting_folder
        / "aligned_recordings"
        / "{}.pickle".format(recording_row.recording_id)
    )
    # ensure recording already has been generated
    if not recording_pickle_loc.is_file():
        print(
            "recording pickle does not exist, skipping: {}".format(recording_pickle_loc)
        )
        return
    with open(recording_pickle_loc, "rb") as input_file:
        recording = pickle.load(input_file)

    if "audio_events" not in recording.keys():
        print("No audio events found for: {}".format(recording_row.recording_id))
        return

    if len(recording["audio_events"]) == 0:
        print("No audio events: {}".format(recording_row.recording_id))
        return

    # grab event information
    FPGA_events = recording["FPGA_events"]
    audio_events = recording["audio_events"]
    network_events = recording["network_events"]
    FPGA_events["recording_id"] = recording_row.recording_id
    network_events["recording_id"] = recording_row.recording_id
    audio_events["recording_id"] = recording_row.recording_id

    # first, match the pyoperant behavior to the open ephys trials
    network_events_trial = network_events[
        np.array(["trial" in i for i in network_events.text.values])
    ]
    network_events_trial["time"] = [i[6:] for i in network_events_trial.text.values]
    network_events_trial_merged = pd.merge(
        left=network_events_trial, right=behavior_df, how="inner", on=["time"]
    )

    # get the stim corresponding to each trial
    network_events["stim"] = None
    network_events["timestamp_stim"] = None
    network_events["metadata_stim"] = None
    for idx, recording_row in tqdm(
        network_events.iterrows(), total=len(network_events)
    ):
        if "trial" in recording_row.text:
            if idx >= len(network_events) - 1:
                continue
            if "stim" not in network_events.loc[idx + 1].text:
                print("No stim following trial")
                break
            else:
                network_events.loc[idx, "stim"] = network_events.loc[idx + 1].text
                network_events.loc[idx, "timestamp_stim"] = network_events.loc[
                    idx + 1
                ].timestamps
                network_events.loc[idx, "metadata_stim"] = network_events.loc[
                    idx + 1
                ].metadata
                continue

    # subset trials from playbacks
    trial_mask = np.array(
        [
            "trial" in recording_row.text
            for idx, recording_row in tqdm(
                network_events.iterrows(), total=len(network_events)
            )
        ]
    )
    network_events_trial = network_events[trial_mask]

    # merge audio events all on stim timestamp_stim metadata_stim
    audio_events["metadata_stim"] = audio_events["metadata"]
    audio_events_network_events_merged = pd.merge(
        left=audio_events,
        right=network_events_trial,
        how="left",
        on=["stim", "timestamp_stim", "metadata_stim"],
    )

    # merge audio events all on stim timestamp_stim metadata_stim
    full_events = pd.merge(
        left=audio_events_network_events_merged,
        right=network_events_trial_merged,
        how="left",
        on="text",
    )

    # subset trials and passive playbacks
    trial_events = full_events[
        (full_events.text.isnull() == False) & (full_events.stim.isnull() == False)
    ]
    passive_events = full_events[
        full_events.text.isnull() & (full_events.stim.isnull() == False)
    ]

    print(trial_events_loc)
    print("Saving event info: {}".format(recording_row.recording_id))
    trial_events.to_pickle(trial_events_loc)
    passive_events.to_pickle(passive_events_loc)
    network_events.to_pickle(network_events_loc)
    FPGA_events.to_pickle(FPGA_events_loc)


def find_edges_from_binary_data(
    recording,
    recording_si,
    sample_rate=30000,
    subsample=1000,
    binary_threshold=2000,
    min_sound_clip_len_s=0.5,
    window_length_s=0.45,
):
    """
    Find edges from binary data if timestamp alignment has failed
    """
    # assume audio binary channel is third ADC channel
    audio_binary_channel_num = recording["ADC_data_channels"][
        ADC_CHANNELS["audio_binary"]
    ]

    audio_on_channel = recording_si.get_traces()[:, audio_binary_channel_num]

    print("loading subsampled binary: {}".format(datetime.datetime.now()))

    audio_on_channel_subsample_binary = (
        progress_load_memmap_into_ram(audio_on_channel[::subsample]) > binary_threshold
    )

    # find edges
    audio_starts, audio_stops = find_edges(audio_on_channel_subsample_binary)
    audio_starts *= subsample
    audio_stops *= subsample
    audio_lengths = (audio_stops - audio_starts) / sample_rate
    if np.any(audio_lengths < min_sound_clip_len_s):
        print(
            {
                "Binary edge detection error."
                + " Likely due to error in binary signal generation. "
                + " Smoothing edges to improve signal. "
            }
        )
        window_length_frames = int(window_length_s * sample_rate / subsample)
        print(window_length_frames)
        win = np.ones(window_length_frames)
        convolved_binary = (
            signal.convolve(audio_on_channel_subsample_binary, win, mode="same")
            / sum(win)
            > 0.01
        )

        audio_starts, audio_stops = find_edges(convolved_binary)
        audio_starts += int(len(win) / 2)
        audio_stops -= int(len(win) / 2)
        audio_starts *= subsample
        audio_stops *= subsample
        audio_lengths = (audio_stops - audio_starts) / sample_rate
    return audio_starts, audio_stops, audio_lengths


def progress_load_memmap_into_ram(memmap_array, blocksize=100000):
    y = np.zeros(np.shape(memmap_array))
    n_blocks = int(np.ceil(memmap_array.shape[0] / blocksize))
    for b in tqdm(range(n_blocks)):
        y[b * blocksize : (b + 1) * blocksize] = memmap_array[
            b * blocksize : (b + 1) * blocksize
        ]
    return y


def find_edges(edge_signal):
    audio_starts = np.where((edge_signal[:-1] == 0) & (edge_signal[1:] == 1))[0]
    audio_stops = np.where((edge_signal[:-1] == 1) & (edge_signal[1:] == 0))[0]
    playback_at_start = edge_signal[0]
    playback_at_end = edge_signal[-1]
    if playback_at_start:
        audio_stops = audio_stops[1:]
    if playback_at_end:
        audio_starts = audio_starts[:-1]
    return audio_starts, audio_stops


def ensure_sine_channel(
    sine_channel_num,
    audio_starts,
    audio_stops,
    ADC_data_channels,
    recording_si,
    sample_rate,
    pad=0.35,
    tolerance=0.05,
):
    """Ensures that the ADC channel with the sine wave is correct, otherwise
    looks for a new sine wave channel in the ADC channels.
    """
    # ensure that the first few playbacks aren't corrupted

    for i in tqdm(range(1000)):
        # determine that the audio start/stop is a good one
        for audio_stop, audio_start in zip(audio_stops, audio_starts):
            if audio_stop - audio_start > sample_rate:
                break
        sine_wave = recording_si.get_traces(
            start_frame=audio_start, end_frame=audio_stop
        )[:, sine_channel_num][int(sample_rate * pad) : -int(sample_rate * pad)]

        expected_nzc = int(1000 * len(sine_wave) / sample_rate)
        true_nzc = ((sine_wave[:-1] > 0) & (sine_wave[1:] < 0)).sum()
        if (np.abs(expected_nzc - true_nzc) / expected_nzc) > tolerance:
            print(
                """sine channel number is incorrect. attemping \
                to find sine channel in ADC channels: {} {}""".format(
                    expected_nzc, true_nzc
                )
            )
            for sine_channel_num in ADC_data_channels:
                sine_wave = recording_si.get_traces(
                    start_frame=audio_start, end_frame=audio_stop
                )[:, sine_channel_num][int(sample_rate * pad) : -int(sample_rate * pad)]

                true_nzc = ((sine_wave[:-1] > 0) & (sine_wave[1:] < 0)).sum()

                if (np.abs(expected_nzc - true_nzc) / expected_nzc) <= tolerance:
                    return sine_channel_num, True
        else:
            return sine_channel_num, False

    raise ValueError("Sine channel not found")
