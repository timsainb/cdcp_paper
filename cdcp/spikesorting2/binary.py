from cdcp.paths import ensure_dir
import json
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from tqdm.autonotebook import tqdm
import warnings
import sys, os
import datetime
import pickle

RPIOPERANT_DIGITAL_CHANNELS = {
    1: "Left Peck",
    2: "Center Peck",
    3: "Right Peck",
    4: "Hopper",
    5: "TXD",
    6: "GPIO22",
    7: "GPIO27",
    8: "GPIO8EXT",
}

ADC_CHANNELS = {"sine": 0, "audio": 1, "audio_binary": 2}


def retrive_experiments(experiment_paths, acute=False):
    """Grab all experiments from paths"""
    all_dat_files = []
    all_info_files = []
    all_recording_message_files = []

    data_dirs = np.concatenate(
        [list(experiment_loc.glob("20*/")) for experiment_loc in experiment_paths]
    )
    if acute:
        recording_dirs = np.concatenate(
            [list(dir_.glob("**/experiment*/recording*/")) for dir_ in tqdm(data_dirs)]
        )
    else:
        recording_dirs = np.concatenate(
            [
                list(dir_.glob("Record Node 10*/experiment*/recording*/"))
                for dir_ in tqdm(data_dirs)
            ]
        )
    for experiment_loc in tqdm(recording_dirs, leave=False):
        # e.g. B1432/2021-03-02_14-29-37_2000/Record Node 103/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat
        dat_files = list(
            (experiment_loc / "continuous").glob("Rhythm_FPGA*/continuous.dat")
        )
        # e.g. B1432/2021-03-02_14-29-37_2000/Record Node 103/experiment1/recording1/structure.oebin
        info_files = [experiment_loc / "structure.oebin"]
        # e.g. B1432/2021-03-02_14-29-37_2000/Record Node 103/experiment1/recording1/sync_messages.oebin
        recording_message_files = [experiment_loc / "sync_messages.txt"]

        all_dat_files += dat_files
        all_info_files += info_files
        all_recording_message_files += recording_message_files

    return all_dat_files, all_info_files, all_recording_message_files


def get_recording_ttl_events(dat_file, FPGA_events_folder):
    """load and grab FPGA TTL events from corresponding dat file.
    Open Ephys has a bug in which .npy files are often corrupted.
    """
    FPGA_events = dat_file.parent.parent.parent / "events" / FPGA_events_folder
    try:
        channel_states = np.load(FPGA_events / "channel_states.npy")
    except Exception as e:
        warnings.warn("{} loading: {}".format(e, FPGA_events / "channel_states.npy"))
        channel_states = []
    try:
        channels = np.load(FPGA_events / "channels.npy")
    except Exception as e:
        warnings.warn("{} loading: {}".format(e, FPGA_events / "channels.npy"))
        channels = []
    try:
        full_words = np.load(FPGA_events / "full_words.npy")
    except Exception as e:
        warnings.warn("{} loading: {}".format(e, FPGA_events / "full_words.npy"))
        full_words = []
    try:
        timestamps = np.load(FPGA_events / "timestamps.npy")
    except Exception as e:
        warnings.warn("{} loading: {}".format(e, FPGA_events / "timestamps.npy"))
        timestamps = []

    TTL_events = pd.DataFrame(
        {
            "channels": channels,
            "channel_states": channel_states,
            "timestamps": timestamps,
            "full_words": full_words,
        }
    )

    # merge with channel information
    TTL_events["channel_ID"] = [
        RPIOPERANT_DIGITAL_CHANNELS[i] for i in TTL_events["channels"]
    ]

    return TTL_events


def parse_Network_Events(recording, load_txt=True):
    """goes through Network Events folder and grabs all relevant data"""
    # search through recording for network events
    network_events_dfs = []
    for event_folder in recording["info"]["events"]:
        if "Network_Events" in event_folder["folder_name"]:
            if "TEXT_group" in event_folder["folder_name"]:

                # load text and timestamp events
                zmq_text_events_folder = (
                    recording["dat_file"].parents[2]
                    / "events"
                    # / event_folder["folder_name"]
                )
                channels_loc = [
                    i
                    for i in zmq_text_events_folder.glob(
                        "Network_Events*/TEXT_group_*/channels.npy"
                    )
                ][0]
                metadata_loc = [
                    i
                    for i in zmq_text_events_folder.glob(
                        "Network_Events*/TEXT_group_*/metadata.npy"
                    )
                ][0]
                text_loc = [
                    i
                    for i in zmq_text_events_folder.glob(
                        "Network_Events*/TEXT_group_*/text.npy"
                    )
                ][0]
                timestamps_loc = [
                    i
                    for i in zmq_text_events_folder.glob(
                        "Network_Events*/TEXT_group_*/timestamps.npy"
                    )
                ][0]

                channels = np.load(channels_loc)
                metadata = np.load(metadata_loc)
                timestamps = np.load(timestamps_loc)

                network_events_df = pd.DataFrame(
                    {
                        "channels": channels,
                        "metadata": metadata,
                        "timestamps": timestamps,
                    }
                )
                if load_txt:
                    text = np.load(text_loc)
                    network_events_df["text"] = text.astype("str")
                network_events_dfs.append(network_events_df)
    if len(network_events_dfs) > 0:
        return pd.concat(network_events_dfs)


def find_first_peak(data, thresh_factor=0.3):
    """Find the first peak in a sine wave
    Originally by Zeke Arneodo
    """
    data = data - np.mean(data)
    thresh = np.max(data) * thresh_factor
    a = data[1:-1] - data[2:]
    b = data[1:-1] - data[:-2]
    c = data[1:-1]
    max_pos = np.where((a >= 0) & (b > 0) & (c > thresh))[0] + 1
    return int(max_pos[0])


def detect_audio_events(
    recording,
    sine_channel_num=None,
    audio_binary_channel_num=None,
    signal_pad_s=0.1,
    subsample=1000,
    chunk_size=10000,
    binary_threshold=2000,
    verbose=True,
):
    """Find audio events given a sine channel and an audio onset/offset signal.
    Subsample speeds up thresholding for very large memmaps
    """

    # get recording info
    dat_file = recording["dat_file"]
    sample_rate = recording["info"]["continuous"][0]["sample_rate"]
    num_channels = recording["info"]["continuous"][0]["num_channels"]

    # load recording data into spikeinterface
    recording_si = si.core.BinaryRecordingExtractor(
        dat_file, sampling_frequency=sample_rate, num_chan=num_channels, dtype="int16"
    )
    if verbose:
        print("loading timestamps")
    recording_timestamps = np.load(dat_file.parent / "timestamps.npy", mmap_mode="r")

    # assume sine channel is first ADC channel
    if sine_channel_num is None:
        sine_channel_num = recording["ADC_data_channels"][ADC_CHANNELS["sine"]]
    # assume audio binary channel is third ADC channel
    if audio_binary_channel_num is None:
        audio_binary_channel_num = recording["ADC_data_channels"][
            ADC_CHANNELS["audio_binary"]
        ]

    if verbose:
        print("loading sine and binary channels")
    # get sine channel
    sine_channel = recording_si.get_traces()[sine_channel_num]
    # get audio_on channel
    audio_on_channel = recording_si.get_traces()[audio_binary_channel_num]

    if verbose:
        print("finding audio on/off events")
    # get starts and ends of playback
    if True:

        if verbose:
            print("loading subsampled binary channel into memory")
        # subsample channel for quicker search
        audio_on_channel_subsample = np.array(audio_on_channel[::subsample])
        # get starts and ends of playback
        audio_starts = np.where(
            (audio_on_channel_binary[:-1] == 0) & (audio_on_channel_binary[1:] == 1)
        )[0]
        audio_stops = np.where(
            (audio_on_channel_binary[:-1] == 1) & (audio_on_channel_binary[1:] == 0)
        )[0]

    if False:
        if verbose:
            print("loading subsampled binary channel into memory")
        # make binary
        audio_on_channel_binary = audio_on_channel / np.max(audio_on_channel) > 0.1

        audio_starts = np.zeros(len(recording["network_events"]), dtype=int)
        audio_stops = np.zeros(len(recording["network_events"]), dtype=int)
        audio_start_i = 0
        audio_stop_i = 0
        for i in tqdm(
            range(1, len(audio_on_channel_subsample) - 1),
            desc="finding audio binary on/off",
            leave=False,
            # miniters=10000,
        ):
            prev_samp, samp, next_samp = audio_on_channel_subsample[i - 1 : i + 2]

            if (samp > binary_threshold) & (prev_samp < binary_threshold):
                audio_starts[audio_start_i] = int(i * subsample)
                audio_start_i += 1
            if (samp > binary_threshold) & (next_samp < binary_threshold):
                audio_stops[audio_stop_i] = int(i * subsample)
                audio_stop_i += 1
        audio_stops = audio_stops[:audio_stop_i]
        audio_starts = audio_starts[:audio_start_i]
    if False:
        if verbose:
            print("loading subsampled binary channel into memory")
        # subsample channel for quicker search
        audio_on_channel_subsample = np.array(audio_on_channel[::subsample])

        all_sample_starts = []
        all_sample_ends = []
        n_chunks = np.ceil(len(audio_on_channel_subsample) / chunk_size).astype(int)
        for chunk in tqdm(
            range(n_chunks), leave=False, desc="search binary audio via chunks"
        ):
            # get chunk bounds
            chunk_start = chunk_size * chunk
            chunk_end = chunk_size * (chunk + 1)
            if chunk_end > len(audio_on_channel_subsample):
                chunk_end = len(audio_on_channel_subsample)
            # overlap by one in case edge falls on the sample bounds
            if chunk_start > 0:
                chunk_start -= 1
            chunk = np.array(audio_on_channel_subsample[chunk_start:chunk_end])
            # find starts and ends
            sample_starts = (
                np.where(
                    (chunk[1:] > binary_threshold) & (chunk[:-1] < binary_threshold)
                )[0]
                + chunk_start
            ) * subsample
            sample_ends = (
                np.where(
                    (chunk[1:] < binary_threshold) & (chunk[:-1] > binary_threshold)
                )[0]
                + chunk_start
            ) * subsample
            all_sample_starts.append(sample_starts)
            all_sample_ends.append(sample_ends)
        audio_starts = np.concatenate(all_sample_starts)
        audio_stops = np.concatenate(all_sample_ends)

    # set equal, in case we begin or end in the middle of a playback
    audio_stops = audio_stops[-len(audio_starts) :]
    audio_starts = audio_starts[: len(audio_stops)]

    # sync audio playback with sine channel
    signal_pad_frames = signal_pad_s * sample_rate

    # get where the sine wave begins in that region
    start_timestamps = np.zeros(len(audio_starts))
    end_timestamps = np.zeros(len(audio_stops))
    start_frames = np.zeros(len(audio_starts))
    end_frames = np.zeros(len(audio_stops))
    # get sine beginning and ending
    for playback_i, (start, stop) in tqdm(
        enumerate(zip(audio_starts, audio_stops)),
        total=len(audio_starts),
        leave=False,
        desc="finding sine",
    ):
        window_start = start + signal_pad_frames
        window_end = stop - signal_pad_frames
        window = sine_channel[int(window_start) : int(window_end)]
        fp = find_first_peak(window) + window_start
        lp = len(window) - find_first_peak(window[::-1]) + window_start
        start_frames[playback_i] = int(fp)
        end_frames[playback_i] = int(lp)

        ### NOTE: this is a hack, I'm not sure why timestamps doesn't have a 1-1
        ### coorespondance with the length of the recording...
        if int(fp) > len(recording_timestamps):
            start_timestamps[playback_i] = (
                recording_timestamps[-1] + int(fp) - len(recording_timestamps)
            )
        else:
            start_timestamps[playback_i] = recording_timestamps[int(fp)]
        if int(lp) > len(recording_timestamps):
            end_timestamps[playback_i] = (
                recording_timestamps[-1] + int(lp) - len(recording_timestamps)
            )
        else:
            end_timestamps[playback_i] = recording_timestamps[int(lp)]

    playback_df = pd.DataFrame(
        {"timestamp_begin": start_timestamps, "timestamp_end": end_timestamps}
    )

    return playback_df


def merge_network_audio_events_with_FPGA_sine(recording, stim_search_padding_s=0.4):
    """Given a dataset of network events over ZMQ, and a dataset of
    detected sine audio events, merge the events
    """

    # stim declaration via zmq can precede stim playback by up to this amount
    sample_rate = recording["info"]["continuous"][0]["sample_rate"]
    stim_search_padding_frames = stim_search_padding_s * sample_rate

    stim_mask = ["stim" in i for i in recording["network_events"].text]
    audio_network_events = recording["network_events"][stim_mask]

    if len(audio_network_events) == 0:
        print("No audio network events found for this recording")

    # for each stim playback, if a zmq audio event exists (within
    #     stim_search_padding_s) , merge it
    recording["audio_events"]["stim"] = None
    recording["audio_events"]["metadata"] = None
    recording["audio_events"]["channels"] = None
    recording["audio_events"]["timestamp_stim"] = None
    for idx, row in tqdm(
        recording["audio_events"].iterrows(),
        desc="matching audio",
        total=len(recording["audio_events"]),
        leave=False,
    ):

        starts_before = audio_network_events.timestamps.values > (
            row.timestamp_begin - stim_search_padding_frames
        )
        ends_after = audio_network_events.timestamps.values < (row.timestamp_end)

        if np.any(starts_before & ends_after):
            matching_row = audio_network_events.iloc[
                np.where(starts_before & ends_after)[0][0]
            ]
            recording["audio_events"].loc[idx, "channel"] = matching_row.channels
            recording["audio_events"].loc[idx, "stim"] = matching_row.text
            recording["audio_events"].loc[idx, "metadata"] = matching_row.metadata
            recording["audio_events"].loc[
                idx, "timestamp_stim"
            ] = matching_row.timestamps

    return recording["audio_events"]


def create_recordings_dict(
    dat_files,
    info_files,
    sync_files,
    spikesorting_folder=None,
    rewrite_pickle=False,
    get_audio_events=True,
    verbose=True,
):
    """create a dictionary of all recordings and parameters"""

    recordings = {}
    for ri, (dat_file, info_file, sync_file) in tqdm(
        enumerate(zip(dat_files, info_files, sync_files)),
        desc="creating dictionary for each recording file",
        total=len(dat_files),
        leave=False,
    ):
        if spikesorting_folder is not None:
            # if the dictionary already exists, load it
            pickle_loc = (
                spikesorting_folder
                / "recording_dicts"
                / "{}.pickle".format(
                    "_".join(
                        [
                            dat_file.parents[5].stem,
                            dat_file.parents[3].stem,
                            dat_file.parents[2].stem,
                        ]
                    )
                )
            )
            if pickle_loc.exists() and (rewrite_pickle == False):
                # try:
                # Load data (deserialize)
                with open(pickle_loc, "rb") as handle:
                    recordings[ri] = pickle.load(handle)
                continue
                # except:
                #    print("loading {} failed".format(pickle_loc))

        try:

            recordings[ri] = {
                "dat_file": dat_file,
                "info_file": info_file,
                "sync_file": sync_file,
            }

            # add info from .oebin
            with open(info_file.as_posix(), "r") as f:
                info = f.read()
            recordings[ri]["info"] = json.loads(info)

            # add sync messages
            with open(sync_file.as_posix(), "r") as f:
                sync_message = f.read()
            recordings[ri]["sync_message"] = sync_message

            # grab TTL events from FPGA events folder
            FPGA_events_folder = recordings[ri]["info"]["events"][0]["folder_name"]
            recordings[ri]["FPGA_events"] = get_recording_ttl_events(
                dat_file, FPGA_events_folder
            )

            # specify neural data channels
            channel_desc = recordings[ri]["info"]["continuous"][0]["channels"]
            recordings[ri]["neural_data_channels"] = [
                i["source_processor_index"]
                for i in channel_desc
                if i["description"] == "Headstage data channel"
            ]
            recordings[ri]["AUX_data_channels"] = [
                i["source_processor_index"]
                for i in channel_desc
                if i["description"] == "Auxiliar data channel"
            ]

            recordings[ri]["ADC_data_channels"] = [
                i["source_processor_index"]
                for i in channel_desc
                if i["description"] == "ADC data channel"
            ]

            # parse network events for each recording
            recordings[ri]["network_events"] = parse_Network_Events(recordings[ri])

            if get_audio_events:
                ## get audio events
                recordings[ri]["audio_events"] = detect_audio_events(recordings[ri])

                ## merge audio events with network events
                recordings[ri][
                    "audio_events"
                ] = merge_network_audio_events_with_FPGA_sine(recordings[ri])
            if spikesorting_folder is not None:
                ensure_dir(pickle_loc)
                with open(pickle_loc, "wb") as handle:
                    pickle.dump(
                        recordings[ri], handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
                print("pickle {} saved".format(pickle_loc))
            # merge network events with TTL events
            # TODO
        except Exception as e:
            print("loading {} failed".format(dat_file))
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            recordings.pop(ri)
            continue

    return recordings


def create_recording_summary_dataframe(recordings):
    recording_summary_df = pd.DataFrame(
        columns=[
            "recording_ID",
            "experiment_num",
            "recording_num",
            "datetime",
            "date_str",
            "n_channels",
            "dat_size_gb",
            "n_ttl_events",
            "n_trials",
            "n_playbacks",
            "n_response",
            "n_punish",
            "n_reward",
            "n_hours",
        ]
    )

    for key in tqdm(recordings.keys()):
        if recordings[key]["network_events"] is None:
            n_trials = n_playbacks = n_response = n_punish = n_reward = np.nan
        else:
            n_trials = np.sum(
                ["trial" in i for i in recordings[key]["network_events"].text]
            )
            n_playbacks = np.sum(
                ["stim" in i for i in recordings[key]["network_events"].text]
            )
            n_response = np.sum(
                ["response" in i for i in recordings[key]["network_events"].text]
            )
            n_punish = np.sum(
                ["punish" in i for i in recordings[key]["network_events"].text]
            )
            n_reward = np.sum(
                ["reward" in i for i in recordings[key]["network_events"].text]
            )

        dat_size = os.path.getsize(recordings[key]["dat_file"]) / 1e9
        time_string = recordings[key]["dat_file"].parents[5].stem
        try:
            time_datetime = datetime.strptime(time_string, "%Y-%m-%d_%H-%M-%S_%f")
        except ValueError:
            try:
                time_string = recordings[key]["dat_file"].parents[4].stem
                time_datetime = datetime.strptime(time_string, "%Y-%m-%d_%H-%M-%S")
            except ValueError as e:
                print("Time does not match format: {}".format(e))
                continue

        n_channels = len(recordings[key]["neural_data_channels"])

        n_ttl_events = len(
            get_recording_ttl_events(
                recordings[key]["dat_file"],
                recordings[key]["info"]["events"][0]["folder_name"],
            )
        )

        recording = recordings[key]
        dat_file = recording["dat_file"]
        sample_rate = recording["info"]["continuous"][0]["sample_rate"]
        num_channels = recording["info"]["continuous"][0]["num_channels"]
        # load recording data into spikeinterface
        recording_si = si.core.BinaryRecordingExtractor(
            dat_file,
            sampling_frequency=sample_rate,
            num_chan=num_channels,
            dtype="int16",
        )
        n_hours = recording_si.get_num_samples(segment_index=0) / sample_rate / 60 / 60

        experiment_num = int(recordings[key]["dat_file"].parents[3].stem[10:])
        recording_num = int(recordings[key]["dat_file"].parents[2].stem[9:])

        recording_summary_df.loc[len(recording_summary_df)] = [
            key,
            experiment_num,
            recording_num,
            time_datetime,
            time_string,
            n_channels,
            dat_size,
            n_ttl_events,
            n_trials,
            n_playbacks,
            n_response,
            n_punish,
            n_reward,
            n_hours,
        ]
    return recording_summary_df


from tqdm.autonotebook import tqdm


def get_recording_IDs(recording_pair, recording_summary_df):
    """Get find row of a recording dictionary based upon the recording experiment
    number, recording number, and datestring
    """
    try:
        recording_IDs = [
            recording_summary_df[
                (
                    recording_summary_df.experiment_num
                    == recording_dict["experiment_num"]
                )
                & (
                    recording_summary_df.recording_num
                    == recording_dict["recording_num"]
                )
                & (recording_summary_df.date_str == recording_dict["date_str"])
            ]
            .iloc[0]
            .recording_ID
            for recording_dict in recording_pair
        ]
    except Exception as e:
        for recording_dict in recording_pair:
            if (
                len(
                    recording_summary_df[
                        (
                            recording_summary_df.experiment_num
                            == recording_dict["experiment_num"]
                        )
                        & (
                            recording_summary_df.recording_num
                            == recording_dict["recording_num"]
                        )
                        & (recording_summary_df.date_str == recording_dict["date_str"])
                    ]
                )
                < 1
            ):
                print(recording_dict)
        raise e

    return recording_IDs


def make_recording_info_dataframe(recordings, recordings_to_sort, recording_summary_df):
    """Create a dataframe containing relevant recording information"""
    # make a dataframe of recording info
    recording_df = pd.DataFrame(
        columns=[
            "dat_file",
            "sample_rate",
            "channels",
            "n_samples",
            "experiment_num",
            "recording_num",
            "date_str",
            "num_channels_total",
            "ADC_data_channels",
            "site",
            "AP",
            "ML",
            "depth",
            "hemisphere",
            "probes",
            "bad_channels",
        ]
    )

    # get recordings in recordings_to_sort list
    recording_IDs = get_recording_IDs(recordings_to_sort, recording_summary_df)
    recordings = [recordings[i] for i in recording_IDs]

    for ri, (recording, recording_info) in tqdm(
        enumerate(zip(recordings, recordings_to_sort)), total=len(recordings)
    ):
        # load data
        dat_file = recording["dat_file"]
        sample_rate = recording["info"]["continuous"][0]["sample_rate"]
        num_channels = recording["info"]["continuous"][0]["num_channels"]

        recording_si = si.core.BinaryRecordingExtractor(
            dat_file,
            sampling_frequency=sample_rate,
            num_chan=num_channels,
            dtype="int16",
        )
        if "bad_channels" in recording_info:
            bad_channels = recording_info["bad_channels"]
        else:
            bad_channels = []
        n_channels = recording_si.get_num_channels()
        n_samples = recording_si.get_num_samples(segment_index=0)
        recording_df.loc[len(recording_df)] = [
            dat_file,
            sample_rate,
            np.unique(recording["neural_data_channels"]),
            n_samples,
            recording_info["experiment_num"],
            recording_info["recording_num"],
            recording_info["date_str"],
            num_channels,
            recording["ADC_data_channels"],
            recording_info["site"],
            recording_info["AP"],
            recording_info["ML"],
            recording_info["depth"],
            recording_info["hemisphere"],
            recording_info["probes"],
            bad_channels,
        ]

    for site in recording_df.site.unique():
        if len(recording_df[recording_df.site == site].sample_rate.unique()) > 1:
            raise ValueError("Sample rates do not match for grouped recordings")
        if (
            len(
                np.unique(
                    [
                        len(i)
                        for i in recording_df[recording_df.site == site].channels.values
                    ]
                )
            )
            > 1
        ):
            raise ValueError(
                "Channels / number of channels do not match for grouped recordings"
            )

    #
    recording_df = pd.merge(
        left=recording_df,
        right=recording_summary_df,
        # axis=1,
        how="inner",
        on=["recording_num", "experiment_num", "date_str"],
    ).sort_values(by="datetime")

    recording_df["recording_id"] = [
        "exp{}_rec{}_dat{}".format(row.experiment_num, row.recording_num, row.date_str)
        for idx, row in tqdm(recording_df.iterrows(), total=len(recording_df))
    ]
    return recording_df


from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_sorted_recordings_vs_all(
    recording_summary_df, recording_df_sort, time_format="%Y-%m-%d_%H-%M-%S_%f"
):

    fig, ax = plt.subplots(figsize=(40, 5))
    # all recordings
    recording_times = [
        datetime.strptime(i, time_format) for i in recording_summary_df.date_str.values
    ]
    for rec_time, rec_len in zip(recording_times, recording_summary_df.n_hours.values):
        ax.fill_between(
            [rec_time, rec_time + timedelta(hours=rec_len)], [1, 1], [2, 2], alpha=0.5
        )

    # recordings to use
    recording_times = [
        datetime.strptime(i, time_format) for i in recording_df_sort.date_str.values
    ]
    for rec_time, rec_len in zip(recording_times, recording_df_sort.n_hours.values):
        ax.fill_between(
            [rec_time, rec_time + timedelta(hours=rec_len)], [0, 0], [1, 1], alpha=0.5
        )

    hours = mdates.HourLocator(interval=12)
    h_fmt = mdates.DateFormatter("%m/%d@%H")
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    ax.tick_params(axis="x", labelrotation=45)
