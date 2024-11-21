import numpy as np
from tqdm.autonotebook import tqdm


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


def sync_binary_with_sine(
    audio_starts,
    audio_stops,
    sine_channel,
    recording_timestamps,
    signal_pad_s=0.15,
    sample_rate=30000,
):
    """The binary on/off is a courase method fir grabbing the
    onset and offset of stimulus. We then use. the sine wave to refine
    that estimate.
    """
    timestamps_len = len(recording_timestamps)
    last_timestamp = recording_timestamps[-1]

    # sync audio playback with sine channel
    signal_pad_frames = signal_pad_s * sample_rate

    # get where the sine wave begins in that region
    start_frames = []
    end_frames = []
    start_timestamps = []
    end_timestamps = []

    corrupted_sine_sections = 0
    # get sine beginning and ending
    for playback_i, (start, stop) in tqdm(
        enumerate(zip(audio_starts, audio_stops)), total=len(audio_starts), leave=False
    ):
        window_start = start + signal_pad_frames
        window_end = stop - signal_pad_frames
        if window_end <= window_start:
            continue
        window = sine_channel[int(window_start) : int(window_end)]
        if len(window) == 0:
            corrupted_sine_sections += 1
            continue
        if np.max(window) == 0:
            corrupted_sine_sections += 1
            continue
        fp = find_first_peak(window) + window_start
        lp = len(window) - find_first_peak(window[::-1]) + window_start
        start_frames.append(int(fp))
        end_frames.append(int(lp))

        ### NOTE: this is a hack, I'm not sure why timestamps doesn't have a 1-1
        ### coorespondance with the length of the recording...
        if int(fp) > timestamps_len:
            start_timestamps.append(last_timestamp + int(fp) - timestamps_len)
        else:
            start_timestamps.append(recording_timestamps[int(fp)])
        if int(lp) > timestamps_len:
            end_timestamps.append(last_timestamp + int(lp) - timestamps_len)
        else:
            end_timestamps.append(recording_timestamps[int(lp)])
    print("corrupted_sine_sections: {}".format(corrupted_sine_sections))
    return start_frames, end_frames, start_timestamps, end_timestamps


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
