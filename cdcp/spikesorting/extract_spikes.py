# This file contains functions for grabbing behavior info
import numpy as np
from tqdm.autonotebook import tqdm
from cdcp.paths import DATA_DIR
from joblib import Parallel, delayed
import spikeextractors as se
from spikeextractors.extraction_tools import read_binary
import scipy.signal as ss
import socket
from pathlib2 import Path
import copy

hostname = socket.gethostname()


def prepare_bandpass(sample_rate, bandpass_min, bandpass_max):
    fn = sample_rate / 2
    band = np.array([bandpass_min, bandpass_max]) / fn
    _b, _a = ss.butter(3, band, btype="bandpass")
    return _b, _a


def move_spike_to_binary_chunk(
    spike_time_chunk,
    spike_time_idx_chunk,
    spike_dat_memmap,
    binary_file_loc,
    bandpass_filter_pad_frames,
    snippet_len_before,
    snippet_len_after,
    n_channels,
    num_channels_total,
    subset_channel_ids,
    _a,
    _b,
):

    # read the binary file
    binary_file = read_binary(
        binary_file_loc,
        numchan=num_channels_total,
        dtype="int16",
        time_axis=0,
        offset=0,
    )

    spike_chunk = np.zeros(
        [
            len(spike_time_chunk),
            len(subset_channel_ids),
            snippet_len_before + snippet_len_after,
        ],
        dtype="int16",
    )

    for sixi, (spike_time) in enumerate(spike_time_chunk):
        try:
            # if the spike occurs too early, pad with zeros
            if spike_time < (bandpass_filter_pad_frames + snippet_len_before):
                # TODO

                spike_all_chan = np.zeros(
                    (
                        n_channels,
                        (
                            snippet_len_before
                            + snippet_len_after
                            + bandpass_filter_pad_frames * 2
                        ),
                    )
                )
                spike_all_chan[
                    :,
                    int(
                        (snippet_len_before + bandpass_filter_pad_frames) - spike_time
                    ) :,
                ] = np.array(
                    binary_file[
                        :n_channels,
                        : int(
                            spike_time + snippet_len_after + bandpass_filter_pad_frames
                        ),
                    ]
                )

            else:
                spike_all_chan = np.array(
                    binary_file[
                        :n_channels,
                        int(
                            spike_time - snippet_len_before - bandpass_filter_pad_frames
                        ) : int(
                            spike_time + snippet_len_after + bandpass_filter_pad_frames
                        ),
                    ]
                )

            # filter
            bp_filtered = ss.filtfilt(_b, _a, spike_all_chan, axis=1)

            # common average
            cmr_filtered = (bp_filtered - np.median(bp_filtered, axis=0))[
                :, bandpass_filter_pad_frames:-bandpass_filter_pad_frames
            ]

            # subset channels
            spike_subset = cmr_filtered[np.array(subset_channel_ids)].astype("int16")

            # append to chunk
            spike_chunk[sixi] = spike_subset
        except Exception as e:
            print("Chunk failed: {}".format(e))

    # save the subsetted channels to a new binary file
    spike_dat_memmap[
        spike_time_idx_chunk : spike_time_idx_chunk + len(spike_chunk)
    ] = spike_chunk


def write_spikes_to_dat(
    dat_file_loc,
    sample_rate,
    sort_dat_name,
    spike_times_relative_dat_loc,  # sample relative to .dat file of spike
    channel_group,  # the channel group
    subset_channel_ids,  # which channels are in this channel group
    n_neural_channels,  # number of neural_channels
    n_channels_total,
    bandpass_filter_pad_frames=3000,  # padding for computing bandpass filter (3000 in spike interface)
    snippet_len_before=30,  # number of frames before spike to sample
    snippet_len_after=60,  # number of after before spike to sample
    bandpass_min=300,
    bandpass_max=6000,
    chunk_size=1000,
    n_jobs=-1,
    recompute=True,
):
    print(sort_dat_name)
    # prepare bandpass filter components
    _b, _a = prepare_bandpass(sample_rate, bandpass_min, bandpass_max)

    ####### TODO, write this relative to machine

    # get the dat file, relative to the machine being used
    dat_file_loc = Path(dat_file_loc)
    # dat_file_loc = Path(
    #    *[i if i is not "cube" else "sphere" for i in dat_file_loc.parts]
    # )

    # spike times numpy file, relative to the current dat file
    spike_times_relative_dat_loc = Path(spike_times_relative_dat_loc)

    if hostname in ["pakhi", "txori"]:
        if "cube" in dat_file_loc.parts:
            dat_file_loc = Path(
                *[i if i is not "cube" else "sphere" for i in dat_file_loc.parts]
            )

    if "ssrde" in hostname:
        spike_times_relative_dat_loc = Path(
            *["/cube/bigbird"] + list(spike_times_relative_dat_loc.parts[3:])
        )
        dat_file_loc = Path(
            *["/home/AD/tsainbur/tmp_spikesorting"] + list(dat_file_loc.parts[6:])
        )

    print(spike_times_relative_dat_loc)
    spike_times = copy.deepcopy(np.load(spike_times_relative_dat_loc))

    # set a location for where to save the .dat file
    spike_dat_file_loc = spike_times_relative_dat_loc.parent / sort_dat_name
    print(spike_dat_file_loc)
    if spike_dat_file_loc.is_file() and not recompute:
        print("Binary file already exists for this sort")
        return

    # create .dat file
    spike_dat_memmap_shape = (
        len(spike_times),
        len(subset_channel_ids),
        snippet_len_before + snippet_len_after,
    )
    spike_dat_memmap = np.memmap(
        spike_dat_file_loc, dtype="int16", mode="w+", shape=spike_dat_memmap_shape
    )
    print("dat file: {}".format(spike_dat_file_loc))
    # how many chunks do we need to write
    n_chunks = int(np.ceil(len(spike_times) / chunk_size))

    Parallel(n_jobs=n_jobs, verbose=5)(  # , prefer="threads")(
        delayed(move_spike_to_binary_chunk)(
            spike_times[chunk_i * chunk_size : (chunk_i + 1) * chunk_size],
            chunk_i * chunk_size,
            spike_dat_memmap,
            dat_file_loc,
            bandpass_filter_pad_frames,
            snippet_len_before,
            snippet_len_after,
            n_neural_channels,
            n_channels_total,
            subset_channel_ids,
            _a,
            _b,
        )
        for chunk_i in tqdm(range(n_chunks), total=n_chunks)
    )
