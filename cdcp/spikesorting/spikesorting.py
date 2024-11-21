# This file contains functions for grabbing behavior info
import shutil
import sys
import numpy as np
import socket
import tempfile
from pathlib2 import Path
import pickle
import pandas as pd
import spikeextractors as se
import spikeinterface.toolkit as st
import spikesorters as ss
import os
import datetime
from tqdm.autonotebook import tqdm
import logging
from cdcp.paths import DATA_DIR
from cdcp.paths import DATA_DIR, ensure_dir


hostname = socket.gethostname()
if "ssrde" in hostname:
    # os.environ["NPY_MATLAB_PATH"] = "/usr/local/bin/matlab20a"
    # os.environ["matlab"] = "/usr/local/bin/matlab20a"
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort2"
    )
    # ss.Kilosort2_5Sorter.set_kilosort2_5_path(
    #    "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort"
    # )
elif hostname in ["pakhi", "txori"]:
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/mnt/cube/tsainbur/Projects/github_repos/Kilosort2"
    )
    # ss.Kilosort2_5Sorter.set_kilosort2_5_path(
    #    "/mnt/cube/tsainbur/Projects/github_repos/Kilosort"
    # )


def cube_on_txori_to_ssrde(path):

    if "ssrde" in hostname:
        path = Path(path)
        if len(list(path.parents)) > 3:
            if list(path.parents)[-3].as_posix() == "/mnt/cube":
                return Path(*(["/cube", "bigbird"] + list(path.parts[3:])))
        return path
    else:
        return path


def create_logger(file_loc, log_name="spike_logger"):
    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(file_loc)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def sort_batch(
    tmp_loc,
    save_directory,
    batch_num,
    sample_start,
    sample_end,
    freq_min=300,
    freq_max=6000,
):

    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # the directory to save output into
    save_directory = Path(save_directory)

    # if on slurm server, change path names to match mount point on server
    if "ssrde" in hostname:
        save_directory = cube_on_txori_to_ssrde(save_directory)
        tmp_loc = cube_on_txori_to_ssrde(tmp_loc)

    # if the sort has already been performed, end function
    sort_pickle = save_directory / "sort_{}.pickle".format(str(int(batch_num)).zfill(4))
    if sort_pickle.is_file():
        return

    # create a logger
    logger = create_logger(save_directory / "sorting_{}.log".format(batch_num))
    logger.info("started {}".format(str(datetime.datetime.now())))
    logger.info(hostname)
    if os.getenv("SLURM_JOB_ID") is not None:
        logger.info("slurm job id: {}".format(os.environ["SLURM_JOB_ID"]))
    logger.info(save_directory.as_posix())

    # load recording info
    recording_df_sort = pd.read_pickle(save_directory / "recording_df.pickle")

    # load the geometry
    geom = np.load(save_directory / "geom.npy")

    # merge all recordings into a super recording
    multirec = merge_recordings(recording_df_sort, geom)

    # grab a recording of the relevant samples
    sort_recording = se.SubRecordingExtractor(
        parent_recording=multirec, start_frame=sample_start, end_frame=sample_end
    )

    # preprocess recording
    recording_f = st.preprocessing.bandpass_filter(
        sort_recording, freq_min=freq_min, freq_max=freq_max
    )
    recording_cmr = st.preprocessing.common_reference(recording_f)

    # make a temporary directory
    tmpdir = tempfile.mkdtemp(dir=tmp_loc)

    logger.info("starting spikesort {}".format(str(datetime.datetime.now())))

    try:
        # perform kilosort sorting
        sorting_KS = ss.run_kilosort2(
            recording_cmr,
            car=True,
            output_folder=Path(tmpdir) / "tmp_KS2",
            parallel=False,
            verbose=True,
        )
        # save recording
        with open(sort_pickle, "wb") as output:
            pickle.dump(sorting_KS, output, pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Sorting output saved to {}".format(save_directory / "KS2_sort.pickle")
        )
        logger.info("starting templates {}".format(str(datetime.datetime.now())))
        # get templates and max channel
        templates = st.postprocessing.get_unit_templates(
            recording_cmr,
            sorting_KS,
            max_spikes_per_unit=200,
            save_as_property=True,
            verbose=True,
        )
        max_chan = st.postprocessing.get_unit_max_channels(
            recording_cmr, sorting_KS, save_as_property=True, verbose=True
        )
        logger.info("saving sort {}".format(str(datetime.datetime.now())))
        # save recording
        with open(sort_pickle, "wb") as output:
            pickle.dump(sorting_KS, output, pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Sorting output saved to {}".format(save_directory / "KS2_sort.pickle")
        )
    except:
        e = sys.exc_info()
        logger.info(
            "Spike sorting failed: {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            )
        )
        logger.info(e)
        print(e)

        return

    # clean up temprary directory
    shutil.rmtree(tmpdir, ignore_errors=True)


def merge_recordings(
    recording_df_sort,
    channel_group_geom,
    neural_channels_only=True,
    recording_dtype="int16",
):
    """From a dataset of recordings, merge into one recording"""
    recording_list = []
    for idx, row in tqdm(recording_df_sort.iterrows()):
        geom_with_padding = np.vstack(
            [
                channel_group_geom,
                np.zeros((row.num_channels_total - (len(row.channels)), 2)).astype(
                    "int"
                ),
            ]
        )

        dat_file = cube_on_txori_to_ssrde(row.dat_file)

        recording = se.BinDatRecordingExtractor(
            dat_file,
            geom=geom_with_padding,
            sampling_frequency=row.sample_rate,
            numchan=row.num_channels_total,
            dtype=recording_dtype,
        )
        if neural_channels_only:
            recording_list.append(
                se.SubRecordingExtractor(
                    parent_recording=recording, channel_ids=row.channels
                )
            )
        else:
            recording_list.append(recording)
    multirec = se.MultiRecordingTimeExtractor(recording_list)
    return multirec


def sort_group(
    save_directory, tmp_loc, run_car=True, preprocess=True, chunksize=300000
):
    starttime = datetime.datetime.now()
    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # the directory to save output into
    save_directory = Path(save_directory)

    # cube_on_txori_to_ssrde
    if "ssrde" in hostname:
        save_directory = cube_on_txori_to_ssrde(save_directory)

        if type(tmp_loc) == str:
            tmp_loc = cube_on_txori_to_ssrde(tmp_loc)

    logger = create_logger(save_directory / "sorting.log")
    logger.info(save_directory.as_posix())
    print(save_directory)
    logger.info(hostname)
    logger.info("start: {}".format(starttime.strftime("%Y-%m-%d_%H-%M-%S_%f")))
    # load the geometry
    geom = np.load(save_directory / "geom.npy")

    # load the save_directory
    recording_df = pd.read_pickle(save_directory / "recording_df.pickle")
    if "ssrde" in hostname:
        bin_loc_list = [cube_on_txori_to_ssrde(i) for i in recording_df.dat_file.values]
    else:
        bin_loc_list = recording_df.dat_file.values

    # save start time
    np.savetxt(
        save_directory / "runtimes.txt",
        np.array([hostname, starttime.strftime("%Y-%m-%d_%H-%M-%S_%f")]),
        delimiter=" ",
        fmt="%s",
    )

    # load the geometry
    channels = np.load(save_directory / "channels.npy")

    # make a folder in /tmp/

    tmpdir = tempfile.mkdtemp(dir=tmp_loc)
    print(tmpdir)
    logger.info("Moving .dat files to tmp dir: {}".format(tmpdir))
    # create a memmap file merging channels
    merged_memmap = merge_bins_into_memmap(
        save_loc=Path(tmpdir) / "merged_recordings.dat",
        total_channels_list=recording_df.num_channels_total.values,
        channels_list=[channels for i in range(len(recording_df))],
        samples_list=recording_df.n_samples.values,
        bin_loc_list=bin_loc_list,
        chunksize=chunksize,
        sample_rate=recording_df.sample_rate.values[0],
        recording_dtype="int16",
        preprocess=preprocess,
    )

    # get channel geometry
    channel_group_geom = geom[channels]

    # load as numpy
    recording_merged = se.NumpyRecordingExtractor(
        timeseries=merged_memmap,
        geom=channel_group_geom,
        sampling_frequency=recording_df.sample_rate.values[0],
    )

    logger.info(
        "Spike sorting: {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        )
    )

    try:
        # perform kilosort sorting
        sorting_KS = ss.run_kilosort2(
            # sorting_KS = ss.run_kilosort2(
            recording_merged,
            car=run_car,
            output_folder=Path(tmpdir) / "tmp_KS2",
            parallel=False,
            verbose=True,
        )
    except:
        e = sys.exc_info()
        logger.info(
            "Spike sorting failed: {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            )
        )
        logger.info(e)
        print(e)
        return
    # get templates and max channel
    templates = st.postprocessing.get_unit_templates(
        recording_merged,
        sorting_KS,
        max_spikes_per_unit=200,
        save_as_property=True,
        verbose=True,
    )
    max_chan = st.postprocessing.get_unit_max_channels(
        recording_merged, sorting_KS, save_as_property=True, verbose=True
    )

    # clean up temprary directory
    shutil.rmtree(tmpdir, ignore_errors=True)

    # save the resulting sorting
    with open(save_directory / "KS2_sort.pickle", "wb") as output:
        pickle.dump(sorting_KS, output, pickle.HIGHEST_PROTOCOL)
    logger.info("Sorting output saved to {}".format(save_directory / "KS2_sort.pickle"))

    endtime = datetime.datetime.now()
    logger.info(endtime)
    # save start time
    np.savetxt(
        save_directory / "runtimes.txt",
        np.array(
            [
                hostname,
                starttime.strftime("%Y-%m-%d_%H-%M-%S_%f"),
                endtime.strftime("%Y-%m-%d_%H-%M-%S_%f"),
            ]
        ),
        delimiter=" ",
        fmt="%s",
    )


def get_recording_IDs(recording_pair, recording_summary_df):
    """Get find row of a recording dictionary based upon the recording experiment
    number, recording number, and datestring
    """
    recording_IDs = [
        recording_summary_df[
            (recording_summary_df.experiment_num == recording_dict["experiment_num"])
            & (recording_summary_df.recording_num == recording_dict["recording_num"])
            & (recording_summary_df.date_str == recording_dict["date_str"])
        ]
        .iloc[0]
        .recording_ID
        for recording_dict in recording_pair
    ]
    return recording_IDs


def make_recording_info_dataframe(recordings, recordings_info):
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
        ]
    )
    for ri, (recording, recording_info) in tqdm(
        enumerate(zip(recordings, recordings_info))
    ):
        # load data
        dat_file = recording["dat_file"]
        sample_rate = recording["info"]["continuous"][0]["sample_rate"]
        num_channels = recording["info"]["continuous"][0]["num_channels"]
        recording_si = se.BinDatRecordingExtractor(
            dat_file,
            sampling_frequency=sample_rate,
            numchan=num_channels,
            dtype="int16",
        )
        n_channels, n_samples = recording_si._timeseries.shape
        recording_df.loc[len(recording_df)] = [
            dat_file,
            sample_rate,
            np.unique(recording["neural_data_channels"]),
            n_samples,
            recording_info["experiment_num"],
            recording_info["recording_num"],
            recording_info["date_str"],
            num_channels,
        ]

    if len(recording_df.sample_rate.unique()) > 1:
        raise ValueError("Sample rates do not match for grouped recordings")
    if len(np.unique([len(i) for i in recording_df.channels.values])) > 1:
        raise ValueError(
            "Channels / number of channels do not match for grouped recordings"
        )

    return recording_df


def merge_bins_into_memmap(
    save_loc,
    total_channels_list,
    channels_list,
    samples_list,
    bin_loc_list,
    chunksize=32768,
    sample_rate=30000,
    recording_dtype="int16",
    preprocess=True,
):
    """Merge multiple memmaps into a a single"""
    merged_memmap = np.memmap(
        save_loc,
        dtype=recording_dtype,
        mode="w+",
        shape=(len(channels_list[0]), int(sum(samples_list))),
        order="C",
    )
    start_sample = 0
    for bin_file, n_channels, data_channels, rec_len in zip(
        bin_loc_list, total_channels_list, channels_list, samples_list
    ):
        se_recording = se.BinDatRecordingExtractor(
            bin_file,
            sampling_frequency=sample_rate,
            numchan=n_channels,
            dtype=recording_dtype,
        )
        if preprocess:
            # preprocessing
            se_recording = st.preprocessing.bandpass_filter(
                se_recording, freq_min=300, freq_max=6000
            )
            se_recording = st.preprocessing.common_reference(se_recording)
        data_channels = np.array(data_channels)
        chunk_start = 0
        iterator = range(np.ceil(rec_len / chunksize).astype("int"))
        if "ssrde" not in hostname:
            iterator = tqdm(iterator, leave=False, desc="copying memmap chunks")
        for chunk in iterator:
            if chunk_start + chunksize > rec_len:
                chunk_end = rec_len
            else:
                chunk_end = chunk_start + chunksize

            if preprocess:
                merged_memmap[
                    :, start_sample + chunk_start : start_sample + chunk_end
                ] = se_recording.get_traces(data_channels, chunk_start, chunk_end)
            else:
                merged_memmap[
                    :, start_sample + chunk_start : start_sample + chunk_end
                ] = se_recording._timeseries[data_channels, chunk_start:chunk_end]

            chunk_start = chunk_end
        start_sample += rec_len
    return merged_memmap
