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
import spikeinterface.sorters as ss
import os
import datetime
from tqdm.autonotebook import tqdm
import logging
from cdcp.paths import DATA_DIR, ensure_dir
from oebinarytools.binary import ADC_CHANNELS


import multiprocessing

N_JOBS_MAX = multiprocessing.cpu_count()


def choose_GPU():
    import nvidia_smi

    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    free_gpu_memory_per_device = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Total memory:", info.total / 1e9)
        print("Free memory:", info.free / 1e9)
        print("Used memory:", info.used / 1e9)
        free_gpu_memory_per_device.append(info.free)
    GPU_to_use = np.argmax(free_gpu_memory_per_device)
    print("GPU to use: {}".format(GPU_to_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_to_use)


hostname = socket.gethostname()

if "ssrde" in hostname:
    # os.environ["NPY_MATLAB_PATH"] = "/usr/local/bin/matlab20a"
    # os.environ["matlab"] = "/usr/local/bin/matlab20a"
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort2"
    )
    ss.Kilosort2_5Sorter.set_kilosort2_5_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort"
    )
    ss.IronClustSorter.set_ironclust_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/ironclust"
    )

elif hostname in ["pakhi", "txori"]:
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/mnt/cube/tsainbur/Projects/github_repos/Kilosort2"
    )
    ss.Kilosort2_5Sorter.set_kilosort2_5_path(
        "/mnt/cube/tsainbur/Projects/github_repos/Kilosort"
    )
    ss.IronClustSorter.set_ironclust_path(
        "/mnt/cube/tsainbur/Projects/github_repos/ironclust"
    )
    try:
        choose_GPU()
    except:
        print("could not import nvidia-smi")


def sort_batch(
    bird,
    timestamp,
    batch_num,
    sample_start,
    sample_end,
    sorting_method="kilosort2",
    grouping_property=None,  # 'channels'
    freq_min=300,
    freq_max=6000,
):

    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # get folders to save results to, as well as to save temporary data to
    spikesorting_directory, tmp_loc, raw_data_folder = get_spikesorting_directories(
        hostname, bird, timestamp
    )

    # get the name of the output sort
    sort_pickle = (
        spikesorting_directory
        / "sort_{}".format(sorting_method)
        / "sort_{}.pickle".format(str(int(batch_num)).zfill(4))
    )
    # skip if sort already exists
    if sort_pickle.is_file():
        return
    ensure_dir(spikesorting_directory / "sort_{}".format(sorting_method))
    # create a logger
    ensure_dir(spikesorting_directory / "sorting_logs")
    logger = create_logger(
        spikesorting_directory
        / "sorting_logs"
        / "sorting_{}_{}.log".format(sorting_method, str(int(batch_num)).zfill(4))
    )

    # log basic info
    logger.info("Started: {}".format(str(datetime.datetime.now())))
    logger.info("Host: {}".format(hostname))
    logger.info("Save to: {}".format(spikesorting_directory.as_posix()))

    # load recording info
    recording_df_sort = pd.read_pickle(spikesorting_directory / "recording_df.pickle")

    # load the geometry
    geom = np.load(spikesorting_directory / "geom.npy")

    # load channel_groups
    channel_groups = np.load(spikesorting_directory / "channel_groups.npy")
    channel_group_channels = channel_groups[0]
    channel_group_groups = channel_groups[1]

    # merge all recordings into a super recording
    multirec = merge_recordings(recording_df_sort, raw_data_folder, geom)
    multirec.set_channel_groups(channel_group_groups, channel_group_channels)

    # merge all recordings into a super recording

    # grab a recording of the relevant samples
    sort_recording = se.SubRecordingExtractor(
        parent_recording=multirec, start_frame=sample_start, end_frame=sample_end
    )

    # preprocess recording
    recording_f = st.preprocessing.bandpass_filter(
        sort_recording, freq_min=freq_min, freq_max=freq_max
    )
    recording_cmr = st.preprocessing.common_reference(recording_f)

    # make a temporary directory to store neural data while spikesorting
    tmpdir = tempfile.mkdtemp(dir=tmp_loc)

    logger.info("Temp directory: {}".format(tmpdir))
    logger.info("Starting spikesort: {}".format(str(datetime.datetime.now())))

    # run the spike sorter
    run_spikesort(
        recording_cmr,
        sort_pickle,
        logger,
        tmpdir,
        grouping_property=grouping_property,
        sorting_method=sorting_method,
    )

    return


def run_spikesort(
    recording,
    sort_pickle,
    logger,
    tmpdir,
    grouping_property=None,
    sorting_method="kilosort2",
    n_jobs_bin=N_JOBS_MAX,
    chunk_mb=4000,
    **sort_kwargs
):

    logger.info("Grouping property: {}".format(grouping_property))
    logger.info("sorting method: {}".format(sorting_method))

    try:
        if sorting_method == "kilosort2":
            # perform kilosort sorting
            sort = ss.run_kilosort2(
                recording,
                car=True,
                output_folder=Path(tmpdir) / "tmp_KS2",
                parallel=True,
                verbose=True,
                grouping_property=grouping_property,
                chunk_mb=chunk_mb,
                n_jobs_bin=n_jobs_bin,
                **sort_kwargs
            )
        elif sorting_method == "kilosort2_5":
            # perform kilosort sorting
            sort = ss.run_kilosort2_5(
                recording,
                car=True,
                output_folder=Path(tmpdir) / "tmp_KS2",
                parallel=True,
                verbose=True,
                grouping_property=grouping_property,
                chunk_mb=chunk_mb,
                n_jobs_bin=n_jobs_bin,
                preclust_threshold=6,  # [8]
                projection_threshold=[8, 4],  # [10, 4]
                minFR=0.001,  # 0.01
                **sort_kwargs
            )
        elif sorting_method == "ironclust":
            # perform kilosort sorting
            sort = ss.run_ironclust(
                recording,
                output_folder=Path(tmpdir) / "tmp_IC",
                parallel=True,
                verbose=True,
                grouping_property=grouping_property,
                # chunk_mb=chunk_mb,
                # n_jobs_bin=n_jobs_bin,
                **sort_kwargs
            )
        elif sorting_method == "mountainsort4":
            # perform kilosort sorting
            sort = ss.run_mountainsort4(
                recording,
                output_folder=Path(tmpdir) / "tmp_MS4",
                parallel=True,
                verbose=True,
                grouping_property=grouping_property,
                **sort_kwargs
            )
        else:
            logger.info("Sorter not implemented: {}".format(sorting_method))
            raise NotImplementedError(
                "Sorter not implemented: {}".format(sorting_method)
            )

        # save sort
        logger.info("Saving sort {}".format(str(datetime.datetime.now())))
        with open(sort_pickle, "wb") as output:
            pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Sorting output saved to {}".format(sort_pickle))

        # get templates and max channel
        logger.info("Starting templates: {}".format(str(datetime.datetime.now())))
        templates = st.postprocessing.get_unit_templates(
            recording,
            sort,
            max_spikes_per_unit=200,
            save_as_property=True,
            verbose=True,
            n_jobs=n_jobs_bin,
            grouping_property=grouping_property,
        )
        max_chan = st.postprocessing.get_unit_max_channels(
            recording, sort, save_as_property=True, verbose=True, n_jobs=n_jobs_bin
        )

        # save sort
        logger.info("Saving sort {}".format(str(datetime.datetime.now())))
        with open(sort_pickle, "wb") as output:
            pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Sorting output saved to {}".format(sort_pickle))
        """except:
            e = sys.exc_info()
            logger.info(
                "Spike sorting failed: {}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                )
            )
            logger.info(e)
            print(e)

            # read failed ks2 logs and append to log
            logs = list((Path(tmpdir) / "tmp_KS2").glob("**/kilosort2.log"))
            for log in logs:
                logger.info(log)
                with open(log, "r") as f:
                    for line in f:
                        logger.info(line)

            return False"""
    except Exception as e:
        logger.info(e)
        logger.info("failed, cleaning up")
        print("failed, cleaning up")

    shutil.rmtree(tmpdir)

    return True


def cube_on_txori_to_ssrde(path):

    if "ssrde" in hostname:
        path = Path(path)
        if len(list(path.parents)) > 3:
            if list(path.parents)[-3].as_posix() == "/mnt/cube":
                return Path(*(["/cube", "bigbird"] + list(path.parts[3:])))
        return path
    else:
        return path


def get_spikesorting_directories(hostname, bird, timestamp):
    # if on slurm server, change path names to match mount point on server
    if "ssrde" in hostname:
        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp
        tmp_dir = Path("/home/AD/tsainbur/tmp_spikesorting/tmp")

        raw_data_folder = (
            list(
                Path("/sphere/gentnerlab_NB/RawData/").glob(
                    "Samamba/ephys/{}/".format(bird)
                )
            )
            + list(
                Path("/sphere/gentnerlab_NB/RawData/").glob("nyoni/{}/".format(bird))
            )
        )[0]
        # raw_data_folder = Path("/sphere/gentnerlab_NB/RawData/Samamba/ephys") / bird

    elif ("txori" in hostname) or ("pakhi" in hostname):
        # tmp_dir = DATA_DIR / "tmp"
        tmp_dir = Path("/mnt/sphere/spikesorting_tmp")
        # tmp_dir = Path("/home/AD/tsainbur/spikesorting_tmp/tmp")
        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        raw_data_folder = (
            list(Path("/mnt/sphere/RawData/").glob("Samamba/ephys/{}/".format(bird)))
            + list(Path("/mnt/sphere/RawData/").glob("nyoni/{}/".format(bird)))
        )[0]
        # raw_data_folder = Path("/mnt/sphere/RawData/Samamba/ephys") / bird

    else:
        raise ValueError

    # make sure the folder exists
    ensure_dir(spikesorting_directory)
    ensure_dir(tmp_dir)

    return spikesorting_directory, tmp_dir, raw_data_folder


def merge_recordings(
    recording_df_sort,
    raw_data_folder,
    channel_group_geom,
    neural_channels_only=True,
    recording_dtype="int16",
):
    recording_list = []
    for idx, row in tqdm(recording_df_sort.iterrows(), total=len(recording_df_sort)):
        # pad geometry with additional blank channels for e.g. audio channel
        channel_group_geom = channel_group_geom[: len(row.channels)]
        geom_with_padding = np.vstack(
            [
                channel_group_geom,
                np.zeros((row.num_channels_total - (len(row.channels)), 2)).astype(
                    "int"
                ),
            ]
        )
        dat_file = Path(*[raw_data_folder] + list(row.dat_file.parts[-7:]))
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


import logging


def merge_recordings_audio(
    recording_df_sort,
    raw_data_folder,
    channel_group_geom,
    neural_channels_only=True,
    recording_dtype="int16",
):
    recording_list = []
    for idx, row in tqdm(recording_df_sort.iterrows(), total=len(recording_df_sort)):

        dat_file = Path(*[raw_data_folder] + list(row.dat_file.parts[-7:]))

        recording = se.BinDatRecordingExtractor(
            dat_file,
            sampling_frequency=row.sample_rate,
            numchan=row.num_channels_total,
            dtype=recording_dtype,
        )

        recording_list.append(
            se.SubRecordingExtractor(
                parent_recording=recording,
                channel_ids=[
                    row.ADC_data_channels[ADC_CHANNELS["audio"]],
                    row.ADC_data_channels[ADC_CHANNELS["sine"]],
                ],
                renamed_channel_ids=[1, 2],
            )
        )
    multirec = se.MultiRecordingTimeExtractor(recording_list)
    return multirec


def create_logger(file_loc, log_name="spike_logger"):
    # create logger
    logger = logging.getLogger(log_name)
    print(logger)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    # create file handler which logs even debug messages
    fh = logging.FileHandler(file_loc)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


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
            "ADC_data_channels",
        ]
    )
    for ri, (recording, recording_info) in tqdm(
        enumerate(zip(recordings, recordings_info)), total=len(recordings)
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
            recording["ADC_data_channels"],
        ]

    if len(recording_df.sample_rate.unique()) > 1:
        raise ValueError("Sample rates do not match for grouped recordings")
    if len(np.unique([len(i) for i in recording_df.channels.values])) > 1:
        raise ValueError(
            "Channels / number of channels do not match for grouped recordings"
        )

    return recording_df


def sort_recording(
    bird,
    timestamp,
    experiment_num,
    recording_num,
    date_str,
    sorting_method="kilosort2",
    grouping_property=None,  # 'channels'
    freq_min=300,
    freq_max=6000,
    recording_dtype="int16",
    resort_recordings=False,
    recording_subset_frames=None,
):

    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # get folders to save results to, as well as to save temporary data to
    spikesorting_directory, tmp_loc, raw_data_folder = get_spikesorting_directories(
        hostname, bird, timestamp
    )

    # load recording info
    recording_df_sort = pd.read_pickle(spikesorting_directory / "recording_df.pickle")

    # load the geometry
    geom = np.load(spikesorting_directory / "geom.npy")

    # get a recording_id
    recording_ID = "exp{}_rec{}_dat{}".format(experiment_num, recording_num, date_str)

    # load channel_groups
    channel_groups = np.load(spikesorting_directory / "channel_groups.npy")
    channel_group_channels = channel_groups[0]
    channel_group_groups = channel_groups[1]

    # differ spikesorting by method and
    spikesorting_directory = (
        spikesorting_directory / str(grouping_property) / sorting_method
    )
    print(spikesorting_directory)

    # get the name of the output sort
    sort_pickle = spikesorting_directory / "{}.pickle".format(recording_ID)
    print("sort pickle: {}".format(sort_pickle))

    # skip if sort already exists
    if sort_pickle.is_file() and not resort_recordings:
        print("pickle already exists")
        return
    ensure_dir(spikesorting_directory / "sort_{}".format(sorting_method))
    # create a logger
    ensure_dir(spikesorting_directory / "sorting_logs")
    logger = create_logger(
        spikesorting_directory / "sorting_logs" / "{}.log".format(recording_ID)
    )

    # log basic info
    logger.info("Started: {}".format(str(datetime.datetime.now())))
    logger.info("Host: {}".format(hostname))
    logger.info("Save to: {}".format(spikesorting_directory.as_posix()))

    # get the dat file
    recording_row = recording_df_sort[
        (recording_df_sort.experiment_num == experiment_num)
        & (recording_df_sort.recording_num == recording_num)
        & (recording_df_sort.date_str == date_str)
    ].iloc[0]

    # create dummy zeros for non-neural channels
    geom_with_padding = np.vstack(
        [
            geom,
            np.zeros(
                (recording_row.num_channels_total - (len(recording_row.channels)), 2)
            ).astype("int"),
        ]
    )

    dat_file = Path(*[raw_data_folder] + list(recording_row.dat_file.parts[-7:]))
    print(
        "geom_with_padding shape: {} {} {}".format(
            geom_with_padding.shape,
            recording_row.num_channels_total,
            len(recording_row.channels),
        )
    )
    recording = se.BinDatRecordingExtractor(
        dat_file,
        geom=geom_with_padding,
        sampling_frequency=recording_row.sample_rate,
        numchan=recording_row.num_channels_total,
        dtype=recording_dtype,
    )
    # subset neural channels
    recording = se.SubRecordingExtractor(
        parent_recording=recording, channel_ids=recording_row.channels
    )
    recording.set_channel_groups(channel_group_groups, channel_group_channels)

    if recording_subset_frames is not None:
        recording = se.SubRecordingExtractor(
            parent_recording=recording,
            start_frame=recording_subset_frames[0],
            end_frame=recording_subset_frames[1],
        )

    # preprocess recording
    recording_f = st.preprocessing.bandpass_filter(
        recording, freq_min=freq_min, freq_max=freq_max
    )
    recording_cmr = st.preprocessing.common_reference(recording_f)

    # make a temporary directory to store neural data while spikesorting
    tmpdir = tempfile.mkdtemp(dir=tmp_loc)

    logger.info("Temp directory: {}".format(tmpdir))
    logger.info("Starting spikesort: {}".format(str(datetime.datetime.now())))

    # run the spike sorter
    run_spikesort(
        recording_cmr,
        sort_pickle,
        logger,
        tmpdir,
        grouping_property=grouping_property,
        sorting_method=sorting_method,
    )
