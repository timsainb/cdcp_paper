import matplotlib

import time
import sys

print(matplotlib.get_backend())
import matplotlib.pyplot as plt
from cdcp.paths import ensure_dir
import multiprocessing
import pickle
import shutil
from probeinterface import Probe

# import spiketoolkit as st
# import spikeextractors as se
# import spikesorters as ss
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface as si
from pathlib2 import Path
import numpy as np
import pandas as pd
import socket
from cdcp.paths import DATA_DIR
import psutil
from spikeinterface import append_recordings, concatenate_recordings

# from cdcp.spikesorting.sort import get_spikesorting_directories
from oebinarytools.probe_maps import probe_maps

# from structurednoisereduction.remove_noise_artifacts import suppress_artifacts
# from structurednoisereduction.spectral_gating import spectral_gating
import tempfile
import logging
from datetime import datetime

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
    ss.Kilosort3Sorter.set_kilosort3_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/kilosort3/Kilosort"
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
    ss.Kilosort3Sorter.set_kilosort3_path(
        "/mnt/cube/tsainbur/Projects/github_repos/kilosort3/Kilosort"
    )
    try:
        choose_GPU()
    except:
        print("could not import nvidia-smi")


def get_spikesorting_directories(hostname, bird, timestamp):
    # if on slurm server, change path names to match mount point on server
    if "ssrde" in hostname:
        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        try:
            sphere_n_available_GB = (
                psutil.disk_usage("/sphere/gentnerlab_NB/tmp_spikesorting").free
                / 1000000000
            )
        except PermissionError:
            print("Permission error reaching sphere")
            sphere_n_available_GB = 0

        ssrde_n_available_GB = (
            psutil.disk_usage("/home/AD/tsainbur/tmp_spikesorting/").free / 1000000000
        )
        if ssrde_n_available_GB > 1000:
            tmp_dir = Path("/home/AD/tsainbur/tmp_spikesorting")
        elif sphere_n_available_GB > 1000:
            tmp_dir = Path("/sphere/gentnerlab_NB/tmp_spikesorting")
        else:
            raise ValueError("Not enough disk space anywhere")

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

        scratch_n_available_GB = psutil.disk_usage("/scratch").free / 1000000000
        try:
            sphere_n_available_GB = (
                psutil.disk_usage("/mnt/sphere/tmp_spikesorting").free / 1000000000
            )
        except PermissionError:
            print("Permission error reaching sphere")
            sphere_n_available_GB = 0
        if scratch_n_available_GB > 800:
            tmp_dir = Path("/scratch/tsainbur")
        elif sphere_n_available_GB > 1000:
            tmp_dir = Path("/mnt/sphere/spikesorting_tmp")
        else:
            raise ValueError("Not enough disk space anywhere")

        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        raw_data_folder = (
            list(Path("/mnt/sphere/RawData/").glob("Samamba/ephys/{}/".format(bird)))
            + list(Path("/mnt/sphere/RawData/").glob("nyoni/{}/".format(bird)))
        )[0]

    else:
        raise ValueError

    # make sure the folder exists
    ensure_dir(spikesorting_directory)
    ensure_dir(tmp_dir)

    print("spikesorting_directory: {}".format(spikesorting_directory))
    print("tmp_dir: {}".format(tmp_dir))
    print("raw_data_folder: {}".format(raw_data_folder))

    return spikesorting_directory, tmp_dir, raw_data_folder


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


# create the recording object
def create_merged_recording(
    prev_row, row, raw_data_folder, max_n_minutes_overlap=120, recording_dtype="int16"
):

    # prepare the recording
    recording_list = []

    for ri, row in enumerate([prev_row, row]):
        if row is None:
            n_frames_prepended = 0
            continue
        geom = probe_maps[row.probe]["geom"]

        channel_groups = probe_maps[row.probe]["channel_groups"]

        # pad geometry with additional blank channels for e.g. audio channel
        geom_with_padding = np.vstack(
            [
                geom,
                np.zeros((row.num_channels_total - (len(row.channels)), 2)).astype(
                    "int"
                ),
            ]
        )
        dat_file = Path(*[raw_data_folder] + list(row.dat_file.parts[-7:]))

        recording = si.core.BinaryRecordingExtractor(
            dat_file,
            # geom=geom_with_padding,
            sampling_frequency=row.sample_rate,
            num_chan=row.num_channels_total,
            dtype=recording_dtype,
        )
        recording.set_channel_locations(geom_with_padding)
        if row.recording_subset_frames is not None:
            recording = se.SubRecordingExtractor(
                parent_recording=recording,
                start_frame=row.recording_subset_frames[0],
                end_frame=row.recording_subset_frames[1],
            )
        # get only the neural data channels
        recording = si.core.ChannelSliceRecording(
            parent_recording=recording, channel_ids=row.channels
        )

        # if this is the revious recording, and the recording is longer than
        # . max_n_minutes_overlap, subset this recording to make it shorter
        if ri == 0:
            recording_n_minutes = recording.get_num_frames() / row.sample_rate
            # if this recording is longer than 2 hrs, get the last 2 hrs
            if recording_n_minutes > max_n_minutes_overlap:
                start_frame = (
                    recording.get_num_frames() - row.sample_rate * max_n_minutes_overlap
                )
                end_frame = recording.get_num_frames()
                recording = se.SubRecordingExtractor(
                    parent_recording=recording,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                n_frames_prepended = end_frame - start_frame
            else:
                n_frames_prepended = recording.get_num_frames()
        # set channel groups
        channel_group_channels = np.concatenate(
            [channel_groups[i]["channels"] for i in channel_groups]
        )
        channel_group_groups = np.concatenate(
            [np.repeat(i, len(channel_groups[i]["channels"])) for i in channel_groups]
        )
        recording.set_channel_groups(channel_group_groups, channel_group_channels)

        recording_list.append(recording)

    # merge recordings
    # multirec = se.MultiRecordingTimeExtractor(recording_list)
    multirec = concatenate_recordings(recording_list)

    return multirec, n_frames_prepended


def sort_overlapping(
    bird,
    timestamp,
    merge_recording_id,
    sorter,
    denoised,
    grouping,
    overwrite_sort=False,
    max_n_minutes_overlap=120,
    freq_min=300,
    freq_max=6000,
    max_spikes_per_unit=250,
    suppress_artifacts_args={
        "thresh": 1500,
        "ms_surrounding": 250,
        "fill_mode": "noise",
    },
    spectral_gating_args={
        "time_constant": 0.01,
        "hop_length_ms": 0.5,
        "win_length_ms": 1.0,
        "n_fft": 32,
        "thresh_n_mult": 1,
        "sigmoid_slope": 3,
        "filter_padding": 3000,
        "chunk_size": 30000 * 10,
    },
):
    # time code
    t0 = time.time()
    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # get folders to save results to, as well as to save temporary data to
    spikesorting_directory, tmp_loc, raw_data_folder = get_spikesorting_directories(
        hostname, bird, timestamp
    )

    sort_loc = (
        spikesorting_directory
        / "merged_sorts"
        / merge_recording_id
        / sorter
        / "denoised_{}".format(denoised)
        / "grouping_{}".format(grouping)
    )
    ensure_dir(sort_loc)

    # check if sort already exists
    sort_pickle = sort_loc / "sort.pickle"
    if sort_pickle.exists() and (overwrite_sort is False):
        print("Sort exists. Quitting.")
        return

    logger = create_logger(sort_loc / "{}.log".format(sorter))

    # log basic info
    logger.info("sys prefix: {}".format(sys.prefix))
    logger.info("tmp directory: {}".format(tempfile.gettempdir()))
    logger.info("Started: {}".format(str(datetime.now())))
    logger.info("Host: {}".format(hostname))
    logger.info("Save to: {}".format(sort_loc.as_posix()))

    # load rows to merge and sort info
    rows_to_merge = pd.read_pickle(sort_loc / "rows_to_merge.pickle").iloc[0]
    sort_method = pd.read_pickle(sort_loc / "sort_method.pickle").iloc[0]

    # get sort dataframe
    recording_df = pd.read_pickle(sort_loc.parents[4] / "recording_df.pickle")

    # get the rows
    row = recording_df[
        recording_df["recording_id"].values == rows_to_merge["recording_id"]
    ].iloc[0]
    if rows_to_merge.recording_id_prev is not None:
        prev_row = recording_df[
            recording_df.recording_id.values == rows_to_merge["recording_id_prev"]
        ].iloc[0]
    else:
        prev_row = None

    # make recording to sort
    recording, n_frames_prepended = create_merged_recording(
        prev_row,
        row,
        raw_data_folder=raw_data_folder,
        max_n_minutes_overlap=max_n_minutes_overlap,
    )

    # set the geom/probe
    geom = probe_maps[row.probe]["geom"]
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=geom, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(len(geom)))
    recording.set_probe(probe)

    # save the number of frames preceding the recording of interest
    np.save(sort_loc / "n_frames_prepended.npy", np.array([n_frames_prepended]))

    # preprocess recording
    recording = st.preprocessing.bandpass_filter(
        recording, freq_min=freq_min, freq_max=freq_max
    )
    recording = st.preprocessing.common_reference(recording)

    # remove noise artifacts with amplitude above clip_val
    # if sort_method.artifacts_removed:
    #    recording_artifacts_removed = suppress_artifacts(
    #        recording, **suppress_artifacts_args
    #    )
    # if sort_method.denoised:
    #    gated_recording = spectral_gating(recording, **spectral_gating_args)

    # make a temporary directory to store neural data while spikesorting
    # tmpdir = tempfile.mkdtemp(dir=tmp_loc)
    with tempfile.TemporaryDirectory(dir=tmp_loc) as tmpdir:

        logger.info("Temp directory: {}".format(tmpdir))
        logger.info("Starting spikesort: {}".format(str(datetime.now())))

        logger.info("Grouping property: {}".format(sort_method.grouping))
        logger.info("sorting method: {}".format(sort_method.sorter))

        n_minutes_total = (
            recording.get_num_frames(segment_index=0)
            / recording.get_sampling_frequency()
            / 60
        )

        np.save(sort_loc / "n_minutes_total.npy", np.array([n_minutes_total]))

        logger.info("sorting length (min): {}".format(str(round(n_minutes_total))))

        N_JOBS_MAX = multiprocessing.cpu_count()

        if "n_jobs_bin" in sort_method.kw_args:
            if sort_method.kw_args["n_jobs_bin"] == "MAX":
                sort_method.kw_args["n_jobs_bin"] = N_JOBS_MAX

        try:
            sort = ss.run_sorter(
                sorter_name=sort_method.sorter,
                recording=recording,
                # grouping_property=sort_method.grouping,
                output_folder=Path(tmpdir),
                # parallel=True,
                verbose=True,
                **sort_method.kw_args
            )
            # save sort
            logger.info("Saving sort {}".format(str(datetime.now())))
            with open(sort_pickle, "wb") as output:
                pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
            logger.info("Sorting output saved to {}".format(sort_pickle))
            # get templates and max channel
            logger.info("Starting templates: {}".format(str(datetime.now())))
            templates = st.postprocessing.get_unit_templates(
                recording,
                sort,
                max_spikes_per_unit=max_spikes_per_unit,
                save_as_property=True,
                verbose=True,
                n_jobs=N_JOBS_MAX,
                grouping_property=sort_method.grouping,
            )
            max_chan = st.postprocessing.get_unit_max_channels(
                recording, sort, save_as_property=True, verbose=True, n_jobs=N_JOBS_MAX
            )
            # save sort
            logger.info("Saving sort {}".format(str(datetime.now())))
            with open(sort_pickle, "wb") as output:
                pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
            logger.info(
                "Sorting (with templates) output saved to {}".format(sort_pickle)
            )

        except Exception as e:
            logger.info(e)
            logger.info("failed, cleaning up")

        t1 = time.time()
        total_time = np.array([t1 - t0])
        logger.info("Sorting time: {}".format(total_time))
        np.save(sort_loc / "sort_time.npy", total_time)

    # delete the folder
    try:
        logger.info("Trying to remove directory: {}".format(total_time))
        shutil.rmtree(tmpdir)
    except:
        return

