import subprocess as sp
import os
import stat
import multiprocessing
import time
import socket
import sys
import tempfile
import spikeinterface as si
from spikeinterface import append_recordings, concatenate_recordings
import spikeinterface.toolkit as st
from cdcp.spikesorting2.suppress_artifacts_si2 import suppress_artifacts
import shutil
from spikeinterface import WaveformExtractor, extract_waveforms
from cdcp.spikesorting2.directories import get_spikesorting_directories
from cdcp.spikesorting2.utils import create_logger
import pandas as pd
from cdcp.paths import ensure_dir
from cdcp.spikesorting2.probe_maps import make_probe_group
import probeinterface as pi
import numpy as np
from datetime import datetime
from pathlib2 import Path
import spikeinterface.sorters as ss
from cdcp.spikesorting2.utils import choose_gpu
from cdcp.paths import ALTERNATIVE_DATA_DIRS

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
        choose_gpu()
    except:
        print("could not choose gpu")


def prep_sort(
    spikesorting_folder,
    sort_method_row,
    recording_row,
    recording_df_sort,
    overwrite_sort=False,
):

    # get folder to save sort info
    sort_loc = (
        spikesorting_folder
        / "sorts"
        / recording_row.recording_id
        / sort_method_row["sorter"]
        / "denoised_{}".format(sort_method_row["denoised"])
        / "grouping_{}".format(str(sort_method_row["grouping"]))
    )
    ensure_dir(sort_loc)

    # check if sort already exists
    sort_exists = (sort_loc / "sort.npz").exists()
    if sort_exists and not overwrite_sort:
        print(
            "sort exists: sorter:{} | denoised: {} | grouping: {}".format(
                sort_method_row["sorter"],
                sort_method_row["denoised"],
                str(sort_method_row["grouping"]),
            )
        )
        return

    # let other sort jobs know if a sorting is currently happening
    is_sorting_file = sort_loc / "is_sorting.npy"
    if not is_sorting_file.exists():
        sorting_file_array = np.array(
            [False, datetime.now().strftime("%m/%d/%y %H:%M:%S")]
        )
        np.save(is_sorting_file, sorting_file_array)

    # save the sort parameters
    sort_method_row_df = pd.DataFrame(sort_method_row).T
    sort_method_row_df.to_pickle(sort_loc / "sort_method.pickle")

    # save the probe
    probe_list = recording_df_sort[
        recording_df_sort.recording_id == recording_row.recording_id
    ].probes.values[0]
    probe_group = make_probe_group(probe_list)
    pi.io.write_prb(sort_loc / "probe.prb", probe_group)

    print(
        "Recording length (hrs): {} ({})".format(
            round(recording_row.n_hours, 3), recording_row.recording_id
        )
    )

    denoised = sort_method_row["denoised"]
    grouping = sort_method_row["grouping"]
    sorter = sort_method_row["sorter"]

    return pd.DataFrame(
        [[sort_loc, recording_row.recording_id, denoised, grouping, sorter]],
        columns=["sort_loc", "recording_id", "denoised", "grouping", "sorter"],
    )


def run_sort(
    recording_id,
    sorter,
    denoised,
    grouping,
    bird,
    timestamp,
    freq_min=300,
    freq_max=6000,
    max_spikes_per_unit=1000,
    last_frame=None,
    overwrite_sort=False,
    n_hours_currently_running=12,
    n_jobs=-1,
    suppress_artifacts_kwarge={
        "fill_mode": "noise",
        "thresh": 1500,
        "ms_surrounding": 150,
    },
    user="tsainbur",
):

    # time code
    t0 = time.time()
    # are we in txori, pakhi, or ssrde
    hostname = socket.gethostname()

    # get folders to save results to, as well as to save temporary data to
    (
        spikesorting_directory,
        tmp_loc,
        raw_data_folder,
        raw_data_folder2,
    ) = get_spikesorting_directories(hostname, bird, timestamp, user=user)

    # get folder to save sort info
    sort_loc = (
        spikesorting_directory
        / "sorts"
        / recording_id
        / sorter
        / "denoised_{}".format(denoised)
        / "grouping_{}".format(grouping)
    )

    # check if sort already exists
    sort_exists = (sort_loc / "sort.npz").exists()
    if sort_exists and not overwrite_sort:
        print(
            "sort exists: sorter:{} | denoised: {} | grouping: {}".format(
                sorter, denoised, grouping
            )
        )
        return

    # check if sort exists in alternative sort spots
    if not overwrite_sort:
        timestamp = spikesorting_directory.stem
        bird = spikesorting_directory.parent.stem

        for alt_data_dir in ALTERNATIVE_DATA_DIRS:
            spikesorting_folder_alt = alt_data_dir / "spikesorting" / bird / timestamp
            sort_loc_alt = (
                spikesorting_folder_alt
                / "sorts"
                / recording_id
                / sorter
                / "denoised_{}".format(denoised)
                / "grouping_{}".format(grouping)
            )
            sort_exists_alt = (sort_loc_alt / "sort.npz").exists()
            if sort_exists_alt:
                print(
                    "sort exists in an alternative folder: folder:{} \n\t sorter:{} | denoised: {} | grouping: {}".format(
                        sort_loc_alt,
                        sorter,
                        denoised,
                        grouping,
                    )
                )
                return

    # check if sort is currently running
    is_sorting_file = sort_loc / "is_sorting.npy"
    if is_sorting_file.exists():
        sorting_file_array = np.load(is_sorting_file)
        # if there's a sort currently running
        if sorting_file_array[0] in ["True", True]:
            if len(sorting_file_array) == 2:
                # if it's been more than a day you can start a new sort
                started_time = datetime.strptime(
                    sorting_file_array[1], "%m/%d/%y %H:%M:%S"
                )
                hours_since_sort_started = (
                    (datetime.now() - started_time).seconds / 60 / 60
                )

                print(
                    "\nHours since sort started: {} | started: {}, now: {}".format(
                        hours_since_sort_started, started_time, datetime.now()
                    )
                )
                if hours_since_sort_started < n_hours_currently_running:
                    if not overwrite_sort:
                        print(
                            "sort currently running. sorter:{} | denoised: {} | grouping: {} | started: {}".format(
                                sorter, denoised, grouping, sorting_file_array[1]
                            )
                        )
                        return
                else:
                    print(
                        "sort says its running, but is taking too long. Re-sorting. sorter:{} | denoised: {} | grouping: {} | started: {}".format(
                            sorter, denoised, grouping, sorting_file_array[1]
                        )
                    )

    sorting_file_array = np.array([True, datetime.now().strftime("%m/%d/%y %H:%M:%S")])
    np.save(is_sorting_file, sorting_file_array)

    logger = create_logger(sort_loc / "{}.log".format(sorter))
    # log basic info
    logger.info("sys prefix: {}".format(sys.prefix))
    logger.info("tmp directory: {}".format(tempfile.gettempdir()))
    logger.info("Started: {}".format(str(datetime.now())))
    logger.info("Host: {}".format(hostname))
    logger.info("Save to: {}".format(sort_loc.as_posix()))

    # load rows to merge and sort info
    sort_method = pd.read_pickle(sort_loc / "sort_method.pickle").iloc[0]

    # get sort dataframe
    recording_df = pd.read_pickle(sort_loc.parents[4] / "recording_df.pickle")

    # get the rows
    sort_row = recording_df[recording_df["recording_id"].values == recording_id].iloc[0]

    # add the probe info
    probe_group = pi.io.read_prb(sort_loc / "probe.prb")

    dat_file = Path(*[raw_data_folder] + list(sort_row.dat_file.parts[-7:]))
    if not dat_file.exists():
        dat_file = Path(*[raw_data_folder2] + list(sort_row.dat_file.parts[-7:]))

    # prepare the recording
    recording = si.core.BinaryRecordingExtractor(
        dat_file,
        sampling_frequency=sort_row.sample_rate,
        num_chan=sort_row.num_channels_total,
        dtype="int16",
    )

    # if we are subsetting this recording (getting rid of bad frames toward the end of
    # the recording), slice frames
    if last_frame is not None:
        recording = recording.frame_slice(0, last_frame)

    recording = recording.set_probes(probe_group)

    logger.info("Created recording")

    if "bad_channels" in sort_row:
        # remove bad channels
        if len(sort_row["bad_channels"]) > 0:
            good_channels = [
                i for i in list(sort_row.channels) if i not in sort_row.bad_channels
            ]
            recording = si.core.ChannelSliceRecording(
                recording,
                channel_ids=good_channels,
            )

    # preprocess recording
    recording = st.preprocessing.bandpass_filter(
        recording, freq_min=freq_min, freq_max=freq_max
    )

    recording = st.preprocessing.common_reference(recording)
    # suppress noise artifacts that might screw up sorting
    recording = suppress_artifacts(recording, **suppress_artifacts_kwarge)

    logger.info("Preprocessed recording")

    # run spikesort
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

        if n_jobs == -1:
            n_jobs = N_JOBS_MAX = np.min([10, multiprocessing.cpu_count()])

        # if "n_jobs_bin" in sort_method.kw_args:
        #    if sort_method.kw_args["n_jobs_bin"] == "MAX":
        #        sort_method.kw_args["n_jobs_bin"] = n_jobs

        sort_method.kw_args["n_jobs_bin"] = n_jobs
        # sort_method.kw_args["engine"] = "joblib"
        # sort_method.kw_args["engine_kwargs"] = {"n_jobs": n_jobs}
        print("n_jobs: {}".format(n_jobs))
        try:
            logger.info("starting run_sorter")
            sort = si.sorters.run_sorter(
                sorter_name=sort_method.sorter,
                recording=recording,
                output_folder=Path(tmpdir),
                verbose=True,
                **sort_method.kw_args
            )

            # save sort
            logger.info("Saving sort {}".format(str(datetime.now())))
            si.core.NpzSortingExtractor.write_sorting(sort, sort_loc / "sort.npz")
            logger.info("Sorting output saved to {}".format(sort_loc / "sort.npz"))

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
    except Exception as e:
        print("deleting folder failed: {}".format(e))
        logger.info("deleting folder failed: {}".format(e))

    try:
        # extract spikes
        t0 = time.time()
        logger.info("Extracting spike waveforms")
        sort = si.core.NpzSortingExtractor(sort_loc / "sort.npz")

        waveform_extractor_folder = sort_loc / "waveforms_{}".format(
            max_spikes_per_unit
        )
        # start extraction
        logger.info("Starting extraction: {}".format(str(datetime.now())))
        we = WaveformExtractor.create(
            recording, sort, waveform_extractor_folder, remove_if_exists=False
        )
        we.set_params(
            ms_before=1.5, ms_after=2, max_spikes_per_unit=max_spikes_per_unit
        )
        we.run(n_jobs=n_jobs, chunk_size=300000, progress_bar=True)
        print(we)
        t1 = time.time()
        total_time = np.array([t1 - t0])
        logger.info("Extraction time: {}".format(total_time))

        # t0 = time.time()
        # logger.info("Grabbing spike PCA")

        #  get principal components
        # pc = st.compute_principal_components(
        #    we, load_if_exists=True, n_components=10, mode="by_channel_local"
        # )
        # t0 = time.time()
        # logger.info("PCA time: {}".format(total_time))
    except Exception as e:
        print("Waveform extraction failed: {}".format(e))
        logger.info("Waveform extraction failed: {}".format(e))

    # let other sorters know this is no longer running
    np.save(is_sorting_file, np.array([False]))
