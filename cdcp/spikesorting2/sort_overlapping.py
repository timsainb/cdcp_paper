import numpy as np
from cdcp.spikesorting2.probe_maps import make_probe_group
import probeinterface as pi
from datetime import datetime
from cdcp.paths import ensure_dir
import pandas as pd
import time
import socket
import sys
from cdcp.spikesorting2.directories import get_spikesorting_directories
from cdcp.spikesorting2.utils import create_logger
import tempfile
import spikeinterface as si
import spikeinterface.toolkit as st
from cdcp.spikesorting2.suppress_artifacts_si2 import suppress_artifacts
import multiprocessing
from spikeinterface import WaveformExtractor, extract_waveforms
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from cdcp.spikesorting2.utils import choose_gpu
import spikeinterface.sorters as ss
import stat

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


def run_sort_overlap(
    merge_recording_id,
    recording_id_prev,
    recording_id_next,
    sorter,
    denoised,
    grouping,
    bird,
    timestamp,
    freq_min=300,
    freq_max=6000,
    overwrite_sort=False,
    max_spikes_per_unit=1000,
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
        / "overlap_sorts"
        / merge_recording_id
        / sorter
        / "denoised_{}".format(denoised)
        / "grouping_{}".format(grouping)
    )
    print(sort_loc)

    # check if sort already exists
    sort_exists = (sort_loc / "sort.npz").exists()
    if sort_exists and not overwrite_sort:
        print(
            "sort exists: sorter:{} | denoised: {} | grouping: {}".format(
                sorter, denoised, grouping
            )
        )
        return

    # check if sort is currently running
    is_sorting_file = sort_loc / "is_sorting.npy"
    if is_sorting_file.exists() and not overwrite_sort:
        sorting_file_array = np.load(is_sorting_file)
        # if there's a sort currently running
        if sorting_file_array[0] in ["True", True]:
            if len(sorting_file_array) == 2:
                # if it's been more than a day you can start a new sort
                if (
                    datetime.now()
                    - datetime.strptime(sorting_file_array[1], "%m/%d/%y %H:%M:%S")
                ).seconds / 60 / 60 < n_hours_currently_running:

                    print(
                        "sort currently running. sorter:{} | denoised: {} | grouping: {}".format(
                            sorter, denoised, grouping
                        )
                    )
                    return
    sorting_file_array = np.array([True, datetime.now().strftime("%m/%d/%y %H:%M:%S")])
    np.save(is_sorting_file, sorting_file_array)

    # log basic info
    logger = create_logger(sort_loc / "{}.log".format(sorter))
    logger.info("sys prefix: {}".format(sys.prefix))
    logger.info("tmp directory: {}".format(tempfile.gettempdir()))
    logger.info("Started: {}".format(str(datetime.now())))
    logger.info("Host: {}".format(hostname))
    logger.info("Save to: {}".format(sort_loc.as_posix()))

    # load rows to merge and sort info
    sort_method = pd.read_pickle(sort_loc / "sort_method.pickle").iloc[0]

    # get sort dataframe
    recording_df_sort = pd.read_pickle(sort_loc.parents[4] / "recording_df.pickle")

    # add the probe info
    probe_group = pi.io.read_prb(sort_loc / "probe.prb")

    # get row info
    recording_row_prev = recording_df_sort[
        recording_df_sort.recording_id == recording_id_prev
    ].iloc[0]

    recording_row_next = recording_df_sort[
        recording_df_sort.recording_id == recording_id_next
    ].iloc[0]

    # prepare the recording
    dat_file_prev = Path(
        *[raw_data_folder] + list(recording_row_prev.dat_file.parts[-7:])
    )
    if not dat_file_prev.exists():
        dat_file_prev = Path(
            *[raw_data_folder2] + list(recording_row_prev.dat_file.parts[-7:])
        )
    recording_prev = si.core.BinaryRecordingExtractor(
        dat_file_prev,
        sampling_frequency=recording_row_prev.sample_rate,
        num_chan=recording_row_prev.num_channels_total,
        dtype="int16",
    )

    # prepare the recording
    dat_file_next = Path(
        *[raw_data_folder] + list(recording_row_next.dat_file.parts[-7:])
    )
    if not dat_file_next.exists():
        dat_file_next = Path(
            *[raw_data_folder2] + list(recording_row_next.dat_file.parts[-7:])
        )
    recording_next = si.core.BinaryRecordingExtractor(
        dat_file_next,
        sampling_frequency=recording_row_next.sample_rate,
        num_chan=recording_row_next.num_channels_total,
        dtype="int16",
    )

    # get the frames to subset
    prev_row_sample_start, next_row_sample_end = np.load(
        sort_loc / "sample_start_end.npy"
    )
    prev_row_sample_start, next_row_sample_end

    # print(prev_row_sample_start, recording_prev.get_num_frames(segment_index=0))
    # frame sli
    recording_prev = recording_prev.frame_slice(
        prev_row_sample_start, recording_prev.get_num_frames(segment_index=0)
    )
    recording_next = recording_next.frame_slice(0, next_row_sample_end)

    # get only data channels
    recording_prev = recording_prev.channel_slice(recording_row_prev.channels)
    recording_next = recording_next.channel_slice(recording_row_prev.channels)

    recording = si.concatenate_recordings([recording_prev, recording_next])

    logger.info("Created recording")

    recording = st.preprocessing.bandpass_filter(
        recording, freq_min=freq_min, freq_max=freq_max
    )
    recording = st.preprocessing.common_reference(recording)
    # suppress noise artifacts that might screw up sorting
    recording = suppress_artifacts(recording, **suppress_artifacts_kwarge)
    recording = recording.set_probes(probe_group)
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

        N_JOBS_MAX = multiprocessing.cpu_count()

        if n_jobs == -1:
            n_jobs = N_JOBS_MAX = np.min([10, multiprocessing.cpu_count()])

        # if "n_jobs_bin" in sort_method.kw_args:
        #    if sort_method.kw_args["n_jobs_bin"] == "MAX":
        #        sort_method.kw_args["n_jobs_bin"] = n_jobs

        sort_method.kw_args["n_jobs_bin"] = n_jobs
        # sort_method.kw_args["engine"] = "joblib"
        # sort_method.kw_args["engine_kwargs"] = {"n_jobs": n_jobs}
        print("n_jobs: {}".format(n_jobs))

        # try:
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

        # except Exception as e:
        #    logger.info(e)
        #    logger.info("failed, cleaning up")

        t1 = time.time()
        total_time = np.array([t1 - t0])
        logger.info("Sorting time: {}".format(total_time))
        np.save(sort_loc / "sort_time.npy", total_time)

    # extract waveforms
    logger.info("Extracting waveforms")
    extract_waveforms_from_sort(
        sort_loc,
        recording,
        max_spikes_per_unit=max_spikes_per_unit,
        logger=logger,
        n_jobs=n_jobs,
    )
    logger.info("Waveforms extracted")

    sorting_file_array = np.array([False, datetime.now().strftime("%m/%d/%y %H:%M:%S")])
    np.save(is_sorting_file, sorting_file_array)


def extract_waveforms_from_sort(
    sort_loc, recording, max_spikes_per_unit, logger, n_jobs
):
    try:
        # extract spikes
        t0 = time.time()
        logger.info("Extracting spike waveforms")
        sort = si.core.NpzSortingExtractor(sort_loc / "sort.npz")

        waveform_extractor_folder = sort_loc / "waveforms_{}".format(
            max_spikes_per_unit
        )
        # get templates and max channel
        logger.info("Starting templates: {}".format(str(datetime.now())))
        we = WaveformExtractor.create(
            recording, sort, waveform_extractor_folder, remove_if_exists=False
        )
        we.set_params(
            ms_before=3.0, ms_after=4.0, max_spikes_per_unit=max_spikes_per_unit
        )
        we.run(n_jobs=n_jobs, chunk_size=30000, progress_bar=True)
        print(we)
        t1 = time.time()
        total_time = np.array([t1 - t0])
        logger.info("Extraction time: {}".format(total_time))

        t0 = time.time()
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


def prep_merged_sort(
    sort_method_row,
    merge_row,
    recording_df_sort,
    spikesorting_folder,
    minutes_to_overlap=2,
    overwrite_is_sorting=False,
):
    # get folder to save sort info
    sort_loc = (
        spikesorting_folder
        / "overlap_sorts"
        / merge_row.merge_recording_id
        / sort_method_row["sorter"]
        / "denoised_{}".format(sort_method_row["denoised"])
        / "grouping_{}".format(str(sort_method_row["grouping"]))
    )
    ensure_dir(sort_loc)

    # let other sort jobs know if a sorting is currently happening
    is_sorting_file = sort_loc / "is_sorting.npy"
    if not is_sorting_file.exists() or overwrite_is_sorting:
        sorting_file_array = np.array(
            [False, datetime.now().strftime("%m/%d/%y %H:%M:%S")]
        )
        np.save(is_sorting_file, sorting_file_array)

    # save the sort parameters
    sort_method_row_df = pd.DataFrame(sort_method_row).T
    sort_method_row_df.to_pickle(sort_loc / "sort_method.pickle")

    recording_row_prev = recording_df_sort[
        recording_df_sort.recording_id == merge_row.recording_id_prev
    ].iloc[0]

    recording_row_next = recording_df_sort[
        recording_df_sort.recording_id == merge_row.recording_id_next
    ].iloc[0]

    # save the probe
    probe_list = recording_row_prev.probes
    probe_group = make_probe_group(probe_list)
    pi.io.write_prb(sort_loc / "probe.prb", probe_group)

    #
    samples_to_overlap = minutes_to_overlap * 60 * recording_row_prev.sample_rate

    # determine what sample number to start subset for prev_row
    if samples_to_overlap > recording_row_prev.n_samples:
        prev_row_sample_start = 0
    else:
        prev_row_sample_start = recording_row_prev.n_samples - samples_to_overlap

    # determine what sample number to start subset for next
    if samples_to_overlap > recording_row_next.n_samples:
        next_row_sample_end = recording_row_next.n_samples
    else:
        next_row_sample_end = samples_to_overlap
    prev_row_sample_start, next_row_sample_end

    np.save(
        sort_loc / "sample_start_end.npy",
        np.array([prev_row_sample_start, next_row_sample_end]),
    )

    return pd.DataFrame(
        [
            [
                sort_loc,
                merge_row.recording_id_prev,
                merge_row.recording_id_next,
                merge_row.merge_recording_id,
                sort_method_row.denoised,
                sort_method_row.grouping,
                sort_method_row.sorter,
                prev_row_sample_start,
                next_row_sample_end,
                samples_to_overlap,
                recording_row_prev.n_samples,
            ]
        ],
        columns=[
            "sort_loc",
            "recording_id_prev",
            "recording_id_next",
            "merge_recording_id",
            "denoised",
            "grouping",
            "sorter",
            "prev_row_sample_start",
            "next_row_sample_end",
            "samples_to_overlap",
            "recording_prev_n_samples_total",
        ],
    )
