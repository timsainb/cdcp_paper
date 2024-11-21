from spikeinterface import WaveformExtractor, extract_waveforms
import datetime
import numpy as np
from spikeinterface.toolkit.postprocessing.template_tools import (
    get_template_best_channels,
)
from spikeinterface.core.job_tools import ensure_n_jobs, ensure_chunk_size, devide_recording_into_chunks

from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from tqdm.auto import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, desc=None, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm, desc=self._desc, total=self._total
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield np.array(lst[i : i + n])


def grab_spike_chunk(
    spike_times,
    spike_labels,
    nbefore,
    nafter,
    max_channels_per_template,
    waveform_extractor_path,
    best_channels_index,
    preload_trace=False,  # whether to grab spikes individually (better for sparse) or grab spikes as part of a preloaded trace (better for dense)
):
    # load the extractor
    we = WaveformExtractor.load_from_folder(waveform_extractor_path)
    recording = we.recording

    shape = (spike_times.size, nbefore + nafter, max_channels_per_template)
    unit_spike_waveforms = np.zeros(dtype="float32", shape=shape)

    t0 = spike_times[0]
    t1 = spike_times[-1]

    if preload_trace:
        trace = recording.get_traces(start_frame=t0 - nbefore, end_frame=t1 + nafter)
        for spike_i, (spike_time, spike_label) in enumerate(
            zip(
                tqdm(spike_times, desc="extract batch spikes", leave=False),
                spike_labels,
            )
        ):
            unit_spike_waveforms[spike_i] = trace[
                t0 - spike_time : t0 - spike_time + nbefore + nafter,
                best_channels_index[spike_label][:max_channels_per_template],
            ]
    else:
        for spike_i, (spike_time, spike_label) in enumerate(
            zip(
                tqdm(spike_times, desc="extract batch spikes", leave=False),
                spike_labels,
            )
        ):
            unit_spike_waveforms[spike_i] = recording.get_traces(
                start_frame=spike_time - nbefore,
                end_frame=spike_time + nafter,
                channel_ids=best_channels_index[spike_label][
                    :max_channels_per_template
                ],
            )

    return unit_spike_waveforms, spike_labels


def sample_spikes(
    waveform_extractor_path,
    best_channels_index=None,
    n_spikes_to_sample=5000,
    max_channels_per_template=8,
    n_jobs=1,
    good_unit_labels=None,
    chunk_by_spikes=False,
    chunk_size=None,
    total_memory=None,
    chunk_memory=None,
    template_best_channels_args={"peak_sign": "neg"},
    verbose=False,
):

    if verbose:
        start_time = datetime.datetime.now()
        print("Loading waveform extractor: {}".format(start_time))

    # load waveform extractor
    we = WaveformExtractor.load_from_folder(waveform_extractor_path)

    # get recording and sorter
    sorting = we.sorting
    recording = we.recording

    assert recording.get_num_segments() == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)

    # get chunk size for iterating over waveforms (or spikes)
    if chunk_size is None:
        if chunk_by_spikes:
            chunk_size = 1000
    else:
        chunk_size = ensure_chunk_size(recording,
                                            total_memory=total_memory, chunk_size=chunk_size,
                                            chunk_memory=chunk_memory, n_jobs=n_jobs)

    # get num samples
    recording_samples = recording.get_num_samples(segment_index=0)

    # get all spikes
    spike_times = sorting._sorting_segments[0].spike_indexes
    spike_labels = sorting._sorting_segments[0].spike_labels

    if verbose:
        print("Sampling spikes: {}".format(datetime.datetime.now() - start_time))
    # grab sample of spikes
    subset_spike_index = np.unique(
        np.random.randint(len(spike_times), size=n_spikes_to_sample)
    )
    spike_times = spike_times[subset_spike_index]
    spike_labels = spike_labels[subset_spike_index]

    # subset only good units, if a set of good units is provided
    if good_unit_labels is not None:
        good_unit_mask = np.array([i in good_unit_labels for i in spike_labels])
        spike_times = spike_times[good_unit_mask]
        spike_labels = spike_labels[good_unit_mask]

    # ensure that the full spike template can be grabbed
    nafter = we.nafter
    nbefore = we.nbefore
    spike_mask = np.array(
        (spike_times > nbefore) & (spike_times < (recording_samples - we.nbefore))
    )
    spike_times = spike_times[spike_mask]
    spike_labels = spike_labels[spike_mask]

    # get best channel index if it does not exist
    if best_channels_index is None:
        max_channels_per_template = min(
            max_channels_per_template, we.recording.get_num_channels()
        )
        best_channels_index = get_template_best_channels(
            we, max_channels_per_template, **template_best_channels_args
        )


    # split samples into chunks
    if chunk_by_spikes:
        spike_times_chunks = list(chunk_list(spike_times, chunk_size))
        spike_labels_chunks = list(chunk_list(spike_labels, chunk_size))
    else:
        spike_time_bins = np.arange(spike_times[0], spike_times[-1], chunk_size)
        spike_bin_index = np.digitize(spike_times, spike_time_bins)
        spike_times_chunks = [
            spike_times[spike_bin_index == i]
            for i in range(len(spike_time_bins))
            if np.sum(spike_bin_index == i) > 0
        ]
        spike_labels_chunks = [
            spike_labels[spike_bin_index == i]
            for i in range(len(spike_time_bins))
            if np.sum(spike_bin_index == i) > 0
        ]
    # print([len(i) for i in spike_times_chunks])
    if verbose:
        print("Grabbing chunks: {}".format(datetime.datetime.now() - start_time))
    # grab samples
    spike_waveforms_labels = ProgressParallel(
        n_jobs=n_jobs,
        verbose=20,
        #prefer="threads",
        desc="batch sample spikes",
        total=len(spike_times_chunks),
    )(
        delayed(grab_spike_chunk)(
            spike_times_chunk,
            spike_labels_chunk,
            nbefore,
            nafter,
            max_channels_per_template,
            waveform_extractor_path,
            best_channels_index,
        )
        for spike_times_chunk, spike_labels_chunk in #tqdm(
            zip(spike_times_chunks, spike_labels_chunks),
            #total=len(spike_times_chunks),
            #desc="batch sample spikes",
            # leave=False,
        #)
    )

    # seperate labels from waveforms
    spike_waveforms = np.vstack([i[0] for i in spike_waveforms_labels])
    spike_labels = np.concatenate([i[1] for i in spike_waveforms_labels])

    if verbose:
        print("Completed: {}".format(datetime.datetime.now() - start_time))

    return spike_waveforms, spike_labels
