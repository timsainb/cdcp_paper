import numpy as np
from scipy.signal import convolve

from spikeinterface.toolkit.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)

from spikeinterface.toolkit.utils import get_closest_channels


class SuppressArtifactsRecording(BasePreprocessor):
    """
    Re-references the recording extractor traces.
    Parameters
    ----------
    Finds and removes artifacts by thresholding voltage and replacing the time around
    that artifact with either zeros, or noise (matched to the data). This is meant to
    be applied after bandpass filtering and common average referencing.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    thresh: float  (default 2000)
        Threshold for determining noise artifact
    ms_surrounding: float  (default 50)
        Surrounding milliseconds from artifact to remove
    fill_mode: str (default 'noise')
        Fill removed artifacts with either 0s ('zeros') or uniform noise ('noise')
    noise_fill_std: float (default 3)
        Number of standard deviations the noise should be if fill_mode is 'noise'
    Returns
    -------
    rescaled_traces: SuppressArtifactsRecording
        The suppressed artifacts recording extractor object
    """

    name = "suppress_artifacts"

    def __init__(
        self,
        recording,
        thresh=2000,
        ms_surrounding=50,
        fill_mode="noise",
        noise_fill_std=3,
        verbose=False,
    ):

        neighbors = None
        # some checks
        if fill_mode not in ("zeros", "noise"):
            raise ValueError("'fill_mode' must be either 'zeros', or 'noise'")

        BasePreprocessor.__init__(self, recording)

        for parent_segment in recording._recording_segments:
            rec_segment = SuppressArtifactsRecordingSegment(
                parent_segment,
                thresh=thresh,
                ms_surrounding=ms_surrounding,
                fill_mode=fill_mode,
                noise_fill_std=noise_fill_std,
                fs=recording.get_sampling_frequency(),
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording.to_dict(),
            thresh=thresh,
            ms_surrounding=ms_surrounding,
            fill_mode=fill_mode,
            noise_fill_std=noise_fill_std,
        )


class SuppressArtifactsRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        fs,
        thresh=2000,
        ms_surrounding=50,
        fill_mode="noise",
        noise_fill_std=3,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self._thresh = thresh
        self._ms_surrounding = ms_surrounding
        self._fill_mode = fill_mode
        self._noise_fill_std = noise_fill_std
        self._fs = fs

    def get_traces(self, start_frame, end_frame, channel_indices):
        # need input trace
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices
        ).T

        # remove any nans
        traces = np.nan_to_num(traces)

        # get the # of frames surrounding noise artifacts to suppress
        frames_surrounding = int(self._ms_surrounding * self._fs / 1000)

        above_thresh = np.abs(traces) > self._thresh

        # convolve with surrounding frames
        above_thresh = (
            convolve(above_thresh, np.ones((1, int(frames_surrounding))), mode="same")
            > 1e-10
        )

        trace_stds = np.array(
            [
                1e-1 + np.std(trace[thresh == False])
                for trace, thresh in zip(traces, above_thresh)
            ]
        )

        # add a small amount of noise to any zeros, so variance is not zero if a patch of zeros exists
        for i in range(len(traces)):
            n_zeros = np.sum(traces[i] == 0)
            traces[i, traces[i] == 0] = np.random.normal(loc=0, scale=trace_stds[i])

        trace_means = np.array(
            [
                np.mean(trace[thresh == False])
                for trace, thresh in zip(traces, above_thresh)
            ]
        )
        # add gaussian noise
        noise = (
            (
                (
                    np.random.normal(
                        loc=0, scale=1, size=np.product(np.shape(traces))
                    ).reshape(np.shape(traces)[::-1])
                )
                * np.expand_dims(trace_stds, 0)
                + np.expand_dims(trace_means, 0)
            )
            .astype(traces.dtype)
            .T
        )
        if False:
            noise = (
                (
                    (
                        np.random.rand(np.product(np.shape(traces))).reshape(
                            np.shape(traces)[::-1]
                        )
                        - 0.5
                    )
                    * np.expand_dims(trace_stds, 0)
                    * self._noise_fill_std
                    + np.expand_dims(trace_means, 0)
                )
                .astype(traces.dtype)
                .T
            )

        traces[above_thresh] = noise[above_thresh]

        return traces.T


# function for API
def suppress_artifacts(*args, **kwargs):
    return SuppressArtifactsRecording(*args, **kwargs)


suppress_artifacts.__doc__ = SuppressArtifactsRecording.__doc__
