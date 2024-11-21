from typing import List, Union
from .mytypes import ChannelId, SampleIndex, ChannelIndex, Order, SamplingFrequencyHz

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


class FrameSliceRecording(BaseRecording):
    """
    Class to slice a Recording object based on channel_ids.
    """

    def __init__(self, parent_recording, start_frame, end_frame):

        self._parent_recording = parent_recording
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._channel_ids = parent_recording.get_channel_ids()

        # some checks
        # TODO assert that all frames are within bounds of recording

        BaseRecording.__init__(self,
                               sampling_frequency=parent_recording.get_sampling_frequency(),
                               channel_ids=self._channel_ids,
                               dtype=parent_recording.get_dtype())


        # link recording segment
        for parent_segment in self._parent_recording._recording_segments:
            sub_segment = FrameSliceRecordingSegment(parent_segment, start_frame, end_frame)
            self.add_recording_segment(sub_segment)


        # update dump dict
        self._kwargs = {'parent_recording': parent_recording.to_dict()}


class FrameSliceRecordingSegment(BaseRecordingSegment):
    """
    Class to return a sliced segment traces.
    """

    def __init__(self, parent_recording_segment, start_frame, end_frame):
        BaseRecordingSegment.__init__(self)
        self._parent_recording_segment = parent_recording_segment

    def get_num_samples(self) -> SampleIndex:
        # TODO set self.total_length
        #return self._parent_recording_segment.get_num_samples()
