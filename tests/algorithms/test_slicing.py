"""A record cut into segments: every one the same number of samples, whatever the step."""

import numpy as np

from sigpipe.algorithms.segmentation.slice import slice_segments
from sigpipe.base.stream import Stream


def test_segments_stack_when_the_step_is_no_whole_number_of_samples(stream: Stream) -> None:
    # 100 Hz: a step of 0.105 s is 10.5 samples, which rounding alone made 10 or 11 apart and the
    # segments 20 or 21 samples long, unstackable.
    segments = slice_segments(stream, segment_duration=0.2, segment_step=0.105)

    assert len(segments) == 3
    assert {segment.xt.shape for segment in segments} == {(4, 21)}
    np.stack([segment.xt for segment in segments])
    # Each starts at the sample nearest its time: within half a sample of it.
    starts = [float(segment.ts[0]) for segment in segments]
    np.testing.assert_allclose(starts, [0.0, 0.105, 0.21], atol=0.5 / stream.sampling_freq + 1e-6)
