"""The phase-weighted stack: the same, whether it measures its streams' phases or is given them."""

import numpy as np

from sigpipe.algorithms.stacking.stream.phase_weighted import (
    instantaneous_phase,
    stack_phase_weighted,
    stack_phase_weighted_given,
)
from sigpipe.base.stream import Stream


def test_the_stack_given_its_phases_is_the_one_it_makes(stream: Stream) -> None:
    rng = np.random.default_rng(3)
    streams = [
        Stream(
            xt=rng.standard_normal(stream.xt.shape).astype(np.float32),
            ts=stream.ts,
            sampling_freq=stream.sampling_freq,
            acquisition=stream.acquisition,
        )
        for _ in range(7)
    ]

    made = stack_phase_weighted(streams, nu=3)
    given = stack_phase_weighted_given(
        streams, [instantaneous_phase(one.xt) for one in streams], nu=3
    )

    np.testing.assert_array_equal(given.xt, made.xt)
