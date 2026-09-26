from dataclasses import replace

from sigpipe.base.stream import Stream


def seen_from_first_receiver(stream: Stream) -> Stream:
    """
    A gather correlated on its last receiver, then flipped in space: its zero-lag trace, the
    virtual source, is now its first, so its offsets count from the first receiver. The flip
    keeps the receivers in place, which is exact for evenly spaced receivers.
    """
    receivers = stream.acquisition.receivers
    acquisition = type(stream.acquisition)(source=receivers[0], receivers=receivers)
    return replace(stream, acquisition=acquisition)
