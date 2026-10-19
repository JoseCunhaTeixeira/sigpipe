"""A gather as PAC's gather view draws it: on a figure of its own, or on axes given, a panel of a
larger figure."""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from sigpipe.base.stream import Stream
from sigpipe.dataio.stream.plotting import draw_stream, plot_stream


def test_a_gather_drawn_on_given_axes_as_on_its_own_figure(stream: Stream) -> None:
    figure, (panel, other) = plt.subplots(1, 2)
    own = plot_stream(stream)

    draw_stream(panel, stream)

    for axes in (panel, own.axes[0]):
        (lobes,) = [one for one in axes.collections if isinstance(one, PolyCollection)]
        (wiggles,) = [one for one in axes.collections if isinstance(one, LineCollection)]
        assert len(lobes.get_paths()) == len(wiggles.get_segments()) == 4
    assert panel.get_xlim() == own.axes[0].get_xlim()
    assert panel.get_ylim() == own.axes[0].get_ylim()
    assert not other.collections
    plt.close(figure)
    plt.close(own)
