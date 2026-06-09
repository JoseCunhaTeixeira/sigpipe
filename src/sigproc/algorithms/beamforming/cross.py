import numpy as np
import torch

from sigproc.base.beamforming import Beam
from sigproc.base.coordinate import coordinates_to_tuples
from sigproc.base.dispersion import DispersionCurve
from sigproc.base.stream import Stream


def beamform_cross(
    stream: Stream,
    medium_velocity: float | DispersionCurve,
    xgrid_lims: tuple[float, float],
    ygrid_lims: tuple[float, float],
    grid_spacing: float,
    fmin: float | None,
    fmax: float | None,
) -> Beam:

    xgrid_coords = torch.arange(xgrid_lims[0], xgrid_lims[1], grid_spacing)
    ygrid_coords = torch.arange(ygrid_lims[0], ygrid_lims[1], grid_spacing)

    X, Y = torch.meshgrid(xgrid_coords, ygrid_coords, indexing="ij")
    gridpoints = torch.stack([X.ravel(), Y.ravel()], dim=1)

    stations = torch.Tensor(coordinates_to_tuples(stream.acquisition.receivers))[:, :2]

    distances_to_all_gridpoints = torch.linalg.norm(
        gridpoints[:, None, :] - stations[None, :, :], dim=2
    )

    xt = torch.from_numpy(stream.xt.copy())
    _, nt = xt.shape

    freqs = torch.fft.rfftfreq(nt, d=1 / stream.sampling_freq)
    omegas = 2 * torch.pi * freqs

    if fmin is not None:
        fmin = max(fmin, min(freqs))
    else:
        fmin = min(freqs)

    if fmax is not None:
        fmax = min(fmax, max(freqs))
    else:
        fmax = max(freqs)

    if isinstance(medium_velocity, DispersionCurve):
        fmin = max(fmin, min(medium_velocity.fs))
        fmax = min(fmax, max(medium_velocity.fs))

    freq_idx = torch.where((freqs >= fmin) & (freqs <= fmax))[0]
    freqs_lim = freqs[freq_idx]
    omegas_lim = omegas[freq_idx]

    if isinstance(medium_velocity, DispersionCurve):
        vs = torch.from_numpy(
            np.interp(freqs_lim, medium_velocity.fs, medium_velocity.vs)
        ).to(freqs_lim.device)
        traveltimes = distances_to_all_gridpoints[:, :, None] / vs
        greens_functions = torch.exp(
            -1j * omegas_lim[None, None, :] * traveltimes[:, :, :]
        ).type(torch.complex64)
    else:
        traveltimes = distances_to_all_gridpoints / medium_velocity
        greens_functions = torch.exp(
            -1j * omegas_lim[None, None, :] * traveltimes[:, :, None]
        ).type(torch.complex64)

    xf = torch.fft.rfft(xt, dim=1)
    xf_lim = xf[:, freq_idx]

    K = xf_lim[:, None, :] * xf_lim.conj()[None, :, :]
    diag_idxs = torch.arange(K.shape[0])
    K[diag_idxs, diag_idxs, :] = torch.zeros(omegas_lim.shape, dtype=torch.complex64)

    beampowers = torch.zeros(gridpoints.shape[0], dtype=torch.float32)
    for iw in range(len(omegas_lim)):
        g = greens_functions[:, :, iw]
        Kw = K[:, :, iw]
        tmp = torch.matmul(g.conj(), Kw)
        beampowers += (tmp * g).sum(dim=1).real

    bp = beampowers.reshape(len(xgrid_coords), len(ygrid_coords))
    bp = abs(bp)
    bp = (bp - bp.min()) / (bp.max() - bp.min())

    return Beam(
        xy_map=bp.cpu().numpy().astype(np.float32),
        xs=xgrid_coords.cpu().numpy().astype(np.float32),
        ys=bp.cpu().numpy().astype(np.float32),
        acquisition=stream.acquisition,
    )
