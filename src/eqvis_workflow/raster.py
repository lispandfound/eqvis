"""Turning scattered stations into a raster, and choosing its colour levels.

Stations sit on a rotated grid that is already masked to land, so the map is
built by interpolating station values onto a regular lon/lat grid and blanking
cells with no station nearby -- the coastline emerges from the data.
"""

import numpy as np
import shapely
import typer
from matplotlib.ticker import (
    LogLocator,
    MaxNLocator,
)
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .geography import land_mask


def rasterise(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    coastline: shapely.MultiPolygon | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Interpolate scattered station values onto a regular lon/lat grid.

    With a ``coastline``, off-land cells are masked and the rest is left to the
    interpolator: cells outside the station footprint (the domain edge) come
    back as NaN and are masked, while small interior gaps -- lakes and rivers
    that the station grid omits -- are interpolated across rather than punched
    out as holes.

    Without a coastline, there is no land boundary to lean on, so cells further
    from every station than 1.5x that station's own local grid spacing are
    masked instead. This keeps the domain edges blank (the station grid mixes
    resolutions) but leaves the ragged station-grid boundary and small gaps.
    """
    aspect = np.cos(np.radians(lat.mean()))
    # Work in locally-isotropic coordinates for distances.
    pts = np.column_stack([lon * aspect, lat])
    tree = cKDTree(pts)
    # Distance from each station to its nearest neighbour: the local spacing.
    local_spacing = tree.query(pts, k=2)[0][:, 1]
    spacing = float(np.median(local_spacing))

    dlat = spacing
    dlon = spacing / aspect
    grid_lon = np.arange(lon.min(), lon.max() + dlon, dlon)
    grid_lat = np.arange(lat.min(), lat.max() + dlat, dlat)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    grid = griddata((lon, lat), values, (mesh_lon, mesh_lat), method="linear")
    if coastline is not None:
        masked = land_mask(mesh_lon, mesh_lat, coastline)
    else:
        dist, nearest = tree.query(
            np.column_stack([mesh_lon.ravel() * aspect, mesh_lat.ravel()])
        )
        masked = (dist > 1.5 * local_spacing[nearest]).reshape(mesh_lon.shape)
    return grid_lon, grid_lat, np.ma.masked_invalid(np.ma.masked_where(masked, grid))


def discrete_norm(
    values: np.ndarray, n_levels: int, log: bool, vmin: float | None, vmax: float | None
) -> np.ndarray:
    """Level boundaries covering the robust (1-99%) range of the data."""
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        raise typer.BadParameter("no finite data to plot")
    lo = vmin if vmin is not None else float(np.quantile(finite, 0.01))
    hi = vmax if vmax is not None else float(np.quantile(finite, 0.99))
    if not hi > lo:
        hi = lo + (abs(lo) or 1.0) * 0.1
    if log:
        levels = LogLocator(subs=(1.0, 2.0, 5.0)).tick_values(lo, hi)
        levels = levels[(levels >= lo / 2) & (levels <= hi * 2)]
        if len(levels) < 5:
            levels = np.geomspace(lo, hi, n_levels + 1)
    else:
        levels = MaxNLocator(n_levels).tick_values(lo, hi)
    return levels


def symmetric_norm(values: np.ndarray, n_levels: int) -> np.ndarray:
    """Zero-centred level boundaries for a diverging (diff) map."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise typer.BadParameter("no finite data to plot")
    limit = float(np.quantile(np.abs(finite), 0.99))
    if limit <= 0:
        limit = 0.1
    levels = MaxNLocator(n_levels, symmetric=True).tick_values(-limit, limit)
    return levels


def fixed_symmetric_norm(limit: float, n_levels: int) -> np.ndarray:
    """Zero-centred levels in round steps, ending exactly on +/-``limit``.

    Like :func:`symmetric_norm`, but for a scale pinned to a limit the caller
    chose rather than one read off the data, so the outermost level has to land
    on the limit itself: the round step dividing it evenly that comes closest
    to ``n_levels`` bins wins.
    """
    decade = 10.0 ** np.floor(np.log10(limit))
    steps = [f * decade for f in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)]
    exact = [s for s in steps if abs(limit / s - round(limit / s)) < 1e-9]
    step = min(exact or steps, key=lambda s: abs(2 * limit / s - n_levels))
    bins = round(limit / step)
    # Rounded, so the zero level is exactly zero rather than float dust.
    return np.round(step * np.arange(-bins, bins + 1), 12)
