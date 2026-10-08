"""``hazard-map``: probabilistic seismic hazard in plan view.

Every other map in this package draws one earthquake. This one draws all of
them at once: the ground motion a site expects to exceed once in a given number
of years, from a rupture set whose members each carry an annual rate::

    eqvis hazard-map ims_lite.duckdb --ruptures ruptures.parquet \\
        --nshm hazard_all_ruptures.h5 --output-dir hazard_maps

The figure is up to four panels of the same quantity from four sources, drawn
on one grid and one colour scale so they can be read against each other:

1. **NSHM2022** -- every rupture in the national model, empirical ground motion.
2. **NSHM2022, simulated subset** -- the same empirical model restricted to the
   ruptures the simulation actually ran, which is the only like-for-like
   reference the pilot has.
3. **Pilot simulation** -- those ruptures as physics rather than as a
   regression.
4. **Augmented** -- the national model with the simulated ruptures' empirical
   contribution swapped out for the simulated one.

Panels 3 and 4 carry the differences that would otherwise want panels of their
own: at each site of the empirical model a triangle is drawn, coloured by the
log ratio to its reference, exactly as ``map`` colours a recording by its
misfit against the simulation. Panel 3's triangles are the like-for-like ratio
against panel 2 -- the hazard deficiency -- and panel 4's are the net effect of
the update against panel 1.

Where the empirical model has nothing to say (PGA, PGV, durations, or a pSA
period off its grid) the figure is the simulation panel alone.

Two things about the numbers, both of which the caption repeats. The simulated
rupture set is a *sample*: its rates sum to a fraction of the national model's,
so the simulation panel is a hazard from the ruptures that were run and not an
estimate of the hazard. And the simulation carries one realisation per rupture
with no ground-motion variability, while the empirical model integrates over
its own scatter -- so the deficiency panel 3 measures is a deficiency of the
comparison as much as of the physics.
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import duckdb
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import typer
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm

from .console import console_warn
from .constants import (
    DEFAULT_COMPONENT,
    IM_UNITS,
    LOG_SCALED,
    SCALAR_IMS,
    SUBSCRIPTS,
    UNIT_LABEL,
)
from .display import Display
from .geography import (
    basins_in_view,
    draw_basins,
    draw_coastline,
    land_mask,
    load_basins,
    load_coastline,
)
from .raster import discrete_norm, fixed_symmetric_norm

# The NSHM hazard file is one group per fault system, each holding
# hazard[threshold, rupture, site, period]. The values are already rate
# contributions -- rupture rate times probability of exceedance -- so a hazard
# curve is their plain sum over ruptures, with no rate to multiply back in.
FAULT_SYSTEMS = ("crustal", "hikurangi", "puysegur")


# Only crustal ruptures were simulated. The rupture ids are unique within a
# fault system and *not* across them, so an event id that also names a Puysegur
# rupture would silently pick up the wrong rate without this.
SIMULATED_SYSTEM = 3


# Ruptures read from the hazard file at a time. The array is 10 GB and the
# reduction over ruptures is the whole point, so it is summed a slab at a time
# rather than loaded; 128 keeps the working set a few hundred megabytes.
RUPTURE_CHUNK = 128


# The aggregation above is a minute of I/O for a result of a few hundred
# kilobytes, so it is kept.
CACHE_DIR = Path.home() / ".cache" / "eqvis"


# Tables a lite IM database has to have. `store.connect` wants the full
# composite schema -- run_labels and the rest -- which a distribution-only
# database does not carry, so this command checks for what it actually reads.
REQUIRED_TABLES = ("runs", "stations", "psa_log", "scalars_log", "periods", "meta")


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    """Open an IM database read-only, refusing one that cannot answer this."""
    con = duckdb.connect(str(path), read_only=True)
    con.execute("SET enable_progress_bar = false")
    present = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    missing = [name for name in REQUIRED_TABLES if name not in present]
    if missing:
        raise typer.BadParameter(
            f"{path} is missing the {', '.join(missing)} table"
            f"{'s' if len(missing) > 1 else ''}. It holds: "
            f"{', '.join(sorted(present)) or '(nothing)'}"
        )
    return con


def quantisation_step(con, key: str) -> float:
    """The log10 step the quantised IM tables were stored with.

    Read rather than assumed: the database records it in `meta` precisely so a
    reader does not hard-code a constant that a rebuild could change.
    """
    row = con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
    if row is None:
        raise typer.BadParameter(f"{key} is not recorded in the database's meta table")
    return float(row[0])


def rupture_rates(path: Path) -> tuple[dict[int, float], float]:
    """Annual rate of each crustal rupture, and the whole set's rate.

    Only the columns needed: the file also carries the rupture geometry, which
    is most of its size and none of its use here. The total covers every fault
    system, because what the caption has to say is how much of the *national*
    rate the simulated ruptures speak for.
    """
    table = pq.read_table(path, columns=["rate", "fault_system", "rupture"])
    system = np.asarray(table.column("fault_system"))
    rupture = np.asarray(table.column("rupture"))
    rate = np.asarray(table.column("rate"))
    crustal = system == SIMULATED_SYSTEM
    return (
        dict(zip(rupture[crustal].tolist(), rate[crustal].tolist())),
        float(rate.sum()),
    )


def run_rates(con, rates: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """``(run_id, rate)`` for every run whose event names a rupture with a rate.

    A run whose event is not in the rupture set has no rate and so cannot
    contribute to a hazard: it is dropped, loudly, rather than counted at zero.
    """
    runs = con.execute('SELECT run_id, "event" FROM runs ORDER BY run_id').fetchall()
    kept, rate = [], []
    missing = []
    for run_id, event in runs:
        try:
            found = rates[int(event)]
        except (KeyError, ValueError):
            missing.append(str(event))
            continue
        kept.append(run_id)
        rate.append(found)
    if missing:
        console_warn(
            f"{len(missing)} of {len(runs)} runs name no rupture in the rupture "
            f"set and are left out: {', '.join(missing[:5])}"
            f"{' ...' if len(missing) > 5 else ''}"
        )
    return np.array(kept, dtype=np.int32), np.array(rate, dtype=np.float64)


def station_coordinates(con) -> dict[str, np.ndarray]:
    """Every station's name and mean position.

    The position is averaged over the runs because the solvers snap stations to
    their own grids: the database's own note puts the disagreement at up to
    0.0016 degrees, which is far below anything a national-scale map resolves,
    so one position per station is the right simplification here.
    """
    rows = con.execute(
        """
        SELECT s.station_id, s.station, avg(rs.longitude), avg(rs.latitude)
        FROM run_stations rs JOIN stations s ON s.station_id = rs.station_id
        GROUP BY 1, 2 ORDER BY 1
        """
    ).fetchall()
    return {
        "id": np.array([r[0] for r in rows], dtype=np.int32),
        "name": np.array([r[1] for r in rows], dtype=object),
        "lon": np.array([r[2] for r in rows], dtype=np.float64),
        "lat": np.array([r[3] for r in rows], dtype=np.float64),
    }


def measure_sql(im: str, period_index: int | None, component: str) -> str:
    """The query giving one ``(station_id, quantised value, rate)`` row per
    station and run, ordered so that each station's motions run downwards.

    Ordered in the database rather than in numpy because the sort is the
    expensive part of the whole computation and DuckDB will spill it to disk;
    sorting fourteen million rows in memory here is exactly what this command
    is trying not to do.
    """
    if period_index is not None:
        # DuckDB lists are 1-based; period_index is the 0-based grid position.
        return f"""
            SELECT p.station_id AS station_id,
                   p.spec[{period_index + 1}]::SMALLINT AS q,
                   r.rate::FLOAT AS rate
            FROM psa_log p JOIN run_rate r ON r.run_id = p.run_id
            ORDER BY station_id, q DESC
        """
    return f"""
        SELECT s.station_id AS station_id,
               s.{im}_q::SMALLINT AS q,
               r.rate::FLOAT AS rate
        FROM scalars_log s JOIN run_rate r ON r.run_id = s.run_id
        WHERE s.component = '{component}' AND s.{im}_q IS NOT NULL
        ORDER BY station_id, q DESC
    """


def segment_starts(keys: np.ndarray) -> np.ndarray:
    """Index where each run of equal ``keys`` begins, for a sorted array."""
    if keys.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(([0], np.flatnonzero(keys[1:] != keys[:-1]) + 1))


def exceedance_rate(rate: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Rate of exceeding each motion, within each station's own block.

    The rows arrive strongest motion first, so the running total of rate down a
    station's block *is* its hazard curve: the rate at which something at least
    this strong happens. Done as one cumulative sum with each block's opening
    total subtracted back off, rather than a Python loop over a hundred
    thousand stations.
    """
    total = np.cumsum(rate)
    counts = np.diff(np.append(starts, rate.size))
    opening = np.concatenate(([0.0], total[starts[1:] - 1])) if starts.size > 1 else np.zeros(1)
    return total - np.repeat(opening, counts)


def level_at_rate(
    motion: np.ndarray, cumulative: np.ndarray, starts: np.ndarray, target: float
) -> np.ndarray:
    """The motion each station exceeds at ``target`` per year.

    The curve is a staircase -- one step per rupture -- so the answer is read
    off by interpolating between the steps that straddle the target rate,
    straight-line in log motion against log rate, which is the shape a hazard
    curve actually has.

    Two edges. A station whose whole rupture set does not add up to ``target``
    has no answer at all and comes back NaN. A station that passes ``target`` on
    its very first step has one only as a bound -- the target rate is reached by
    the single strongest motion alone -- and comes back as that motion, which
    understates it.
    """
    reached = np.where(cumulative >= target, np.arange(cumulative.size), cumulative.size)
    first = np.minimum.reduceat(reached, starts)
    level = np.full(starts.size, np.nan)

    bounded = first == starts  # crossed on the first step
    if bounded.any():
        level[bounded] = motion[first[bounded]]

    inside = (first > starts) & (first < cumulative.size)
    if inside.any():
        upper = first[inside]  # weaker motion, higher rate
        lower = upper - 1  # stronger motion, lower rate
        with np.errstate(divide="ignore", invalid="ignore"):
            span = np.log(cumulative[upper]) - np.log(cumulative[lower])
            fraction = np.where(
                span == 0, 0.0, (np.log(target) - np.log(cumulative[lower])) / span
            )
            level[inside] = np.exp(
                np.log(motion[lower])
                + fraction * (np.log(motion[upper]) - np.log(motion[lower]))
            )
    return level


def rate_above(
    motion: np.ndarray, rate: np.ndarray, starts: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """``(station, threshold)`` rate of exceeding each threshold.

    The counterpart of :func:`level_at_rate`, reading the curve the other way
    up. Needed to put the simulation onto the empirical model's threshold grid,
    which is what makes the two addable in the augmented panel.
    """
    out = np.empty((starts.size, thresholds.size))
    for column, threshold in enumerate(thresholds):
        out[:, column] = np.add.reduceat(np.where(motion > threshold, rate, 0.0), starts)
    return out


@dataclass(frozen=True)
class Curve:
    """One station's hazard curve per block, for a whole measure at once.

    Kept as the sorted motions and their running rate rather than as an answer,
    because a run asks the same curve for more than one return period and the
    sort behind it is the expensive part: reading it twice costs nothing,
    building it twice costs everything.
    """

    station: np.ndarray  # station id, one per block
    motion: np.ndarray
    rate: np.ndarray
    cumulative: np.ndarray
    starts: np.ndarray

    def level(self, target: float) -> np.ndarray:
        """The motion each station exceeds at ``target`` per year."""
        return level_at_rate(self.motion, self.cumulative, self.starts, target)

    def above(self, thresholds: np.ndarray) -> np.ndarray:
        """Each station's rate of exceeding each of ``thresholds``."""
        return rate_above(self.motion, self.rate, self.starts, thresholds)


def simulated_curve(
    con, im: str, period_index: int | None, component: str, step: float
) -> Curve:
    """Hazard curves from the simulated rupture set, at every station it covers."""
    table = con.execute(measure_sql(im, period_index, component)).fetch_arrow_table()
    station = table.column("station_id").to_numpy(zero_copy_only=False)
    quantised = table.column("q").to_numpy(zero_copy_only=False)
    rate = table.column("rate").to_numpy(zero_copy_only=False).astype(np.float64)
    # Freed before the decoded motions are built, so the Arrow copy and the
    # float64 copy of fourteen million rows are never both resident.
    del table
    if station.size == 0:
        raise typer.BadParameter(f"the database holds no {im} for component {component}")

    # Quantised values are monotone in the motion they encode, so the ordering
    # the database did is already the ordering the curve needs; only the values
    # that survive into an answer have to be decoded.
    motion = np.power(10.0, quantised.astype(np.float64) * step)
    del quantised
    starts = segment_starts(station)
    return Curve(
        station=station[starts], motion=motion, rate=rate,
        cumulative=exceedance_rate(rate, starts), starts=starts,
    )


def nshm_hazard(path: Path, events: np.ndarray) -> dict[str, np.ndarray]:
    """Hazard curves from the national model, all ruptures and the simulated few.

    One pass over the file, summing the rate contributions over ruptures, which
    turns ten gigabytes into a few hundred kilobytes; the result is cached
    against the file and the rupture list that produced it.
    """
    signature = hashlib.sha256(
        f"{path.resolve()}:{path.stat().st_mtime_ns}:{np.sort(events).tobytes()!r}".encode()
    ).hexdigest()[:16]
    cache = CACHE_DIR / f"nshm2_{signature}.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        return {key: stored[key] for key in stored.files}

    with h5py.File(path, "r") as handle:
        groups = [name for name in FAULT_SYSTEMS if name in handle]
        if not groups:
            raise typer.BadParameter(
                f"{path} has no fault system groups ({', '.join(FAULT_SYSTEMS)}); "
                f"it holds {', '.join(handle.keys())}"
            )
        first = handle[groups[0]]
        thresholds = first["threshold"][:]
        periods = first["period"][:]
        sites = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in first["site"][:]],
            dtype=object,
        )
        shape = (thresholds.size, sites.size, periods.size)
        every = np.zeros(shape)
        simulable = np.zeros(shape)
        subset = np.zeros(shape)
        for name in groups:
            group = handle[name]
            block = group["hazard"]
            ruptures = group["rupture"][:]
            # Only one fault system was simulated at all, so its total is kept
            # apart: it is the difference between hazard the pilot *could* have
            # covered and hazard no crustal simulation could ever reach.
            crustal = float(group["fault_system"][()]) == SIMULATED_SYSTEM
            simulated = np.isin(ruptures, events) if crustal else None
            for start in range(0, block.shape[1], RUPTURE_CHUNK):
                slab = block[:, start : start + RUPTURE_CHUNK, :, :]
                np.nan_to_num(slab, copy=False)
                total = slab.sum(axis=1)
                every += total
                if crustal:
                    simulable += total
                    chosen = simulated[start : start + RUPTURE_CHUNK]
                    if chosen.any():
                        subset += slab[:, chosen, :, :].sum(axis=1)
                del slab, total

    result = {
        "threshold": thresholds,
        "period": periods,
        "site": sites,
        "every": every,
        "simulable": simulable,
        "subset": subset,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def curve_levels(
    thresholds: np.ndarray, curves: np.ndarray, target: float
) -> np.ndarray:
    """Read ``(n, threshold)`` hazard curves at ``target`` per year.

    Log-log interpolation between the bracketing thresholds. NaN where the
    target rate falls off the grid the curves were computed on, at either end,
    which is a real limit of the file rather than a gap to paper over: an
    empirical curve stops at its first and last threshold and nothing here
    knows its shape outside them. A simulation curve has no such limit -- it is
    built from the motions themselves -- which is why only the empirical side
    comes back with holes in it.

    Vectorised because the augmented panel reads one of these per station, and
    a Python loop over a hundred thousand of them is the difference between a
    map that takes a second and one that takes a minute.
    """
    rates = np.asarray(curves, dtype=np.float64)
    logs = np.log(thresholds)
    level = np.full(rates.shape[0], np.nan)

    # Curves fall as the threshold rises, so the crossing is the first column
    # at or below the target -- and there is only an answer if the curve starts
    # above the target and has somewhere to fall to.
    below = rates <= target
    crossing = np.argmax(below, axis=1)
    usable = below.any(axis=1) & (rates[:, 0] >= target) & (crossing > 0)
    if not usable.any():
        return level

    rows = np.flatnonzero(usable)
    upper = crossing[rows]
    lower = upper - 1
    low_rate = rates[rows, lower]
    high_rate = rates[rows, upper]
    # A curve that reaches zero carries no information about where between the
    # two thresholds it got there, so it is left unanswered rather than guessed.
    positive = (low_rate > 0) & (high_rate > 0)
    rows, upper, lower = rows[positive], upper[positive], lower[positive]
    low_rate, high_rate = low_rate[positive], high_rate[positive]

    span = np.log(high_rate) - np.log(low_rate)
    fraction = np.where(span == 0, 0.0, (np.log(target) - np.log(low_rate)) / span)
    level[rows] = np.exp(logs[lower] + fraction * (logs[upper] - logs[lower]))
    return level


def spread_to_stations(
    site_lon: np.ndarray,
    site_lat: np.ndarray,
    values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Carry per-site hazard curves onto the simulation's stations.

    The empirical model is held at a few hundred sites and the simulation at a
    hundred thousand, and the augmented panel has to add them, so one of them
    has to move. Interpolating the *coarse* field is the right direction --
    hazard varies smoothly at the scale those sites are spaced, while the
    simulation's fine structure is the thing worth keeping.

    In log space, because hazard curves span decades, and filled from the
    nearest site beyond the sites' convex hull, where a linear interpolant has
    nothing to work with.
    """
    aspect = np.cos(np.radians(site_lat.mean()))
    source = np.column_stack([site_lon * aspect, site_lat])
    target = np.column_stack([lon * aspect, lat])
    with np.errstate(divide="ignore"):
        logs = np.log(np.where(values > 0, values, np.nan))
    good = np.all(np.isfinite(logs), axis=1)
    if good.sum() < 4:
        raise typer.BadParameter("too few empirical sites resolve to interpolate")
    inside = LinearNDInterpolator(source[good], logs[good])(target)
    outside = NearestNDInterpolator(source[good], logs[good])(target)
    return np.exp(np.where(np.isfinite(inside), inside, outside))


def build_grid(
    lon: np.ndarray, lat: np.ndarray, coastline: shapely.MultiPolygon | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lon/lat grid every panel is drawn on, and the cells to blank.

    One grid for the whole figure rather than one per panel: the panels are
    meant to be read against each other, and two fields resampled onto two
    different grids differ by their grids as well as by themselves.

    Spacing follows the station spacing the way :func:`raster.rasterise`
    chooses it, so a hazard map has the same grain as the intensity measure
    maps drawn from the same stations.
    """
    aspect = np.cos(np.radians(lat.mean()))
    points = np.column_stack([lon * aspect, lat])
    spacing = float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))
    grid_lon = np.arange(lon.min(), lon.max() + spacing / aspect, spacing / aspect)
    grid_lat = np.arange(lat.min(), lat.max() + spacing, spacing)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    blank = (
        land_mask(mesh_lon, mesh_lat, coastline)
        if coastline is not None
        else np.zeros(mesh_lon.shape, dtype=bool)
    )
    return grid_lon, grid_lat, blank


def interpolate_field(
    triangulation: Delaunay,
    values: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    aspect: float,
    blank: np.ndarray,
) -> np.ma.MaskedArray:
    """One field onto the shared grid, through a triangulation built once.

    The dense panels share a hundred-thousand-point triangulation that costs
    more to build than to use, and the figure interpolates several fields
    through it, so it is passed in rather than rebuilt per field.
    """
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    flat = LinearNDInterpolator(triangulation, values)(
        np.column_stack([mesh_lon.ravel() * aspect, mesh_lat.ravel()])
    )
    return np.ma.masked_invalid(np.ma.masked_where(blank, flat.reshape(mesh_lon.shape)))


def im_label(im: str, period: float | None) -> str:
    """How an intensity measure is written on an axis."""
    if im == "pSA" and period is not None:
        return f"pSA({period:g} s)"
    return SUBSCRIPTS.get(im, im)


def return_period_text(years: float) -> str:
    """The return period, and the exceedance probability engineers read it as."""
    probability = 100 * (1 - np.exp(-50 / years))
    return f"{years:g}-year return period ({probability:.0f}% in 50 years)"


# pSA periods drawn when nothing else says which. A broadband set: the national
# model stops at 1 s, but the simulation does not -- it still resolves a median
# 0.0008 g at 20 s -- and long-period hazard is exactly what a physics-based
# simulation is worth running for. With --nshm the fifteen periods the national
# model is tabulated on are used instead, so that a comparison is a lookup
# rather than an interpolation.
STANDARD_PERIODS = (
    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0,
    1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0,
)


# `stations.draw_observed` sizes its triangles for one event's handful of
# recordings; several hundred of them on a map of the whole country have to be
# smaller than that or they cover the field they are drawn over.
SITE_MARKER = 14


# Inches across for one panel, and inches of everything that is not a panel:
# the title above, two colour bars below, and the caption under those.
PANEL_WIDTH = 3.2
FURNITURE = 2.3


# The two levels a New Zealand design decision is made at: 10% and 2% chance of
# being exceeded in fifty years.
DEFAULT_RETURN_PERIODS = (475.0, 2475.0)


@dataclass(frozen=True)
class Ground:
    """What every panel of every map in one run of the command shares.

    The grid, the triangulations and the empirical curves cost more to build
    than to use and none of them depend on the intensity measure, so they are
    built once and carried rather than rebuilt per figure.
    """

    con: duckdb.DuckDBPyConnection
    stations: dict[str, np.ndarray]
    dense: Delaunay
    grid_lon: np.ndarray
    grid_lat: np.ndarray
    blank: np.ndarray
    aspect: float
    coast: shapely.MultiPolygon | None
    outlines: list | None
    covered: float
    total_rate: float
    step: dict[str, float]
    empirical: dict[str, np.ndarray] | None = None
    site_at: np.ndarray | None = None  # station row for each empirical site
    site_lon: np.ndarray | None = None
    site_lat: np.ndarray | None = None
    site_tri: Delaunay | None = None

    @property
    def view(self) -> tuple[float, float, float, float]:
        return (
            float(self.grid_lon.min()), float(self.grid_lat.min()),
            float(self.grid_lon.max()), float(self.grid_lat.max()),
        )


@dataclass(frozen=True)
class Style:
    """How the figure is drawn, as opposed to what it is drawn from."""

    output_dir: Path
    levels: int
    residual_limit: float
    cmap: str | None
    dpi: int
    display_height: float | None
    viewing_distance: float | None


def rate_table(run_id: np.ndarray, rate: np.ndarray):
    """The run-to-rate map, as something the database can join against."""
    return pa.table({"run_id": pa.array(run_id, pa.int32()),
                     "rate": pa.array(rate, pa.float64())})


def draw_sites(ax, lon, lat, values, colormap, norm, display) -> object | None:
    """The empirical model's sites, as triangles over the field.

    The same shape and the same meaning as the recordings in ``map`` -- a
    triangle is a place where something other than the raster knows the answer
    -- and the same convention for one that does not: drawn hollow rather than
    left off, so a site that resolves nowhere still shows that it was asked.
    """
    known = np.isfinite(values)
    size = display.mark(SITE_MARKER)
    drawn = None
    if known.any():
        drawn = ax.scatter(
            lon[known], lat[known], c=values[known], cmap=colormap, norm=norm,
            marker="^", s=size, ec="black", lw=display.mark(0.3), zorder=6,
        )
    if (~known).any():
        ax.scatter(
            lon[~known], lat[~known], marker="^", s=size, fc="none",
            ec="black", lw=display.mark(0.3), zorder=6,
        )
    return drawn


def draw_panel(ax, panel, ground, colormap, norm, ratio_cmap, ratio_norm, display):
    """One source's hazard, with the empirical model's sites over it."""
    mesh = ax.pcolormesh(
        ground.grid_lon, ground.grid_lat, panel["field"],
        cmap=colormap, norm=norm, rasterized=True,
    )
    view = ground.view
    if ground.coast is not None:
        draw_coastline(ax, ground.coast, view, display)
    if ground.outlines:
        draw_basins(ax, ground.outlines, shapely.box(*view),
                    (view[2] - view[0]) / 400, display=display)

    sites = None
    marks = panel.get("sites")
    if marks is not None:
        # A panel with a reference colours its sites by the log ratio to it; a
        # panel that *is* a reference colours them by their own value on the
        # raster's own scale, so the triangles read as samples of the field
        # rather than as a second quantity.
        drawn = draw_sites(
            ax, marks["lon"], marks["lat"], marks["value"],
            ratio_cmap if marks["ratio"] else colormap,
            ratio_norm if marks["ratio"] else norm,
            display,
        )
        if marks["ratio"]:
            sites = drawn

    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_aspect(1 / np.cos(np.radians((view[1] + view[3]) / 2)))
    degrees = FuncFormatter(lambda v, _: f"{v:g}°")
    ax.xaxis.set_major_formatter(degrees)
    ax.yaxis.set_major_formatter(degrees)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=display.ticks(3)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=display.ticks(5)))
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(display.mark(0.6))
    ax.set_title(panel["title"], fontsize=9, linespacing=1.3)
    return mesh, sites


def empirical_panels(
    ground: Ground, period: float, target: float, like_for_like: bool
) -> tuple[list[dict], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """The national model's panels at one period, and the curves behind them.

    Returns the panels, the per-site levels for the whole model and for the
    simulated subset, and the two curve sets the augmented panel needs. An
    empty list means the model says nothing at this period, which is the normal
    case for PGA, PGV, the durations and most of the pSA grid.
    """
    empirical = ground.empirical
    if empirical is None:
        return [], None, None, None
    match = np.flatnonzero(np.abs(empirical["period"] - period) < 1e-9)
    if match.size == 0:
        return [], None, None, None
    column = int(match[0])

    thresholds = empirical["threshold"]
    every = empirical["every"][:, :, column].T  # (site, threshold)
    subset = empirical["subset"][:, :, column].T
    level_every = curve_levels(thresholds, every, target)
    level_subset = curve_levels(thresholds, subset, target)

    panels = [
        {
            "title": "NSHM2022\nevery rupture",
            "values": level_every,
            "at": "sites",
            "sites": {"lon": ground.site_lon, "lat": ground.site_lat,
                      "value": level_every, "ratio": False},
        }
    ]
    if like_for_like:
        panels.append(
            {
                "title": "NSHM2022\nthe simulated ruptures only",
                "values": level_subset,
                "at": "sites",
                "sites": {"lon": ground.site_lon, "lat": ground.site_lat,
                          "value": level_subset, "ratio": False},
            }
        )
    return panels, level_every, level_subset, (every, subset)


def augmented_curve(
    ground: Ground,
    curves: tuple[np.ndarray, np.ndarray],
    simulated: np.ndarray,
    update: str,
) -> np.ndarray:
    """The national model's curves with the simulation folded in, per station.

    ``substitute`` takes the simulated ruptures' empirical contribution out of
    the national curve and puts the simulated one in its place. Nothing else
    moves: every rupture that was not simulated keeps exactly the rate and the
    ground motion the national model gave it, so the update is a swap of one
    term in a sum and needs no weight to be chosen.

    ``ratio`` instead reads the simulated-to-empirical ratio as a correction
    that applies to the whole model, unsimulated ruptures included. That is a
    much stronger claim -- it says the pilot's disagreement generalises -- and
    it is the right one only once the simulated set is big enough to stand for
    the rest.

    Clipped at zero because the empirical pair is interpolated from a few
    hundred sites onto a hundred thousand, and interpolation does not respect
    the fact that a subset cannot exceed its whole.
    """
    every, subset = curves
    placed = np.isfinite(ground.site_lon)
    lon, lat = ground.stations["lon"], ground.stations["lat"]
    every_here = spread_to_stations(
        ground.site_lon[placed], ground.site_lat[placed], every[placed], lon, lat
    )
    subset_here = spread_to_stations(
        ground.site_lon[placed], ground.site_lat[placed], subset[placed], lon, lat
    )
    if update == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(subset_here > 0, simulated / subset_here, np.nan)
        return every_here * np.clip(factor, 1e-6, 1e6)
    return np.clip(every_here - subset_here, 0.0, None) + simulated


@dataclass
class Drawing:
    """One figure's worth of answers, computed but not yet drawn.

    Computing and drawing are separated because the colour scale is a property
    of the *set*: a run that draws fifteen periods at two return periods should
    put them all on one scale, and it cannot know what that scale is until the
    last of them has been worked out.
    """

    im: str
    period: float | None
    return_period: float
    panels: list[dict]

    @property
    def measured(self) -> np.ndarray:
        """Every value that has to fit on the measure's colour scale."""
        finite = [p["values"][np.isfinite(p["values"])] for p in self.panels]
        return np.concatenate(finite) if finite else np.zeros(0)


def compute_map(
    ground: Ground,
    im: str,
    period: float | None,
    period_index: int | None,
    curve: Curve,
    return_period: float,
    update: str,
    like_for_like: bool,
) -> Drawing | None:
    """Work out every panel of one figure, without drawing any of it."""
    label = im_label(im, period)
    target = 1.0 / return_period

    panels, level_every, level_subset, curves = empirical_panels(
        ground, period if period is not None else -1.0, target, like_for_like
    )
    thresholds = ground.empirical["threshold"] if curves is not None else None

    count = ground.stations["id"].size
    rows = np.searchsorted(ground.stations["id"], curve.station)
    simulated = np.full(count, np.nan)
    simulated[rows] = curve.level(target)
    if not np.isfinite(simulated).any():
        console_warn(
            f"no station reaches a {return_period:g}-year {label}; nothing to draw"
        )
        return None
    resolved = int(np.isfinite(simulated).sum())
    if resolved < count // 2:
        console_warn(
            f"{label}: the simulated rupture set reaches {return_period:g} years at "
            f"only {resolved} of {count} stations -- its rates sum to "
            f"{ground.covered:.3g}/yr, so it cannot express a rate below that"
        )

    simulation_sites = None
    if level_subset is not None:
        at_site = np.where(ground.site_at >= 0, simulated[ground.site_at.clip(0)], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            simulation_sites = {
                "lon": ground.site_lon, "lat": ground.site_lat,
                "value": np.log(at_site / level_subset), "ratio": True,
            }
    panels.append(
        {
            "title": "Pilot simulation"
            + ("\n▲ ln(simulated / empirical, same ruptures)" if simulation_sites else ""),
            "values": simulated,
            "at": "stations",
            "sites": simulation_sites,
        }
    )

    if curves is not None:
        # The query only returns stations the simulation actually reached, and
        # the empirical background covers every station, so the simulated
        # curves are placed into a full-length array first. A station the
        # simulation never reached keeps a zero simulated rate, which is what
        # it means: no simulated rupture shook it that hard.
        reached = np.zeros((count, thresholds.size))
        reached[rows] = curve.above(thresholds)
        augmented = curve_levels(
            thresholds, augmented_curve(ground, curves, reached, update), target
        )
        at_site = np.where(ground.site_at >= 0, augmented[ground.site_at.clip(0)], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            marks = {
                "lon": ground.site_lon, "lat": ground.site_lat,
                "value": np.log(at_site / level_every), "ratio": True,
            }
        panels.append(
            {
                "title": f"Augmented ({update})\n▲ ln(augmented / NSHM2022)",
                "values": augmented,
                "at": "stations",
                "sites": marks,
            }
        )
        for panel in panels:
            if panel["at"] != "sites":
                continue
            known = int(np.isfinite(panel["values"]).sum())
            if known < panel["values"].size // 2:
                console_warn(
                    f"{label}: \"{panel['title'].splitlines()[-1]}\" resolves at only "
                    f"{known} of {panel['values'].size} sites at "
                    f"{return_period:g} years -- the empirical curves are "
                    f"tabulated from {thresholds[0]:g} to {thresholds[-1]:g} g and "
                    "the answer falls outside that"
                )
    return Drawing(im=im, period=period, return_period=return_period, panels=panels)


def measure_scales(
    drawings: list[Drawing], count: int
) -> dict[str, np.ndarray]:
    """One set of colour levels per measure, over every map of it in the run.

    A scale read off a single map makes the set incomparable: the same colour
    means a different ground motion at every period and every return period, so
    a reader comparing two of them is comparing two colour bars. Pooling fixes
    the meaning of a colour for a whole measure at once.

    The range is the pooled 0.1st to 99.9th percentile rather than the extremes,
    which is wide enough that nothing a reader would look at is clipped, and
    narrow enough that one station sitting on a fault trace does not spend a
    decade of the scale on itself.
    """
    scales: dict[str, np.ndarray] = {}
    for im in dict.fromkeys(drawing.im for drawing in drawings):
        pooled = np.concatenate(
            [d.measured for d in drawings if d.im == im] or [np.zeros(0)]
        )
        log = im in LOG_SCALED
        if log:
            pooled = pooled[pooled > 0]
        if pooled.size == 0:
            continue
        low, high = np.percentile(pooled, [0.1, 99.9])
        scales[im] = discrete_norm(pooled, count, log, float(low), float(high))
    return scales


def draw_map(ground: Ground, style: Style, drawing: Drawing, boundaries: np.ndarray):
    """Draw one figure and write it out."""
    panels = drawing.panels
    im, period = drawing.im, drawing.period
    label = im_label(im, period)

    # Every panel onto the one shared grid, so that what differs between them
    # is the hazard and not the resampling.
    placed = np.isfinite(ground.site_lon) if ground.site_lon is not None else None
    for panel in panels:
        at_sites = panel["at"] == "sites"
        panel["field"] = interpolate_field(
            ground.site_tri if at_sites else ground.dense,
            panel["values"][placed] if at_sites else panel["values"],
            ground.grid_lon, ground.grid_lat, ground.aspect, ground.blank,
        )

    colormap = plt.get_cmap(style.cmap or "magma_r")
    norm = BoundaryNorm(boundaries, colormap.N, extend="both")
    ratio_levels = fixed_symmetric_norm(style.residual_limit, style.levels)
    ratio_cmap = plt.get_cmap("RdBu_r")
    ratio_norm = BoundaryNorm(ratio_levels, ratio_cmap.N, extend="both")

    # The country is taller than it is wide once latitude is accounted for, so
    # the canvas follows the panels rather than the other way round: a fixed
    # height would leave a hand's width of nothing above the maps.
    view = ground.view
    shape = ((view[3] - view[1]) / (view[2] - view[0])) / np.cos(
        np.radians((view[1] + view[3]) / 2)
    )
    # The heading is wrapped before the height is fixed, because a one-panel
    # figure is narrow enough that the return period runs off both ends of it,
    # and a second line of title needs a second line of room.
    width = PANEL_WIDTH * len(panels) + 0.4
    heading = textwrap.wrap(
        f"{label} at a {return_period_text(drawing.return_period)}",
        width=max(24, int(10.9 * width)),
    )
    design = (width, PANEL_WIDTH * shape + FURNITURE + 0.22 * (len(heading) - 1))
    display = Display.for_figure(
        design, style.dpi, style.display_height, style.viewing_distance
    )
    fig, axes = plt.subplots(
        1, len(panels), figsize=display.size, dpi=display.dpi,
        layout="constrained", squeeze=False,
    )
    mesh = sites = None
    for column, (ax, panel) in enumerate(zip(axes[0], panels)):
        drawn, marked = draw_panel(
            ax, panel, ground, colormap, norm, ratio_cmap, ratio_norm, display
        )
        # Every panel is the same piece of the country on the same grid, so one
        # set of latitudes serves them all; repeating it four times spends
        # width on nothing.
        if column:
            ax.tick_params(labelleft=False)
        mesh = drawn
        sites = marked or sites

    units = UNIT_LABEL.get(IM_UNITS.get(im, ""), IM_UNITS.get(im, ""))
    # Constrained layout stacks colour bars away from the axes in the order
    # they are made, so the ratio scale is made first to end up underneath: the
    # measure's own scale belongs against the maps, being what most of the
    # panels are actually showing, and the ratio reads as a gloss on it.
    if sites is not None:
        add_colorbar(fig, axes[0], sites, ratio_levels,
                     "▲  ln ratio to the panel's reference", display, len(panels))
    add_colorbar(fig, axes[0], mesh, boundaries,
                 f"{label} ({units})" if units else label, display, len(panels))

    fig.suptitle("\n".join(heading), fontsize=12, linespacing=1.3)
    if display.detailed:
        # fig.text sits outside the layout engine and would be drawn straight
        # through the colour bars, so the room it needs is taken off the
        # layout's rectangle first and the text placed inside what was taken.
        lines = textwrap.wrap(caption(ground, len(panels) > 1),
                              width=int(14 * display.size[0]))
        reserved = (0.16 * len(lines) + 0.06) / display.size[1]
        fig.get_layout_engine().set(rect=(0.004, reserved, 0.996, 0.996))
        fig.text(
            0.5, reserved / 2, "\n".join(lines), ha="center", va="center",
            fontsize=7, color="#4a4a4a", linespacing=1.4,
        )

    name = f"hazard_{im}" + (f"_{period:g}s" if period is not None else "")
    path = style.output_dir / f"{name}_{drawing.return_period:g}yr.png"
    fig.savefig(path, dpi=display.dpi)
    plt.close(fig)
    tqdm.write(f"wrote {path}")


# Roughly the width of a digit as a fraction of its point size, plus the gap
# two neighbouring tick labels need to read as two labels.
DIGIT_WIDTH = 0.6
TICK_GAP = 8.0


def style_ticks(boundaries: np.ndarray, display: Display, inches: float) -> list:
    """As many of a colour bar's boundaries as the bar can physically carry.

    ``Display.ticks`` thins for *enlarged text* and returns the count unchanged
    at natural size, which is the right answer for an axis -- matplotlib picks
    those tick positions itself -- and the wrong one here, where the ticks are
    the level boundaries and there can be twenty of them. A measure spanning
    four decades gets a boundary at every 1, 2 and 5, and their labels are the
    widest ones too, so they overlap into a smear exactly when the scale is
    most in need of reading. So the room is measured rather than assumed.
    """
    labels = [f"{b:g}" for b in boundaries]
    per_tick = max(len(text) for text in labels) * DIGIT_WIDTH * 7.0 + TICK_GAP
    room = max(2, int(inches * 72.0 / per_tick))
    room = max(3, min(room, display.ticks(len(boundaries))))

    # Thinned by a stride rather than spaced from the ends. A BoundaryNorm bar
    # gives every bin the same width, so evenly spaced *indices* are the only
    # way to get evenly spaced *labels*; picking a count and spreading it from
    # both ends leaves some neighbours one bin apart, which is narrower than a
    # label and puts them back on top of each other. A stride also keeps the
    # middle boundary of a symmetric scale whenever it keeps any, so a
    # diverging bar does not lose the zero it is read against.
    count = len(boundaries)
    stride = max(1, -(-(count - 1) // max(1, room - 1)))
    kept = list(range(0, count, stride))
    if kept[-1] != count - 1:
        # The last boundary carries the scale's limit, so it is always shown --
        # but not crowded against the one before it.
        if len(kept) > 1 and (count - 1) - kept[-1] < stride:
            kept.pop()
        kept.append(count - 1)
    return [boundaries[i] for i in kept]


def add_colorbar(fig, axes, mappable, boundaries, label, display, panels=1):
    """A horizontal bar under the whole row, ticked at its own levels.

    Sized against the row it spans rather than at a fixed fraction: half the
    width of four panels is a generous bar, and half the width of one is a
    stub with its tick labels on top of each other.
    """
    shrink = 0.92 if panels < 3 else 0.55
    bar = fig.colorbar(
        mappable, ax=list(axes), orientation="horizontal",
        shrink=shrink, pad=0.02, aspect=display.mark(26 if panels < 3 else 50),
    )
    bar.set_label(label, fontsize=9)
    shown = style_ticks(boundaries, display, display.size[0] * shrink)
    bar.set_ticks(shown)
    bar.set_ticklabels([f"{b:g}" for b in shown], fontsize=7)
    bar.ax.tick_params(length=display.mark(2.5), pad=1)
    bar.outline.set_linewidth(display.mark(0.5))
    return bar


def caption(ground: Ground, compared: bool) -> str:
    """What a reader has to know before believing the figure.

    Both sentences are about the same thing -- that the simulation panel is a
    sample and not an estimate -- said once about its rates and once about its
    scatter, because the two understate the hazard for quite different reasons
    and a reader who fixes one in their head will not think of the other.
    """
    share = 100 * ground.covered / ground.total_rate
    text = (
        f"The simulated rupture set carries {ground.covered:.3g}/yr, {share:.1f}% of "
        f"the rupture set's {ground.total_rate:.3g}/yr, and all of it crustal: its "
        "panel is the hazard from the ruptures that were run, not an estimate of "
        "the hazard."
    )
    if compared:
        text += (
            " It holds one realisation per rupture with no ground-motion "
            "variability, while the empirical model integrates over its own "
            "scatter, so part of every ratio drawn here is that difference "
            "rather than a difference of physics."
        )
    return text


def hazard_map(
    db: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Intensity measure database"),
    ],
    ruptures: Annotated[
        Path,
        typer.Option(
            "--ruptures", exists=True, dir_okay=False,
            help="Rupture set as parquet, carrying each rupture's annual rate",
        ),
    ],
    nshm: Annotated[
        Path | None,
        typer.Option(
            "--nshm", exists=True, dir_okay=False,
            help="NSHM2022 hazard file: rate contributions by threshold, rupture, "
            "site and period. Without it every figure is the simulation alone",
        ),
    ] = None,
    im: Annotated[
        list[str] | None,
        typer.Option("--im", help="Intensity measure; repeat. Default: all of them"),
    ] = None,
    period: Annotated[
        list[float] | None,
        typer.Option("--period", help="pSA period in seconds; repeat"),
    ] = None,
    all_periods: Annotated[
        bool, typer.Option("--all-periods", help="Every pSA period the database holds")
    ] = False,
    return_period: Annotated[
        list[float] | None,
        typer.Option(
            "--return-period",
            help="Years between exceedances; repeat. Every return period asked "
            "for is read off the same curves and drawn on the same colour "
            "scale, so the set is comparable. Default: 475 and 2475, the "
            "10%% and 2%% in 50 years design levels",
        ),
    ] = None,
    update: Annotated[
        str,
        typer.Option(
            "--update",
            help="How the simulation updates the national model in the last panel. "
            "'substitute' swaps the simulated ruptures' empirical contribution "
            "for the simulated one, which is exact and leaves every other "
            "rupture alone. 'ratio' carries the simulated-to-empirical ratio "
            "across to the unsimulated ruptures too: a stronger claim, and the "
            "one to make once the rupture set is big enough to support it",
        ),
    ] = "substitute",
    like_for_like: Annotated[
        bool,
        typer.Option(
            "--like-for-like/--no-like-for-like",
            help="Draw the empirical model restricted to the simulated ruptures as "
            "its own panel. It is the only reference the simulation can be "
            "judged against fairly, but its curves are short and may not reach "
            "the return period asked for",
        ),
    ] = True,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Folder to write the maps into")
    ] = Path("hazard_maps"),
    levels: Annotated[int, typer.Option(help="Approximate number of colour bins")] = 10,
    residual_limit: Annotated[
        float, typer.Option("--residual-limit", help="Log-ratio scale limit")
    ] = 1.5,
    cmap: Annotated[str | None, typer.Option(help="Matplotlib colormap name")] = None,
    basins: Annotated[
        bool, typer.Option("--basins/--no-basins", help="Draw the basin outlines")
    ] = False,
    basin_file: Annotated[
        Path | None,
        typer.Option("--basin-file", exists=True, dir_okay=False, help="Basin outlines"),
    ] = None,
    coastline: Annotated[
        Path | None,
        typer.Option("--coastline", exists=True, dir_okay=False, help="Coastline blob"),
    ] = None,
    memory_limit: Annotated[
        str,
        typer.Option(
            "--memory-limit",
            help="Ceiling on the database's working memory. The per-station sort is "
            "the largest thing this command does, and DuckDB will spill it to "
            "disk rather than exceed this",
        ),
    ] = "2GB",
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution")] = 200,
    display_height: Annotated[
        float | None, typer.Option("--display-height", help="Height in cm")
    ] = None,
    viewing_distance: Annotated[
        float | None, typer.Option("--viewing-distance", help="Metres")
    ] = None,
) -> None:
    """Draw probabilistic seismic hazard in plan view, one file per measure."""
    if update not in ("substitute", "ratio"):
        raise typer.BadParameter("--update is either 'substitute' or 'ratio'")
    periods_wanted = sorted(return_period or DEFAULT_RETURN_PERIODS)
    if any(years <= 0 for years in periods_wanted):
        raise typer.BadParameter("a return period is a number of years, so positive")

    con = connect(db)
    con.execute(f"SET memory_limit = '{memory_limit}'")
    rates, total_rate = rupture_rates(ruptures)
    run_id, rate = run_rates(con, rates)
    if run_id.size == 0:
        raise typer.BadParameter(f"no run in {db} names a rupture in {ruptures}")
    con.register("run_rate", rate_table(run_id, rate))
    covered = float(rate.sum())
    print(
        f"{run_id.size} simulated ruptures carrying {covered:.4g}/yr, "
        f"{100 * covered / total_rate:.1f}% of the set's {total_rate:.4g}/yr; "
        f"any one of them is a {1 / covered:.0f}-year event"
    )

    stations = station_coordinates(con)
    coast = load_coastline(coastline)
    grid_lon, grid_lat, blank = build_grid(stations["lon"], stations["lat"], coast)
    aspect = float(np.cos(np.radians(stations["lat"].mean())))
    print(
        f"{stations['id'].size} stations on a {grid_lat.size}x{grid_lon.size} grid "
        f"({int(blank.size - blank.sum())} cells on land)"
    )
    dense = Delaunay(np.column_stack([stations["lon"] * aspect, stations["lat"]]))

    empirical = site_at = site_lon = site_lat = site_tri = None
    if nshm is not None:
        events = np.array(
            [int(e) for (e,) in con.execute('SELECT "event" FROM runs').fetchall()]
        )
        empirical = nshm_hazard(nshm, events)
        index = {str(name): row for row, name in enumerate(stations["name"])}
        site_at = np.array([index.get(str(s), -1) for s in empirical["site"]])
        if (site_at < 0).any():
            console_warn(
                f"{int((site_at < 0).sum())} of {site_at.size} empirical sites are "
                "not stations in the database, so they carry no simulated value"
            )
        placed = site_at >= 0
        site_lon = np.where(placed, stations["lon"][site_at.clip(0)], np.nan)
        site_lat = np.where(placed, stations["lat"][site_at.clip(0)], np.nan)
        site_tri = Delaunay(
            np.column_stack([site_lon[placed] * aspect, site_lat[placed]])
        )
        print(
            f"{int(placed.sum())} empirical sites on "
            f"{empirical['period'].size} periods, "
            f"{empirical['threshold'][0]:g}-{empirical['threshold'][-1]:g} g"
        )

    outlines = None
    if basins:
        loaded = load_basins(basin_file)
        if loaded:
            view = (grid_lon.min(), grid_lat.min(), grid_lon.max(), grid_lat.max())
            outlines = basins_in_view(loaded, view)

    ground = Ground(
        con=con, stations=stations, dense=dense, grid_lon=grid_lon,
        grid_lat=grid_lat, blank=blank, aspect=aspect, coast=coast,
        outlines=outlines, covered=covered, total_rate=total_rate,
        step={
            "psa": quantisation_step(con, "psa_log_step"),
            "scalars": quantisation_step(con, "scalars_log_step"),
        },
        empirical=empirical, site_at=site_at, site_lon=site_lon,
        site_lat=site_lat, site_tri=site_tri,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    style = Style(
        output_dir=output_dir, levels=levels, residual_limit=residual_limit,
        cmap=cmap, dpi=dpi, display_height=display_height,
        viewing_distance=viewing_distance,
    )

    grid = con.execute("SELECT period, period_index FROM periods ORDER BY period").df()

    # The whole work list first, so the bar knows how far it has to go and can
    # name what it is on: these are minutes apiece, and a bar that only counts
    # is no help in telling a slow map from a stuck one.
    work: list[tuple[str, float | None, int | None]] = []
    for measure in list(im) if im else ["pSA", *SCALAR_IMS]:
        if measure != "pSA":
            work.append((measure, None, None))
            continue
        if all_periods:
            chosen = list(grid["period"])
        elif period:
            chosen = list(period)
        elif empirical is not None:
            chosen = list(empirical["period"])
        else:
            chosen = list(STANDARD_PERIODS)
        for wanted in chosen:
            row = int((grid["period"] - wanted).abs().idxmin())
            exact = float(grid["period"][row])
            if abs(exact - wanted) > 1e-9:
                console_warn(f"no pSA at {wanted:g} s; drawing {exact:g} s instead")
            work.append(("pSA", exact, int(grid["period_index"][row])))

    # Worked out first and drawn second. Every return period comes off one
    # curve, so asking for two costs one sort rather than two, and the colour
    # scale can be pooled over the whole set once the last one is known.
    drawings: list[Drawing] = []
    computing = tqdm(work, unit="measure", desc="computing")
    for measure, at_period, period_index in computing:
        spoken = measure + (f" {at_period:g} s" if at_period is not None else "")
        computing.set_description(f"computing {spoken:>12}")
        curve = simulated_curve(
            con, measure, period_index,
            DEFAULT_COMPONENT.get(measure, "geom"),
            ground.step["psa" if period_index is not None else "scalars"],
        )
        for years in periods_wanted:
            drawn = compute_map(ground, measure, at_period, period_index, curve,
                                years, update, like_for_like)
            if drawn is not None:
                drawings.append(drawn)
        del curve
    computing.close()
    con.close()

    scales = measure_scales(drawings, levels)
    for measure, boundaries in scales.items():
        tqdm.write(
            f"{measure}: colour scale {boundaries[0]:g} to {boundaries[-1]:g} "
            f"over {len(boundaries) - 1} bins, shared by "
            f"{sum(1 for d in drawings if d.im == measure)} maps"
        )

    drawing_bar = tqdm(drawings, unit="map", desc="drawing")
    for drawing in drawing_bar:
        spoken = drawing.im + (
            f" {drawing.period:g} s" if drawing.period is not None else ""
        )
        drawing_bar.set_description(f"drawing {spoken:>12} {drawing.return_period:g}yr")
        boundaries = scales.get(drawing.im)
        if boundaries is None:
            continue
        draw_map(ground, style, drawing, boundaries)
    drawing_bar.close()


# Where a deficiency is read. The share the pilot carries barely moves with
# ground motion -- 9.4% at 0.1 g against 11.0% at 2 g for pSA(1 s) -- so the
# choice is not critical, and the lowest tabulated threshold is the one every
# site resolves at.
DEFAULT_DEFICIT_THRESHOLD = 0.1


def deficit_shares(
    empirical: dict[str, np.ndarray], column: int, level: int
) -> list[dict]:
    """Split the national hazard three ways, as shares of itself.

    Deficiency is read as a **ratio of rates at a fixed ground motion**, not as
    a hazard level at a fixed rate, and that choice is what makes it answerable
    at all. A level at a fixed rate has to be found by searching the tabulated
    curve, which fails wherever the answer falls off either end of the table --
    the reason the like-for-like panel of ``hazard-map`` is mostly empty. A
    ratio at a tabulated threshold is defined at every site and every period,
    because both curves are tabulated there by construction.

    It also cancels the ground motion model. The same empirical model is on the
    top and the bottom of the ratio, so what is left is not physics against
    regression, but purely which ruptures were run and where they mattered.

    The three shares add to one at every site: what the pilot carries, what it
    missed by not simulating other crustal ruptures, and what no crustal
    simulation could reach.
    """
    every = empirical["every"][level, :, column]
    simulable = empirical["simulable"][level, :, column]
    pilot = empirical["subset"][level, :, column]
    with np.errstate(divide="ignore", invalid="ignore"):
        carried = np.where(every > 0, pilot / every, np.nan)
        crustal_gap = np.where(every > 0, (simulable - pilot) / every, np.nan)
        other_gap = np.where(every > 0, (every - simulable) / every, np.nan)
    return [
        {
            "title": "Carried by the pilot\nsimulated crustal ruptures",
            "values": 100 * carried,
            "at": "sites",
            "sites": None,
        },
        {
            "title": "Missing: crustal ruptures\nthat were not simulated",
            "values": 100 * crustal_gap,
            "at": "sites",
            "sites": None,
        },
        {
            "title": "Missing: not crustal\n(Hikurangi, Puysegur)",
            "values": 100 * other_gap,
            "at": "sites",
            "sites": None,
        },
    ]


def hazard_deficit(
    db: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Intensity measure database"),
    ],
    nshm: Annotated[
        Path,
        typer.Option(
            "--nshm", exists=True, dir_okay=False,
            help="NSHM2022 hazard file, which supplies both sides of the ratio",
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Ground motion the shares are read at, in the measure's units. "
            "Snapped to the nearest tabulated threshold",
        ),
    ] = DEFAULT_DEFICIT_THRESHOLD,
    period: Annotated[
        list[float] | None,
        typer.Option("--period", help="pSA period in seconds; repeat"),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Folder to write the maps into")
    ] = Path("hazard_maps"),
    levels: Annotated[int, typer.Option(help="Approximate number of colour bins")] = 10,
    coastline: Annotated[
        Path | None,
        typer.Option("--coastline", exists=True, dir_okay=False, help="Coastline blob"),
    ] = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution")] = 200,
    display_height: Annotated[
        float | None, typer.Option("--display-height", help="Height in cm")
    ] = None,
    viewing_distance: Annotated[
        float | None, typer.Option("--viewing-distance", help="Metres")
    ] = None,
) -> None:
    """Map where the simulated rupture set is missing hazard, and to what."""
    con = connect(db)
    events = np.array(
        [int(e) for (e,) in con.execute('SELECT "event" FROM runs').fetchall()]
    )
    empirical = nshm_hazard(nshm, events)
    level = int(np.abs(empirical["threshold"] - threshold).argmin())
    at = float(empirical["threshold"][level])
    if abs(at - threshold) > 1e-9:
        console_warn(f"no tabulated threshold at {threshold:g}; reading at {at:g}")

    stations = station_coordinates(con)
    index = {str(name): row for row, name in enumerate(stations["name"])}
    site_at = np.array([index.get(str(s), -1) for s in empirical["site"]])
    placed = site_at >= 0
    if not placed.any():
        raise typer.BadParameter("no empirical site is a station in the database")
    site_lon = stations["lon"][site_at[placed]]
    site_lat = stations["lat"][site_at[placed]]
    con.close()

    coast = load_coastline(coastline)
    grid_lon, grid_lat, blank = build_grid(site_lon, site_lat, coast)
    aspect = float(np.cos(np.radians(site_lat.mean())))
    triangulation = Delaunay(np.column_stack([site_lon * aspect, site_lat]))
    view = (grid_lon.min(), grid_lat.min(), grid_lon.max(), grid_lat.max())

    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(period) if period else list(empirical["period"])
    colormap = plt.get_cmap("magma_r")
    # A share of a whole, so the scale is the whole: fixed 0-100% rather than
    # read off the data, or the same colour would mean a different share on
    # every panel of every period.
    boundaries = np.linspace(0.0, 100.0, 11)
    norm = BoundaryNorm(boundaries, colormap.N)

    progress = tqdm(wanted, unit="map", desc="deficit")
    for asked in progress:
        match = np.flatnonzero(np.abs(empirical["period"] - asked) < 1e-9)
        if match.size == 0:
            console_warn(
                f"the national model has no {asked:g} s; it is tabulated at "
                f"{empirical['period'][0]:g}-{empirical['period'][-1]:g} s"
            )
            continue
        progress.set_description(f"deficit pSA {asked:g} s")
        panels = deficit_shares(empirical, int(match[0]), level)
        for panel in panels:
            panel["field"] = interpolate_field(
                triangulation, panel["values"][placed],
                grid_lon, grid_lat, aspect, blank,
            )

        width = PANEL_WIDTH * len(panels) + 0.4
        shape = ((view[3] - view[1]) / (view[2] - view[0])) / np.cos(
            np.radians((view[1] + view[3]) / 2)
        )
        design = (width, PANEL_WIDTH * shape + FURNITURE - 0.5)
        display = Display.for_figure(design, dpi, display_height, viewing_distance)
        fig, axes = plt.subplots(
            1, len(panels), figsize=display.size, dpi=display.dpi,
            layout="constrained", squeeze=False,
        )
        ground = Ground(
            con=None, stations=stations, dense=None, grid_lon=grid_lon,
            grid_lat=grid_lat, blank=blank, aspect=aspect, coast=coast,
            outlines=None, covered=0.0, total_rate=1.0, step={},
        )
        mesh = None
        for column, (ax, panel) in enumerate(zip(axes[0], panels)):
            mesh, _ = draw_panel(
                ax, panel, ground, colormap, norm, colormap, norm, display
            )
            if column:
                ax.tick_params(labelleft=False)
        add_colorbar(
            fig, axes[0], mesh, boundaries,
            f"share of the NSHM2022 exceedance rate at {at:g} g (%)",
            display, len(panels),
        )
        fig.suptitle(
            f"Where the pilot is missing hazard — pSA({asked:g} s) at {at:g} g",
            fontsize=12,
        )
        lines = textwrap.wrap(
            "Rates, not levels: each panel is a share of the national model's "
            "annual rate of exceeding this ground motion, so the three add to "
            "100% at every site. The same empirical model is on both sides of "
            "the ratio, so this measures which ruptures were run and where they "
            "mattered — not simulation against regression.",
            width=int(14 * display.size[0]),
        )
        reserved = (0.16 * len(lines) + 0.06) / display.size[1]
        fig.get_layout_engine().set(rect=(0.004, reserved, 0.996, 0.996))
        fig.text(0.5, reserved / 2, "\n".join(lines), ha="center", va="center",
                 fontsize=7, color="#4a4a4a", linespacing=1.4)

        path = output_dir / f"deficit_pSA_{asked:g}s_{at:g}g.png"
        fig.savefig(path, dpi=display.dpi)
        plt.close(fig)
        tqdm.write(f"wrote {path}")
    progress.close()
