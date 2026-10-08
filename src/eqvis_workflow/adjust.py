"""``adjust``: apply a spatial bias field to a simulation's intensity measures.

A non-ergodic adjustment model is a map, per period, of how much a ground motion
model runs low or high at a place -- the part of the residual that belongs to
the site and the path rather than to the earthquake. It is fitted once over a
whole catalogue and is then a correction any ergodic prediction can be given::

    eqvis adjust sw4/im.h5 spatial_bias_field.parquet -o sw4/im_adjusted.h5

What comes out is an IM file of the same shape, so everything else reads it
without knowing an adjustment happened, and the comparison the adjustment has
to earn is one command::

    eqvis bias sw4/im_adjusted.h5 --observed flatfiles.zip --diff sw4/im.h5
    eqvis map sw4/im_adjusted.h5 pSA --period 3 --diff sw4/im.h5

That second one is the adjustment itself, drawn: the log difference between the
adjusted run and the raw one is the field, resampled onto the simulation's own
stations.

**Which column, and why it matters.** These fields usually decompose into a
constant per period and a zero-mean map -- ``a_total = c(period) + a_smooth``.
The constant is an event term belonging to the earthquakes the model was fitted
on, and adding it to a *different* earthquake asserts that this event is biased
the same way the fitting set was on average, which is not something the field
knows. So ``--column`` defaults to the spatial part alone, and the command says
what it is applying and what the alternative would have done to the level.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr
from qcore import coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .console import console_warn
from .data import open_ims

# The grids these fields come on are a few kilometres across, so a station more
# than about two cells from the nearest one is not somewhere the model has an
# opinion about -- it is off the edge of the fitted region, or out at sea.
DEFAULT_MAX_DISTANCE = 15.0

# A period is the same period if the two grids agree to this. The same rule
# ``bias`` matches recordings onto a run's ordinates with.
PERIOD_TOLERANCE = 0.05

# A period whose whole map varies by less than this is not a spatial model --
# it is a plane of zeros where a fit did not take. Applied without comment it
# leaves a hole in the correction, which on a plot against period reads as a
# band where the adjustment genuinely does nothing rather than as a band where
# the model has nothing to say.
DEGENERATE_SPREAD = 0.01

# The broadband blend, as `workflow.scripts.bb_sim` builds it: an order-4
# Butterworth applied forward and backward, so effective order 8, with the
# corner shifted so that two passes give exactly 1/sqrt(2) at the frequency
# handed in. Mirrored here as a closed form rather than by running the filter;
# `tests/test_adjust.py` checks it against `scipy.signal` and fails if it
# drifts.
BLEND_ORDER = 4
BLEND_PASSES = 2

# Beyond the field's last period there is no map. Holding the last one and
# fading it out avoids a step in the corrected spectrum, but it is an
# extrapolation, and on the one dataset this has been measured against it made
# the fit worse where it was used (10-20 s, mean |bias| 0.190 raw against 0.223
# held). So the default is to stop at the field's own edge and take the step:
# between a discontinuity and a correction that is wrong, the discontinuity is
# the one that does not pretend. `--fade-octaves` turns the fade back on.
FADE_OCTAVES = 0.0


def blend_weight(
    periods: np.ndarray, corner: float, order: int = BLEND_ORDER,
    passes: int = BLEND_PASSES,
) -> np.ndarray:
    """How much of the ground motion at each period is the deterministic leg.

    A hybrid run is two solutions blended by complementary Butterworth filters,
    and only the low-frequency one is the deterministic solution a field like
    this is compatible with -- the stochastic leg carries whatever site
    amplification *it* was built with. So the correction is weighted by the
    low-frequency leg's share of the power at that period: one at long period,
    a half at the crossover, and zero well above it.

    This replaces a hard period cutoff. The cutoff was defensible but it put a
    step into the corrected spectrum, and there is nothing in the physics that
    steps -- the two legs trade off smoothly, and so should the correction.
    """
    shift = (np.sqrt(2.0) - 1.0) ** (1.0 / (2 * order))
    frequency = 1.0 / np.asarray(periods, dtype=float)
    with np.errstate(over="ignore"):
        high = (frequency / (corner * shift)) ** (2 * order)
        low = (frequency / (corner / shift)) ** (2 * order)
        # Two passes, in power: (|H|^passes)^2.
        hf = np.where(np.isfinite(high), (high / (1.0 + high)) ** passes, 1.0)
        lf = (1.0 / (1.0 + low)) ** passes
    total = lf + hf
    return np.where(total > 0, lf / total, 0.0)


def fade_weight(
    periods: np.ndarray, last: float, octaves: float = FADE_OCTAVES
) -> np.ndarray:
    """One up to the field's last period, fading to zero over ``octaves`` past it."""
    if octaves <= 0:
        return (periods <= last).astype(float)
    over = np.log2(np.maximum(np.asarray(periods, dtype=float), last) / last)
    return np.clip(1.0 - over / octaves, 0.0, 1.0)


def read_field(path: Path) -> pd.DataFrame:
    """The adjustment model as a tidy frame, whichever way it was written."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"period", "nztm_e", "nztm_n"} - set(frame.columns)
    if missing:
        raise typer.BadParameter(
            f"{path} is missing {sorted(missing)}; a bias field needs a period "
            "and an NZTM easting and northing per row"
        )
    return frame


def field_grid(
    frame: pd.DataFrame, column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The field as a dense ``(period, northing, easting)`` cube.

    These models are written out as one row per occupied cell, which is a
    fraction of the bounding grid -- the sea is simply absent. Densifying it
    puts NaN in the gaps, which is what makes a linear interpolation refuse to
    invent a value across the coastline rather than spanning it.
    """
    eastings = np.sort(frame["nztm_e"].unique())
    northings = np.sort(frame["nztm_n"].unique())
    periods = np.sort(frame["period"].unique())
    cube = np.full((periods.size, northings.size, eastings.size), np.nan)
    cube[
        np.searchsorted(periods, frame["period"].values),
        np.searchsorted(northings, frame["nztm_n"].values),
        np.searchsorted(eastings, frame["nztm_e"].values),
    ] = frame[column].values
    return periods, northings, eastings, cube


def sample_field(
    cube: np.ndarray,
    northings: np.ndarray,
    eastings: np.ndarray,
    slice_index: int,
    station_n: np.ndarray,
    station_e: np.ndarray,
    tree: cKDTree,
    occupied: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """One period of the field at the stations: linear inside, nearest at the edge.

    Linear between cells, because the field is smooth and a nearest-cell lookup
    would put the model's own grid into the output as visible blocks. A station
    whose surrounding cells are not all occupied -- anywhere along a coast --
    gets the nearest cell instead, which is the difference between an adjustment
    that stops at the shoreline and one that has a ragged hole in it.
    """
    plane = cube[slice_index]
    linear = RegularGridInterpolator(
        (northings, eastings), plane, method="linear",
        bounds_error=False, fill_value=np.nan,
    )(np.column_stack([station_n, station_e]))

    edge = ~np.isfinite(linear)
    if edge.any():
        distance, index = tree.query(np.column_stack([station_e[edge], station_n[edge]]))
        values = plane.ravel()[occupied[index]]
        linear[edge] = np.where(distance <= max_distance * 1000.0, values, np.nan)
    return linear


def adjust(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    field: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Spatial bias field: parquet or CSV of period, nztm_e, nztm_n "
            "and one or more adjustment columns",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Adjusted intensity measure file to write"),
    ],
    column: Annotated[
        str,
        typer.Option(
            "--column",
            help="Adjustment column to apply. The spatially varying part by "
            "default; a column carrying a per-period constant as well asserts "
            "this event is biased like the fitting set was on average",
        ),
    ] = "a_smooth",
    max_distance: Annotated[
        float,
        typer.Option(
            "--max-distance",
            help="How far (km) a station may be from the field before it is "
            "left unadjusted rather than extrapolated to",
        ),
    ] = DEFAULT_MAX_DISTANCE,
    blend_corner: Annotated[
        float | None,
        typer.Option(
            "--blend-corner",
            help="Hybrid blend corner in Hz. Given, the correction is tapered "
            "in by the deterministic leg's share of the power, instead of "
            "being applied whole at every period",
        ),
    ] = None,
    fade_octaves: Annotated[
        float,
        typer.Option(
            "--fade-octaves",
            help="Past the field's last period, hold its last map and fade the "
            "correction out over this many octaves. Off by default: it is an "
            "extrapolation, and it measured worse than stopping",
        ),
    ] = FADE_OCTAVES,
):
    """Apply a spatial bias field to a run's pSA, writing a new IM file.

    The adjustment is additive in log space -- ``pSA`` is multiplied by
    ``exp(a)`` -- and is applied to every pSA component in the file. Fields of
    this kind are fitted on one component, usually RotD50, so applying it to the
    others assumes the site and path terms it captures do not depend on the
    component convention; that is the usual assumption and it is worth knowing
    it is being made.

    Only pSA is adjusted, because that is the ordinate the field is tabulated
    on. Every other measure is copied through untouched, and the file records
    which ones those are rather than leaving a reader to assume the whole file
    was corrected.

    ``--blend-corner`` exists because a field like this is fitted on somebody
    else's runs, and a hybrid run is two solutions blended together. Where the
    stochastic high-frequency leg still contributes, the field carries whatever
    site amplification *that* leg used, and applying it to a run built with a
    different one adds a correction for a model this simulation never had. Given
    the blend's corner, the correction is weighted by the deterministic leg's
    share of the power at each period -- so it tapers in exactly as the leg it
    belongs to takes over, rather than switching on at a period somebody chose.
    """
    frame = read_field(field)
    if column not in frame.columns:
        options = [
            name
            for name in frame.columns
            if name not in ("period", "nztm_e", "nztm_n")
        ]
        raise typer.BadParameter(
            f"{field} has no column {column!r}. Available: {options}"
        )

    tree = open_ims(im_file)
    if "pSA" not in tree.children:
        raise typer.BadParameter(f"{im_file} has no pSA to adjust")
    node = tree["pSA"]
    periods = node["period"].values

    field_periods, northings, eastings, cube = field_grid(frame, column)
    flat = np.array(
        [float(np.nanstd(plane)) < DEGENERATE_SPREAD for plane in cube]
    )
    if flat.any():
        listed = ", ".join(f"{value:g}" for value in field_periods[flat][:12])
        console_warn(
            f"{int(flat.sum())} of {field_periods.size} periods in {field.name} "
            f"carry no spatial variation at all ({listed}"
            f"{', …' if int(flat.sum()) > 12 else ''}): the fit did not take "
            "there, and those periods are left unadjusted rather than "
            "corrected by a plane of zeros"
        )
    occupied_mask = np.isfinite(cube[0]).ravel()
    occupied = np.flatnonzero(occupied_mask)
    grid_e, grid_n = np.meshgrid(eastings, northings)
    lookup = cKDTree(np.column_stack([grid_e.ravel()[occupied], grid_n.ravel()[occupied]]))

    nztm = coordinates.wgs_depth_to_nztm(
        np.column_stack(
            [
                node["latitude"].values,
                node["longitude"].values,
                np.zeros(node["latitude"].size),
            ]
        )
    )
    station_n, station_e = nztm[:, 0], nztm[:, 1]

    # One column of the adjustment per simulated period, NaN where the field
    # has nothing to say -- either off its grid or off the end of its periods.
    adjustment = np.full((station_n.size, periods.size), np.nan)
    matched = np.zeros(periods.size, bool)
    longest = float(field_periods.max())
    reach = longest * 2.0**fade_octaves if fade_octaves > 0 else longest
    for index, period in enumerate(periods):
        nearest = int(np.abs(field_periods - period).argmin())
        within = abs(field_periods[nearest] - period) / period <= PERIOD_TOLERANCE
        # Past the field's last period the map is held rather than matched, and
        # faded out by the weight below; beyond the fade there is nothing.
        held = not within and longest < period <= reach
        if not (within or held):
            continue
        if held:
            nearest = int(np.argmax(field_periods))
        if flat[nearest]:
            continue
        matched[index] = True
        adjustment[:, index] = sample_field(
            cube, northings, eastings, nearest,
            station_n, station_e, lookup, occupied, max_distance,
        )

    if not matched.any():
        raise typer.BadParameter(
            f"{field} covers {field_periods.min():g}-{field_periods.max():g} s "
            f"and the run carries {periods.min():g}-{periods.max():g} s; "
            "no period is shared"
        )
    # Two tapers, both replacing an edge that would otherwise be a step: the
    # blend at the short end, and the field simply running out at the long one.
    weight = np.ones(periods.size)
    if blend_corner is not None:
        weight *= blend_weight(periods, blend_corner)
    weight *= fade_weight(periods, float(field_periods.max()), fade_octaves)
    adjustment *= weight[None, :]

    outside = periods[~matched]
    if outside.size:
        console_warn(
            f"{outside.size} of {periods.size} periods are outside the field "
            f"({outside.min():g}-{outside.max():g} s) and are left unadjusted"
        )
    uncovered = int((~np.isfinite(adjustment[:, matched])).all(axis=1).sum())
    if uncovered:
        console_warn(
            f"{uncovered} of {station_n.size} stations are further than "
            f"{max_distance:g} km from the field and are left unadjusted"
        )

    # An unadjusted cell is multiplied by one, not dropped: the output has to
    # keep the shape every other command expects to read.
    factor = np.exp(np.where(np.isfinite(adjustment), adjustment, 0.0))
    applied = {}
    for name, variable in node.dataset.data_vars.items():
        if "period" not in variable.dims:
            continue
        applied[name] = variable * xr.DataArray(
            factor if variable.dims == ("station", "period") else factor.T,
            dims=variable.dims,
            coords={dim: variable[dim] for dim in variable.dims if dim in variable.coords},
        )
    if not applied:
        raise typer.BadParameter(f"{im_file} pSA carries no component over period")

    adjusted = tree.copy()
    adjusted["pSA"] = node.dataset.assign(applied)
    untouched = sorted(set(tree.children) - {"pSA"})
    adjusted.attrs = dict(tree.attrs) | {
        "spatial_bias_field": str(field),
        "spatial_bias_column": column,
        "spatial_bias_unadjusted_measures": ",".join(untouched),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_netcdf(output, engine="h5netcdf")

    covered = int(np.isfinite(adjustment[:, matched]).all(axis=1).sum())
    median = float(np.nanmedian(adjustment[:, matched]))
    spread = float(np.nanstd(adjustment[:, matched]))
    print(
        f"applied {column} from {field.name} to pSA at "
        f"{covered}/{station_n.size} stations over "
        f"{int(matched.sum())}/{periods.size} periods"
    )
    print(
        f"  adjustment: median {median:+.3f}, s.d. {spread:.3f} in ln "
        f"(a factor of {np.exp(median):.3f} typical)"
    )
    if untouched:
        print(f"  carried through unadjusted: {', '.join(untouched)}")
    print(f"wrote {output}")
