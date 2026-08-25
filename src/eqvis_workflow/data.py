"""Reading an intensity measure file, and naming what comes out of it.

The IM file is a netCDF/HDF5 datatree: one group per intensity measure, each
holding one variable per component of motion and carrying the station
coordinates, the source distances and the site terms. Which components exist
varies by measure -- no rotd on the durations, eas only on FAS -- so choosing
one is a lookup against the file rather than a constant.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import shapely
import typer
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

from .console import console_warn
from .constants import SUBSCRIPTS, UNIT_LABEL


def run_names(names: list[str] | None, files: list[Path]) -> list[str]:
    """Display name for each simulation: ``--name`` values in order, else the path.

    Falling back to the whole path rather than the stem keeps two runs
    distinguishable when they are the same filename in different directories.
    """
    names = names or []
    if len(names) > len(files):
        raise typer.BadParameter(
            f"got {len(names)} --name values for {len(files)} simulation(s)"
        )
    return [
        names[index] if index < len(names) else str(path)
        for index, path in enumerate(files)
    ]


def comparison_labels(names: list[str] | None, defaults: tuple[str, str]) -> list[str]:
    """Names for the two series being compared, falling back to ``defaults``.

    Unlike :func:`run_names` these end up as short in-plot labels, so an unnamed
    run gets a generic word rather than its path.
    """
    names = names or []
    if len(names) > 2:
        raise typer.BadParameter(f"got {len(names)} --name values for 2 series")
    return [names[index] if index < len(names) else defaults[index] for index in (0, 1)]


def run_title(names: list[str], named: bool) -> str:
    """Title fragment naming the runs, superscripted to match the ln[1/2] label.

    A single run only earns a slot in the title when it was actually named --
    otherwise this would paste the input path over every plain plot.
    """
    if len(names) < 2:
        return names[0] if names and named else ""
    return " vs ".join(f"{name}$^{index}$" for index, name in enumerate(names, 1))


def open_ims(path: Path) -> xr.DataTree:
    return xr.open_datatree(path, engine="h5netcdf", mask_and_scale=False)


def select_empirical(
    tree: xr.DataTree, im: str, model: str, selection: dict[str, float]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean and total sigma of ln(IM) from an empirical model, matched to the IM.

    Empirical predictions hang off the measure they predict, one group per
    model, and are stored in log space: ``mean`` is the mean of ln(IM) and
    ``std_Total`` its total standard deviation. The same period or frequency
    already resolved for the simulation is taken here, so the two are
    comparable.
    """
    node = tree[im]
    if "empirical" not in node.children:
        available = [
            name for name in tree.children if "empirical" in tree[name].children
        ]
        raise typer.BadParameter(
            f"{im} carries no empirical predictions. Available for: {available}"
        )
    models = node["empirical"]
    if model not in models.children:
        raise typer.BadParameter(
            f"unknown empirical model {model!r} for {im}. "
            f"Available: {list(models.children)}"
        )
    group = models[model]
    mean, sigma = group["mean"], group["std_Total"]
    for dim, value in selection.items():
        mean = mean.sel({dim: value}, method="nearest")
        sigma = sigma.sel({dim: value}, method="nearest")
    return mean, sigma


def empirical_loess(
    distance: np.ndarray,
    mean: np.ndarray,
    log_x: bool,
    frac: float = 0.2,
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """LOESS fit of the empirical prediction against distance, and its spread.

    Fitted in log-log space -- log distance against the model's mean, which is
    already the mean of ln(IM) -- so the fit works in the space the plot is
    drawn in and the ground motion scaling is close to linear there.

    The spread returned is the standard deviation of the station predictions
    about the fitted curve, i.e. how much site conditions move a prediction at
    a given distance. That is deliberately not the model's own sigma, which is
    far larger and stays where it belongs, as the z-score denominator.

    ``delta`` collapses points that fall within 1% of the x range into a linear
    interpolation, which is what makes this tractable: without it a fit over
    tens of thousands of stations takes ~300x longer for the same curve.
    """
    keep = np.isfinite(distance) & np.isfinite(mean)
    if log_x:
        keep &= distance > 0
    if keep.sum() < 20:
        return None
    d, m = distance[keep], mean[keep]
    x = np.log(d) if log_x else d
    fitted = lowess(m, x, frac=frac, it=3, delta=0.01 * np.ptp(x), return_sorted=True)

    spread = float(np.std(m - np.interp(x, fitted[:, 0], fitted[:, 1])))
    grid = np.linspace(x.min(), x.max(), points)
    centre = np.interp(grid, fitted[:, 0], fitted[:, 1])
    return (np.exp(grid) if log_x else grid), centre, spread


def im_label(im: str, selection: dict[str, float]) -> str:
    label = SUBSCRIPTS.get(im, im)
    if "period" in selection:
        return f"{label}({selection['period']:g}s)"
    if "frequency" in selection:
        return f"{label}({selection['frequency']:g}Hz)"
    return label


def residual_label(name: str) -> str:
    """Axis label for the misfit against the recordings: ln(sim / obs).

    Simulation over observation, so a positive residual is the simulation
    running high -- the direction to read when the simulation is the subject.
    One function, so the convention is written once and the map's colour bar,
    the distance panels and the bias sweep cannot drift apart.
    """
    return f"ln[{name}$_{{sim}}$ / {name}$_{{obs}}$]"


def default_title(attrs: dict, runs: str) -> str:
    """Event, magnitude and run names: the title a map gets when none is given.

    Shared with the picker so that opening it, changing nothing and saving
    reproduces the title the command line would have drawn by itself.
    """
    parts = []
    if event := attrs.get("event"):
        parts.append(str(event))
    if magnitude := attrs.get("magnitude"):
        parts.append(f"M$_w$ {float(magnitude):.1f}")
    parts.append(runs)
    return "  ".join(filter(None, parts))


def select_im(
    tree: xr.DataTree,
    im: str,
    component: str,
    period: float | None,
    frequency: float | None,
) -> tuple[xr.DataArray, dict[str, float]]:
    """Select a single (station,) slice of an IM, resolving defaults.

    Each intensity measure is its own group, holding one variable per component
    of motion and carrying the station coordinates -- position, distances, site
    terms -- so the returned array is self-contained.
    """
    if im not in tree.children:
        raise typer.BadParameter(
            f"{im!r} not in file. Available: {list(tree.children)}"
        )
    node = tree[im]
    if component not in node.data_vars:
        raise typer.BadParameter(
            f"component {component!r} not in {im}. "
            f"Available: {[str(c) for c in node.data_vars]}"
        )
    da = node[component]
    selection: dict[str, float] = {}
    if "period" in da.dims:
        da = da.sel(period=period if period is not None else 1.0, method="nearest")
        selection["period"] = float(da.period)
        da = da.drop_vars("period")
    if "frequency" in da.dims:
        da = da.sel(
            frequency=frequency if frequency is not None else 1.0, method="nearest"
        )
        selection["frequency"] = float(da.frequency)
        da = da.drop_vars("frequency")
    return da, selection


# The SW4 supergrid absorbing layer, as it arrives on an IM file: the
# penetration of each station into the layer, in metres and in grid points.
# Written by SW4 as SGDEPTH/SGDEPTHGP and renamed by the workflow's LF reader,
# so these two names are the whole contract between the solver and every
# command here.
SUPERGRID_DEPTH = "supergrid_depth"
SUPERGRID_DEPTH_GP = "supergrid_depth_gp"


class Screen(str, Enum):
    """What a command does with a station inside the supergrid absorbing layer.

    ``exclude`` drops it, ``mark`` keeps it but draws and reports it as
    suppressed, ``keep`` treats it as any other station. The three values mean
    the same thing in every command that offers ``--supergrid``; only which one
    is the default differs, and it differs for one reason -- a command that
    prints a number defaults to ``exclude``, because a printed statistic is a
    claim, while a command that draws a picture defaults to ``mark``.
    """

    exclude = "exclude"
    mark = "mark"
    keep = "keep"


@dataclass(frozen=True)
class Supergrid:
    """How far into the absorbing layer each station of a run sits.

    Inside the SW4 supergrid layer the solver deliberately integrates a damped,
    coordinate-stretched equation rather than the wave equation, so a trace from
    in there is not a ground motion prediction at all. The depth is carried
    rather than a boolean because the severity scales with it.

    Three states, and keeping them apart is the whole point:

    ``0.0``
        interior: a clean station, and positively stated to be clean.
    ``> 0``
        inside the layer, by that many metres.
    ``NaN``
        unknown -- the run did not report it. Not the same as clean.

    ``stated`` is deliberately separate from the per-station arrays, because
    "this file does not say" is a fact about a *file* while "this station is
    fine" is a claim about a *station*, and a solver with no absorbing layer at
    all makes the first statement and not the second.
    """

    depth: np.ndarray
    gridpoints: np.ndarray
    stated: bool

    @property
    def flagged(self) -> np.ndarray:
        """Stations inside the layer, whose traces are not ground motion.

        ``depth > 0`` is ``False`` for NaN, and that single expression is the
        entire backward-compatibility story: a file that carries no flag has an
        all-NaN depth, flags nothing, and every command behaves exactly as it
        did before the flag existed.
        """
        return self.depth > 0

    @property
    def clean(self) -> np.ndarray:
        """Stations the run positively reports as interior."""
        return self.depth == 0

    @property
    def unknown(self) -> np.ndarray:
        """Stations the run said nothing about."""
        return ~np.isfinite(self.depth)


def unfilled(da: xr.DataArray) -> np.ndarray:
    """``da``'s values as float, with any fill value it declares turned to NaN.

    :func:`open_ims` passes ``mask_and_scale=False`` so that the file is read as
    it was written and nothing silently rescales -- which means xarray does
    *not* apply ``_FillValue`` here, and the fill arrives as a number. For a
    depth into the absorbing layer that is a trap either way round and a silent
    one both times: netCDF's default float fill, 9.969e36, is greater than zero
    and would flag every station in the file, while a hand-rolled -9999 is not
    and would report every missing station as positively clean. So the fill is
    honoured here instead of relying on the reader to do it.
    """
    values = np.asarray(da.values, dtype=np.float64)
    for key in ("_FillValue", "missing_value"):
        if key not in da.attrs:
            continue
        fills = np.atleast_1d(np.asarray(da.attrs[key], dtype=np.float64)).ravel()
        for fill in fills:
            # A NaN fill is already NaN; comparing against it never matches.
            if not np.isfinite(fill):
                continue
            # atol=0 so that a real depth of exactly zero -- a station the run
            # positively calls interior -- can never match a nonzero fill and
            # be turned into "unknown".
            values = np.where(
                np.isclose(values, fill, rtol=1e-6, atol=0.0), np.nan, values
            )
    return values


def supergrid_terms(
    source: xr.Dataset | xr.DataArray, stations: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """The supergrid coordinates ``source`` carries, if it carries any.

    Read with ``in .coords`` and matched to ``stations`` by name, exactly as the
    site terms and the source distances are: an optional per-station term is a
    coordinate on this file, and coordinates ride the pipeline from the solver
    to here for free while data variables do not.
    """
    terms: dict[str, np.ndarray] = {}
    for name in (SUPERGRID_DEPTH, SUPERGRID_DEPTH_GP):
        if name not in source.coords:
            continue
        da = source[name]
        if stations is not None and not np.array_equal(da.station.values, stations):
            # Matched on the station name, so a station the donor group has
            # never heard of comes back unknown rather than taking a
            # neighbour's depth.
            da = da.drop_vars(
                [other for other in da.coords if other != "station"]
            ).reindex(station=stations)
        terms[name] = unfilled(da)
    return terms


def supergrid(tree: xr.DataTree, da: xr.DataArray | None = None) -> Supergrid:
    """What ``tree`` says about the absorbing layer, aligned to ``da``.

    Resolved from the array's own coordinates first, and only then from any
    group of the file that carries them. That fallback is not defensive
    tidiness: ``bias`` reads pSA and FAS and never touches PGA, while ``ingest``
    anchors on PGA alone, so a flag that landed on one group and not the other
    would be invisible to the command that most needs it, or would stage as
    all-NULL, and in both cases silently. Looking in the other groups costs one
    pass over the file's metadata and removes that whole class of failure.

    Where two groups disagree the first is used and the disagreement is
    reported once, rather than per group: it means the file was assembled from
    two runs, which is one fact, not one fact per intensity measure.
    """
    stations = None
    if da is not None and "station" in da.coords:
        stations = da.station.values

    found = supergrid_terms(da) if da is not None else {}
    if not found:
        gathered = []
        for name in tree.children:
            node = tree[name]
            if "station" not in node.dims:
                continue
            if terms := supergrid_terms(node.dataset, stations):
                gathered.append((name, terms))
        if gathered:
            first, found = gathered[0]
            for name, terms in gathered[1:]:
                keys = set(found) | set(terms)
                if any(not _same_term(found.get(key), terms.get(key)) for key in keys):
                    console_warn(
                        f"{first} and {name} disagree about the supergrid "
                        f"layer; using {first}'s"
                    )
                    break

    depth = found.get(SUPERGRID_DEPTH)
    gridpoints = found.get(SUPERGRID_DEPTH_GP)
    if depth is None and gridpoints is not None:
        # SW4 writes both or neither, and the workflow reader reads them under
        # one guard, so grid points alone is a corrupt or hand-made file. The
        # threshold is stated in metres, and grid points cannot be converted to
        # metres without the grid spacing, so the honest answer is "unknown".
        console_warn(
            f"{SUPERGRID_DEPTH_GP} is present without {SUPERGRID_DEPTH}; "
            "the supergrid layer cannot be read from grid points alone"
        )
    size = _station_count(tree, da, found)
    absent = np.full(size, np.nan)
    return Supergrid(
        depth=absent if depth is None else depth,
        gridpoints=absent if gridpoints is None else gridpoints,
        stated=bool(found),
    )


def _same_term(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    """Whether two groups state the same thing, counting NaN as agreement."""
    if left is None or right is None:
        return left is right
    return np.array_equal(left, right, equal_nan=True)


def _station_count(
    tree: xr.DataTree, da: xr.DataArray | None, found: dict[str, np.ndarray]
) -> int:
    """How many stations a :class:`Supergrid` has to cover.

    Needed because an absent flag still has to come back as an array of the
    right length: the whole design is that a file without the flag takes the
    same code path as one with it.
    """
    if da is not None and "station" in da.dims:
        return int(da.sizes["station"])
    for values in found.values():
        return int(values.size)
    for name in tree.children:
        if "station" in tree[name].dims:
            return int(tree[name].sizes["station"])
    return 0


def supergrid_note(sg: Supergrid) -> str | None:
    """One line saying what a run reports about its absorbing layer, or nothing.

    ``None`` when the run said nothing at all, so that a file from a solver with
    no absorbing layer does not gain a line claiming it has a clean one.
    """
    if not sg.stated:
        return None
    total = int(sg.depth.size)
    flagged = int(sg.flagged.sum())
    unknown = int(sg.unknown.sum())
    trailing = f", {unknown} not reported" if unknown else ""
    if not flagged:
        return f"no stations inside the absorbing layer, of {total}{trailing}"
    deepest = float(np.nanmax(sg.depth))
    return (
        f"{flagged} of {total} stations inside the absorbing layer, up to "
        f"{deepest:.0f} m in -- not ground motion{trailing}"
    )


def in_domain(
    lon: np.ndarray,
    lat: np.ndarray,
    attrs: dict,
    bound_lon: np.ndarray,
    bound_lat: np.ndarray,
) -> np.ndarray:
    """Boolean mask for stations inside the simulation domain.

    Uses the domain polygon when the file carries one, and the extent of
    ``bound_lon``/``bound_lat`` -- the raster grid, or the simulation stations
    themselves -- otherwise.
    """
    if domain := attrs.get("domain"):
        polygon = shapely.from_wkt(domain)
        # A domain that straddles the antimeridian stores vertices either
        # side of the +/-180 cut (e.g. 177 next to -179), which is a ~357
        # degree span read literally -- containment against it is inverted,
        # true almost everywhere except the small local area the domain
        # actually covers. Detected by that oversized span and fixed by
        # unwrapping both the polygon and the query points onto one
        # continuous strip (west side shifted past 180) before testing.
        poly_lon = np.asarray(polygon.exterior.coords)[:, 0]
        if poly_lon.max() - poly_lon.min() > 180:
            polygon = shapely.transform(
                polygon,
                lambda coords: np.column_stack(
                    [np.where(coords[:, 0] < 0, coords[:, 0] + 360, coords[:, 0]), coords[:, 1]]
                ),
            )
            lon = np.where(lon < 0, lon + 360, lon)
        return shapely.contains_xy(polygon, lon, lat)
    return (
        (lon >= bound_lon.min())
        & (lon <= bound_lon.max())
        & (lat >= bound_lat.min())
        & (lat <= bound_lat.max())
    )


def restrict_to_domain(
    observed: dict[str, np.ndarray],
    attrs: dict,
    bound_lon: np.ndarray,
    bound_lat: np.ndarray,
    source: Path,
) -> dict[str, np.ndarray]:
    """Drop observed stations that fall outside the simulation domain."""
    inside = in_domain(observed["lon"], observed["lat"], attrs, bound_lon, bound_lat)
    if not inside.any():
        console_warn(f"no {source} stations inside the domain")
    return {key: value[inside] for key, value in observed.items()}


def print_info(path: Path) -> None:
    tree = open_ims(path)
    print(f"{path}")
    if event := tree.attrs.get("event"):
        print(f"  event:      {event}")
    if magnitude := tree.attrs.get("magnitude"):
        print(f"  magnitude:  {float(magnitude):.2f}")
    if "hypo_lat" in tree.attrs:
        print(
            f"  hypocentre: {float(tree.attrs['hypo_lat']):.3f}, "
            f"{float(tree.attrs['hypo_lon']):.3f}"
        )
    for name in tree.children:
        print(f"  stations:   {tree[name].sizes['station']}")
        break
    # Whether any station sat in the absorbing layer decides whether the rest
    # of this file means anything, so it is said here rather than only where a
    # command happens to act on it: `--info` is the one-file screen.
    if note := supergrid_note(supergrid(tree)):
        print(f"  supergrid:  {note}")
    print("  intensity measures:")
    for name in tree.children:
        node = tree[name]
        # Components vary by measure: no rotd for the durations, eas only on FAS.
        components = [str(c) for c in node.data_vars]
        units = ""
        if components:
            raw = node[components[0]].attrs.get("units", "")
            units = UNIT_LABEL.get(raw, raw)
        extra = ""
        if "period" in node.dims:
            periods = node.period.values
            extra = f", {len(periods)} periods {periods.min():g}-{periods.max():g}s"
        if "frequency" in node.dims:
            freqs = node.frequency.values
            extra = f", {len(freqs)} frequencies {freqs.min():g}-{freqs.max():g}Hz"
        description = node.attrs.get("description", "")
        print(f"    {name:<6} {description} ({units}{extra})")
        print(f"           components: {' '.join(components)}")
