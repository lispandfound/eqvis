"""The composite database as a data source for the figures.

Everything that draws in this package reads an HDF5 datatree through
:mod:`~.data` and a flatfile archive through :mod:`~.flatfile`. Neither can
answer a question across runs, which is what the composite database exists for
-- so this module is the other half of those two: the same shapes, read out of
SQL instead of out of a file. A command that reads the database says
``from .store import select_im`` and one that reads a file says
``from .data import select_im``, so one grep says which source a figure came
from.

The other half of the module is the run *vocabulary*, and it is the reason the
analysis commands are reusable at all. What distinguishes one run of an event
from another -- solver, layers, mesh, whatever the ingest's ``--extract`` regex
captured -- lives in ``run_labels`` as ``(run_id, name, value)`` rows rather
than as columns, because those dimensions belong to the caller's tree and not to
the schema. So nothing here names a dimension: the names come out of
:func:`dimensions`, the values out of the data, and both reach SQL as bind
parameters. Point these commands at a database ingested along different axes and
they group by *its* dimensions with no change here.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import duckdb
import numpy as np
import typer
import xarray as xr

from .console import console_warn
from .constants import GRID_COMPONENT, SCALAR_IMS, SPECTRAL_AXIS
from .database import sql_literal

# The one dimension the schema needs by name, because it is what pairs a
# simulation with the recordings of the same earthquake -- so it is a column on
# `runs` rather than a row in `run_labels`. A caller grouping runs has no reason
# to care which of the two shapes a dimension is stored in, so everything below
# accepts `event` alongside whatever the regex named. This is the only dimension
# name written down anywhere in this package's analysis code.
EVENT = "event"

# The tables a reader here needs, checked at connect rather than at the first
# query so that a database built by an older schema says so up front.
REQUIRED_TABLES = ("runs", "run_labels", "stations", "run_stations", "scalars", "psa")

# Where a covariate may come from, finest grain first. A name in two tables
# resolves to the finest that has it: a coordinate that is per-run must not be
# read as though it were per-station.
COVARIATE_TABLES = ("run_stations", "stations", "runs")

# Columns of those tables that are keys or bookkeeping rather than covariates.
NOT_COVARIATES = frozenset(
    {
        "run_id", "station", "run_key", "event", "kind", "im_file",
        "file_size", "file_mtime_ns", "ingested_at", "observed_component",
        "n_stations", "n_observation_sites",
        "source_wkt", "trace_wkt", "domain_wkt",
        "x", "y", "is_observation_site",
    }
)


def connect(path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a composite IM database, refusing one that is not one.

    Read-only by default because every analysis command is a reader, and a
    read-only handle can be opened while something else holds the file.

    The tables are checked here rather than at the first query, so a path that
    happens to be some other DuckDB file, or a database built before a schema
    change, fails with a message naming what it did find instead of with a
    binder error from the middle of a join.
    """
    con = duckdb.connect(str(path), read_only=read_only)
    present = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    missing = [name for name in REQUIRED_TABLES if name not in present]
    if missing:
        raise typer.BadParameter(
            f"{path} is missing the {', '.join(missing)} table"
            f"{'s' if len(missing) > 1 else ''}, so it is not a composite "
            f"intensity measure database. It holds: "
            f"{', '.join(sorted(present)) or '(nothing)'}"
        )
    return con


def dimensions(con) -> dict[str, list[str]]:
    """Every run dimension in the database, and the values each takes.

    Read out of ``run_labels``, never assumed, with ``event`` spliced in from
    ``runs``. This is what makes the analysis commands general: a tree ingested
    with a different ``--extract`` regex describes itself here, and every
    command takes its option values from this rather than from a constant.
    """
    found = {
        str(name): [str(value) for value in values]
        for name, values in con.execute(
            """
            SELECT name, list(DISTINCT value ORDER BY value)
            FROM run_labels GROUP BY name ORDER BY name
            """
        ).fetchall()
    }
    events = [
        str(row[0])
        for row in con.execute(
            "SELECT DISTINCT event FROM runs ORDER BY event"
        ).fetchall()
    ]
    return {EVENT: events, **found}


def parse_label(text: str) -> tuple[str, str]:
    """A ``name=value`` option value, refusing one that is not.

    Split on the first ``=`` only: a value may contain one, a name may not --
    and no regex group name could anyway.
    """
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise typer.BadParameter(
            f"--label takes name=value, not {text!r}, where the name is a "
            "dimension the ingest's --extract regex captured. Run "
            "`eqvis runs DB` to see which this database has -- naming one here "
            "as an example would be a guess about someone else's tree"
        )
    return name, value


def check_names(con, names: Iterable[str], option: str) -> None:
    """Refuse a dimension the database does not have, listing the ones it does."""
    available = dimensions(con)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise typer.BadParameter(
            f"{option} names {', '.join(repr(n) for n in unknown)}, which this "
            f"database has no dimension for. It has: "
            f"{', '.join(f'{k} ({len(v)} values)' for k, v in available.items())}"
        )


def selection_sql(labels: Sequence[tuple[str, str]]) -> tuple[str, list]:
    """The WHERE fragment and bind parameters for a label selection.

    A dimension named more than once is a **disjunction** over its values; two
    different dimensions **intersect**. So ``solver=emod3d solver=sw4`` reads as
    "either solver" and ``solver=sw4 layers=full`` as "both". Intersecting the
    repeats instead would return nothing for exactly the invocation someone
    comparing two cells of one dimension writes first.

    Names and values are bound, never interpolated, so no dimension name or
    value ever appears as SQL text.
    """
    wanted: dict[str, list[str]] = {}
    for name, value in labels:
        wanted.setdefault(name, []).append(value)

    clauses, parameters = [], []
    for name, values in wanted.items():
        placeholders = ", ".join("?" for _ in values)
        if name == EVENT:
            clauses.append(f"r.event IN ({placeholders})")
            parameters.extend(values)
        else:
            clauses.append(
                f"EXISTS (SELECT 1 FROM run_labels l WHERE l.run_id = r.run_id "
                f"AND l.name = ? AND l.value IN ({placeholders}))"
            )
            parameters.append(name)
            parameters.extend(values)
    return "".join(f" AND {clause}" for clause in clauses), parameters


def coverage(con) -> list[tuple[str, int, str]]:
    """Per event, how many simulated runs there are and what they are.

    The factorial is not assumed to be full -- the schema's own ``coverage``
    note says so -- and a comparison that averaged over a different denominator
    per cell without saying would be wrong in a way no figure would reveal.
    """
    return con.execute(
        """
        SELECT r.event, count(*) AS n,
               string_agg(d.dimensions, ', ' ORDER BY d.dimensions) AS present
        FROM runs r
        JOIN (
            SELECT run_id, string_agg(value, '/' ORDER BY name) AS dimensions
            FROM run_labels GROUP BY run_id
        ) d USING (run_id)
        WHERE r.kind = 'simulated'
        GROUP BY r.event ORDER BY r.event
        """
    ).fetchall()


def select_runs(
    con,
    labels: Sequence[tuple[str, str]] = (),
    kind: str = "simulated",
    pairable: bool = True,
) -> list[dict]:
    """Every run matching a conjunction of dimension constraints.

    ``pairable`` keeps only runs whose event has recordings, which is what every
    residual figure needs and is not the same as every simulated run: a
    simulated event with no recordings joins to nothing, and would otherwise
    turn up as an empty panel rather than as a message.

    Each run comes back as ``{"run_id", "run_key", "event", "labels"}``, the
    last a dict of that run's own dimensions. ``run_key`` is the stable natural
    key -- ``run_id`` is renumbered by a rebuild -- so it is what the panels and
    the provenance strings carry.
    """
    check_names(con, {name for name, _ in labels}, "--label")
    where, parameters = selection_sql(labels)
    if pairable:
        where += (
            " AND EXISTS (SELECT 1 FROM runs o WHERE o.kind = 'observed'"
            " AND o.event = r.event)"
        )
    rows = con.execute(
        f"""
        SELECT r.run_id, r.run_key, r.event
        FROM runs r WHERE r.kind = ?{where}
        ORDER BY r.event, r.run_key
        """,
        [kind, *parameters],
    ).fetchall()

    if not rows:
        raise typer.BadParameter(
            "no run matches that selection. The database holds:\n"
            + "\n".join(f"  {event:<14} {n}  {present}" for event, n, present in coverage(con))
        )

    labelled = {
        run_id: {} for run_id, _, _ in rows
    }
    for run_id, name, value in con.execute(
        "SELECT run_id, name, value FROM run_labels ORDER BY run_id, name"
    ).fetchall():
        if run_id in labelled:
            labelled[run_id][str(name)] = str(value)
    return [
        {
            "run_id": run_id,
            "run_key": str(run_key),
            "event": str(event),
            "labels": {EVENT: str(event), **labelled[run_id]},
        }
        for run_id, run_key, event in rows
    ]


def varying(runs: Sequence[dict]) -> list[str]:
    """The dimensions that actually differ across a run set, in name order.

    The default for ``--group-by``: grouping by a dimension every run shares
    gives one cell and says nothing, and naming the dimensions by hand is the
    thing this package is trying not to make anyone do.
    """
    seen: dict[str, set[str]] = {}
    for run in runs:
        for name, value in run["labels"].items():
            seen.setdefault(name, set()).add(value)
    return sorted(name for name, values in seen.items() if len(values) > 1)


def group_runs(
    con, runs: Sequence[dict], by: Sequence[str]
) -> list[tuple[tuple[str, ...], list[dict]]]:
    """Runs bucketed into the cells of ``by``, in sorted key order.

    Sorted rather than in discovery order, so a re-run gives the same cells the
    same colours and the same panel positions. A figure that renumbered its
    series between runs could not be compared against the one already in the
    report.
    """
    check_names(con, by, "--group-by")
    cells: dict[tuple[str, ...], list[dict]] = {}
    for run in runs:
        key = tuple(run["labels"].get(name, "") for name in by)
        cells.setdefault(key, []).append(run)
    return sorted(cells.items())


def cell_label(by: Sequence[str], key: Sequence[str]) -> str:
    """How a cell is named in a legend.

    Its values alone when one dimension varies, ``name=value`` when several do
    and the bare values would not say which was which.
    """
    if len(by) == 1:
        return str(key[0])
    return " ".join(f"{name}={value}" for name, value in zip(by, key))


def available_covariates(con) -> dict[str, str]:
    """Every station- or run-level column a caller may condition on.

    Discovered by describing the tables rather than listed here, so the columns
    a later schema adds become options with no code change -- the same argument
    ``run_labels`` makes for the run dimensions, applied to the site and source
    terms. Returns ``{column: table}``.
    """
    found: dict[str, str] = {}
    for table in COVARIATE_TABLES:
        for row in con.execute(f"DESCRIBE {table}").fetchall():
            column = str(row[0])
            if column in NOT_COVARIATES:
                continue
            if column in found:
                console_warn(
                    f"{column!r} is in both {found[column]} and {table}; reading "
                    f"it from {found[column]}, the finer grain"
                )
                continue
            found[column] = table
    return found


def measure_source(im: str) -> tuple[str, str, str | None]:
    """Which table a measure lives in, its value column, and its ordinate axis.

    ``FAS`` is refused rather than served. The simulated and observed Fourier
    grids share no value at all -- 100 points to 100 Hz against 240 to 24.5 Hz --
    so a Fourier residual out of this database is an empty join. Bridging them is
    log-log interpolation, which the schema deliberately leaves outside itself
    (its own ``frequency grids`` note says so), and doing it here would smuggle a
    second data source's worth of choices into a figure.
    """
    if im in SPECTRAL_AXIS:
        axis = SPECTRAL_AXIS[im]
        if axis == "frequency":
            raise typer.BadParameter(
                f"{im} cannot be paired against the recordings from this "
                "database: the simulated and observed frequency grids share no "
                "value, so the join is empty. Resampling one onto the other is "
                "interpolation and is deliberately left outside the database -- "
                "see its 'frequency grids' note"
            )
        return im.lower(), im, axis
    if im in SCALAR_IMS:
        return "scalars", im, None
    raise typer.BadParameter(
        f"{im!r} is not a measure this database holds. Spectral: "
        f"{', '.join(SPECTRAL_AXIS)}; scalar: {', '.join(SCALAR_IMS)}"
    )


def label_columns(names: Sequence[str]) -> str:
    """SQL widening ``run_labels`` into one column per dimension.

    ``run_labels`` is a join table on purpose, so a query wanting the dimensions
    side by side has to widen it. DuckDB's ``PIVOT`` needs the value list spelled
    out, and the value list is exactly what this code must not know -- so the
    widening is one correlated subquery per *discovered* name instead. The names
    come from :func:`dimensions`; they are escaped even though a regex group name
    cannot contain anything dangerous, because the day the names stop coming from
    a regex is not the day to discover that.
    """
    return "".join(
        f", (SELECT l.value FROM run_labels l WHERE l.run_id = r.run_id "
        f"AND l.name = '{sql_literal(name)}') AS \"{sql_literal(name)}\""
        for name in names
        if name != EVENT
    )


def read_residuals(
    con,
    runs: Sequence[dict],
    im: str = "pSA",
    component: str | None = None,
    ordinate_range: tuple[float | None, float | None] = (None, None),
    covariates: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """The long-form residual table: one row per (run, station, ordinate).

    ``ln(sim / obs)``, the repo's sign rather than the literature's, so a
    **positive residual is the simulation over-predicting**. Lee et al. (2022)
    define it the other way round, which inverts every "over-" and
    "under-prediction" sentence in that paper against this column.
    :func:`eqvis_workflow.data.residual_label` renders the axis for it, and this
    is the one place the direction is decided.

    The observed side is joined on the component the observed run *records
    itself as carrying* -- ``runs.observed_component``, the assertion the ingest
    was given -- rather than on the caller's simulated component. The recordings
    have no component column of their own, so that field is the only statement
    of the convention there is, and pairing through it means a database ingested
    under a different assertion pairs correctly with no change here.

    Spectral measures join on the exact stored ordinate. That is a DOUBLE
    equality, not an interpolation: the simulated and observed pSA grids are the
    same 111 values, which is why
    :func:`eqvis_workflow.bias.match_columns` -- the nearest-column matcher the
    flatfile path needs -- has no counterpart here.

    Every run dimension arrives as its own column, so a caller groups on a
    dimension by name without this function, or the schema, knowing what the
    dimensions are. Returned as a dict of bare arrays, the shape
    :func:`eqvis_workflow.flatfile.read_observed` already uses, because
    everything downstream is numpy.
    """
    table, value, axis = measure_source(im)
    component = component or GRID_COMPONENT.get(im, "geom")
    names = [name for name in dimensions(con) if name != EVENT]

    available = available_covariates(con)
    unknown = [name for name in covariates if name not in available]
    if unknown:
        raise typer.BadParameter(
            f"no such covariate: {', '.join(unknown)}. This database offers "
            f"{', '.join(sorted(available))}"
        )
    extra = "".join(
        f", {'g' if available[name] == 'run_stations' else 'st' if available[name] == 'stations' else 'r'}"
        f".\"{sql_literal(name)}\" AS \"{sql_literal(name)}\""
        for name in covariates
    )

    ordinate = f", p.{axis} AS ordinate" if axis else ""
    on_ordinate = f" AND o.{axis} = p.{axis}" if axis else ""
    bounds, parameters = "", []
    low, high = ordinate_range
    if axis and low is not None:
        bounds += f" AND p.{axis} >= ?"
        parameters.append(low)
    if axis and high is not None:
        bounds += f" AND p.{axis} <= ?"
        parameters.append(high)

    keys = [run["run_id"] for run in runs]
    placeholders = ", ".join("?" for _ in keys)
    rows = con.execute(
        f"""
        SELECT r.run_key, r.event, p.station{ordinate},
               ln(p.{value} / o.{value}) AS residual
               {label_columns(names)}{extra}
        FROM {table} p
        JOIN runs r ON r.run_id = p.run_id
        JOIN runs ro ON ro.kind = 'observed' AND ro.event = r.event
        JOIN {table} o ON o.run_id = ro.run_id AND o.station = p.station
                      AND o.component = ro.observed_component{on_ordinate}
        JOIN run_stations g ON g.run_id = p.run_id AND g.station = p.station
        JOIN stations st ON st.station = p.station
        WHERE p.run_id IN ({placeholders}) AND p.component = ?
          AND p.{value} > 0 AND o.{value} > 0{bounds}
        ORDER BY r.run_key, p.station{', p.' + axis if axis else ''}
        """,
        [*keys, component, *parameters],
    ).to_arrow_table()

    if rows.num_rows == 0:
        raise typer.BadParameter(
            f"no {im} residuals for component {component!r}. The simulated "
            f"components in this database are "
            f"{', '.join(str(c[0]) for c in con.execute(f'SELECT DISTINCT component FROM {table} ORDER BY 1').fetchall())}"
        )

    out: dict[str, np.ndarray] = {}
    for name in rows.column_names:
        column = rows.column(name)
        out[name] = (
            column.to_numpy()
            if column.type.id in (8, 9, 10, 11)  # integer / floating
            else column.to_numpy(zero_copy_only=False)
        )
    return out


# Station coordinates the drawing helpers read by name off the returned array.
# Every one is emitted even where a run has no value for it, as NaN: SW4 carries
# no grid indices and some runs have no rx/ry, and a helper reading `da.rx` off a
# coordinate that is simply absent fails on a missing attribute rather than on a
# NaN it already handles. Same move as :func:`eqvis_workflow.ingest.convert_run`.
RUN_COORDS = ("latitude", "longitude", "rrup", "rjb", "rx", "ry", "hyp", "epi")
SITE_COORDS = ("vs30", "z1pt0", "z2pt5", "elevation", "basin")

# How the WKT columns are named on the IM file's root, which is what
# `data.default_title`, `data.in_domain` and `geography.draw_geometry` read.
ATTR_RENAMES = {
    "domain_wkt": "domain",
    "source_wkt": "source",
    "trace_wkt": "trace",
}
ATTR_PLAIN = ("event", "magnitude", "hypo_lat", "hypo_lon", "hypo_depth")


def resolve_ordinate(con, im: str, wanted: float | None) -> tuple[str | None, float | None]:
    """The stored ordinate nearest what was asked for, and its axis name.

    Resolved before the query rather than inside it, so the join can be the
    exact DOUBLE equality the simulated and observed grids share. A tolerance in
    the join would silently pair neighbouring periods instead.
    """
    _, _, axis = measure_source(im)
    if axis is None:
        return None, None
    table = "periods" if axis == "period" else "frequencies"
    stored = np.array(
        [row[0] for row in con.execute(f"SELECT DISTINCT {axis} FROM {table}").fetchall()]
    )
    if stored.size == 0:
        raise typer.BadParameter(f"this database holds no {axis} grid")
    target = 1.0 if wanted is None else wanted
    return axis, float(stored[np.abs(stored - target).argmin()])


def run_attrs(con, run_key: str) -> dict[str, str]:
    """A run's root attributes, under the keys ``tree.attrs`` uses.

    :func:`eqvis_workflow.data.default_title`,
    :func:`eqvis_workflow.data.in_domain`,
    :func:`eqvis_workflow.data.restrict_to_domain` and
    :func:`eqvis_workflow.geography.draw_geometry` all read a plain dict of
    strings off the IM file's root, so building the same dict out of ``runs`` is
    the whole of what those four need in order to work against the database. Only
    the WKT columns are renamed.

    The values are stringified because that is what an HDF5 attribute is and
    what the consumers parse -- ``float(attrs["hypo_lon"])`` in
    ``draw_geometry``. Returning floats would work today and break the first
    time one of them was put in a title.
    """
    columns = [*ATTR_PLAIN, *ATTR_RENAMES]
    row = con.execute(
        f"SELECT {', '.join(columns)} FROM runs WHERE run_key = ?", [run_key]
    ).fetchone()
    if row is None:
        raise typer.BadParameter(f"no run with key {run_key!r}")
    named = {}
    for column, value in zip(columns, row):
        if value is None:
            continue
        named[ATTR_RENAMES.get(column, column)] = str(value)
    return named


def select_im(
    con,
    run_key: str,
    im: str,
    component: str,
    period: float | None = None,
    frequency: float | None = None,
) -> tuple[xr.DataArray, dict[str, float]]:
    """One run's intensity measure as the ``(station,)`` array the figures read.

    The database-backed twin of :func:`eqvis_workflow.data.select_im`, down to
    the station coordinates it carries, the ``attrs["units"]`` it sets and the
    ``selection`` dict it reports back. That identity is the point:
    :func:`eqvis_workflow.raster.rasterise`,
    :func:`eqvis_workflow.attenuation.select_distance` and every drawing helper
    downstream work on either source without knowing which they were given.
    """
    table, value, axis = measure_source(im)
    resolved_axis, ordinate = resolve_ordinate(con, im, period if axis == "period" else frequency)
    where = f" AND p.{resolved_axis} = ?" if resolved_axis else ""
    parameters = [run_key, component] + ([ordinate] if resolved_axis else [])

    rows = con.execute(
        f"""
        SELECT p.station, p.{value} AS value,
               {', '.join(f'g.{name}' for name in RUN_COORDS)},
               {', '.join(f'st.{name}' for name in SITE_COORDS)}
        FROM {table} p
        JOIN runs r ON r.run_id = p.run_id
        JOIN run_stations g ON g.run_id = p.run_id AND g.station = p.station
        LEFT JOIN stations st ON st.station = p.station
        WHERE r.run_key = ? AND p.component = ?{where}
        ORDER BY p.station
        """,
        parameters,
    ).to_arrow_table()

    if rows.num_rows == 0:
        available = [
            str(c[0])
            for c in con.execute(
                f"SELECT DISTINCT component FROM {table} p JOIN runs r "
                "ON r.run_id = p.run_id WHERE r.run_key = ? ORDER BY 1",
                [run_key],
            ).fetchall()
        ]
        raise typer.BadParameter(
            f"no {im} for component {component!r} in run {run_key!r}. "
            f"Available: {available}"
        )

    def column(name):
        return rows.column(name).to_numpy(zero_copy_only=False)

    station = column("station").astype(str)
    coords = {"station": ("station", station)}
    for name in RUN_COORDS:
        coords[name] = ("station", column(name).astype(float))
    for name in SITE_COORDS:
        raw = column(name)
        coords[name] = (
            "station",
            raw.astype(object) if name == "basin" else raw.astype(float),
        )

    unit = con.execute("SELECT unit FROM im_units WHERE im = ?", [im]).fetchone()
    array = xr.DataArray(
        column("value").astype(float),
        dims="station",
        coords=coords,
        name=im,
        attrs={"units": unit[0] if unit else ""},
    )
    selection = {resolved_axis: ordinate} if resolved_axis else {}
    return array, selection


def read_observed(
    con,
    run_key: str,
    im: str,
    component: str,
    selection: dict[str, float],
    metric: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The recordings paired with a simulated run, in the shape the maps read.

    The twin of :func:`eqvis_workflow.flatfile.read_observed`:
    ``{"name", "lon", "lat", "value"}``, plus ``"distance"`` when a ``metric``
    is asked for, and the resolved selection.

    ``run_key`` names the **simulated** run, and that is not a convenience. An
    observed run has no ``run_stations`` rows at all -- zero, for every one of
    them, which the schema's ``site terms vs geometry`` note states -- so there
    is no coordinate on the observed side to read. The recording is at the same
    site as the simulated station of the same name, so the geometry is the
    simulation's and the pairing is an exact join on that name. No
    nearest-station search, which the file-based path has to do because the
    flatfile's coordinates are the network's and the simulation's are its grid's.

    The observed side is matched on the component the observed run records
    itself as carrying, not on the caller's: the recordings have no component
    column, so that assertion is the only statement of the convention there is.

    A measure the recordings do not carry comes back all-NaN with a warning,
    exactly as the flatfile version does, leaving the caller drawing bare station
    markers. PGD is the standing case -- the recordings have no displacement
    column at all.
    """
    table, value, axis = measure_source(im)
    ordinate = selection.get(axis) if axis else None
    where = f" AND o.{axis} = ? AND p.{axis} = ?" if axis else ""
    parameters = [run_key] + ([ordinate, ordinate] if axis else [])
    distance = f", g.{metric} AS distance" if metric else ""

    if metric and metric not in RUN_COORDS:
        raise typer.BadParameter(
            f"{metric!r} is not a distance this database holds. Available: "
            f"{[m for m in RUN_COORDS if m not in ('latitude', 'longitude')]}"
        )

    rows = con.execute(
        f"""
        SELECT DISTINCT p.station AS name, g.longitude AS lon, g.latitude AS lat,
               o.{value} AS value{distance}
        FROM runs r
        JOIN runs ro ON ro.kind = 'observed' AND ro.event = r.event
        JOIN {table} o ON o.run_id = ro.run_id
                      AND o.component = ro.observed_component
        JOIN {table} p ON p.run_id = r.run_id AND p.station = o.station
        JOIN run_stations g ON g.run_id = r.run_id AND g.station = o.station
        WHERE r.run_key = ?{where}
        ORDER BY p.station
        """,
        parameters,
    ).to_arrow_table()

    if rows.num_rows == 0:
        console_warn(f"no recordings pair with run {run_key!r}")
        empty = np.array([])
        observed = {"name": empty.astype(str), "lon": empty, "lat": empty,
                    "value": empty}
        if metric:
            observed["distance"] = empty
        return observed, selection

    observed = {
        "name": rows.column("name").to_numpy(zero_copy_only=False).astype(str),
        "lon": rows.column("lon").to_numpy().astype(float),
        "lat": rows.column("lat").to_numpy().astype(float),
        "value": rows.column("value").to_numpy(zero_copy_only=False).astype(float),
    }
    if not np.any(np.isfinite(observed["value"])):
        console_warn(
            f"the recordings carry no {im}; plotting stations only. "
            "PGD is the standing case -- there is no observed displacement"
        )
    if metric:
        observed["distance"] = (
            rows.column("distance").to_numpy(zero_copy_only=False).astype(float)
        )
    return observed, selection


# What identifies one residual observation across the ordinate axis: an
# earthquake recorded at a station. Not the station alone -- a station records
# many earthquakes, and pairing on it would compare an event against every other
# event's ground motion, which is the trap the schema's `one observed run per
# event` note exists to prevent.
CELL_KEYS = (EVENT, "station")


def pivot_ordinates(
    data: dict[str, np.ndarray], rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A long residual table reshaped to ``(cell, ordinate)``, the shape the
    statistics take.

    :func:`eqvis_workflow.bias.bias_statistics` reduces over axis 0 of a
    ``(station, period)`` table, so the long form has to be laid out that way
    before any of the existing drawing code can touch it. The rows are
    ``(event, station)`` cells rather than stations, for the reason
    :data:`CELL_KEYS` gives.

    Returns the matrix, the ordinate grid it is columned by, and the row keys.
    Missing combinations are NaN rather than dropped, so every cell of every
    series is columned by the same grid and two series can be differenced by
    position. ``rows`` fixes the row order, which is what lets a paired
    difference line two cells up without a join.

    A scalar measure has no ordinate, and comes back as a single column so that
    exactly the same statistics and the same panels serve both.
    """
    residual = np.asarray(data["residual"], dtype=float)
    if "ordinate" in data:
        ordinates = np.unique(np.asarray(data["ordinate"], dtype=float))
        column = np.searchsorted(ordinates, np.asarray(data["ordinate"], dtype=float))
    else:
        ordinates = np.array([np.nan])
        column = np.zeros(len(residual), dtype=np.intp)

    keys = np.array(
        [
            "\x00".join(str(value) for value in pair)
            for pair in zip(*(data[name] for name in CELL_KEYS))
        ]
    )
    if rows is None:
        rows = np.unique(keys)
    index = {key: position for position, key in enumerate(rows)}
    row = np.array([index.get(key, -1) for key in keys], dtype=np.intp)

    matrix = np.full((len(rows), len(ordinates)), np.nan)
    keep = row >= 0
    matrix[row[keep], column[keep]] = residual[keep]
    return matrix, ordinates, rows


def common_rows(cells: Sequence[np.ndarray]) -> np.ndarray:
    """The ``(event, station)`` cells every series has, in sorted order.

    The factorial is ragged -- on the reference data one event was run under
    only half the configurations -- so a series scored over cells another series
    does not have differs from it partly by those cells' own bias rather than by
    the thing being compared. Intersecting first is what makes the comparison a
    comparison.
    """
    shared = set(cells[0])
    for other in cells[1:]:
        shared &= set(other)
    return np.array(sorted(shared))


def runs(
    db: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Composite intensity measure database"
        ),
    ],
    label: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            help="Restrict to runs matching name=value; repeat. A name repeated "
            "keeps several of its values, two different names must both hold",
        ),
    ] = None,
    pairable: Annotated[
        bool,
        typer.Option(
            "--pairable/--all",
            help="Only runs whose event has recordings. Off by default: the "
            "first question is what is in here, not what can be plotted",
        ),
    ] = False,
) -> None:
    """Report what a composite database holds: its dimensions and its coverage."""
    con = connect(db)
    available = dimensions(con)
    print(f"{db}\n")
    print("dimensions")
    for name, values in available.items():
        shown = ", ".join(values[:8]) + (" ..." if len(values) > 8 else "")
        print(f"  {name:<12} {len(values):>4}  {shown}")

    labels = [parse_label(text) for text in (label or [])]
    found = select_runs(con, labels, pairable=pairable)
    print(f"\n{len(found)} simulated runs" + (" with recordings" if pairable else ""))
    print(f"  varying: {', '.join(varying(found)) or '(nothing)'}")

    print("\ncoverage")
    counts = {event: n for event, n, _ in coverage(con)}
    full = max(counts.values()) if counts else 0
    for event, n, present in coverage(con):
        flag = "" if n == full else f"  <- {full - n} missing"
        print(f"  {event:<14} {n}  {present}{flag}")

    observed = con.execute(
        "SELECT count(*) FROM runs WHERE kind = 'observed'"
    ).fetchone()[0]
    pairs = con.execute(
        """
        SELECT count(*) FROM runs s WHERE s.kind = 'simulated'
          AND EXISTS (SELECT 1 FROM runs o
                      WHERE o.kind = 'observed' AND o.event = s.event)
        """
    ).fetchone()[0]
    print(f"\n{observed} observed runs; {pairs} simulated runs have recordings")

    site = con.execute(
        """
        SELECT count(*), count(vs30), count(elevation), count(basin)
        FROM stations
        """
    ).fetchone()
    print(
        f"\nsite terms over {site[0]:,} stations: vs30 {site[1]:,}, "
        f"elevation {site[2]:,}, basin {site[3]:,}"
    )
    if not site[2]:
        console_warn(
            "no station elevations; re-run the ingest with --dem to fill them"
        )
    if not site[3]:
        console_warn(
            "no station basins; re-run the ingest with basin outlines available"
        )

    print(f"\ncovariates: {', '.join(sorted(available_covariates(con)))}")
    topics = [row[0] for row in con.execute("SELECT topic FROM notes").fetchall()]
    print(f"\n{len(topics)} notes -- read them first: {', '.join(topics)}")
    con.close()
