"""Converting a tree of simulation IM files into staged parquet.

The input is a tree of HDF5 datatrees, one per run. How that tree is organised
is not assumed: an ``--extract`` regex is searched against each file's path, and
every named group it captures becomes a dimension of that run. The default
reproduces what the cylc workflow uploads --
``<event>/[R<n>/]<solver>_<layers>/intensity_measures.h5`` -- but a tree
organised along different axes needs a different regex and no code change,
because the dimensions land in a join table rather than in columns.

The one dimension the database needs by name is the event, since that is what
pairs a simulation with the recordings of the same earthquake. Capture it in the
regex, or leave it out and it is read from the IM file's own root attribute.

:mod:`~.database` then assembles a database from the whole staged set. Staging
rather than writing straight into the database buys two things. The conversion
is the expensive half -- tens of gigabytes of compressed HDF5 -- and a run whose
file has not changed since it was last converted is skipped, so adding one event
to a finished tree costs one run's work rather than all of them. And a crash
leaves a directory of complete parquet files plus a manifest saying which runs
are done, rather than a half-written database.

Not every station carries every component. Storing all seven components of pSA
at every station of every run would be some 500 million rows for ten events and
seven billion for eighty-one, so the spectra follow a tier rule:

* ``rotd50`` (pSA) and ``eas`` (FAS) at **every** station, which is what a map
  or a distance plot needs;
* **every** component, but only at stations that are observation sites, which is
  what a residual against a recording needs.

The rule is recorded per station in ``run_stations.is_observation_site`` rather
than left as folklore.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from . import database
from .console import console_warn
from .constants import GRID_COMPONENT, SCALAR_IMS, SPECTRAL_AXIS
from .data import open_ims

# The filename the workflow writes. Discovery goes through --extract, so this
# is only a convenience for callers building a path.
IM_FILE = "intensity_measures.h5"

# Root attributes worth carrying onto the run: the source parameters a
# regression wants as predictors, and the geometry as WKT.
RUN_ATTRS_NUMERIC = (
    "magnitude",
    "hypo_lat",
    "hypo_lon",
    "hypo_depth",
    "rake",
    "dip",
    "ztor",
    "zbot",
)
RUN_ATTRS_TEXT = ("tect_type",)
RUN_ATTRS_WKT = {"source": "source_wkt", "trace": "trace_wkt", "domain": "domain_wkt"}

# Site terms are properties of the site, not of the run: they were checked to
# be identical across all four configurations of an event, so they normalise
# out of the per-run table into a global one.
SITE_TERMS = ("vs30", "z1pt0", "z2pt5")

# Per-run geometry. Unlike the site terms these do differ between solvers,
# which snap stations to their own grids -- up to 0.0016 degrees, and 0.14 km
# of rrup.
RUN_GEOMETRY = ("latitude", "longitude", "rrup", "rjb", "rx", "ry", "hyp", "epi")


@dataclass(frozen=True)
class Run:
    """One simulation: an event, and whatever dimensions distinguish it.

    Beyond the event, a run's dimensions are whatever ``--extract`` named --
    ``solver`` and ``layers`` for the current workflow, but nothing here
    assumes that. They are carried as sorted ``(name, value)`` pairs and land
    in the ``run_labels`` join table, so a tree organised along different axes
    needs a different regex and no schema change.

    A tuple of pairs rather than a dict so the run stays frozen, hashable and
    picklable -- it has to cross into a worker process.
    """

    event: str
    labels: tuple[tuple[str, str], ...]
    im_file: Path

    @property
    def key(self) -> str:
        """A filesystem-safe, collision-free name for this run.

        The label *names* are in the key, not just their values: two runs whose
        regexes matched different dimensions could otherwise produce the same
        string and quietly share a staging directory.
        """
        parts = [f"event={self.event}"] + [f"{k}={v}" for k, v in self.labels]
        return re.sub(r"[^A-Za-z0-9=+.-]+", "-", "__".join(parts))

    @property
    def label(self) -> str:
        """How the run is named in a message to the reader."""
        dimensions = " ".join(f"{k}={v}" for k, v in self.labels)
        return f"{self.event} {dimensions}".strip()


# The event is the one dimension the database needs by name: it is what pairs a
# simulation with the recordings of the same earthquake. Everything else is a
# label. If the regex does not name an event, it is read from the IM file's own
# root attributes, which carry it.
EVENT_GROUP = "event"

# Reproduces the layout the current workflow uploads:
# <event>/[R<n>/]<solver>_<layers>/intensity_measures.h5
DEFAULT_EXTRACT = (
    r"(?P<event>[^/]+)/(?:(?P<realisation>R\d+)/)?"
    r"(?P<solver>[^/_]+)_(?P<layers>[^/]+)/intensity_measures\.h5$"
)


def event_from_file(path: Path) -> str:
    """The event name out of an IM file's root attributes.

    The fallback for a regex that does not name an event -- which is the
    common case when the tree is organised by configuration rather than by
    earthquake. Read with h5py rather than xarray: only the root attributes
    are wanted, and this is on the discovery path for every file.
    """
    with h5py.File(path, "r") as handle:
        if EVENT_GROUP not in handle.attrs:
            raise ValueError(
                f"{path} has no 'event' root attribute and --extract does not "
                "capture one, so there is no way to tell which earthquake it is"
            )
        value = handle.attrs[EVENT_GROUP]
    return value.decode() if isinstance(value, bytes) else str(value)


def compile_extract(extract: str) -> re.Pattern:
    """Compile an ``--extract`` regex, refusing one that cannot work."""
    try:
        pattern = re.compile(extract)
    except re.error as error:
        raise typer.BadParameter(f"--extract is not a valid regex: {error}") from error
    if not pattern.groupindex:
        raise typer.BadParameter(
            "--extract has no named groups, so it cannot say what distinguishes "
            "one run from another. Name at least one, e.g. "
            r"'(?P<solver>[^/_]+)_(?P<layers>[^/]+)/intensity_measures\.h5$'"
        )
    return pattern


def discover(results_dir: Path, extract: str = DEFAULT_EXTRACT) -> list[Run]:
    """Every run under ``results_dir``, in a stable order.

    The regex decides both which files are runs and how each one decomposes:
    it is searched against each file's path relative to ``results_dir``, and
    every named group it captures becomes a dimension of that run. A file the
    regex does not match is not a run.

    ``search`` rather than ``fullmatch``, so a regex describing only the tail
    of the path -- which is what anyone writes first -- does the obvious thing.
    """
    pattern = compile_extract(extract)
    runs: dict[str, Run] = {}
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        match = pattern.search(path.relative_to(results_dir).as_posix())
        if match is None:
            continue
        captured = {k: v for k, v in match.groupdict().items() if v is not None}
        event = captured.pop(EVENT_GROUP, None) or event_from_file(path)
        run = Run(event, tuple(sorted(captured.items())), path)
        if run.key in runs:
            raise typer.BadParameter(
                f"--extract does not tell {path} apart from "
                f"{runs[run.key].im_file}: both come out as {run.label!r}. "
                "Capture whatever distinguishes them"
            )
        runs[run.key] = run
    return list(runs.values())


def parse_only(texts: list[str] | None) -> tuple[tuple[str, str], ...] | None:
    """The run selector from ``--only name=value`` options.

    Each option names a run dimension -- one the ``--extract`` regex captured,
    or ``event`` -- and the value it must take. Repeated names are a disjunction
    over their values and different names intersect, matching how the analysis
    commands read ``--label``.
    """
    if not texts:
        return None
    pairs = []
    for text in texts:
        name, sep, value = text.partition("=")
        if not sep or not name:
            raise typer.BadParameter(
                f"--only takes name=value, not {text!r}, where the name is a "
                "dimension the --extract regex captures, or 'event'"
            )
        pairs.append((name, value))
    return tuple(pairs)


def selected_runs(
    runs: list[Run], only: tuple[tuple[str, str], ...] | None
) -> set[str]:
    """The keys of the runs ``only`` selects, or every run's key.

    ``event`` is the one dimension carried on the run itself rather than as a
    label, so it is matched against ``Run.event``; every other name is matched
    against the run's labels.
    """
    if not only:
        return {run.key for run in runs}
    wanted: dict[str, set[str]] = {}
    for name, value in only:
        wanted.setdefault(name, set()).add(value)

    def matches(run: Run) -> bool:
        labels = dict(run.labels)
        for name, values in wanted.items():
            if name == EVENT_GROUP:
                if run.event not in values:
                    return False
            elif labels.get(name) not in values:
                return False
        return True

    return {run.key for run in runs if matches(run)}


def stamp(path: Path) -> dict:
    """What decides whether a converted run is still current."""
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def attr_float(attrs: dict, name: str) -> float | None:
    """A root attribute as a float, tolerating its absence or its being text.

    The IM writer stores these as strings, and not every event carries every
    one, so a missing or unparsable attribute is a null rather than an error.
    """
    if name not in attrs:
        return None
    try:
        return float(attrs[name])
    except (TypeError, ValueError):
        return None


def populated(dataset, component: str, sample: int = 500) -> bool:
    """Whether a component of a group holds anything at all.

    Some groups carry a component that is entirely fill -- the first
    ``sample`` stations decide it, since reading every station of every
    component would mean reading the file twice over just to learn its shape.
    """
    probe = dataset[component].isel(station=slice(0, sample))
    return bool(probe.notnull().any())


DICT_STRING = pa.dictionary(pa.int32(), pa.string())


def repeated(value: str | None, length: int) -> pa.Array:
    """A column of one value repeated, dictionary-encoded.

    The spectral tables repeat a run key and a component name once per row, and
    at millions of rows a plain string column of those is both slow to build
    and large enough that pyarrow chunks it -- which record_batch will not
    take. One dictionary entry and a run of zero indices says the same thing.
    """
    return pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(length, dtype=np.int32)), pa.array([value])
    )


def dictionary_of(values: np.ndarray, repeat: int) -> pa.Array:
    """``values`` each repeated ``repeat`` times, dictionary-encoded."""
    indices = np.repeat(np.arange(len(values), dtype=np.int32), repeat)
    return pa.DictionaryArray.from_arrays(pa.array(indices), pa.array(values))


def float_array(values: np.ndarray, dtype=pa.float32()) -> pa.Array:
    """A float column with the fill value as SQL NULL rather than NaN.

    The IM files use NaN as their fill, and a component a measure does not
    define is entirely fill. Carried into the database as NaN it would poison
    every aggregate that touched it -- AVG over a column holding one NaN is
    NaN, where over a column holding one NULL it is the mean of the rest -- so
    the fill becomes NULL here, once, rather than in every query later.
    """
    values = np.asarray(values)
    return pa.array(values, type=dtype, mask=np.isnan(values))


def convert_run(
    run: Run, observation_sites: frozenset[str], build_dir: Path, with_fas: bool
) -> dict:
    """Convert one run's IM file into parquet under ``build_dir``.

    Returns the manifest entry for the run: its stamp, its station count, and
    the row count of each table it wrote.
    """
    out = build_dir / run.key
    out.mkdir(parents=True, exist_ok=True)
    tree = open_ims(run.im_file)
    attrs = dict(tree.attrs)

    groups = list(tree.children)
    if "PGA" not in groups:
        raise ValueError(f"{run.im_file} has no PGA group; is it an IM file?")
    anchor = tree["PGA"].to_dataset()
    station = anchor.station.values.astype(str)
    n_stations = len(station)
    is_site = np.isin(station, np.array(sorted(observation_sites), dtype=object))

    rows = {}

    # -- the run itself -------------------------------------------------
    run_row = {
        "run_key": run.key,
        "event": run.event,
        "kind": "simulated",
        "observed_component": None,
        "im_file": str(run.im_file),
        "file_size": stamp(run.im_file)["size"],
        "file_mtime_ns": stamp(run.im_file)["mtime_ns"],
        "n_stations": n_stations,
        "n_observation_sites": int(is_site.sum()),
        **{name: attr_float(attrs, name) for name in RUN_ATTRS_NUMERIC},
        **{name: attrs.get(name) for name in RUN_ATTRS_TEXT},
        **{column: attrs.get(name) for name, column in RUN_ATTRS_WKT.items()},
    }
    pq.write_table(pa.table({k: [v] for k, v in run_row.items()}), out / "run.parquet")
    # The dimensions the regex named, as rows rather than columns, so a tree
    # organised along different axes needs no schema change.
    pq.write_table(
        pa.table(
            {
                "run_key": pa.array([run.key] * len(run.labels)),
                "name": pa.array([name for name, _ in run.labels]),
                "value": pa.array([value for _, value in run.labels]),
            }
        ),
        out / "run_labels.parquet",
    )

    # -- site terms, which are global, and geometry, which is not -------
    site = {"station": pa.array(station)}
    for term in SITE_TERMS:
        site[term] = float_array(
            anchor[term].values.astype(np.float32)
            if term in anchor.coords
            else np.full(n_stations, np.nan, dtype=np.float32)
        )
    pq.write_table(pa.table(site), out / "stations.parquet")
    rows["stations"] = n_stations

    geometry = {
        "run_key": repeated(run.key, n_stations),
        "station": pa.array(station),
    }
    for axis in ("x", "y"):
        # EMOD3D carries the grid indices it sampled at; SW4 does not.
        geometry[axis] = pa.array(
            anchor[axis].values.astype(np.int32)
            if axis in anchor.coords
            else np.full(n_stations, None, dtype=object),
            type=pa.int32(),
        )
    for term in RUN_GEOMETRY:
        geometry[term] = float_array(
            anchor[term].values.astype(np.float64)
            if term in anchor.coords
            else np.full(n_stations, np.nan),
            pa.float64(),
        )
    geometry["is_observation_site"] = pa.array(is_site)
    pq.write_table(pa.table(geometry), out / "run_stations.parquet")
    rows["run_stations"] = n_stations

    # -- scalars: every station, every component, the measures as columns
    components: list[str] = []
    for im in SCALAR_IMS:
        if im in groups:
            for component in tree[im].to_dataset().data_vars:
                if component not in components:
                    components.append(str(component))
    blocks = []
    for component in components:
        block = {
            "run_key": repeated(run.key, n_stations),
            "station": pa.array(station),
            "component": repeated(component, n_stations),
        }
        for im in SCALAR_IMS:
            values = np.full(n_stations, np.nan, dtype=np.float32)
            if im in groups:
                dataset = tree[im].to_dataset()
                if component in dataset.data_vars:
                    values = dataset[component].values.astype(np.float32)
            block[im] = float_array(values)
        blocks.append(pa.table(block))
    scalars = pa.concat_tables(blocks)
    pq.write_table(scalars, out / "scalars.parquet")
    rows["scalars"] = scalars.num_rows

    # -- spectra, under the tier rule ----------------------------------
    for group, axis in SPECTRAL_AXIS.items():
        if group == "FAS" and not with_fas:
            continue
        if group not in groups:
            console_warn(f"{run.label} has no {group} group")
            continue
        rows[group] = write_spectra(
            tree[group].to_dataset(),
            axis,
            group,
            run.key,
            station,
            is_site,
            out / f"{group.lower()}.parquet",
            run.label,
        )

    tree.close()
    return {"stamp": stamp(run.im_file), "n_stations": n_stations, "rows": rows}


def write_spectra(
    dataset,
    axis: str,
    value_column: str,
    run_key: str,
    station: np.ndarray,
    is_site: np.ndarray,
    path: Path,
    label: str,
) -> int:
    """Write one spectral group to parquet, one component at a time.

    Component by component so the peak memory is one component's block rather
    than the whole group: the largest run here is 60,334 stations, which at 111
    periods is 6.7 million rows for a single component.
    """
    ordinates = dataset[axis].values.astype(np.float64)
    site_index = np.flatnonzero(is_site)
    grid_component = GRID_COMPONENT[value_column]
    if grid_component not in dataset.data_vars:
        console_warn(
            f"{label} {value_column} has no {grid_component!r} component; "
            "no grid-wide spectra for this run"
        )

    schema = pa.schema(
        [
            ("run_key", DICT_STRING),
            ("station", DICT_STRING),
            ("component", DICT_STRING),
            (axis, pa.float64()),
            (value_column, pa.float32()),
        ]
    )
    written = 0
    with pq.ParquetWriter(path, schema) as writer:
        for component in dataset.data_vars:
            component = str(component)
            if not populated(dataset, component):
                continue
            # Everywhere for the grid component; only at observation sites for
            # the rest, which is what keeps this table a hundredth the size.
            index = slice(None) if component == grid_component else site_index
            if not isinstance(index, slice) and index.size == 0:
                continue
            values = (
                dataset[component]
                .transpose("station", axis)
                .values[index]
                .astype(np.float32)
            )
            names = station[index]
            batch = pa.record_batch(
                [
                    repeated(run_key, values.size),
                    dictionary_of(names, len(ordinates)),
                    repeated(component, values.size),
                    pa.array(np.tile(ordinates, len(names))),
                    float_array(values.ravel()),
                ],
                schema=schema,
            )
            writer.write_batch(batch)
            written += batch.num_rows
    return written


def load_manifest(build_dir: Path) -> dict:
    path = build_dir / "manifest.json"
    if not path.exists():
        return {"runs": {}}
    return json.loads(path.read_text())


def save_manifest(build_dir: Path, manifest: dict) -> None:
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _convert(args) -> tuple[str, dict]:
    """Trampoline so a worker process can call :func:`convert_run`."""
    run, sites, build_dir, with_fas = args
    return run.key, convert_run(run, sites, build_dir, with_fas)


def stage(
    results_dir: Path,
    build_dir: Path,
    observation_sites: frozenset[str],
    extract: str = DEFAULT_EXTRACT,
    with_fas: bool = True,
    fresh: bool = False,
    only: tuple[tuple[str, str], ...] | None = None,
    force: bool = False,
    jobs: int = 1,
) -> tuple[list[Run], dict]:
    """Convert every run under ``results_dir`` that is not already current.

    Returns the runs found and the updated manifest. A run is current when the
    size and mtime of its IM file match what the manifest recorded, so touching
    a file is enough to force its reconversion and nothing else's.

    ``only`` narrows which runs are considered for conversion without narrowing
    discovery: ``results_dir`` is still scanned in full, so a run outside the
    selection keeps its manifest entry rather than being dropped as vanished.
    ``force`` reconverts the selected runs even when their stamp is unchanged,
    for a download that preserved the file's mtime.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    runs = discover(results_dir, extract)
    if not runs:
        raise ValueError(
            f"no file under {results_dir} matches --extract; the pattern is "
            "searched against each path relative to that directory"
        )
    manifest = {"runs": {}} if fresh else load_manifest(build_dir)

    selected = selected_runs(runs, only)
    if only and not selected:
        console_warn(
            "--only selected none of the discovered runs; nothing will be converted"
        )
    stale = []
    for run in runs:
        if run.key not in selected:
            continue
        entry = manifest["runs"].get(run.key)
        if force or entry is None or entry["stamp"] != stamp(run.im_file):
            stale.append(run)
    selection = f" ({len(selected)} selected)" if only else ""
    print(
        f"{len(runs)} runs found; {len(stale)} to convert, "
        f"{len(runs) - len(stale)} already current{selection}"
    )

    work = [(run, observation_sites, build_dir, with_fas) for run in stale]
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for key, entry in pool.map(_convert, work):
                manifest["runs"][key] = entry
                save_manifest(build_dir, manifest)
                print(f"  converted {key}")
    else:
        for item in work:
            run = item[0]
            key, entry = _convert(item)
            manifest["runs"][key] = entry
            save_manifest(build_dir, manifest)
            print(
                f"  converted {run.label}: {entry['n_stations']} stations, "
                + ", ".join(f"{k}={v:,}" for k, v in entry["rows"].items())
            )

    # A run that has gone from the tree should not linger in the manifest and
    # be assembled into the database off its stale parquet.
    live = {run.key for run in runs}
    for key in [k for k in manifest["runs"] if k not in live]:
        console_warn(f"{key} is in the build directory but no longer on disk; dropping")
        del manifest["runs"][key]
    save_manifest(build_dir, manifest)
    return runs, manifest


def default_jobs() -> int:
    """One worker per core, less a couple, and never more than eight.

    Each worker holds a component's block of spectra in memory, so the cap is
    about memory rather than about cores.
    """
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def ingest(
    results: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory of simulation results: <event>/[R<n>/]<solver>_<layers>/",
        ),
    ],
    out: Annotated[
        Path, typer.Argument(help="Output DuckDB database (rebuilt on every run)")
    ],
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            file_okay=False,
            help="Directory of the raw recording CSVs -- im_obs.csv, one row "
            "per recorded ground motion, with events.csv and stations.csv the "
            "dictionaries its integer ids point into. Its recordings become "
            "one further run per event. Omitted, the database holds only "
            "simulated runs, with no observed side to pair against",
        ),
    ] = None,
    observed_component: Annotated[
        str | None,
        typer.Option(
            "--observed-component",
            help="Which component of motion the observed measures are. There is "
            "no default: im_obs.csv carries no component column and does not "
            "say, so the choice has to be made explicitly and is recorded in "
            "the database. Required when --observed is given",
        ),
    ] = None,
    extract: Annotated[
        str,
        typer.Option(
            "--extract",
            help="Regex naming what distinguishes one run from another. It is "
            "searched against each file's path relative to RESULTS, and every "
            "named group becomes a dimension of that run in the run_labels "
            "table. A file it does not match is not a run. Capture 'event' to "
            "take the earthquake from the path, or leave it out and it is read "
            "from the IM file's own root attribute",
        ),
    ] = DEFAULT_EXTRACT,
    build_dir: Annotated[
        Path | None,
        typer.Option(
            "--build",
            help="Parquet staging directory (default: OUT with a .build suffix)",
        ),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option("--fresh", help="Reconvert every run, ignoring the manifest"),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Restrict conversion to runs matching name=value; repeat. The "
            "name is a dimension the --extract regex captures, or 'event'. Runs "
            "outside the selection are left as-is, but RESULTS is still scanned "
            "in full, so point it at the whole tree",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Reconvert the selected runs even if their stamp is unchanged "
            "(e.g. a download preserved the file's mtime). Use with --only to "
            "redo a single event",
        ),
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option("--jobs", "-j", help="Runs to convert in parallel"),
    ] = None,
    fas: Annotated[
        bool, typer.Option("--fas/--no-fas", help="Include the Fourier spectra")
    ] = True,
    dem: Annotated[
        Path | None,
        typer.Option(
            "--dem",
            exists=True,
            help="NZCVM DEM (zarr) to sample station elevations from. Sampled "
            "at the coordinates already staged, so supplying it costs an "
            "assembly rather than a reconversion. Omitted, stations.elevation "
            "stays NULL",
        ),
    ] = None,
    basin_file: Annotated[
        Path | None,
        typer.Option(
            "--basin-file",
            exists=True,
            dir_okay=False,
            help="Basin outlines (GeoParquet) to assign each station a basin "
            "from. Defaults to the same cached blob the maps draw",
        ),
    ] = None,
    basins: Annotated[
        bool,
        typer.Option(
            "--basins/--no-basins",
            help="Assign each station its basin. --no-basins leaves "
            "stations.basin NULL without trying to fetch the outlines",
        ),
    ] = True,
) -> None:
    """Build a composite intensity measure database from simulation output."""
    build_dir = build_dir or out.with_suffix(out.suffix + ".build")
    if observed is not None and observed_component is None:
        raise typer.BadParameter(
            "--observed-component is required when --observed is given: "
            "im_obs.csv carries no component column, so the choice has to be "
            "made explicitly",
            param_hint="'--observed-component'",
        )
    # Ahead of the conversion, so that a wrong --observed costs a second rather
    # than an hour. The records themselves are checked when they are loaded.
    # Re-raised as a BadParameter: it is the option that is wrong, and a usage
    # error reads better than a traceback.
    if observed is None:
        sites: frozenset[str] = frozenset()
    else:
        try:
            database.check_recordings(observed)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="'--observed'") from error
        sites = database.observation_sites(observed)
        print(f"{len(sites)} observation sites from {observed / database.STATIONS_CSV}")

    runs, manifest = stage(
        results,
        build_dir,
        sites,
        extract=extract,
        with_fas=fas,
        fresh=fresh,
        only=parse_only(only),
        force=force,
        jobs=jobs or default_jobs(),
    )
    database.build(
        out,
        build_dir,
        runs,
        manifest,
        observed,
        observed_component,
        with_fas=fas,
        extract=extract,
        dem=dem,
        basin_file=basin_file,
        with_basins=basins,
    )
