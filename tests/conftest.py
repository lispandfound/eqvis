"""Fixtures shared by the database-backed test modules.

A synthetic results tree and a synthetic set of recording CSVs, built once per
module, run through the real :func:`~eqvis_workflow.ingest.stage` and
:func:`~eqvis_workflow.database.build` so that what the tests assert against is
a real database rather than a mock of one. The awkward parts of the real data
are reproduced deliberately -- a solver whose station set is a subset of the
other's, a measure with no rotd components, observed Fourier frequencies that do
not line up with the simulated ones, and integer ids that are neither contiguous
nor in name order.
"""


import duckdb
import numpy as np
import pytest
import xarray as xr

from eqvis_workflow import database, ingest

PERIODS = np.array([0.01, 0.1, 1.0, 5.0])
SIM_FREQUENCIES = np.array([0.1, 1.0, 10.0])
# As in the real flatfile: a different, finer grid over a shorter range, so a
# join on frequency finds nothing.
OBS_FREQUENCIES = np.array([0.100000015334, 0.5, 2.0])

SCALAR_COMPONENTS = ["000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100"]
DURATION_COMPONENTS = ["000", "090", "ver", "geom"]
FAS_COMPONENTS = ["000", "090", "ver", "geom", "eas"]

# The scalars im_obs carries -- no PGD, as in the real export.
OBSERVED_SCALARS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]

# The dictionaries im_obs points into. The ids are deliberately neither in name
# order nor contiguous: a resolution that lined them up positionally, or used an
# id where it meant a name, would come out wrong rather than merely unlucky.
OBSERVED_EVENTS = {7: "2020p111111", 3: "2022p333333"}  # note: not 2021p222222
OBSERVED_STATIONS = {5: "AAAA", 2: "BBBB"}


def write_im_file(path, stations, seed, with_x=True, supergrid=False):
    """Write one run's IM file in the shape the workflow's im-calc produces.

    ``supergrid`` decides whether the run reports how far each station sits into
    an absorbing layer, and is off by default because that is the real
    asymmetry: SW4 has a supergrid layer to report and EMOD3D has none, so a
    multi-solver ingest sees the term on one solver's files and not the other's.
    """
    rng = np.random.default_rng(seed)
    n = len(stations)
    coords = {
        "station": ("station", list(stations)),
        "latitude": ("station", rng.uniform(-45, -44, n)),
        "longitude": ("station", rng.uniform(170, 172, n)),
        "rrup": ("station", rng.uniform(5, 200, n)),
        "rjb": ("station", rng.uniform(5, 200, n)),
        "rx": ("station", rng.uniform(-50, 50, n)),
        "ry": ("station", rng.uniform(-50, 50, n)),
        "hyp": ("station", rng.uniform(5, 200, n)),
        "epi": ("station", rng.uniform(5, 200, n)),
        # Site terms are a property of the site, so they are seeded off the
        # station name rather than the run: every run must agree on them.
        "vs30": ("station", np.array([300.0 + len(s) for s in stations])),
        "z1pt0": ("station", np.array([0.1 * len(s) for s in stations])),
        "z2pt5": ("station", np.array([1.0 * len(s) for s in stations])),
    }
    if with_x:
        coords["x"] = ("station", np.arange(n, dtype=np.int32))
        coords["y"] = ("station", np.arange(n, dtype=np.int32) * 2)
    if supergrid:
        # The first station sits inside the layer and the rest are positively
        # interior -- 0.0, not NaN, which is what the solver writes for a clean
        # station. Float32 with NaN for absent, never an integer sentinel: the
        # reader opens with mask_and_scale=False, so a sentinel would come back
        # raw and read as a plausible depth.
        depth = np.where(np.arange(n) == 0, 750.0, 0.0).astype(np.float32)
        coords["supergrid_depth"] = ("station", depth)
        # Grid points on a 400 m grid, so the flagged station rounds to 1 and a
        # shallower one would round to 0 -- which is why metres decide.
        coords["supergrid_depth_gp"] = (
            "station",
            np.round(depth / 400.0).astype(np.float32),
        )

    groups = {}
    for im in ("PGA", "PGV", "PGD"):
        groups[im] = xr.Dataset(
            {c: ("station", rng.lognormal(-3, 1, n)) for c in SCALAR_COMPONENTS},
            coords=coords,
        )
    for im in ("CAV", "AI", "Ds575", "Ds595"):
        groups[im] = xr.Dataset(
            {c: ("station", rng.lognormal(1, 0.5, n)) for c in DURATION_COMPONENTS},
            coords=coords,
        )
    groups["pSA"] = xr.Dataset(
        {
            c: (("station", "period"), rng.lognormal(-3, 1, (n, len(PERIODS))))
            for c in SCALAR_COMPONENTS
        },
        coords={**coords, "period": ("period", PERIODS)},
    )
    groups["FAS"] = xr.Dataset(
        {
            c: (
                ("station", "frequency"),
                rng.lognormal(-4, 1, (n, len(SIM_FREQUENCIES))),
            )
            for c in FAS_COMPONENTS
        },
        coords={**coords, "frequency": ("frequency", SIM_FREQUENCIES)},
    )

    tree = xr.DataTree.from_dict({f"/{k}": v for k, v in groups.items()})
    tree.attrs = {
        "event": path.parent.parent.name,
        "magnitude": "6.5",
        "hypo_lat": "-44.5",
        "hypo_lon": "171.0",
        "hypo_depth": "10.0",
        "rake": "90.0",
        "dip": "45.0",
        "ztor": "2.0",
        "zbot": "15.0",
        "tect_type": "active_shallow",
        "domain": "POLYGON ((170 -45, 172 -45, 172 -44, 170 -44, 170 -45))",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_netcdf(path, engine="h5netcdf")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A two-event, four-configuration results tree.

    ``sw4`` sees one station fewer than ``emod3d``, as it does in the real
    output, so the tests exercise a ragged station set. Only ``sw4`` reports a
    supergrid depth, and only ``emod3d`` its grid indices, so both of the
    solver-specific per-station terms are exercised in both directions.
    """
    root = tmp_path_factory.mktemp("results")
    emod3d_stations = ["AAAA", "BBBB", "gridpoint1", "gridpoint22"]
    sw4_stations = emod3d_stations[:-1]
    seed = 0
    for event in ("2020p111111", "2021p222222"):
        for solver, stations in (("emod3d", emod3d_stations), ("sw4", sw4_stations)):
            for layers in ("full", "tomography"):
                seed += 1
                write_im_file(
                    root / event / f"{solver}_{layers}" / ingest.IM_FILE,
                    stations,
                    seed,
                    with_x=(solver == "emod3d"),
                    supergrid=(solver == "sw4"),
                )
    return root


def write_csv(path, header, rows):
    """Write a CSV the way the export does: a header line, then bare values.

    ``str`` of a Python float is its shortest round-tripping form, so a value
    written here comes back out of DuckDB's CSV reader as the same double.
    """
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def write_recordings(directory, records):
    """Write the three raw CSVs for ``records``, a list of (event_id, stat_id).

    The measure values are seeded off the record's position, so the same
    (event_id, stat_id) written twice carries two different sets of values and a
    test can tell which of the pair survived a dedup.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        directory / database.EVENTS_CSV,
        ["event_id", "event_name"],
        list(OBSERVED_EVENTS.items()),
    )
    write_csv(
        directory / database.STATIONS_CSV,
        ["stat_id", "stat_name"],
        list(OBSERVED_STATIONS.items()),
    )
    header = (
        ["gm_id", "event_id", "stat_id"]
        + OBSERVED_SCALARS
        + [f"pSA_{p:.12f}" for p in PERIODS]
        + [f"EAS_{f:.12f}" for f in OBS_FREQUENCIES]
    )
    rows = []
    for gm_id, (event_id, stat_id) in enumerate(records, start=1):
        rng = np.random.default_rng(99 + gm_id)
        rows.append(
            [gm_id, event_id, stat_id]
            + rng.lognormal(-3, 1, len(OBSERVED_SCALARS)).tolist()
            + rng.lognormal(-3, 1, len(PERIODS)).tolist()
            + rng.lognormal(-4, 1, len(OBS_FREQUENCIES)).tolist()
        )
    write_csv(directory / database.IM_OBS_CSV, header, rows)
    return directory


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """A stand-in for the recordings directory: the three raw CSVs.

    Only AAAA and BBBB record, and only for two of the three events -- one of
    which was never simulated -- so the tests can tell a correct per-event
    pairing from one that leaks across events.
    """
    return write_recordings(
        tmp_path_factory.mktemp("obs") / "metadata",
        [(e, s) for e in OBSERVED_EVENTS for s in OBSERVED_STATIONS],
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory, results, observed):
    """The assembled database, and a read-only connection to it."""
    work = tmp_path_factory.mktemp("built")
    out = work / "ims.duckdb"
    sites = database.observation_sites(observed)
    runs, manifest = ingest.stage(results, work / "build", sites)
    database.build(out, work / "build", runs, manifest, observed, "rotd50")
    with duckdb.connect(str(out), read_only=True) as con:
        yield con



@pytest.fixture(scope="module")
def built_path(tmp_path_factory, results, observed):
    """The assembled database's path, for a test that opens it itself."""
    work = tmp_path_factory.mktemp("built")
    out = work / "ims.duckdb"
    sites = database.observation_sites(observed)
    runs, manifest = ingest.stage(results, work / "build", sites)
    database.build(out, work / "build", runs, manifest, observed, "rotd50")
    return out
