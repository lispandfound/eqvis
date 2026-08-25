"""Assembling the composite IM database from staged parquet.

The database holds intensity measures and nothing else: simulated runs from the
staged parquet, and the recordings as further runs, so that a residual is a join
between two rows of the same table rather than a join between two schemas. The
event and site *predictors* stay where they are built, in ``events.duckdb``,
which is attached read-only at analysis time.

The recordings are read from the raw CSVs the ground motion database exports --
``im_obs.csv``, one row per recorded ground motion, beside the two dictionaries
its integer ids point into, ``events.csv`` and ``stations.csv``. Those CSVs are
the earliest form the recordings take, so nothing sits between them and here
that could already have made a choice about them. The cost is that they carry
no schema, so what a schema would have refused -- an id no dictionary defines,
a repeated (event, station) pair -- is looked for in :func:`load_recordings`
instead, reported, and dropped.

There are no comparison views and no residual table. The decomposition this
feeds is a mixed-effects regression written outside the database, and a view
that had already chosen a component, a period and a pairing would be in its way
rather than any help.

The recordings become **one observed run per event**, not one run overall. A
station records many earthquakes, so a single observed run would give every
station as many rows sharing a ``(station, component, period)`` key as it has
recordings, and a simulation joined to it would pair with all of them --
quietly comparing an event against every other event's ground motion. Keying
the observed rows by run, like the simulated ones, is what stops that: the run
carries the event.

Two grids do not line up, and the database says so rather than papering over it.
The pSA periods of the simulations and of the recordings are identical, so a
pSA residual is a clean join on exact equality. The Fourier grids are not: the
simulations carry 100 frequencies out to 100 Hz, the recordings 240 out to 24.5
Hz, and no value is shared exactly. Both grids are stored, tagged in
``frequencies``, and resampling one onto the other is left to whoever needs it
-- it is interpolation, not SQL.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from .console import console_warn
from .constants import IM_UNITS, SCALAR_IMS
from .enrich import enrich_stations

if TYPE_CHECKING:  # pragma: no cover - the runtime dependency runs the other way
    from .ingest import Run

# The three CSVs the recordings arrive as. im_obs is normalised: it names its
# earthquake and its station by integer id, and the other two are the
# dictionaries those ids point into.
EVENTS_CSV = "events.csv"
STATIONS_CSV = "stations.csv"
IM_OBS_CSV = "im_obs.csv"
RECORDING_CSVS = (EVENTS_CSV, STATIONS_CSV, IM_OBS_CSV)

# The columns each file is read for, and the type each is read as. Only the keys
# are pinned; the measures are left to the sniffer, which reads them as DOUBLE.
CSV_KEY_TYPES = {
    EVENTS_CSV: {"event_id": "BIGINT", "event_name": "VARCHAR"},
    STATIONS_CSV: {"stat_id": "BIGINT", "stat_name": "VARCHAR"},
    IM_OBS_CSV: {"gm_id": "BIGINT", "event_id": "BIGINT", "stat_id": "BIGINT"},
}

# The wide spectral columns of im_obs: one per period and one per frequency,
# with the ordinate spelled out in the column name after the prefix.
PSA_PREFIX = "pSA_"
EAS_PREFIX = "EAS_"

# The temp table load_recordings puts them in, ids resolved to names.
RECORDINGS = "recordings"

SCHEMA = f"""
CREATE TABLE runs (
    run_id              INTEGER PRIMARY KEY,
    run_key             VARCHAR UNIQUE,
    event               VARCHAR NOT NULL,
    kind                VARCHAR NOT NULL,   -- 'simulated' | 'observed'
    observed_component  VARCHAR,            -- observed runs only
    im_file             VARCHAR,
    file_size           BIGINT,
    file_mtime_ns       BIGINT,
    n_stations          INTEGER,
    n_observation_sites INTEGER,
    magnitude           DOUBLE,
    hypo_lat            DOUBLE,
    hypo_lon            DOUBLE,
    hypo_depth          DOUBLE,
    rake                DOUBLE,
    dip                 DOUBLE,
    ztor                DOUBLE,
    zbot                DOUBLE,
    tect_type           VARCHAR,
    source_wkt          VARCHAR,
    trace_wkt           VARCHAR,
    domain_wkt          VARCHAR,
    ingested_at         TIMESTAMP
);

-- What distinguishes one run of an event from another: solver, layers, and
-- whatever else --extract named. A join table rather than columns, because the
-- dimensions are the caller's regex to choose, not this schema's to fix.
CREATE TABLE run_labels (
    run_id INTEGER NOT NULL,
    name   VARCHAR NOT NULL,
    value  VARCHAR,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE stations (
    station   VARCHAR PRIMARY KEY,
    vs30      FLOAT,
    z1pt0     FLOAT,
    z2pt5     FLOAT,
    -- Derived from the coordinates already in run_stations rather than staged
    -- with the rest, so they can be filled without reconverting anything. NULL
    -- where the input they need was not supplied: a zero elevation would read
    -- as sea level, and a station's basin is not "none" merely because nobody
    -- tested it. See :mod:`~.enrich`.
    elevation FLOAT,    -- m above sea level, nearest NZCVM DEM node
    basin     VARCHAR   -- the basin outline containing the site, if any
);

CREATE TABLE run_stations (
    run_id              INTEGER NOT NULL,
    station             VARCHAR NOT NULL,
    x                   INTEGER,
    y                   INTEGER,
    latitude            DOUBLE,
    longitude           DOUBLE,
    rrup                DOUBLE,
    rjb                 DOUBLE,
    rx                  DOUBLE,
    ry                  DOUBLE,
    hyp                 DOUBLE,
    epi                 DOUBLE,
    is_observation_site BOOLEAN,
    PRIMARY KEY (run_id, station)
);

CREATE TABLE periods (period_index INTEGER, period DOUBLE);
CREATE TABLE frequencies (grid VARCHAR, frequency_index INTEGER, frequency DOUBLE);

CREATE TABLE scalars (
    run_id    INTEGER NOT NULL,
    station   VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    {", ".join(f"{im} FLOAT" for im in SCALAR_IMS)}
);

CREATE TABLE psa (
    run_id    INTEGER NOT NULL,
    station   VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    period    DOUBLE  NOT NULL,
    pSA       FLOAT
);

CREATE TABLE fas (
    run_id    INTEGER NOT NULL,
    station   VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    frequency DOUBLE  NOT NULL,
    FAS       FLOAT
);

CREATE TABLE im_units (im VARCHAR, unit VARCHAR);
CREATE TABLE notes (topic VARCHAR, note VARCHAR);
"""


def sql_literal(value) -> str:
    """A value escaped for a single-quoted SQL string.

    Needed only where DuckDB will not take a bind parameter -- the parquet
    globs of read_parquet, and the paths of read_csv.
    """
    return str(value).replace("'", "''")


def csv_source(path: Path, types: dict[str, str] | None = None) -> str:
    """A ``read_csv`` call over ``path``, with the key columns' types pinned.

    Pinned rather than sniffed, because the sniffer is right about the measures
    and dangerous about the keys. Event names are mostly GeoNet ids like
    ``2012p001887``, but the older ones are bare numbers, so a file holding only
    those would be sniffed as integers -- and an integer never equals the event
    name carried on a simulated run. That is an empty join, not an error.
    """
    call = f"read_csv('{sql_literal(path)}'"
    if types:
        pinned = ", ".join(f"'{name}': '{kind}'" for name, kind in types.items())
        call += f", types = {{{pinned}}}"
    return call + ")"


def csv_columns(con, path: Path) -> list[str]:
    """The column names of a CSV, without reading the whole of it."""
    return [
        row[0]
        for row in con.execute(f"DESCRIBE SELECT * FROM {csv_source(path)}").fetchall()
    ]


def observed_scalars(columns: Iterable[str]) -> dict[str, str | None]:
    """Which scalar measures the recordings carry, in ``scalars`` column order.

    im_obs names its scalars exactly as the simulations do, so the mapping is
    the identity where the column is there and NULL where it is not -- PGD
    always, since the recordings carry no displacement, and anything else a
    particular export happened to leave out.
    """
    present = set(columns)
    return {im: (im if im in present else None) for im in SCALAR_IMS}


def check_recordings(directory: Path) -> None:
    """Refuse a recordings directory that cannot be read, before conversion.

    Only the layout is checked here -- the files are present, and carry the
    columns the build will ask for -- because this runs ahead of hours of HDF5
    conversion and a mistyped ``--observed`` should not be discovered
    afterwards. What needs the records themselves is checked later, in
    :func:`load_recordings`; reaching that a second time is cheap, since the
    staged parquet survives a failed assembly.
    """
    missing = [name for name in RECORDING_CSVS if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"{directory} is not a recordings directory: no "
            f"{', '.join(missing)}. Expected the raw export -- "
            f"{', '.join(RECORDING_CSVS)}"
        )
    with duckdb.connect() as con:
        for name in RECORDING_CSVS:
            columns = csv_columns(con, directory / name)
            absent = [key for key in CSV_KEY_TYPES[name] if key not in columns]
            if absent:
                raise ValueError(
                    f"{directory / name} has no {', '.join(absent)} column; its "
                    f"columns are {', '.join(columns[:8])}"
                )
            if name == IM_OBS_CSV:
                report_measures(directory / name, columns)


def report_measures(path: Path, columns: list[str]) -> None:
    """Say what ``im_obs.csv`` carries, and refuse one with no response spectra.

    A missing scalar is a null column in the database and worth a word. A
    missing pSA grid means the file is not what it claims to be: the pSA
    residual is the point of pairing recordings against simulations at all.
    """
    periods = [column for column in columns if column.startswith(PSA_PREFIX)]
    frequencies = [column for column in columns if column.startswith(EAS_PREFIX)]
    if not periods:
        raise ValueError(
            f"{path} has no {PSA_PREFIX}* columns, so it carries no response "
            "spectra; is it an im_obs export?"
        )
    absent = [im for im, source in observed_scalars(columns).items() if source is None]
    print(
        f"{path.name}: {len(periods)} periods, {len(frequencies)} frequencies"
        + (f"; no {', '.join(absent)}, which will be NULL" if absent else "")
    )
    if not frequencies:
        console_warn(
            f"{path} has no {EAS_PREFIX}* columns; the observed runs will carry "
            "no Fourier spectra"
        )


def load_recordings(con, directory: Path) -> None:
    """Materialise the recordings into a temp table, ids resolved to names.

    A table rather than a view, because it is read four times below -- to count
    each run's stations, for the scalars, and once for each spectral group --
    and the whole of it is a dozen megabytes, so the CSV is parsed once. The
    integer ids are kept here and only here: the dedup below needs them to pick
    a winner, and the database itself is keyed on names throughout.

    LEFT joins rather than inner ones, so that an id the dictionaries do not
    define arrives as a null name to be counted and reported rather than as a
    row that quietly never turns up.
    """
    con.execute(
        f"""
        CREATE TEMP TABLE {RECORDINGS} AS
        SELECT e.event_name AS event, s.stat_name AS station, o.*
        FROM {csv_source(directory / IM_OBS_CSV, CSV_KEY_TYPES[IM_OBS_CSV])} o
        LEFT JOIN {csv_source(directory / EVENTS_CSV, CSV_KEY_TYPES[EVENTS_CSV])} e
               ON e.event_id = o.event_id
        LEFT JOIN {csv_source(directory / STATIONS_CSV, CSV_KEY_TYPES[STATIONS_CSV])} s
               ON s.stat_id = o.stat_id
        """
    )
    drop_unresolved(con, directory)
    drop_repeated_pairs(con, directory)
    kept = con.execute(f"SELECT count(*) FROM {RECORDINGS}").fetchone()[0]
    print(f"{kept:,} recorded ground motions from {directory}")


def drop_unresolved(con, directory: Path) -> None:
    """Drop and report records whose event or station no dictionary defines."""
    for column, key, dictionary in (
        ("event", "event_id", EVENTS_CSV),
        ("station", "stat_id", STATIONS_CSV),
    ):
        rows = con.execute(
            f"SELECT count(*) FROM {RECORDINGS} WHERE {column} IS NULL"
        ).fetchone()[0]
        if not rows:
            continue
        ids = [
            str(row[0])
            for row in con.execute(
                f"SELECT DISTINCT {key} FROM {RECORDINGS} WHERE {column} IS NULL "
                f"ORDER BY {key} LIMIT 6"
            ).fetchall()
        ]
        console_warn(
            f"{rows} records in {directory / IM_OBS_CSV} name a {key} that "
            f"{dictionary} does not define ({', '.join(ids)}); dropping them. "
            "The three files are meant to come from the same export"
        )
    con.execute(f"DELETE FROM {RECORDINGS} WHERE event IS NULL OR station IS NULL")


def drop_repeated_pairs(con, directory: Path) -> None:
    """Keep one record per (event, station), and say so if there was a choice.

    A repeated pair is the fan-out this schema exists to prevent, one level
    down: the observed run would hold two values per (station, component,
    period), and every simulated row joined to it would pair with both. The
    lowest gm_id wins, so which one survives is at least reproducible.
    """
    pairs, extra = con.execute(
        f"""
        SELECT count(*), coalesce(sum(n) - count(*), 0) FROM (
            SELECT count(*) AS n FROM {RECORDINGS}
            GROUP BY event, station HAVING count(*) > 1
        )
        """
    ).fetchone()
    if not pairs:
        return
    console_warn(
        f"{pairs} (event, station) pairs appear more than once in "
        f"{directory / IM_OBS_CSV}; dropping {extra} records, keeping the lowest "
        "gm_id of each. A repeated pair would give the observed run two values "
        "per (station, component, period), and every simulated row joined to it "
        "would pair with both"
    )
    con.execute(
        f"""
        DELETE FROM {RECORDINGS} WHERE rowid IN (
            SELECT rowid FROM (
                SELECT rowid, row_number() OVER (
                    PARTITION BY event, station ORDER BY gm_id, rowid
                ) AS n FROM {RECORDINGS}
            ) WHERE n > 1
        )
        """
    )


def recording_columns(con) -> list[str]:
    """The columns of the loaded recordings table."""
    return [row[0] for row in con.execute(f"DESCRIBE {RECORDINGS}").fetchall()]


def recorded_events(con) -> list[str]:
    """Every event with recordings, which is one observed run each."""
    rows = con.execute(
        f"SELECT DISTINCT event FROM {RECORDINGS} ORDER BY event"
    ).fetchall()
    return [str(row[0]) for row in rows]


def observed_events(directory: Path) -> list[str]:
    """:func:`recorded_events` for a caller holding no connection of its own."""
    with duckdb.connect() as con:
        load_recordings(con, directory)
        return recorded_events(con)


def observation_sites(directory: Path) -> frozenset[str]:
    """Every station that records anything, from ``stations.csv``.

    Read before any conversion happens, because it decides which stations get
    the full set of spectral components staged for them.

    Taken from the station dictionary rather than from the records themselves:
    stations.csv is written as the sites im_obs refers to, so the two agree, and
    it is three kilobytes against thirty megabytes.
    """
    source = csv_source(directory / STATIONS_CSV, CSV_KEY_TYPES[STATIONS_CSV])
    with duckdb.connect() as con:
        rows = con.execute(f"SELECT DISTINCT stat_name FROM {source}").fetchall()
    return frozenset(str(row[0]) for row in rows)


def observed_key(event: str) -> str:
    """The natural key of an event's observed run."""
    return f"observed__{event}"


def assign_run_ids(runs: list[Run], events: list[str]) -> dict[str, int]:
    """A run_id for every run, simulated and observed.

    Numbered over the sorted union rather than in discovery order, so that
    rebuilding from the same inputs gives the same ids. Adding runs renumbers
    the ones that sort after them, which is why ``run_key`` exists: it is the
    stable natural key, and the one to record in an analysis.
    """
    keys = [observed_key(event) for event in events] + [run.key for run in runs]
    return {key: index for index, key in enumerate(sorted(keys), start=1)}


def build(
    out: Path,
    build_dir: Path,
    runs: list[Run],
    manifest: dict,
    observed_dir: Path | None,
    observed_component: str | None,
    with_fas: bool = True,
    extract: str = "",
    dem: Path | None = None,
    basin_file: Path | None = None,
    with_basins: bool = True,
) -> None:
    """Assemble ``out`` from the staged parquet and the recording CSVs.

    Always a full rebuild of the database file: assembly is cheap next to
    conversion, and it means the database can never hold a run the staging
    directory no longer describes.
    """
    staged = [run for run in runs if run.key in manifest["runs"]]
    if not staged:
        raise ValueError("nothing staged; run the conversion first")

    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    con.execute(SCHEMA)
    # Loaded before the run ids are handed out: which events have recordings is
    # only settled once the records are in hand and the unusable ones are gone.
    # With no --observed, there are no recordings and no observed runs at all.
    if observed_dir is None:
        events: list[str] = []
    else:
        load_recordings(con, observed_dir)
        events = recorded_events(con)
    run_ids = assign_run_ids(staged, events)
    con.execute("CREATE TEMP TABLE run_map (run_key VARCHAR, run_id INTEGER)")
    con.executemany("INSERT INTO run_map VALUES (?, ?)", list(run_ids.items()))

    glob = sql_literal(build_dir / "*")
    now = datetime.now(UTC).replace(tzinfo=None)

    print("assembling runs ...")
    con.execute(
        """
        INSERT INTO runs BY NAME
        SELECT m.run_id, r.* EXCLUDE (run_key), r.run_key, ? AS ingested_at
        FROM read_parquet(?, union_by_name := true) r
        JOIN run_map m USING (run_key)
        """,
        [now, f"{build_dir}/*/run.parquet"],
    )

    con.execute(
        f"""
        INSERT INTO run_labels
        SELECT m.run_id, l.name, l.value
        FROM read_parquet('{glob}/run_labels.parquet') l
        JOIN run_map m USING (run_key)
        """
    )

    # Site terms are global. They were verified identical across an event's
    # configurations, but a disagreement would silently pick a winner here, so
    # it is looked for and reported.
    print("assembling stations ...")
    conflicts = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT station FROM read_parquet('{glob}/stations.parquet')
            GROUP BY station
            HAVING count(DISTINCT vs30) > 1
                OR count(DISTINCT z1pt0) > 1
                OR count(DISTINCT z2pt5) > 1
        )
        """
    ).fetchone()[0]
    if conflicts:
        console_warn(
            f"{conflicts} stations have site terms that differ between runs; "
            "taking the first value. Site terms are meant to be a property of "
            "the site, so this is worth looking into"
        )
    con.execute(
        f"""
        INSERT INTO stations (station, vs30, z1pt0, z2pt5)
        SELECT station, any_value(vs30), any_value(z1pt0), any_value(z2pt5)
        FROM read_parquet('{glob}/stations.parquet')
        GROUP BY station
        """
    )

    print("assembling run_stations ...")
    con.execute(
        f"""
        INSERT INTO run_stations
        SELECT m.run_id, g.station, g.x, g.y, g.latitude, g.longitude,
               g.rrup, g.rjb, g.rx, g.ry, g.hyp, g.epi, g.is_observation_site
        FROM read_parquet('{glob}/run_stations.parquet') g
        JOIN run_map m USING (run_key)
        """
    )

    # After run_stations, because that is where the coordinates it derives from
    # live. Site terms the IM files do not carry; see :mod:`~.enrich`.
    enriched = enrich_stations(con, dem, basin_file, with_basins)
    if enriched:
        print("enriching stations ...")
        if "elevation" in enriched:
            moved = enriched["elevation_coordinate_sensitive"]
            print(
                f"  elevation   {enriched['elevation']:>14,} of "
                f"{enriched['stations']:,} ({moved:,} would change cell at the "
                "extreme of their per-run coordinate spread)"
            )
        if "basin" in enriched:
            print(
                f"  basin       {enriched['basin']:>14,} in a basin, "
                f"{enriched['basin_none']:,} in none"
            )

    print("assembling scalars ...")
    con.execute(
        f"""
        INSERT INTO scalars
        SELECT m.run_id, s.station, s.component,
               {", ".join(f"s.{im}" for im in SCALAR_IMS)}
        FROM read_parquet('{glob}/scalars.parquet') s
        JOIN run_map m USING (run_key)
        """
    )

    # Ordered by (run_id, station) so the repeated keys land in runs that
    # DuckDB's dictionary and RLE encodings can actually compress.
    for group, table, axis in (("pSA", "psa", "period"), ("FAS", "fas", "frequency")):
        if group == "FAS" and not with_fas:
            continue
        print(f"assembling {table} ...")
        con.execute(
            f"""
            INSERT INTO {table}
            SELECT m.run_id, p.station, p.component, p.{axis}, p.{group}
            FROM read_parquet('{glob}/{table}.parquet') p
            JOIN run_map m USING (run_key)
            ORDER BY m.run_id, p.station, p.component, p.{axis}
            """
        )

    if observed_dir is not None:
        add_observed(con, events, run_ids, observed_component, now, with_fas)
    add_axes(con, with_fas)
    add_units_and_notes(con, observed_dir, observed_component, with_fas, extract)
    report(con, out, with_fas)
    con.close()


def add_observed(
    con,
    events: list[str],
    run_ids: dict[str, int],
    component: str,
    now: datetime,
    with_fas: bool,
) -> None:
    """Add the recorded intensity measures, as one run per event.

    im_obs is wide -- one column per period and per frequency -- so it is
    unpivoted into the same long tables the simulations use. Every event with
    recordings gets a run, whether or not it has been simulated yet, so the
    observed side does not have to be rebuilt each time a simulation lands.
    """
    print(f"assembling {len(events)} observed runs ...")
    columns = recording_columns(con)
    con.execute("CREATE TEMP TABLE obs_map (event VARCHAR, run_id INTEGER)")
    con.executemany(
        "INSERT INTO obs_map VALUES (?, ?)",
        [(event, run_ids[observed_key(event)]) for event in events],
    )
    con.execute(
        f"""
        INSERT INTO runs (run_id, run_key, event, kind, observed_component,
                          n_stations, ingested_at)
        SELECT m.run_id, 'observed__' || m.event, m.event, 'observed', ?,
               count(*), ?
        FROM obs_map m JOIN {RECORDINGS} o USING (event)
        GROUP BY m.run_id, m.event
        """,
        [component, now],
    )
    con.execute(
        f"""
        INSERT INTO scalars
        SELECT m.run_id, o.station, ?,
               {", ".join(
                   f"CAST(o.{source} AS FLOAT)" if source else "CAST(NULL AS FLOAT)"
                   for source in observed_scalars(columns).values()
               )}
        FROM {RECORDINGS} o JOIN obs_map m USING (event)
        """,
        [component],
    )
    add_observed_spectra(con, "psa", PSA_PREFIX, component, columns)
    if with_fas:
        add_observed_spectra(con, "fas", EAS_PREFIX, "eas", columns)


def add_observed_spectra(
    con, table: str, prefix: str, component: str, columns: list[str]
) -> None:
    """Unpivot one wide spectral block of im_obs into its long table.

    The ordinate is in the column name, after the prefix -- ``pSA_1.000000000000``
    is the response at one second -- so it is the name that becomes the period
    or the frequency, parsed back out of the text it was written as.
    """
    if not any(column.startswith(prefix) for column in columns):
        console_warn(f"the recordings carry no {prefix}* columns; no observed {table}")
        return
    con.execute(
        f"""
        INSERT INTO {table}
        SELECT m.run_id, u.station, ?,
               CAST(substr(u.ordinate, {len(prefix) + 1}) AS DOUBLE),
               CAST(u.value AS FLOAT)
        FROM (UNPIVOT (SELECT event, station, COLUMNS('^{prefix}')
                       FROM {RECORDINGS})
              ON COLUMNS('^{prefix}') INTO NAME ordinate VALUE value) u
        JOIN obs_map m USING (event)
        WHERE u.value IS NOT NULL
        ORDER BY m.run_id, u.station, 4
        """,
        [component],
    )


def add_axes(con, with_fas: bool) -> None:
    """Populate the period and frequency grids, tagged by where they came from."""
    con.execute(
        """
        INSERT INTO periods
        SELECT row_number() OVER (ORDER BY period) - 1, period
        FROM (SELECT DISTINCT period FROM psa)
        """
    )
    if not with_fas:
        return
    for grid, kind in (("sim", "simulated"), ("obs", "observed")):
        con.execute(
            f"""
            INSERT INTO frequencies
            SELECT '{grid}', row_number() OVER (ORDER BY frequency) - 1, frequency
            FROM (
                SELECT DISTINCT f.frequency FROM fas f JOIN runs r USING (run_id)
                WHERE r.kind = '{kind}'
            )
            """
        )


def add_units_and_notes(
    con, observed_dir: Path | None, component: str | None, with_fas: bool, extract: str = ""
) -> None:
    """Record the units, and the things a reader has to know but cannot see."""
    con.executemany("INSERT INTO im_units VALUES (?, ?)", list(IM_UNITS.items()))

    shared = 0
    if with_fas and observed_dir is not None:
        shared = con.execute(
            """
            SELECT count(*) FROM (SELECT frequency FROM frequencies WHERE grid = 'sim')
            SEMI JOIN (SELECT frequency FROM frequencies WHERE grid = 'obs')
            USING (frequency)
            """
        ).fetchone()[0]

    provenance = (
        "Simulated intensity measures come from a tree of HDF5 datatrees, "
        "one per run, decomposed by the --extract regex recorded in the "
        "'extract pattern' note, staged to parquet by eqvis_workflow.ingest "
        "and assembled by eqvis_workflow.database."
    )
    if observed_dir is not None:
        provenance += (
            f" Observed measures come from the raw CSVs in {observed_dir}: "
            f"{IM_OBS_CSV}, one row per recorded ground motion, with "
            f"{EVENTS_CSV} and {STATIONS_CSV} the dictionaries its event_id and "
            "stat_id point into. Those ids are resolved on the way in and not "
            "kept -- event and station here are the names."
        )
    else:
        provenance += (
            " No --observed was supplied, so this database holds only simulated "
            "runs; there is no observed side to pair against."
        )
    provenance += (
        " No predictors are copied here: attach events.duckdb read-only for "
        "the event and site terms."
    )

    notes = [
        ("provenance", provenance),
        (
            "spectral tiers",
            "Storing every component of pSA at every station would be some 500 "
            "million rows for ten events. Instead pSA rotd50 and FAS eas are "
            "held at every station, and the remaining components only at "
            "stations that record something -- run_stations.is_observation_site "
            "says which. So a GROUP BY component over the whole of psa or fas "
            "sees very different station counts per component; that is the tier "
            "rule, not missing data.",
        ),
        (
            "durations have no rotd",
            "CAV, AI, Ds575 and Ds595 are defined only for 000, 090, ver and "
            "geom -- there is no rotd rotation of a duration -- so those "
            "columns are NULL on every rotd row. If the observed component is "
            "recorded as a rotd one, a duration residual has no simulated "
            "counterpart to pair with; pair durations through geom instead.",
        ),
        (
            "period grid",
            "The simulated and observed pSA period grids are identical -- 111 "
            "periods from 0.01 to 20 s, equal as doubles -- so a pSA residual "
            "is a join on (station, period, component) and needs no "
            "interpolation.",
        ),
        (
            "station elevation",
            "stations.elevation is metres above sea level at the nearest node of "
            "the NZCVM DEM, sampled at the mean of the station's per-run "
            "coordinates. Three things to know. The DEM stores depth positive "
            "down, so the value here is its negation -- read the sign backwards "
            "and every station goes below sea level, but the result still "
            "correlates almost perfectly with the truth, so no plot would catch "
            "it. The grid is 0.005 degrees, about 550 m, which smooths summits: "
            "Aoraki reads about 3507 m against a true 3724 m. And bathymetry is "
            "absent -- the sea is clamped to zero -- so a station over water "
            "reads 0.0 m, which is the DEM's answer rather than a missing value. "
            "NULL means no DEM was supplied to the ingest, or the site lies off "
            "the grid. This is the elevation the simulation itself resolved, "
            "which is the right covariate for a residual: not the elevation that "
            "exists.",
        ),
        (
            "station basin",
            "stations.basin is the sedimentary basin outline containing the site, "
            "from the same GeoParquet the maps draw. The outlines are clipped "
            "against each other by priority, so they do not overlap and a site "
            "falls in at most one. NULL means the site is in no basin, which is "
            "a fact about the site and not a gap in the data -- group on it as a "
            "level of its own. Coverage is uneven and thins fast: most stations "
            "are in no basin at all, and among the stations that actually record "
            "a simulated event the populated basins run to only a handful with "
            "more than ten stations each. A GROUP BY basin will therefore give "
            "many one- and two-station groups; those are not worth interpreting.",
        ),
        (
            "basin depth is not in this database",
            "There is no basin depth here, and z1pt0 and z2pt5 are not a "
            "substitute. They are deterministic empirical functions of vs30: "
            "corr(z1pt0, z2pt5) is exactly 1.000000, and each vs30 value maps to "
            "exactly one of each. Putting either beside vs30 in a regression "
            "gives a collinear, uninterpretable fit, and reading either as a "
            "measured basin depth is simply wrong. stations.basin is a "
            "membership label, not a depth.",
        ),
        (
            "site terms vs geometry",
            "vs30, z1pt0 and z2pt5 were verified identical across all four "
            "configurations of an event, so they live once per station in "
            "stations. latitude, longitude and the source distances are NOT: "
            "EMOD3D and SW4 snap stations to their own grids, differing by up "
            "to 0.0016 degrees and 0.14 km of rrup, so they live per run in "
            "run_stations. Join geometry on (run_id, station), not on station. "
            "Observed runs have no run_stations rows -- the recording is at the "
            "same site, so take its geometry from the simulated run.",
        ),
        (
            "extract pattern",
            "The run dimensions were extracted from each file's path with this "
            f"regex: {extract!r}. Every named group became a row in "
            "run_labels. Rebuilding with a different pattern gives different "
            "dimensions, so this is the record of what the labels mean.",
        ),
        (
            "run dimensions are a join table",
            "What distinguishes one run of an event from another -- solver, "
            "layers, realisation, whatever the --extract regex named -- lives "
            "in run_labels as (run_id, name, value) rather than as columns on "
            "runs, because the dimensions belong to the caller's tree, not to "
            "this schema. To filter on one: JOIN run_labels l ON l.run_id = "
            "r.run_id AND l.name = 'solver' AND l.value = 'sw4'. To get them "
            "side by side: SELECT * FROM (SELECT run_id, name, value FROM "
            "run_labels) PIVOT (any_value(value) FOR name IN (...)).",
        ),
        (
            "coverage",
            "runs lists the simulated runs actually found on disk. A missing "
            "combination is simply absent, so a query comparing configurations "
            "should check what it has rather than assume a full factorial: "
            "SELECT event, count(*) FROM runs WHERE kind = 'simulated' "
            "GROUP BY 1.",
        ),
        (
            "run ids",
            "run_id is assigned by sorted order over the whole run set, so a "
            "rebuild from the same inputs reproduces it, but adding runs "
            "renumbers the ones that sort after them. run_key is the stable "
            "natural key; record that in an analysis, not the integer.",
        ),
        (
            "nulls",
            "The IM files use NaN as their fill value; it is converted to SQL "
            "NULL on the way in, so aggregates skip missing values instead of "
            "returning NaN.",
        ),
        (
            "precision",
            "Intensity measure values are FLOAT: about seven significant "
            "digits, far finer than the simulations resolve, and half the size "
            "at this row count. Periods, frequencies, coordinates and distances "
            "are DOUBLE, so equality against a literal period behaves the way "
            "anyone would expect.",
        ),
        (
            "durations run short",
            "Observed Ds575 and Ds595 run some 7 s and 15 s longer than "
            "simulated at the same station. Both are in seconds; this is a "
            "modelling difference, not a unit mismatch.",
        ),
    ]
    if observed_dir is not None:
        notes += [
            (
                "one record per event and station",
                f"{IM_OBS_CSV} holds one row per recorded ground motion -- one "
                "(event, station) pair, no component column, no repeats -- which "
                "is why an observed run's n_stations is simply its row count. A "
                "repeated pair would give the observed run two values per "
                "(station, component, period) and every simulated row joined to "
                "it would pair with both, so repeats are dropped at build time "
                "with the lowest gm_id winning, and a warning printed. Records "
                "naming an event_id or stat_id the dictionaries do not define "
                "are dropped the same way, so a row count here can be short of "
                "the CSV's; the build log says by how much.",
            ),
            (
                "one observed run per event",
                "The recordings are stored as one run per event, not one run "
                "overall. A station records many earthquakes, so a single "
                "observed run would give it several rows sharing a (station, "
                "component, period) key, and a simulation joined to it would "
                "pair with all of them -- comparing one event against every "
                "other event's ground motion. Always pair through the run: join "
                "the observed run whose event matches the simulated run's event.",
            ),
            (
                "observed component",
                f"The observed measures are recorded as component {component!r}, "
                f"supplied by --observed-component. {IM_OBS_CSV} carries no "
                "component column and its documentation does not say which "
                "convention it uses, so this is an assertion by whoever ran the "
                "ingest, not something read out of the data. Comparing it against "
                "the simulated components does not settle it either: the median "
                "ln(obs/sim) of PGA is smallest for rotd100, but rotd50, 000 and "
                "geom sit within 0.02 of each other, well inside the 0.73 scatter.",
            ),
            (
                "frequency grids",
                "The Fourier grids do NOT match. Simulations carry 100 frequencies "
                "from 0.1 to 100 Hz; recordings carry 240 from 0.1 to 24.5 Hz; "
                f"{shared} values are shared exactly (three agree to within a part "
                "in a million, the rest not at all). Both are stored, tagged in "
                "the frequencies table by grid ('sim', 'obs'). A join on frequency "
                "will therefore return nothing. Resampling one grid onto the other "
                "is log-log interpolation and is deliberately left outside this "
                "database.",
            ),
        ]
    con.executemany("INSERT INTO notes VALUES (?, ?)", notes)


def report(con, out: Path, with_fas: bool) -> None:
    """Print what was built, and what is missing from it."""
    tables = ["runs", "stations", "run_stations", "scalars", "psa"]
    if with_fas:
        tables.append("fas")
    print(f"\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    for table in tables:
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<13} {count:>14,}")

    simulated, observed = con.execute(
        """
        SELECT count(*) FILTER (kind = 'simulated'),
               count(*) FILTER (kind = 'observed') FROM runs
        """
    ).fetchone()
    print(f"\n{simulated} simulated runs, {observed} observed runs")
    dimensions = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT name FROM run_labels ORDER BY name"
        ).fetchall()
    ]
    if dimensions:
        print(f"  dimensions: {', '.join(dimensions)}")
    rows = con.execute(
        """
        SELECT r.event, count(*) AS n, string_agg(d.dimensions, ', ' ORDER BY d.dimensions) AS present
        FROM runs r
        JOIN (
            SELECT run_id, string_agg(value, '/' ORDER BY name) AS dimensions
            FROM run_labels GROUP BY run_id
        ) d USING (run_id)
        WHERE r.kind = 'simulated'
        GROUP BY r.event ORDER BY r.event
        """
    ).fetchall()
    for event, count, present in rows:
        print(f"  {event:<14} {count}  {present}")

    # What can actually be compared: a simulated run whose event has recordings.
    pairable = con.execute(
        """
        SELECT count(*) FROM runs s
        WHERE s.kind = 'simulated'
          AND EXISTS (SELECT 1 FROM runs o
                      WHERE o.kind = 'observed' AND o.event = s.event)
        """
    ).fetchone()[0]
    print(f"  {pairable} of {simulated} simulated runs have recordings to pair with")
