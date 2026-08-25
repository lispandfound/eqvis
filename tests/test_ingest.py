"""Tests for the composite IM database.

Built around a synthetic results tree rather than real output: the fixtures
below write IM files in the same datatree shape the workflow produces, small
enough that a test can assert on every value, and with the awkward parts of the
real data deliberately reproduced -- a solver whose station set is a subset of
the other's, a measure with no rotd components, and observed Fourier
frequencies that do not line up with the simulated ones.

The recordings are written as the three raw CSVs the ground motion database
exports, ids and all, since that is what the ingest reads.
"""

import shutil

import duckdb
import numpy as np
import pytest
import typer
import xarray as xr
from conftest import (
    OBS_FREQUENCIES,
    OBSERVED_EVENTS,
    OBSERVED_STATIONS,
    PERIODS,
    SIM_FREQUENCIES,
    write_csv,
    write_im_file,
    write_recordings,
)

from eqvis_workflow import database, ingest


def labels_of(run):
    """A run's dimensions as a dict, for readable assertions."""
    return dict(run.labels)


class TestDiscovery:
    def test_finds_every_run(self, results):
        runs = ingest.discover(results)
        assert len(runs) == 8
        assert {(labels_of(r)["solver"], labels_of(r)["layers"]) for r in runs} == {
            ("emod3d", "full"), ("emod3d", "tomography"),
            ("sw4", "full"), ("sw4", "tomography"),
        }
        assert {r.event for r in runs} == {"2020p111111", "2021p222222"}

    def test_finds_a_realisation_level(self, tmp_path):
        """The workflow writes R<n>/<variant>/; the upload flattens it away."""
        path = tmp_path / "r" / "2020p111111" / "R2" / "sw4_full" / ingest.IM_FILE
        write_im_file(path, ["AAAA"], 1, with_x=False)
        (run,) = ingest.discover(tmp_path / "r")
        assert run.event == "2020p111111"
        assert labels_of(run) == {"realisation": "R2", "solver": "sw4", "layers": "full"}

    def test_a_custom_pattern_names_its_own_dimensions(self, results):
        """Nothing assumes solver/layers: the regex decides what a run is."""
        runs = ingest.discover(
            results, r"(?P<event>[^/]+)/(?P<variant>[^/]+)/intensity_measures\.h5$"
        )
        assert len(runs) == 8
        assert labels_of(runs[0]).keys() == {"variant"}
        assert {labels_of(r)["variant"] for r in runs} == {
            "emod3d_full", "emod3d_tomography", "sw4_full", "sw4_tomography",
        }

    def test_the_event_falls_back_to_the_file_attribute(self, results):
        """A regex describing only the tail still knows which earthquake it is."""
        runs = ingest.discover(results, r"(?P<variant>[^/]+)/intensity_measures\.h5$")
        assert {r.event for r in runs} == {"2020p111111", "2021p222222"}

    def test_a_pattern_that_matches_nothing_finds_nothing(self, results):
        assert ingest.discover(results, r"(?P<x>nonsense)/never\.h5$") == []

    def test_an_ambiguous_pattern_is_refused(self, results):
        """A regex that cannot tell two files apart would silently lose one."""
        with pytest.raises(typer.BadParameter, match="does not tell"):
            ingest.discover(results, r"(?P<solver>emod3d|sw4)[^/]*/intensity_measures\.h5$")

    def test_a_pattern_with_no_groups_is_refused(self, results):
        with pytest.raises(typer.BadParameter, match="no named groups"):
            ingest.discover(results, r"intensity_measures\.h5$")

    def test_a_bad_regex_is_refused(self, results):
        with pytest.raises(typer.BadParameter, match="not a valid regex"):
            ingest.discover(results, r"(?P<solver>[")

    def test_nothing_to_find_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="matches --extract"):
            ingest.stage(tmp_path / "empty", tmp_path / "b", frozenset())


@pytest.fixture
def scratch_results(tmp_path, results):
    """A private copy of the results tree, for tests that change it."""
    root = tmp_path / "results"
    shutil.copytree(results, root)
    return root


class TestStaging:
    def test_unchanged_runs_are_skipped(self, tmp_path, scratch_results, observed):
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        _, first = ingest.stage(results, build, sites)
        stamps = {k: v["stamp"] for k, v in first["runs"].items()}

        _, second = ingest.stage(results, build, sites)
        assert {k: v["stamp"] for k, v in second["runs"].items()} == stamps

        # Touching one file forces that run to reconvert and no other.
        target = results / "2020p111111" / "sw4_full" / ingest.IM_FILE
        target.touch()
        runs, third = ingest.stage(results, build, sites)
        key = next(r.key for r in runs if r.im_file == target)
        assert third["runs"][key]["stamp"] != stamps[key]
        assert all(
            third["runs"][k]["stamp"] == stamps[k] for k in stamps if k != key
        )

    def test_a_vanished_run_leaves_the_manifest(self, tmp_path, scratch_results, observed):
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        ingest.stage(results, build, sites)
        shutil.rmtree(results / "2021p222222")
        runs, manifest = ingest.stage(results, build, sites)
        assert len(runs) == 4
        assert not any("2021p222222" in key for key in manifest["runs"])

    def test_only_scopes_conversion(self, tmp_path, scratch_results, observed):
        """--only converts the selected runs and leaves the rest untouched."""
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        _, first = ingest.stage(results, build, sites)
        stamps = {k: v["stamp"] for k, v in first["runs"].items()}

        # Touch one file in each event; only the selected event may reconvert.
        target = results / "2020p111111" / "sw4_full" / ingest.IM_FILE
        other = results / "2021p222222" / "sw4_full" / ingest.IM_FILE
        target.touch()
        other.touch()

        runs, third = ingest.stage(
            results, build, sites, only=(("event", "2020p111111"),)
        )
        key = next(r.key for r in runs if r.im_file == target)
        other_key = next(r.key for r in runs if r.im_file == other)
        assert third["runs"][key]["stamp"] != stamps[key]
        assert third["runs"][other_key]["stamp"] == stamps[other_key]

    def test_only_matches_solver_labels(
        self, tmp_path, scratch_results, observed
    ):
        """A solver selection covers both events; the excluded solver is left alone."""
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        _, first = ingest.stage(results, build, sites)
        stamps = {k: v["stamp"] for k, v in first["runs"].items()}

        sw4 = results / "2020p111111" / "sw4_full" / ingest.IM_FILE
        emod3d = results / "2020p111111" / "emod3d_full" / ingest.IM_FILE
        sw4.touch()
        emod3d.touch()

        runs, third = ingest.stage(results, build, sites, only=(("solver", "sw4"),))
        sw4_key = next(r.key for r in runs if r.im_file == sw4)
        emod3d_key = next(r.key for r in runs if r.im_file == emod3d)
        assert third["runs"][sw4_key]["stamp"] != stamps[sw4_key]
        assert third["runs"][emod3d_key]["stamp"] == stamps[emod3d_key]

    def test_force_reconverts_despite_an_unchanged_stamp(
        self, tmp_path, scratch_results, observed, capsys
    ):
        """--force makes the selected runs convert even though nothing changed."""
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        ingest.stage(results, build, sites)

        ingest.stage(
            results, build, sites, only=(("event", "2020p111111"),), force=True
        )
        out = capsys.readouterr().out
        assert "4 to convert" in out
        assert "4 selected" in out

    def test_a_bad_only_value_is_refused(self):
        with pytest.raises(typer.BadParameter, match="name=value"):
            ingest.parse_only(["sw4"])

    def test_only_selecting_nothing_warns(
        self, tmp_path, scratch_results, observed, capsys
    ):
        results = scratch_results
        sites = database.observation_sites(observed)
        build = tmp_path / "build"
        ingest.stage(results, build, sites)
        ingest.stage(results, build, sites, only=(("event", "nope"),))
        assert "selected none" in capsys.readouterr().err


class TestSchema:
    def test_runs_hold_simulated_and_observed(self, built):
        simulated, observed = built.execute(
            "SELECT count(*) FILTER (kind='simulated'), "
            "count(*) FILTER (kind='observed') FROM runs"
        ).fetchone()
        assert simulated == 8
        # One per event with recordings, including the event never simulated.
        assert observed == 2

    def test_source_predictors_come_off_the_file(self, built):
        row = built.execute(
            "SELECT magnitude, dip, ztor, tect_type, domain_wkt FROM runs "
            "WHERE kind='simulated' LIMIT 1"
        ).fetchone()
        assert row[:3] == (6.5, 45.0, 2.0)
        assert row[3] == "active_shallow"
        assert row[4].startswith("POLYGON")

    def test_run_key_is_unique_and_stable(self, built):
        keys = [r[0] for r in built.execute("SELECT run_key FROM runs").fetchall()]
        assert len(keys) == len(set(keys))
        assert "observed__2020p111111" in keys

    def test_site_terms_are_global_and_agree(self, built):
        rows = built.execute(
            "SELECT station, vs30, z1pt0 FROM stations ORDER BY station"
        ).fetchall()
        # The union of both solvers' station sets, once each.
        assert [r[0] for r in rows] == ["AAAA", "BBBB", "gridpoint1", "gridpoint22"]
        assert rows[0][1] == pytest.approx(304.0)  # 300 + len("AAAA")

    def test_geometry_is_per_run(self, built):
        """Each run gets its own row per station: solvers snap to their own grids."""
        counts = built.execute(
            """
            SELECT l.value, count(*) FROM run_stations rs
            JOIN run_labels l ON l.run_id = rs.run_id AND l.name = 'solver'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        assert dict(counts) == {"emod3d": 4 * 4, "sw4": 4 * 3}

    def test_dimensions_are_a_join_table(self, built):
        names = built.execute(
            "SELECT DISTINCT name FROM run_labels ORDER BY name"
        ).fetchall()
        assert [n[0] for n in names] == ["layers", "solver"]
        # Observed runs have no dimensions: there is only one recording.
        orphans = built.execute(
            """
            SELECT count(*) FROM run_labels l JOIN runs r USING (run_id)
            WHERE r.kind = 'observed'
            """
        ).fetchone()[0]
        assert orphans == 0

    def test_sw4_stations_are_a_subset_of_emod3d(self, built):
        extra = built.execute(
            """
            WITH by_solver AS (
                SELECT rs.station, r.event, l.value AS solver
                FROM run_stations rs
                JOIN runs r USING (run_id)
                JOIN run_labels l ON l.run_id = r.run_id AND l.name = 'solver'
            )
            SELECT count(*) FROM (
                SELECT station, event FROM by_solver WHERE solver = 'sw4'
                EXCEPT
                SELECT station, event FROM by_solver WHERE solver = 'emod3d')
            """
        ).fetchone()[0]
        assert extra == 0

    def test_x_and_y_are_null_where_the_solver_omits_them(self, built):
        emod3d, sw4 = built.execute(
            """
            SELECT count(rs.x) FILTER (l.value='emod3d'),
                   count(rs.x) FILTER (l.value='sw4')
            FROM run_stations rs
            JOIN run_labels l ON l.run_id = rs.run_id AND l.name = 'solver'
            """
        ).fetchone()
        assert emod3d == 16
        assert sw4 == 0


class TestTiers:
    def test_grid_component_reaches_every_station(self, built):
        """rotd50 pSA is stored at every station of every run."""
        rows = built.execute(
            """
            SELECT count(DISTINCT (p.run_id, p.station)) FROM psa p
            JOIN runs r USING (run_id)
            WHERE r.kind = 'simulated' AND p.component = 'rotd50'
            """
        ).fetchone()[0]
        assert rows == 4 * 4 + 4 * 3  # every station of every run

    def test_other_components_only_at_observation_sites(self, built):
        stations = built.execute(
            """
            SELECT DISTINCT p.station FROM psa p JOIN runs r USING (run_id)
            WHERE r.kind = 'simulated' AND p.component = '000' ORDER BY 1
            """
        ).fetchall()
        assert [s[0] for s in stations] == ["AAAA", "BBBB"]

    def test_observation_sites_are_flagged(self, built):
        flagged = built.execute(
            "SELECT DISTINCT station FROM run_stations "
            "WHERE is_observation_site ORDER BY 1"
        ).fetchall()
        assert [s[0] for s in flagged] == ["AAAA", "BBBB"]


class TestNulls:
    def test_a_measure_without_rotd_is_null_there(self, built):
        """Durations have no rotd rotation, so those cells must be NULL."""
        total, defined = built.execute(
            """
            SELECT count(*), count(Ds575) FROM scalars s JOIN runs r USING (run_id)
            WHERE r.kind = 'simulated' AND s.component = 'rotd50'
            """
        ).fetchone()
        assert total > 0
        assert defined == 0

    def test_aggregates_are_not_poisoned(self, built):
        """A NaN fill would make this NaN rather than a mean."""
        mean = built.execute(
            "SELECT avg(Ds575) FROM scalars WHERE component = 'geom'"
        ).fetchone()[0]
        assert mean is not None and np.isfinite(mean)


class TestObserved:
    def test_one_run_per_event(self, built):
        rows = built.execute(
            "SELECT event, n_stations FROM runs WHERE kind='observed' ORDER BY event"
        ).fetchall()
        assert rows == [("2020p111111", 2), ("2022p333333", 2)]

    def test_component_is_recorded(self, built):
        components = built.execute(
            "SELECT DISTINCT observed_component FROM runs WHERE kind='observed'"
        ).fetchall()
        assert components == [("rotd50",)]

    def test_pgd_is_null_because_the_recordings_lack_it(self, built):
        pga, pgd = built.execute(
            """
            SELECT count(PGA), count(PGD) FROM scalars s JOIN runs r USING (run_id)
            WHERE r.kind = 'observed'
            """
        ).fetchone()
        assert pga == 4
        assert pgd == 0

    def test_residuals_pair_within_the_event_only(self, built):
        """The bug this schema exists to prevent.

        2021p222222 was simulated but never recorded, and 2022p333333 was
        recorded but never simulated. A pairing that went through the station
        alone would match both events' recordings to every simulation.
        """
        rows = built.execute(
            """
            SELECT rs.event, count(*) FROM psa s
            JOIN runs rs ON rs.run_id = s.run_id AND rs.kind = 'simulated'
            JOIN runs ro ON ro.kind = 'observed' AND ro.event = rs.event
            JOIN psa o ON o.run_id = ro.run_id AND o.station = s.station
                      AND o.period = s.period AND o.component = s.component
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        # Only the one event that was both simulated and recorded, and only its
        # two recording stations, at four periods, over four runs.
        assert rows == [("2020p111111", 2 * len(PERIODS) * 4)]


class TestAxes:
    def test_periods_are_shared_between_simulated_and_observed(self, built):
        sim = built.execute(
            "SELECT DISTINCT p.period FROM psa p JOIN runs r USING (run_id) "
            "WHERE r.kind='simulated' ORDER BY 1"
        ).fetchall()
        obs = built.execute(
            "SELECT DISTINCT p.period FROM psa p JOIN runs r USING (run_id) "
            "WHERE r.kind='observed' ORDER BY 1"
        ).fetchall()
        assert sim == obs == [(p,) for p in PERIODS]

    def test_frequency_grids_are_kept_apart(self, built):
        grids = dict(
            built.execute(
                "SELECT grid, count(*) FROM frequencies GROUP BY 1"
            ).fetchall()
        )
        assert grids == {"sim": len(SIM_FREQUENCIES), "obs": len(OBS_FREQUENCIES)}

    def test_the_frequency_grids_do_not_join(self, built):
        """The mismatch is real and the database must not hide it."""
        shared = built.execute(
            """
            SELECT count(*) FROM (SELECT frequency FROM frequencies WHERE grid='sim')
            SEMI JOIN (SELECT frequency FROM frequencies WHERE grid='obs')
            USING (frequency)
            """
        ).fetchone()[0]
        assert shared == 0
        note = built.execute(
            "SELECT note FROM notes WHERE topic = 'frequency grids'"
        ).fetchone()[0]
        assert "do NOT match" in note


class TestFidelity:
    def test_values_match_the_source_file_exactly(self, built):
        """Every staged value is the h5 value, to float32."""
        checked = 0
        for run_id, path in built.execute(
            "SELECT run_id, im_file FROM runs WHERE kind='simulated'"
        ).fetchall():
            tree = xr.open_datatree(path, engine="h5netcdf", mask_and_scale=False)
            for station, component, period, value in built.execute(
                "SELECT station, component, period, pSA FROM psa WHERE run_id = ?",
                [run_id],
            ).fetchall():
                reference = np.float32(
                    tree["pSA"].to_dataset()[component]
                    .sel(station=station, period=period)
                    .values
                )
                assert np.float32(value) == reference
                checked += 1
            tree.close()
        assert checked > 0


class TestProvenance:
    def test_units_are_recorded(self, built):
        units = dict(built.execute("SELECT im, unit FROM im_units").fetchall())
        assert units["PGA"] == "g"
        assert units["PGV"] == "cm/s"

    def test_the_notes_explain_the_traps(self, built):
        topics = {t[0] for t in built.execute("SELECT topic FROM notes").fetchall()}
        assert {
            "one observed run per event",
            "spectral tiers",
            "observed component",
            "frequency grids",
            "site terms vs geometry",
        } <= topics


class TestRecordings:
    """Reading the raw CSVs: the ids, and what they can get wrong."""

    def test_ids_are_resolved_to_names(self, built):
        """The database is keyed on names throughout; no id reaches it."""
        rows = built.execute(
            "SELECT DISTINCT event FROM runs WHERE kind='observed' ORDER BY 1"
        ).fetchall()
        assert [r[0] for r in rows] == sorted(OBSERVED_EVENTS.values())
        stations = built.execute(
            """
            SELECT DISTINCT s.station FROM scalars s JOIN runs r USING (run_id)
            WHERE r.kind = 'observed' ORDER BY 1
            """
        ).fetchall()
        assert [s[0] for s in stations] == sorted(OBSERVED_STATIONS.values())

    def test_sites_come_from_the_station_dictionary(self, observed):
        assert database.observation_sites(observed) == frozenset(
            OBSERVED_STATIONS.values()
        )

    def test_only_events_with_recordings_get_a_run(self, observed):
        assert database.observed_events(observed) == sorted(OBSERVED_EVENTS.values())

    def test_a_missing_csv_is_refused(self, tmp_path, observed):
        directory = tmp_path / "partial"
        shutil.copytree(observed, directory)
        (directory / database.IM_OBS_CSV).unlink()
        with pytest.raises(ValueError, match="not a recordings directory"):
            database.check_recordings(directory)

    def test_a_csv_missing_a_key_column_is_refused(self, tmp_path, observed):
        directory = tmp_path / "keyless"
        shutil.copytree(observed, directory)
        write_csv(
            directory / database.STATIONS_CSV, ["station"], [("AAAA",), ("BBBB",)]
        )
        with pytest.raises(ValueError, match="has no stat_id"):
            database.check_recordings(directory)

    def test_recordings_without_psa_are_refused(self, tmp_path, observed):
        """A file with no response spectra is not an im_obs export."""
        directory = tmp_path / "no-psa"
        shutil.copytree(observed, directory)
        write_csv(
            directory / database.IM_OBS_CSV,
            ["gm_id", "event_id", "stat_id", "PGA"],
            [(1, 7, 5, 0.1)],
        )
        with pytest.raises(ValueError, match="no pSA_. columns"):
            database.check_recordings(directory)

    def test_an_unresolvable_id_is_dropped(self, tmp_path):
        """A record naming an id no dictionary defines is reported, not silent."""
        directory = write_recordings(
            tmp_path / "orphans", [(7, 5), (7, 999), (404, 2)]
        )
        with duckdb.connect() as con:
            database.load_recordings(con, directory)
            rows = con.execute(
                f"SELECT event, station FROM {database.RECORDINGS}"
            ).fetchall()
        assert rows == [("2020p111111", "AAAA")]

    def test_a_repeated_pair_keeps_the_lowest_gm_id(self, tmp_path):
        """The fan-out this schema exists to prevent, one level down."""
        directory = write_recordings(tmp_path / "repeats", [(7, 5), (7, 5), (3, 2)])
        with duckdb.connect() as con:
            database.load_recordings(con, directory)
            rows = con.execute(
                f"SELECT gm_id, event, station FROM {database.RECORDINGS} "
                "ORDER BY gm_id"
            ).fetchall()
        assert rows == [(1, "2020p111111", "AAAA"), (3, "2022p333333", "BBBB")]

    def test_a_numeric_event_name_stays_text(self, tmp_path, observed):
        """The older events are bare numbers; sniffed as integers they never join."""
        directory = tmp_path / "numeric"
        shutil.copytree(observed, directory)
        write_csv(
            directory / database.EVENTS_CSV,
            ["event_id", "event_name"],
            [(7, 3631363), (3, 3713255)],
        )
        assert database.observed_events(directory) == ["3631363", "3713255"]


class TestRunIds:
    def test_assignment_is_reproducible(self, results, observed):
        runs = ingest.discover(results)
        events = database.observed_events(observed)
        assert database.assign_run_ids(runs, events) == database.assign_run_ids(
            list(reversed(runs)), events
        )
