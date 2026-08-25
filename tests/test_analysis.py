"""Tests for the decisions in the heat map, the decomposition and the map.

The drawing is not tested -- that is matplotlib's -- but every choice these
commands make on the caller's behalf is: how a covariate is binned, when a cell
is too thin to colour, what a NULL means, how the fixed-effect design is coded,
and the two guards that stop a figure being quietly wrong.
"""

import numpy as np
import pytest
import typer

from eqvis_workflow import decompose, heatmap, residuals, store


@pytest.fixture(scope="module")
def con(built_path):
    with store.connect(built_path) as connection:
        yield connection


class TestNumericBins:
    def test_quantile_bins_spread_the_data_evenly(self):
        """Equal-width bins would put most stations in one row: distances and
        elevations are heavily skewed."""
        skewed = np.concatenate([np.linspace(0, 10, 90), np.linspace(200, 400, 10)])
        index, labels = heatmap.numeric_bins(skewed, 4, None)
        counts = np.bincount(index.astype(int), minlength=len(labels))
        assert len(labels) == 4
        assert counts.max() / counts.min() < 3

    def test_explicit_edges_override_the_quantiles(self):
        values = np.array([5.0, 15.0, 50.0, 500.0])
        index, labels = heatmap.numeric_bins(values, 6, [0, 10, 100, 1000])
        assert labels == ["0–10", "10–100", "100–1000"]
        assert list(index) == [0, 1, 1, 2]

    def test_a_non_finite_value_is_not_binned(self):
        """A NULL numeric covariate was never derived, so it has no row."""
        index, _ = heatmap.numeric_bins(np.array([1.0, np.nan, 3.0]), 2, None)
        assert np.isnan(index[1])

    def test_a_covariate_with_one_value_is_refused(self):
        with pytest.raises(typer.BadParameter, match="too few distinct values"):
            heatmap.numeric_bins(np.full(10, 3.0), 4, None)


class TestTextBins:
    def test_the_distinct_values_are_the_rows(self):
        index, labels = heatmap.text_bins(np.array(["b", "a", "b"], dtype=object))
        assert labels == ["a", "b"]
        assert list(index) == [1, 0, 1]

    def test_null_becomes_a_row_of_its_own_and_sorts_last(self):
        """"In no basin" is a fact about a site, not a gap in the data."""
        index, labels = heatmap.text_bins(np.array(["a", None, "b"], dtype=object))
        assert labels == ["a", "b", heatmap.ABSENT]
        assert index[1] == len(labels) - 1


class TestGridMeans:
    def test_a_thinly_scored_cell_is_blanked_not_coloured(self):
        """At one or two stations the colour is that station's noise, and a
        reader cannot tell it from a finding."""
        data = {
            "residual": np.array([1.0, 1.0, 1.0, 5.0]),
            "ordinate": np.array([1.0, 1.0, 1.0, 2.0]),
        }
        mean, count, ordinates = heatmap.grid_means(
            data, np.zeros(4), 1, minimum=3
        )
        assert list(ordinates) == [1.0, 2.0]
        assert mean[0, 0] == pytest.approx(1.0)
        assert np.isnan(mean[0, 1])      # only one residual there
        assert count[0, 1] == 1

    def test_a_scalar_measure_gives_one_column(self):
        data = {"residual": np.array([1.0, 3.0])}
        mean, _, ordinates = heatmap.grid_means(data, np.zeros(2), 1, minimum=1)
        assert mean.shape == (1, 1)
        assert mean[0, 0] == pytest.approx(2.0)
        assert len(ordinates) == 1


class TestFixedDesign:
    def test_a_numeric_term_is_centred_so_the_intercept_stays_readable(self):
        """The bias at the mean site, not the bias at a vs30 of zero."""
        data = {"vs30": np.array([200.0, 400.0, 600.0])}
        design, names = decompose.build_fixed(
            data, np.ones(3, dtype=bool), ["vs30"], {}
        )
        assert names == ["intercept", "vs30 (centred)"]
        assert design[:, 1].sum() == pytest.approx(0.0)

    def test_a_text_term_becomes_contrasts_against_a_named_reference(self):
        data = {"basin": np.array(["A", "B", "C"], dtype=object)}
        design, names = decompose.build_fixed(
            data, np.ones(3, dtype=bool), ["basin"], {"basin": "B"}
        )
        assert names == ["intercept", "basin=A", "basin=C"]
        assert list(design[:, 1]) == [1.0, 0.0, 0.0]

    def test_the_reference_defaults_to_the_first_level(self):
        data = {"basin": np.array(["B", "A"], dtype=object)}
        _, names = decompose.build_fixed(data, np.ones(2, dtype=bool), ["basin"], {})
        assert names == ["intercept", "basin=B"]


class TestGuards:
    def test_several_runs_of_one_event_at_one_station_is_warned_about(self, capsys):
        """The shared-observation trap: those residuals share one recording, so
        a site term fitted over them absorbs the recording, not the site."""
        cell = {
            store.EVENT: np.array(["e", "e"]),
            "station": np.array(["S", "S"]),
            "ordinate": np.array([1.0, 1.0]),
        }
        decompose.warn_shared_observation(cell, "a cell")
        assert "share one" in capsys.readouterr().err

    def test_one_run_per_event_and_station_is_not_warned_about(self, capsys):
        cell = {
            store.EVENT: np.array(["e", "e"]),
            "station": np.array(["S", "T"]),
            "ordinate": np.array([1.0, 1.0]),
        }
        decompose.warn_shared_observation(cell, "a cell")
        assert capsys.readouterr().err == ""

    def test_the_map_refuses_to_mix_events_without_being_asked(self):
        """A map has one domain and one extent, so panels across events would be
        comparing different pieces of the country."""
        with pytest.raises(typer.BadParameter, match="spans 2 events"):
            residuals.check_one_event(["a", "b"], False, ["solver"])

    def test_one_event_is_always_fine(self):
        residuals.check_one_event(["a"], False, ["solver"])

    def test_mixing_events_is_allowed_when_asked_for_but_warned_about(self, capsys):
        residuals.check_one_event(["a", "b"], True, ["solver"])
        assert "not comparable" in capsys.readouterr().err

    def test_grouping_by_event_is_taken_as_asking_for_it(self, capsys):
        """Panels explicitly per event are a legitimate small multiple."""
        residuals.check_one_event(["a", "b"], False, [store.EVENT])
        assert "not comparable" in capsys.readouterr().err


class TestPairing:
    def test_two_cells_are_differenced_at_matched_keys(self, con):
        """The recording cancels identically, which is what makes the paired
        difference exact rather than modelled."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        got = decompose.paired(data, ["solver"], ("sw4",), ("emod3d",))
        assert len(got["residual"]) > 0
        assert set(got) >= {"residual", store.EVENT, "station"}
        # A difference of logs of the same observation cannot depend on it.
        assert np.all(np.isfinite(got["residual"]))

    def test_cells_that_share_nothing_are_refused(self, con):
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        with pytest.raises(typer.BadParameter, match="nothing to pair"):
            decompose.paired(data, ["solver"], ("sw4",), ("nosuchsolver",))


class TestPanelBounds:
    def test_the_view_is_the_stations_padded(self):
        observed = {
            "lon": np.array([172.0, 174.0]),
            "lat": np.array([-43.0, -41.0]),
        }
        west, south, east, north = residuals.panel_bounds(observed, {})
        assert west < 172.0 and east > 174.0
        assert south < -43.0 and north > -41.0

    def test_the_domain_caps_it(self):
        """One distant recording would otherwise zoom the map out until the
        interesting part was a few pixels across."""
        observed = {
            "lon": np.array([160.0, 180.0]),
            "lat": np.array([-50.0, -30.0]),
        }
        attrs = {"domain": "POLYGON ((172 -44, 175 -44, 175 -41, 172 -41, 172 -44))"}
        west, south, east, north = residuals.panel_bounds(observed, attrs)
        assert west >= 171.0 and east <= 176.0
        assert south >= -45.0 and north <= -40.0

    def test_an_unparsable_domain_falls_back_to_the_stations(self):
        observed = {"lon": np.array([172.0, 174.0]), "lat": np.array([-43.0, -41.0])}
        got = residuals.panel_bounds(observed, {"domain": "not wkt at all"})
        assert got[0] < 172.0
