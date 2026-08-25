"""Tests for the spectral head-to-head.

The decisions worth testing here are not the drawing -- that is
:mod:`~eqvis_workflow.bias`'s, reused unchanged -- but the three choices this
command makes on the caller's behalf: which cells there are, which one is the
baseline, and which (event, station) pairs every cell is scored over.
"""

import numpy as np
import pytest
import typer
from conftest import OBSERVED_EVENTS

from eqvis_workflow import compare, store


@pytest.fixture(scope="module")
def con(built_path):
    with store.connect(built_path) as connection:
        yield connection


@pytest.fixture(scope="module")
def cells(con):
    return store.group_runs(con, store.select_runs(con), ["solver"])


class TestBaseline:
    def test_the_default_is_the_first_cell_in_sorted_order(self, cells):
        """So the figure reproduces without the caller having to say."""
        assert compare.resolve_baseline(cells, ["solver"], []) == 0

    def test_a_named_cell_is_found(self, cells):
        assert compare.resolve_baseline(cells, ["solver"], [("solver", "sw4")]) == 1

    def test_a_baseline_outside_the_grouping_is_refused(self, cells):
        """A dimension that is not being compared cannot pick out a cell of the
        comparison; the caller wanted --label."""
        with pytest.raises(typer.BadParameter, match="not among the grouping"):
            compare.resolve_baseline(cells, ["solver"], [("layers", "full")])

    def test_a_baseline_matching_no_cell_lists_the_cells(self, cells):
        with pytest.raises(typer.BadParameter, match="matches none of the cells"):
            compare.resolve_baseline(cells, ["solver"], [("solver", "nonesuch")])


class TestCellColours:
    def test_two_cells_read_like_a_bias_figure(self):
        """The same black and orange `bias` uses for a run and its comparison."""
        assert compare.cell_colours(2) == [
            compare.SIM_ONE_BLACK,
            compare.SIM_TWO_ORANGE,
        ]

    @pytest.mark.parametrize("count", [3, 4, 8])
    def test_more_cells_come_off_a_discrete_ramp(self, count):
        colours = compare.cell_colours(count)
        assert len(colours) == count
        assert len(set(map(str, colours))) == count


class TestBalancing:
    def test_a_cell_is_scored_over_only_the_pairs_every_cell_has(self, con):
        """The ragged factorial made concrete: with a cell missing an event, an
        unbalanced comparison differs partly by that event's own bias."""
        found = store.select_runs(con)
        cells = store.group_runs(con, found, ["solver"])
        data = store.read_residuals(con, found, "pSA", "rotd50")
        per_cell = []
        for _, cell_runs in cells:
            keys = {run["run_key"] for run in cell_runs}
            mask = np.isin(data["run_key"], list(keys))
            per_cell.append({name: col[mask] for name, col in data.items()})
        rows = store.common_rows([store.pivot_ordinates(c)[2] for c in per_cell])
        for cell in per_cell:
            matrix, _, used = store.pivot_ordinates(cell, rows)
            assert list(used) == list(rows)
            assert matrix.shape[0] == len(rows)

    def test_the_pairs_are_events_crossed_with_stations_not_stations_alone(self, con):
        """Pairing on the station alone would compare an event against every
        other event's ground motion."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "pSA", "rotd50")
        _, _, rows = store.pivot_ordinates(data)
        assert len(rows) == len(set(zip(data["event"], data["station"])))
        assert all("\x00" in key for key in rows)


class TestPivot:
    def test_a_spectral_measure_gives_one_column_per_ordinate(self, con):
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "pSA", "rotd50")
        matrix, ordinates, rows = store.pivot_ordinates(data)
        assert matrix.shape == (len(rows), len(ordinates))
        assert np.all(np.diff(ordinates) > 0)

    def test_a_scalar_measure_gives_a_single_column(self, con):
        """So the same statistics and the same panels serve both."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        matrix, ordinates, _ = store.pivot_ordinates(data)
        assert matrix.shape[1] == 1
        assert len(ordinates) == 1

    def test_a_fixed_row_order_lets_two_cells_be_differenced_by_position(self, con):
        """What makes the paired difference a subtraction rather than a join."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "pSA", "rotd50")
        _, _, rows = store.pivot_ordinates(data)
        chosen = rows[:1]
        matrix, _, used = store.pivot_ordinates(data, chosen)
        assert list(used) == list(chosen)
        assert matrix.shape[0] == len(chosen)

    def test_a_missing_combination_is_nan_rather_than_dropped(self, con):
        """Every cell has to be columned by the same grid for a difference to
        line up, so gaps are filled rather than closed."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "pSA", "rotd50")
        invented = np.append(
            store.pivot_ordinates(data)[2], "nosuchevent\x00NOSUCHSTATION"
        )
        matrix, _, _ = store.pivot_ordinates(data, invented)
        assert np.all(np.isnan(matrix[-1]))


class TestMeasures:
    def test_fourier_is_refused_with_the_reason(self, con):
        """The grids share no value, so the join is empty; the message has to say
        why rather than return nothing."""
        with pytest.raises(typer.BadParameter, match="share no"):
            store.measure_source("FAS")

    def test_an_unknown_measure_lists_the_ones_that_exist(self):
        with pytest.raises(typer.BadParameter, match="not a measure"):
            store.measure_source("nonsense")

    @pytest.mark.parametrize(
        "im,table,axis", [("pSA", "psa", "period"), ("PGA", "scalars", None)]
    )
    def test_a_measure_resolves_to_its_table_and_axis(self, im, table, axis):
        assert store.measure_source(im) == (table, im, axis)


class TestResiduals:
    def test_the_sign_is_simulation_over_observation(self, con):
        """The repo's convention, the opposite of the literature's. A positive
        residual is the simulation running high."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        one = 0
        row = con.execute(
            """
            SELECT ln(s.PGA / o.PGA)
            FROM runs r
            JOIN runs ro ON ro.kind = 'observed' AND ro.event = r.event
            JOIN scalars s ON s.run_id = r.run_id AND s.component = 'rotd50'
            JOIN scalars o ON o.run_id = ro.run_id AND o.station = s.station
                          AND o.component = ro.observed_component
            WHERE r.run_key = ? AND s.station = ?
            """,
            [data["run_key"][one], data["station"][one]],
        ).fetchone()
        assert data["residual"][one] == pytest.approx(row[0])

    def test_only_events_with_recordings_contribute(self, con):
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        assert set(data["event"]) <= set(OBSERVED_EVENTS.values())

    def test_every_run_dimension_arrives_as_a_column(self, con):
        """So a caller groups on a dimension by name, without this function
        knowing what the dimensions are."""
        found = store.select_runs(con)
        data = store.read_residuals(con, found, "PGA", "rotd50")
        assert {"solver", "layers"} <= set(data)

    def test_a_covariate_can_be_asked_for_by_name(self, con):
        found = store.select_runs(con)
        data = store.read_residuals(
            con, found, "PGA", "rotd50", covariates=("rrup", "vs30")
        )
        assert np.all(np.isfinite(data["rrup"]))
        assert np.all(np.isfinite(data["vs30"]))

    def test_an_unknown_covariate_lists_the_ones_that_exist(self, con):
        with pytest.raises(typer.BadParameter, match="no such covariate"):
            store.read_residuals(
                con, store.select_runs(con), "PGA", "rotd50", covariates=("nope",)
            )
