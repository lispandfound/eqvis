"""Tests for the arithmetic behind the hazard map.

The drawing is matplotlib's problem. What is tested here is every place the
command turns rates into an answer, because each one has an edge that is silent
if it is wrong: a rupture set that does not add up to the return period asked
for, a curve read off the end of the table it was tabulated on, a rupture id
that means two different ruptures in two different fault systems, and the
identity the augmented panel rests on.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eqvis_workflow import hazard
from eqvis_workflow.display import Display

# A figure drawn at its designed size: no text enlargement, so nothing is
# thinned except by the room the bar actually has.
NATURAL_DISPLAY = Display(size=(3.6, 6.0), dpi=200.0)


def deficit(empirical):
    """The three shares at the only threshold and period these fixtures have."""
    return hazard.deficit_shares(empirical, 0, 0)


class TestExceedanceRate:
    """The running total of rate down a station's block is its hazard curve."""

    def test_each_station_accumulates_from_its_own_zero(self):
        rate = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        starts = np.array([0, 3])
        assert list(hazard.exceedance_rate(rate, starts)) == [1, 3, 6, 10, 30]

    def test_one_station_is_a_plain_cumulative_sum(self):
        rate = np.array([1.0, 2.0, 3.0])
        assert list(hazard.exceedance_rate(rate, np.array([0]))) == [1, 3, 6]


class TestLevelAtRate:
    """Reading the staircase at a rate, and the two ways it can have no answer."""

    motion = np.array([1.0, 0.5, 0.25])
    rate = np.array([0.001, 0.001, 0.001])
    starts = np.array([0])

    def cumulative(self):
        return hazard.exceedance_rate(self.rate, self.starts)

    def test_a_target_landing_on_a_step_gives_that_step(self):
        level = hazard.level_at_rate(
            self.motion, self.cumulative(), self.starts, 0.002
        )
        assert level[0] == pytest.approx(0.5)

    def test_between_two_steps_it_interpolates_in_log_log(self):
        """Halfway between the steps in log rate is halfway in log motion."""
        motion = np.array([1.0, 0.01])
        cumulative = np.array([0.001, 0.1])
        level = hazard.level_at_rate(motion, cumulative, np.array([0]), 0.01)
        assert level[0] == pytest.approx(0.1)  # sqrt(1.0 * 0.01)

    def test_a_rupture_set_too_small_for_the_target_has_no_answer(self):
        """Its rates sum to 0.003/yr, so it cannot speak about 0.01/yr at all.
        That has to be NaN rather than the weakest motion it happens to hold."""
        level = hazard.level_at_rate(
            self.motion, self.cumulative(), self.starts, 0.01
        )
        assert np.isnan(level[0])

    def test_crossing_on_the_first_step_returns_that_motion(self):
        """The target is reached by the strongest motion alone, so the answer is
        a bound: at least this, and the curve says nothing about how much more."""
        level = hazard.level_at_rate(
            self.motion, self.cumulative(), self.starts, 0.0005
        )
        assert level[0] == pytest.approx(1.0)

    def test_stations_are_read_independently(self):
        motion = np.array([1.0, 0.5, 2.0, 1.0])
        rate = np.array([0.001, 0.001, 0.001, 0.001])
        starts = np.array([0, 2])
        cumulative = hazard.exceedance_rate(rate, starts)
        level = hazard.level_at_rate(motion, cumulative, starts, 0.002)
        assert level == pytest.approx([0.5, 1.0])


class TestCurveLevels:
    """The empirical curves are tabulated, so they run out at both ends."""

    thresholds = np.array([0.1, 1.0])

    def test_it_interpolates_between_the_bracketing_thresholds(self):
        curves = np.array([[0.01, 0.0001]])
        level = hazard.curve_levels(self.thresholds, curves, 0.001)
        assert level[0] == pytest.approx(0.316227, rel=1e-4)

    def test_a_target_above_the_first_threshold_has_no_answer(self):
        """The curve starts below the target, so the answer is at a motion
        weaker than anything tabulated -- off the bottom of the table."""
        curves = np.array([[0.001, 0.00001]])
        assert np.isnan(hazard.curve_levels(self.thresholds, curves, 0.1)[0])

    def test_a_target_below_the_last_threshold_has_no_answer(self):
        curves = np.array([[0.01, 0.001]])
        assert np.isnan(hazard.curve_levels(self.thresholds, curves, 1e-9)[0])

    def test_a_curve_that_reaches_zero_is_left_unanswered(self):
        """Zero rate carries no information about where between the thresholds
        it got there, and log-interpolating it would divide by infinity."""
        curves = np.array([[0.01, 0.0]])
        assert np.isnan(hazard.curve_levels(self.thresholds, curves, 0.001)[0])

    def test_rows_are_independent(self):
        curves = np.array([[0.01, 0.0001], [0.001, 0.00001]])
        level = hazard.curve_levels(self.thresholds, curves, 0.001)
        assert np.isfinite(level[0]) and np.isnan(level[1])


class TestRuptureRates:
    """A rupture id names a rupture *within* a fault system, not across them."""

    def parquet(self, tmp_path, systems, ruptures, rates):
        path = tmp_path / "ruptures.parquet"
        pq.write_table(
            pa.table({"fault_system": pa.array(systems, pa.int64()),
                      "rupture": pa.array(ruptures, pa.int64()),
                      "rate": pa.array(rates, pa.float64())}),
            path,
        )
        return path

    def test_only_the_simulated_fault_system_is_read(self, tmp_path):
        """The same id in another fault system is a different rupture with a
        different rate; taking it would silently mis-rate a run."""
        path = self.parquet(tmp_path, [2, 3], [100932, 100932], [5.0, 7.0])
        rates, _ = hazard.rupture_rates(path)
        assert rates == {100932: 7.0}

    def test_the_total_spans_every_fault_system(self, tmp_path):
        """What the caption has to report is the share of the *national* rate."""
        path = self.parquet(tmp_path, [1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0])
        rates, total = hazard.rupture_rates(path)
        assert total == pytest.approx(6.0)
        assert rates == {3: 3.0}


class TestAugmentedCurve:
    """The substitution is a swap of one term in a sum, and has to behave like one."""

    def ground(self):
        # Only the coordinates are read; the rest of Ground is not touched.
        lon = np.array([170.0, 171.0, 170.0, 171.0])
        lat = np.array([-44.0, -44.0, -43.0, -43.0])
        return hazard.Ground(
            con=None, stations={"lon": lon, "lat": lat}, dense=None,
            grid_lon=None, grid_lat=None, blank=None, aspect=1.0, coast=None,
            outlines=None, covered=0.0, total_rate=1.0, step={},
            site_lon=lon, site_lat=lat,
        )

    def test_a_simulation_agreeing_with_the_empirical_subset_changes_nothing(self):
        """If the physics reproduces the regression on the same ruptures, the
        national model must come back untouched -- that is what makes the panel
        readable as an update rather than as a third model."""
        every = np.tile(np.array([0.01, 0.001]), (4, 1))
        subset = np.tile(np.array([0.004, 0.0002]), (4, 1))
        ground = self.ground()
        augmented = hazard.augmented_curve(ground, (every, subset), subset, "substitute")
        assert augmented == pytest.approx(every, rel=1e-6)

    def test_a_weaker_simulation_lowers_the_national_curve(self):
        every = np.tile(np.array([0.01, 0.001]), (4, 1))
        subset = np.tile(np.array([0.004, 0.0002]), (4, 1))
        simulated = subset / 2
        ground = self.ground()
        augmented = hazard.augmented_curve(
            ground, (every, subset), simulated, "substitute"
        )
        assert np.all(augmented < every)

    def test_the_ratio_update_scales_the_whole_curve(self):
        """Unlike the swap, it carries the disagreement onto ruptures that were
        never simulated, so the whole curve moves by the ratio."""
        every = np.tile(np.array([0.01, 0.001]), (4, 1))
        subset = np.tile(np.array([0.004, 0.0002]), (4, 1))
        ground = self.ground()
        augmented = hazard.augmented_curve(
            ground, (every, subset), subset / 2, "ratio"
        )
        assert augmented == pytest.approx(every / 2, rel=1e-6)


class TestMeasureScales:
    """One colour scale per measure, pooled over every map of it in the run."""

    def drawing(self, im, period, years, values):
        return hazard.Drawing(
            im=im, period=period, return_period=years,
            panels=[{"values": np.asarray(values), "at": "stations"}],
        )

    def test_the_scale_spans_every_map_of_the_measure(self):
        """The 2475-year map runs higher than the 475-year one, and both have to
        fit: a scale read off either alone makes the pair incomparable."""
        drawings = [
            self.drawing("pSA", 1.0, 475, [0.01, 0.02, 0.05]),
            self.drawing("pSA", 1.0, 2475, [0.1, 0.2, 0.5]),
        ]
        scale = hazard.measure_scales(drawings, 10)["pSA"]
        assert scale[0] <= 0.01 and scale[-1] >= 0.5

    def test_each_measure_gets_its_own(self):
        """PGA is in g and PGV in cm/s; one scale over both would be meaningless."""
        drawings = [
            self.drawing("PGA", None, 475, [0.01, 0.1]),
            self.drawing("PGV", None, 475, [1.0, 50.0]),
        ]
        scales = hazard.measure_scales(drawings, 10)
        assert set(scales) == {"PGA", "PGV"}
        assert scales["PGV"][-1] > scales["PGA"][-1]

    def test_a_duration_is_scaled_linearly(self):
        """Ds595 is a time, not a quantity spanning decades: log levels would
        put almost the whole country in one bin."""
        drawings = [self.drawing("Ds595", None, 475, np.linspace(5.0, 45.0, 50))]
        scale = hazard.measure_scales(drawings, 10)["Ds595"]
        steps = np.diff(scale)
        assert steps == pytest.approx(steps[0])  # equal steps => linear

    def test_a_spectral_measure_is_scaled_logarithmically(self):
        drawings = [self.drawing("pSA", 1.0, 475, np.geomspace(0.001, 1.0, 50))]
        scale = hazard.measure_scales(drawings, 10)["pSA"]
        assert not np.allclose(np.diff(scale), np.diff(scale)[0])

    def test_a_measure_with_nothing_finite_is_left_out(self):
        """Rather than crashing the whole run on one empty measure."""
        drawings = [self.drawing("PGA", None, 475, [np.nan, np.nan])]
        assert "PGA" not in hazard.measure_scales(drawings, 10)


class TestStyleTicks:
    """A colour bar's ticks are its level boundaries, and there can be too many."""

    def test_a_wide_bar_keeps_every_boundary(self):
        boundaries = np.array([0.1, 0.2, 0.5, 1.0])
        shown = hazard.style_ticks(boundaries, NATURAL_DISPLAY, 12.0)
        assert shown == list(boundaries)

    def test_a_narrow_bar_thins_them(self):
        """Four decades at 1-2-5 is fourteen boundaries, and their labels are
        the widest ones; unthinned they overlap into a smear."""
        boundaries = np.array(
            [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
             0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        )
        shown = hazard.style_ticks(boundaries, NATURAL_DISPLAY, 3.3)
        assert len(shown) < len(boundaries)
        assert shown[0] == boundaries[0] and shown[-1] == boundaries[-1]

    def test_kept_ticks_are_evenly_spaced(self):
        """A BoundaryNorm bar gives every bin the same width, so two ticks one
        bin apart collide however few of them there are. Spacing them from the
        ends leaves exactly that; a stride cannot."""
        boundaries = np.array(
            [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        )
        shown = hazard.style_ticks(boundaries, NATURAL_DISPLAY, 3.3)
        index = [list(boundaries).index(v) for v in shown]
        gaps = set(np.diff(index))
        assert len(gaps) == 1, f"uneven tick spacing: {index}"

    def test_the_limit_is_always_shown(self):
        boundaries = np.geomspace(0.001, 10.0, 13)
        shown = hazard.style_ticks(boundaries, NATURAL_DISPLAY, 3.3)
        assert shown[-1] == boundaries[-1]

    def test_a_symmetric_scale_keeps_its_zero(self):
        """RdBu_r is read against zero; dropping it loses the reference."""
        boundaries = np.round(np.arange(-6, 7) * 0.25, 12)
        shown = hazard.style_ticks(boundaries, NATURAL_DISPLAY, 3.3)
        assert 0.0 in shown


class TestDeficitShares:
    """Deficiency as a ratio of rates at a fixed motion, split three ways."""

    def empirical(self, every, simulable, subset):
        shape = (1, len(every), 1)
        return {
            "threshold": np.array([0.1]), "period": np.array([1.0]),
            "site": np.array(["A", "B"][: len(every)], dtype=object),
            "every": np.reshape(every, shape),
            "simulable": np.reshape(simulable, shape),
            "subset": np.reshape(subset, shape),
        }

    def test_the_three_shares_add_to_a_hundred(self):
        """They are shares of one whole, so anything else is a bookkeeping bug."""
        panels = deficit(self.empirical([1.0, 2.0], [0.4, 1.0], [0.1, 0.25]))
        total = sum(p["values"] for p in panels)
        assert total == pytest.approx([100.0, 100.0])

    def test_it_reads_the_pilot_share(self):
        panels = deficit(self.empirical([1.0], [0.4], [0.1]))
        assert panels[0]["values"][0] == pytest.approx(10.0)
        assert panels[1]["values"][0] == pytest.approx(30.0)  # crustal, unsimulated
        assert panels[2]["values"][0] == pytest.approx(60.0)  # not crustal

    def test_a_site_with_no_hazard_is_not_divided_by_zero(self):
        panels = deficit(self.empirical([0.0], [0.0], [0.0]))
        assert all(np.isnan(p["values"][0]) for p in panels)

    def test_full_coverage_leaves_nothing_missing(self):
        panels = deficit(self.empirical([1.0], [1.0], [1.0]))
        assert panels[0]["values"][0] == pytest.approx(100.0)
        assert panels[1]["values"][0] == pytest.approx(0.0)
        assert panels[2]["values"][0] == pytest.approx(0.0)
