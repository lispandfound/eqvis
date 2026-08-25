"""The arithmetic under the bias sweep, which is the part that makes a claim.

The curves themselves are checked by looking at them. What has to be right here
is what the curve *means*: a mean taken over the stations that actually have a
residual, a spread that says how much they disagree, and a count that says how
much there was to average -- because a large bias over three stations is not the
same finding as the same bias over thirty, and the figure only distinguishes
them if the count is honest.

Every function tested here is pure, so none of it needs a file or a figure.
"""

import numpy as np
import pytest

from eqvis_workflow.bias import bias_statistics, interval_half_width, match_columns


class TestBiasStatistics:
    def test_the_mean_is_taken_over_the_stations_of_each_period(self):
        residual = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        stats = bias_statistics(residual)
        assert list(stats["count"]) == [3, 3]
        assert stats["mean"] == pytest.approx([2.0, 3.0])

    def test_a_missing_station_is_left_out_rather_than_counted_as_zero(self):
        """The whole reason the sums are masked: NaN is absent, not a residual."""
        residual = np.array([[1.0, 1.0], [3.0, np.nan], [np.nan, 3.0]])
        stats = bias_statistics(residual)
        assert list(stats["count"]) == [2, 2]
        assert stats["mean"] == pytest.approx([2.0, 2.0])

    def test_the_spread_is_the_sample_standard_deviation(self):
        residual = np.array([[1.0], [2.0], [3.0], [4.0]])
        stats = bias_statistics(residual)
        # ddof=1: the stations are a sample of the field, not the whole of it.
        assert stats["sd"] == pytest.approx([np.std([1, 2, 3, 4], ddof=1)])
        assert stats["se"] == pytest.approx(stats["sd"] / 2.0)

    def test_the_spread_ignores_the_missing_stations_it_did_not_average(self):
        """Same values, one padded with NaN: the answer must not move."""
        dense = np.array([[1.0], [2.0], [3.0]])
        padded = np.array([[1.0], [np.nan], [2.0], [3.0], [np.nan]])
        assert bias_statistics(padded)["sd"] == pytest.approx(
            bias_statistics(dense)["sd"]
        )
        assert bias_statistics(padded)["mean"] == pytest.approx(
            bias_statistics(dense)["mean"]
        )

    def test_one_station_gives_a_mean_but_no_spread(self):
        """A single station has a residual; it has no scatter about itself."""
        stats = bias_statistics(np.array([[1.5], [np.nan]]))
        assert list(stats["count"]) == [1]
        assert stats["mean"] == pytest.approx([1.5])
        assert np.isnan(stats["sd"]).all()
        assert np.isnan(stats["se"]).all()

    def test_a_period_no_station_reached_is_nan_rather_than_zero(self):
        """An unscored period must be absent from the curve, not sitting at zero."""
        stats = bias_statistics(np.array([[np.nan], [np.nan]]))
        assert list(stats["count"]) == [0]
        assert np.isnan(stats["mean"]).all()
        assert np.isnan(stats["sd"]).all()

    def test_a_wholly_empty_table_still_returns_a_column_per_period(self):
        """What a fully screened run leaves behind, before the refusal fires."""
        stats = bias_statistics(np.full((3, 4), np.nan))
        assert list(stats["count"]) == [0, 0, 0, 0]
        assert np.isnan(stats["mean"]).all()

    def test_dropping_a_station_is_the_same_as_never_having_had_it(self):
        """What ``--supergrid exclude`` does, expressed as an assertion.

        The screen NaNs the flagged rows rather than reindexing the table, so
        the statistics must be identical to those of the table with the rows
        physically removed -- otherwise the count on the figure and the mean
        above it would describe different station sets.
        """
        residual = np.array([[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]])
        flagged = np.array([False, True, False])

        screened = residual.copy()
        screened[flagged] = np.nan
        kept = residual[~flagged]

        for key in ("count", "mean", "sd", "se"):
            assert bias_statistics(screened)[key] == pytest.approx(
                bias_statistics(kept)[key], nan_ok=True
            )


class TestInterval:
    def test_the_interval_narrows_as_stations_are_added(self):
        few = bias_statistics(np.array([[1.0], [2.0], [3.0]]))
        many = bias_statistics(np.array([[1.0], [2.0], [3.0]] * 10))
        assert interval_half_width(many, 0.95) < interval_half_width(few, 0.95)

    def test_a_wider_confidence_level_gives_a_wider_interval(self):
        stats = bias_statistics(np.array([[1.0], [2.0], [3.0], [4.0]]))
        assert interval_half_width(stats, 0.99) > interval_half_width(stats, 0.95)

    def test_the_interval_is_students_rather_than_normal(self):
        """Over three stations the normal interval is less than half the width.

        Which is exactly the situation at long period, where only a handful of
        records are still usable -- so the difference is not academic.
        """
        stats = bias_statistics(np.array([[1.0], [2.0], [3.0]]))
        # t(0.975, 2) = 4.303, against 1.96 for the normal.
        assert interval_half_width(stats, 0.95) == pytest.approx(
            4.302652729911275 * float(stats["se"][0])
        )

    def test_one_station_has_no_interval(self):
        stats = bias_statistics(np.array([[1.5]]))
        assert np.isnan(interval_half_width(stats, 0.95)).all()


class TestMatchColumns:
    """Lining a recording's period grid up with the run's, or refusing to."""

    def test_a_matching_grid_comes_back_unchanged(self):
        values = np.array([[1.0, 2.0, 3.0]])
        periods = np.array([0.1, 1.0, 5.0])
        matched = match_columns(values, periods, periods)
        assert matched == pytest.approx(values)

    def test_a_column_is_taken_from_the_nearest_period_available(self):
        values = np.array([[1.0, 2.0]])
        periods = np.array([0.1, 1.0])
        matched = match_columns(values, periods, np.array([1.02, 0.101]))
        assert matched == pytest.approx(np.array([[2.0, 1.0]]))

    def test_a_period_that_is_not_really_the_same_period_is_dropped(self):
        """Beyond the tolerance the two grids disagree about what is being scored."""
        values = np.array([[1.0, 2.0]])
        periods = np.array([0.1, 1.0])
        matched = match_columns(values, periods, np.array([0.1, 3.0]))
        assert matched[0, 0] == pytest.approx(1.0)
        assert np.isnan(matched[0, 1])

    def test_the_tolerance_is_relative_to_the_reference_period(self):
        """5% of 0.1 s is 5 ms; 5% of 5 s is 250 ms. The same rule, not the same gap."""
        values = np.array([[1.0]])
        assert np.isnan(match_columns(values, np.array([0.1]), np.array([0.11])))
        assert not np.isnan(match_columns(values, np.array([5.0]), np.array([5.1])))
