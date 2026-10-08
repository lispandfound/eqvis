"""The physics and the statistics under the radiation figure.

The drawing is checked by looking at it. What has to be right here is what the
figure *claims*: that the pattern is the double couple's and not something with
the same general shape, that the two nodal planes are one pattern rather than
two, that the strike is recovered from a geometry whose vertex order carries no
convention, and -- the part that stops the figure being read as a result it is
not -- that a sweep peak is scored against the distribution of sweep peaks.
"""

import numpy as np
import pytest
import shapely

from eqvis_workflow.radiation import (
    angular_offset,
    correlation,
    largest_gap,
    log_amplitude,
    mechanism_from_attributes,
    radiation_amplitude,
    strike_sweep,
    sweep_null,
)


class TestRadiationAmplitude:
    """The far-field S pattern, checked against what a double couple must do."""

    def test_the_two_nodal_planes_radiate_the_same_pattern(self):
        """The ambiguity the figure exists to declare, as an assertion.

        A double couple has no preferred plane: NP1 and NP2 of one tensor are
        the same source and radiate identically. If this ever came apart, the
        command would be quietly claiming to tell them apart -- which no
        amplitude measurement can.
        """
        azimuth = np.linspace(0.0, 360.0, 145)
        # The BayesISOLA solution for Milford Sound 2026, both planes as quoted.
        np1 = radiation_amplitude(248.8, 62.5, 134.2, azimuth, 50.0)
        np2 = radiation_amplitude(4.2, 50.5, 36.8, azimuth, 50.0)
        # The planes are quoted to one decimal, so they agree to rounding
        # rather than exactly; the pattern peaks at 1, so this is 0.1%.
        assert np.abs(np1 - np2).max() < 5e-3

    def test_the_pattern_is_non_negative(self):
        """It is a magnitude, sqrt(F_SV^2 + F_SH^2), not a signed lobe."""
        azimuth = np.linspace(0.0, 360.0, 361)
        assert (radiation_amplitude(30.0, 45.0, 90.0, azimuth, 60.0) >= 0).all()

    def test_the_pattern_turns_with_the_strike(self):
        """Rotating the strike rotates the lobes by the same angle."""
        azimuth = np.linspace(0.0, 360.0, 721)
        base = radiation_amplitude(0.0, 60.0, 90.0, azimuth, 55.0)
        turned = radiation_amplitude(40.0, 60.0, 90.0, azimuth + 40.0, 55.0)
        assert base == pytest.approx(turned, abs=1e-12)

    def test_a_pure_strike_slip_source_has_four_lobes(self):
        """Vertical, horizontal slip: the textbook quadrantal pattern.

        Counted on a circle, not on a line: the azimuths are half-open so that
        the lobe sitting on north is one maximum rather than two endpoints.
        """
        azimuth = np.arange(0.0, 360.0, 0.1)
        amplitude = radiation_amplitude(0.0, 90.0, 0.0, azimuth, 90.0)
        higher = (amplitude > np.roll(amplitude, 1)) & (
            amplitude > np.roll(amplitude, -1)
        )
        assert int(higher.sum()) == 4

    def test_the_log_pattern_floors_a_node_rather_than_diverging(self):
        """A node put through a log would take the whole axis with it."""
        azimuth = np.linspace(0.0, 360.0, 721)
        assert np.isfinite(log_amplitude(0.0, 90.0, 0.0, azimuth, 90.0)).all()


class TestLargestGap:
    def test_the_widest_wedge_is_found(self):
        gap, start, end = largest_gap(np.array([0.0, 10.0, 20.0, 200.0]))
        assert gap == pytest.approx(180.0)
        assert start == pytest.approx(20.0)
        assert end == pytest.approx(200.0)

    def test_the_gap_wraps_through_north(self):
        """The gap is usually the sea behind the network, and it usually wraps."""
        gap, start, end = largest_gap(np.array([10.0, 90.0, 170.0]))
        assert gap == pytest.approx(200.0)
        assert start == pytest.approx(170.0)
        assert end == pytest.approx(10.0)

    def test_evenly_spread_stations_leave_only_their_spacing(self):
        gap, _, _ = largest_gap(np.arange(0.0, 360.0, 30.0))
        assert gap == pytest.approx(30.0)


class TestAngularOffset:
    def test_a_plane_and_its_reverse_are_the_same_plane(self):
        """Strike is defined modulo 180 for this purpose, so 250 and 70 agree."""
        assert angular_offset(250.0, 70.0) == pytest.approx(0.0)

    def test_the_offset_is_the_short_way_round(self):
        assert angular_offset(350.0, 10.0) == pytest.approx(20.0)

    def test_the_largest_possible_offset_is_ninety(self):
        assert angular_offset(0.0, 90.0) == pytest.approx(90.0)


class TestMechanismFromAttributes:
    """Recovering a strike from a geometry that does not store one."""

    def _attrs(self, trace, source, dip=62.5, rake=134.2):
        return {
            "trace": shapely.LineString(trace).wkt,
            "source": shapely.Polygon(source).wkt,
            "dip": dip,
            "rake": rake,
        }

    def test_the_strike_leaves_the_dip_ninety_degrees_clockwise(self):
        """A trace running east with the plane dipping south strikes 090."""
        strike, _, _ = mechanism_from_attributes(
            self._attrs(
                [(170.0, -44.0), (170.2, -44.0)],
                [(170.0, -44.0), (170.2, -44.0), (170.2, -44.1), (170.0, -44.1)],
            )
        )
        assert strike == pytest.approx(90.0, abs=1.0)

    def test_reversing_the_stored_trace_does_not_move_the_strike(self):
        """The vertex order is an accident of the file, not a convention.

        This is the whole reason the strike is recovered rather than read: the
        same fault written down backwards has to give the same answer.
        """
        forward = [(170.0, -44.0), (170.2, -44.0)]
        surface = [(170.0, -44.0), (170.2, -44.0), (170.2, -44.1), (170.0, -44.1)]
        assert mechanism_from_attributes(self._attrs(forward, surface))[0] == (
            pytest.approx(
                mechanism_from_attributes(self._attrs(forward[::-1], surface))[0],
                abs=1.0,
            )
        )

    def test_a_plane_dipping_the_other_way_strikes_the_other_way(self):
        north = [(170.0, -44.0), (170.2, -44.0), (170.2, -43.9), (170.0, -43.9)]
        strike, _, _ = mechanism_from_attributes(
            self._attrs([(170.0, -44.0), (170.2, -44.0)], north)
        )
        assert strike == pytest.approx(270.0, abs=1.0)

    def test_dip_and_rake_come_straight_off_the_attributes(self):
        _, dip, rake = mechanism_from_attributes(
            self._attrs(
                [(170.0, -44.0), (170.2, -44.0)],
                [(170.0, -44.0), (170.2, -44.0), (170.2, -44.1), (170.0, -44.1)],
                dip=41.0,
                rake=-90.0,
            )
        )
        assert (dip, rake) == (41.0, -90.0)


class TestCorrelation:
    def test_a_pattern_with_no_variation_has_no_correlation(self):
        """A degenerate pattern must not come back as perfect agreement."""
        assert np.isnan(correlation(np.arange(5.0), np.ones(5)))

    def test_too_few_stations_is_nan_rather_than_a_number(self):
        assert np.isnan(correlation(np.array([1.0, 2.0]), np.array([1.0, 3.0])))

    def test_missing_stations_are_left_out_of_the_pair(self):
        residual = np.array([1.0, 2.0, 3.0, np.nan])
        pattern = np.array([1.0, 2.0, 3.0, 99.0])
        assert correlation(residual, pattern) == pytest.approx(1.0)


class TestSweepNull:
    """Why the tallest bump on the sweep is not by itself a result."""

    def test_searching_every_strike_beats_a_fixed_one_under_pure_noise(self):
        """The multiple-comparison trap the figure would otherwise walk into.

        Shuffled residuals carry no azimuthal signal at all, and yet a sweep
        over 180 strikes still finds a substantial correlation nearly every
        time, because it is free to pick the best of 180 smooth curves. The
        peak null must therefore sit well above the fixed-strike null -- if it
        did not, the command could report a peak against the wrong reference.
        """
        rng = np.random.default_rng(0)
        azimuth = rng.uniform(0.0, 360.0, 60)
        residual = rng.normal(size=60)
        peaks, fixed = sweep_null(residual, azimuth, 62.5, 134.2, 50.0, 200, 1)
        assert np.median(peaks) > np.percentile(fixed, 95)

    def test_the_null_is_reproducible_from_its_seed(self):
        rng = np.random.default_rng(2)
        azimuth = rng.uniform(0.0, 360.0, 40)
        residual = rng.normal(size=40)
        first = sweep_null(residual, azimuth, 60.0, 90.0, 50.0, 50, 7)
        second = sweep_null(residual, azimuth, 60.0, 90.0, 50.0, 50, 7)
        assert first[0] == pytest.approx(second[0])
        assert first[1] == pytest.approx(second[1])

    def test_a_real_signal_clears_the_peak_null(self):
        """The other direction: a pattern that is genuinely there is found."""
        azimuth = np.linspace(0.0, 355.0, 72)
        residual = log_amplitude(120.0, 62.5, 134.2, azimuth, 50.0)
        _, sweep = strike_sweep(residual, azimuth, 62.5, 134.2, 50.0)
        peaks, _ = sweep_null(residual, azimuth, 62.5, 134.2, 50.0, 200, 3)
        assert np.nanmax(sweep) > np.percentile(peaks, 95)


class TestStrikeSweep:
    def test_the_sweep_recovers_a_strike_it_was_given(self):
        """A residual that *is* the pattern must peak at the pattern's strike."""
        azimuth = np.linspace(0.0, 355.0, 72)
        residual = log_amplitude(200.0, 62.5, 134.2, azimuth, 50.0)
        strikes, sweep = strike_sweep(residual, azimuth, 62.5, 134.2, 50.0)
        assert angular_offset(float(strikes[np.nanargmax(sweep)]), 200.0) < 5.0

    def test_the_sweep_covers_a_full_turn(self):
        strikes, sweep = strike_sweep(
            np.random.default_rng(0).normal(size=30),
            np.linspace(0.0, 350.0, 30), 60.0, 90.0, 50.0,
        )
        assert strikes[0] == 0.0 and strikes[-1] < 360.0
        assert strikes.shape == sweep.shape
