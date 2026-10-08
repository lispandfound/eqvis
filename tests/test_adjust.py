"""Sampling a spatial bias field, which is the part that can be silently wrong.

An adjustment that is applied at the wrong place, or spread across a coastline
it should have stopped at, still produces a file of the right shape and a figure
that looks reasonable. So what is checked here is where the values come from:
that the gaps in a field written as one row per occupied cell stay gaps, that a
station inside the field is interpolated rather than snapped to a cell, and that
one outside it is left alone instead of extrapolated to.
"""

import numpy as np
import pandas as pd
import pytest
import typer
from scipy.spatial import cKDTree

from eqvis_workflow.adjust import (
    BLEND_ORDER,
    BLEND_PASSES,
    blend_weight,
    fade_weight,
    field_grid,
    read_field,
    sample_field,
)


def build(values: dict[tuple[int, int], float], period: float = 1.0) -> pd.DataFrame:
    """A field frame from ``{(easting, northing): value}``."""
    return pd.DataFrame(
        [
            {"period": period, "nztm_e": e, "nztm_n": n, "a_smooth": v}
            for (e, n), v in values.items()
        ]
    )


def sampler(frame: pd.DataFrame, column: str = "a_smooth"):
    """Everything :func:`sample_field` needs, built the way the command builds it."""
    periods, northings, eastings, cube = field_grid(frame, column)
    occupied_mask = np.isfinite(cube[0]).ravel()
    occupied = np.flatnonzero(occupied_mask)
    grid_e, grid_n = np.meshgrid(eastings, northings)
    tree = cKDTree(
        np.column_stack([grid_e.ravel()[occupied], grid_n.ravel()[occupied]])
    )
    return periods, northings, eastings, cube, tree, occupied


class TestFieldGrid:
    def test_a_value_lands_at_its_own_easting_and_northing(self):
        """The axis order is the one thing here that has no second chance.

        Easting and northing are both monotonic integers of similar magnitude,
        so transposing them produces a field that is the right shape, the right
        range and entirely in the wrong places.
        """
        frame = build({(1000, 5000): 1.0, (2000, 5000): 2.0, (1000, 6000): 3.0})
        periods, northings, eastings, cube = field_grid(frame, "a_smooth")
        assert list(eastings) == [1000, 2000]
        assert list(northings) == [5000, 6000]
        assert cube[0, 0, 0] == pytest.approx(1.0)  # e=1000, n=5000
        assert cube[0, 0, 1] == pytest.approx(2.0)  # e=2000, n=5000
        assert cube[0, 1, 0] == pytest.approx(3.0)  # e=1000, n=6000

    def test_an_unoccupied_cell_is_nan_rather_than_zero(self):
        """Absent is not "no adjustment here": the sea is missing, not flat."""
        _, _, _, cube = field_grid(
            build({(1000, 5000): 1.0, (2000, 6000): 2.0}), "a_smooth"
        )
        assert np.isnan(cube[0, 1, 0])
        assert np.isnan(cube[0, 0, 1])

    def test_every_period_gets_its_own_plane(self):
        frame = pd.concat(
            [build({(1000, 5000): 1.0}, 0.5), build({(1000, 5000): 9.0}, 3.0)]
        )
        periods, _, _, cube = field_grid(frame, "a_smooth")
        assert list(periods) == [0.5, 3.0]
        assert cube[0, 0, 0] == pytest.approx(1.0)
        assert cube[1, 0, 0] == pytest.approx(9.0)


class TestSampleField:
    def test_a_station_between_cells_is_interpolated(self):
        """Not snapped: a nearest-cell lookup prints the model's grid onto the map."""
        frame = build({
            (0, 0): 0.0, (100, 0): 1.0, (0, 100): 0.0, (100, 100): 1.0,
        })
        periods, northings, eastings, cube, tree, occupied = sampler(frame)
        value = sample_field(
            cube, northings, eastings, 0,
            np.array([50.0]), np.array([50.0]), tree, occupied, 15.0,
        )
        assert value[0] == pytest.approx(0.5)

    def test_a_station_on_a_cell_gets_that_cell(self):
        frame = build({
            (0, 0): 4.0, (100, 0): 1.0, (0, 100): 2.0, (100, 100): 3.0,
        })
        periods, northings, eastings, cube, tree, occupied = sampler(frame)
        value = sample_field(
            cube, northings, eastings, 0,
            np.array([0.0]), np.array([0.0]), tree, occupied, 15.0,
        )
        assert value[0] == pytest.approx(4.0)

    def test_a_station_beyond_the_field_is_left_unadjusted(self):
        """NaN, not the nearest value carried outwards for a hundred kilometres."""
        frame = build({
            (0, 0): 1.0, (100, 0): 1.0, (0, 100): 1.0, (100, 100): 1.0,
        })
        periods, northings, eastings, cube, tree, occupied = sampler(frame)
        far = sample_field(
            cube, northings, eastings, 0,
            np.array([0.0]), np.array([900_000.0]), tree, occupied, 15.0,
        )
        assert np.isnan(far[0])

    def test_a_station_just_off_the_edge_falls_back_to_the_nearest_cell(self):
        """A coastline leaves stations outside the convex hull of occupied cells.

        Linear interpolation returns nothing for them, and dropping them would
        put a ragged hole along every shore -- so within the distance limit they
        take the nearest cell instead.
        """
        frame = build({
            (0, 0): 7.0, (100, 0): 7.0, (0, 100): 7.0, (100, 100): 7.0,
        })
        periods, northings, eastings, cube, tree, occupied = sampler(frame)
        # 5 km beyond the corner, inside a 15 km limit.
        value = sample_field(
            cube, northings, eastings, 0,
            np.array([-3000.0]), np.array([-4000.0]), tree, occupied, 15.0,
        )
        assert value[0] == pytest.approx(7.0)

    def test_interpolation_does_not_span_a_gap_in_the_field(self):
        """The reason the cube is densified with NaN rather than packed.

        A cell missing from the middle of the grid is somewhere the model has no
        value. Linear interpolation over the dense grid returns NaN there, and
        the fallback then takes a real neighbouring cell -- rather than the
        triangulation quietly drawing a plane straight across the hole.
        """
        cells = {(e, n): 1.0 for e in (0, 100, 200) for n in (0, 100, 200)}
        del cells[(100, 100)]
        frame = build(cells)
        periods, northings, eastings, cube, tree, occupied = sampler(frame)
        assert np.isnan(cube[0, 1, 1])
        value = sample_field(
            cube, northings, eastings, 0,
            np.array([100.0]), np.array([100.0]), tree, occupied, 15.0,
        )
        # Not interpolated across the hole; taken from a neighbour that exists.
        assert value[0] == pytest.approx(1.0)


class TestReadField:
    def test_a_frame_without_coordinates_is_refused(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"period": [1.0], "a_smooth": [0.1]}).to_csv(path, index=False)
        with pytest.raises(typer.BadParameter, match="nztm_e"):
            read_field(path)

    def test_parquet_and_csv_read_the_same(self, tmp_path):
        frame = build({(1000, 5000): 0.25, (2000, 5000): -0.25})
        csv, parquet = tmp_path / "f.csv", tmp_path / "f.parquet"
        frame.to_csv(csv, index=False)
        frame.to_parquet(parquet)
        assert read_field(csv)["a_smooth"].tolist() == pytest.approx(
            read_field(parquet)["a_smooth"].tolist()
        )


class TestBlendWeight:
    """Tapering the correction in as the deterministic leg takes over."""

    def test_the_closed_form_is_the_filter_the_blend_actually_uses(self):
        """The whole reason this is allowed to be arithmetic rather than a filter.

        `bb_sim` blends with an order-4 Butterworth through `sosfiltfilt`, so
        two passes, with qcore's corner shift. If either of those changes, the
        weight here stops describing the run it is being applied to -- so it is
        checked against the real thing rather than trusted.
        """
        scipy_signal = pytest.importorskip("scipy.signal")
        corner, dt = 1.0, 0.005
        shift = (np.sqrt(2.0) - 1.0) ** (1.0 / (2 * BLEND_ORDER))
        periods = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 3.0, 10.0])

        responses = {}
        for kind, fc, btype in (
            ("hf", corner * shift, "highpass"),
            ("lf", corner / shift, "lowpass"),
        ):
            sos = scipy_signal.butter(
                BLEND_ORDER, fc, btype=btype, output="sos", fs=1.0 / dt
            )
            freq, gain = scipy_signal.sosfreqz(sos, worN=200_000, fs=1.0 / dt)
            amplitude = np.abs(gain) ** BLEND_PASSES
            responses[kind] = np.array(
                [amplitude[np.abs(freq - 1.0 / p).argmin()] for p in periods]
            )
        measured = responses["lf"] ** 2 / (responses["lf"] ** 2 + responses["hf"] ** 2)
        assert blend_weight(periods, corner) == pytest.approx(measured, abs=2e-4)

    def test_the_crossover_is_an_even_split(self):
        """Both legs are 1/sqrt(2) at the corner, so the power share is a half."""
        assert blend_weight(np.array([1.0]), 1.0) == pytest.approx([0.5])

    def test_long_period_is_all_deterministic_and_short_is_none_of_it(self):
        weight = blend_weight(np.array([100.0, 0.01]), 1.0)
        assert weight[0] == pytest.approx(1.0, abs=1e-9)
        assert weight[1] == pytest.approx(0.0, abs=1e-9)

    def test_the_weight_only_ever_increases_with_period(self):
        """A taper that wobbled would put structure into the corrected spectrum."""
        weight = blend_weight(np.geomspace(0.01, 20.0, 400), 1.0)
        assert (np.diff(weight) >= -1e-12).all()

    def test_a_lower_corner_moves_the_taper_with_it(self):
        assert blend_weight(np.array([2.0]), 0.5) == pytest.approx([0.5])

    def test_short_periods_do_not_overflow_to_nan(self):
        """(f/fc)^8 at 0.001 s is large; the weight still has to be a number."""
        assert np.isfinite(blend_weight(np.geomspace(1e-4, 1e4, 200), 1.0)).all()


class TestFadeWeight:
    def test_inside_the_field_nothing_is_faded(self):
        assert fade_weight(np.array([1.0, 5.0, 10.0]), 10.0) == pytest.approx(1.0)

    def test_one_octave_past_the_last_period_is_zero(self):
        assert fade_weight(np.array([20.0]), 10.0, 1.0) == pytest.approx([0.0])

    def test_half_an_octave_is_half_faded(self):
        assert fade_weight(np.array([10.0 * 2**0.5]), 10.0, 1.0) == pytest.approx([0.5])

    def test_zero_octaves_stops_dead(self):
        weight = fade_weight(np.array([10.0, 10.1]), 10.0, 0.0)
        assert list(weight) == [1.0, 0.0]

    def test_the_fade_never_goes_negative(self):
        assert (fade_weight(np.geomspace(1.0, 1000.0, 200), 10.0, 1.0) >= 0).all()
