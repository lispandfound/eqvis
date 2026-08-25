"""Unit tests for the parts of the package that are decisions rather than drawing.

The figures themselves are checked by looking at them. What is worth testing
here is the arithmetic behind them: which colour bin a magnitude lands in,
whether a scale bar picks a round number, whether a pick list survives a round
trip.
"""

import numpy as np
import pytest

from eqvis_workflow import rupture
from eqvis_workflow.animation import get_nice_vmax
from eqvis_workflow.display import Display
from eqvis_workflow.geography import nice_scale_length
from eqvis_workflow.picks import read_pick_list, write_pick_list
from eqvis_workflow.raster import fixed_symmetric_norm
from eqvis_workflow.stations import corner_anchor


class TestMagnitudeColours:
    def test_levels_land_on_whole_steps(self):
        """Two maps of the same magnitude range must colour it the same way."""
        levels = rupture.magnitude_levels({"a": 6.13, "b": 7.07})
        assert levels[0] == pytest.approx(6.1)
        assert levels[-1] == pytest.approx(7.1)
        assert np.allclose(np.diff(levels), rupture.MAGNITUDE_STEP)

    def test_a_single_magnitude_still_gives_a_bin(self):
        levels = rupture.magnitude_levels({"only": 6.5})
        assert len(levels) == 2

    def test_the_top_of_the_range_stays_inside_the_ramp(self):
        """No bin holds the range's top edge, so it has to be clamped."""
        magnitudes = {"a": 6.0, "b": 6.5}
        levels = rupture.magnitude_levels(magnitudes)
        cmap, norm = rupture.magnitude_ramp(levels)
        colours = rupture.magnitude_colours(magnitudes, cmap, norm)
        assert colours["b"] == cmap(cmap.N - 1)

    @pytest.mark.parametrize(
        ("magnitudes", "order", "expected"),
        [
            ({"a": 6.0}, ["a"], False),  # nothing to compare against
            ({"a": 6.0, "b": 6.04}, ["a", "b"], False),  # one bin: no visible spread
            ({"a": 6.0, "b": 6.5}, ["a", "b"], True),
            ({"a": 6.0}, ["a", "b"], False),  # b has no magnitude
        ],
    )
    def test_colour_is_only_used_when_it_carries_something(
        self, magnitudes, order, expected
    ):
        assert rupture.worth_colouring(magnitudes, order) is expected


class TestDarken:
    def test_a_pale_colour_is_brought_down_to_legibility(self):
        red, green, blue = rupture.darken("#fcffa4")  # the top of the magma ramp
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        assert luminance == pytest.approx(0.42)

    def test_an_already_dark_colour_is_left_alone(self):
        assert rupture.darken("#100020") == pytest.approx(
            (16 / 255, 0.0, 32 / 255), abs=1e-9
        )


class TestScaleBar:
    @pytest.mark.parametrize(
        ("span_km", "expected"),
        [(40, 10), (100, 20), (23, 5), (410, 100), (4.2, 1), (900, 200)],
    )
    def test_lengths_are_round_and_about_a_quarter_of_the_map(self, span_km, expected):
        assert nice_scale_length(span_km) == expected

    def test_the_bar_never_overruns_the_map(self):
        for span in np.geomspace(1, 5000, 200):
            assert nice_scale_length(span) <= span / 4 + 1e-9


class TestDisplay:
    def test_a_figure_at_its_designed_size_is_left_alone(self):
        display = Display.for_figure((9, 9), 300, None, None)
        assert display.scale == 1.0
        assert display.size == (9, 9)
        assert display.detailed

    def test_shrinking_and_standing_back_multiply(self):
        """Half the height, twice the distance: four times the text."""
        natural = 9 * 2.54
        display = Display.for_figure((9, 9), 300, natural / 2, 1.0)
        assert display.scale == pytest.approx(4.0)
        assert display.size == pytest.approx((9 / 4, 9 / 4))
        # The pixel count is held: the file is the same size, drawn for a wall.
        assert display.size[1] * display.dpi == pytest.approx(9 * 300)

    def test_fine_annotation_is_dropped_past_the_limit(self):
        assert not Display.for_figure((9, 9), 300, 5, 3).detailed

    def test_ticks_and_marks_thin_rather_than_vanish(self):
        display = Display.for_figure((9, 9), 300, 9 * 2.54 / 4, None)
        assert display.scale == pytest.approx(4.0)
        assert display.ticks(8) == 2
        assert display.mark(4.0) == pytest.approx(2.0)  # sqrt damping, not linear

    def test_keep_holds_both_ends(self):
        kept = Display(size=(1, 1), dpi=1, scale=3).keep([0, 1, 2, 3, 4, 5, 6], 3)
        assert kept == [0, 3, 6]


class TestCornerAnchor:
    @pytest.mark.parametrize(
        ("rect", "loc", "anchor"),
        [
            ((0.02, 0.02, 0.3, 0.3), "lower left", (0.02, 0.02)),
            ((0.68, 0.02, 0.3, 0.3), "lower right", (0.98, 0.02)),
            ((0.02, 0.68, 0.3, 0.3), "upper left", (0.02, 0.98)),
            ((0.68, 0.68, 0.3, 0.3), "upper right", (0.98, 0.98)),
        ],
    )
    def test_the_pinned_corner_faces_the_middle(self, rect, loc, anchor):
        got_loc, got_anchor = corner_anchor(rect)
        assert got_loc == loc
        assert got_anchor == pytest.approx(anchor)


class TestPickList:
    def test_round_trip(self, tmp_path):
        picked = {
            "stations": {"TUHS": True, "WATZ": False},
            "basins": {"Greater Wellington": True},
            "title": "a title",
        }
        path = tmp_path / "picks.stations"
        write_pick_list(path, picked, "eqvis pick im.h5 PGA")
        assert read_pick_list(path) == picked

    def test_unnamed_marks_a_station_drawn_without_its_label(self, tmp_path):
        path = tmp_path / "picks.stations"
        path.write_text("[stations]\nTUHS\nWATZ: unnamed\n[basins]\n")
        assert read_pick_list(path)["stations"] == {"TUHS": True, "WATZ": False}


class TestNorms:
    def test_a_fixed_symmetric_scale_ends_on_the_limit(self):
        levels = fixed_symmetric_norm(1.0, 10)
        assert levels[0] == pytest.approx(-1.0)
        assert levels[-1] == pytest.approx(1.0)
        assert 0.0 in levels  # the zero a diverging scale is read against

    def test_levels_are_evenly_spaced(self):
        levels = fixed_symmetric_norm(0.75, 6)
        assert np.allclose(np.diff(levels), np.diff(levels)[0])


class TestNiceVmax:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.42, 0.5), (0.9, 1.0), (1.4, 2.0), (2.2, 2.5), (7.0, 10.0), (0.0, 1.0)],
    )
    def test_rounds_up_to_a_clean_bound(self, value, expected):
        assert get_nice_vmax(value) == pytest.approx(expected)

    def test_never_clips_the_data(self):
        for value in np.geomspace(1e-4, 1e4, 300):
            assert get_nice_vmax(value) >= value
