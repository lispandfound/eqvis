"""Unit tests for the slip-panel figure's geometry and coarsening.

What the figure looks like is checked by looking at it. What is worth testing
here is the arithmetic underneath: that an SRF read from HDF5 is the same
rupture as the one written, that an unrolled plane lands where the header says
it should, that a block of subfaults averages to the direction it actually
slipped in rather than to the arithmetic mean of an angle, and that every
block drawn gets exactly one arrow.
"""

import dataclasses
import math
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import pytest
import typer

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eqvis_workflow import slip  # noqa: E402
from eqvis_workflow.display import NATURAL  # noqa: E402

# A two-plane rupture with a plane's worth of geometry but few enough
# subfaults to reason about by hand. The planes are unequal on purpose: a
# layout that assumed them equal would still pass on equal ones.
PLANE_LENGTHS_KM = (20.0, 12.0)
PLANE_STRIKE_CELLS = (10, 6)
DIP_CELLS = 4
PLANE_WIDTH_KM = 8.0
DIP_DEGREES = 30.0
TOP_DEPTH_KM = 2.0
RAKE_TOLERANCE_DEGREES = 10.0


def plane_header(
    index: int,
    shyp: float,
    dhyp: float,
    strike_cells: tuple[int, ...] = PLANE_STRIKE_CELLS,
    dip_cells: int = DIP_CELLS,
) -> pd.Series:
    """One plane's row of a synthetic SRF header."""
    return pd.Series(
        {
            "elon": 172.0 + index,
            "elat": -43.0 - index,
            "nstk": strike_cells[index],
            "ndip": dip_cells,
            "len": PLANE_LENGTHS_KM[index],
            "wid": PLANE_WIDTH_KM,
            "stk": 210.0,
            "dip": DIP_DEGREES,
            "dtop": TOP_DEPTH_KM,
            "shyp": shyp,
            "dhyp": dhyp,
        }
    )


def plane_points(header: pd.Series, rake: float = 90.0) -> pd.DataFrame:
    """One plane's subfaults, in SRF order (along strike fastest)."""
    nstk, ndip = int(header["nstk"]), int(header["ndip"])
    down_dip = (np.arange(ndip) + 0.5) * float(header["wid"]) / ndip
    depth = float(header["dtop"]) + down_dip * math.sin(math.radians(DIP_DEGREES))
    count = nstk * ndip
    return pd.DataFrame(
        {
            "lat": np.linspace(-43.0, -43.5, count),
            "lon": np.linspace(172.0, 172.5, count),
            "dep": np.repeat(depth, nstk),
            "slip": np.linspace(0.0, 100.0, count),
            "rise": np.full(count, 1.5),
            "tinit": np.linspace(0.0, 10.0, count),
            "rake": np.full(count, rake),
        }
    )


def unrolled_panels(
    rake: float = 90.0,
    strike_cells: tuple[int, ...] = PLANE_STRIKE_CELLS,
    dip_cells: int = DIP_CELLS,
) -> list[slip.Panel]:
    """The synthetic rupture, laid out the way the fault layout lays it out.

    The cell counts are a parameter because the coarse default is convenient
    for reasoning about geometry by hand and useless for contouring.
    """
    panels = []
    offset = 0.0
    for index in range(len(PLANE_LENGTHS_KM)):
        header = plane_header(
            index, 0.0, 4.0 if index else -1.0, strike_cells, dip_cells
        )
        panels.append(slip.unroll(header, plane_points(header, rake), offset))
        offset += float(header["len"]) * 1000.0
    return panels


def framed(panels):
    """A figure and axes shaped the way the fault layout shapes them.

    The axes box is given the data's own aspect ratio, so that
    ``set_aspect("equal")`` has nothing to adjust. ``plt.subplots`` leaves a
    box of whatever shape it likes and matplotlib shrinks the drawn axes
    inside it at draw time, which puts ``ax.get_position()`` -- what the label
    arithmetic reads -- and the rendered geometry out of step.
    """
    bounds = slip.panel_bounds(panels, slip.FAULT_MARGINS)
    aspect = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    panel = np.array([7.2, 7.2 * aspect])
    margin = np.array([0.7, 0.5])
    size = panel + 2 * margin
    fig = plt.figure(figsize=tuple(size))
    ax = fig.add_axes(np.concatenate([margin / size, panel / size]))
    slip.frame(ax, bounds, bottom_row=True, display=NATURAL)
    return fig, ax, bounds


class TestReadingHdf5:
    """The HDF5 form of an SRF, including the form written without its slip
    time functions -- which is how a rupture of a couple of million subfaults
    is normally stored, and which ``SrfFile.from_hdf5`` cannot read."""

    def srf(self):
        from source_modelling import srf as srf_module

        headers = [plane_header(index, 0.0, 4.0) for index in range(2)]
        return srf_module.SrfFile(
            version="1.0",
            header=pd.DataFrame(headers).reset_index(drop=True),
            points=pd.concat(
                [plane_points(header) for header in headers], ignore_index=True
            ),
            slipt1_array=None,
        )

    def test_a_file_without_slip_time_functions_round_trips(self, tmp_path):
        written = self.srf()
        path = tmp_path / "rupture.h5"
        written.to_xarray(include_slip_time_function=False).to_netcdf(
            path, engine="h5netcdf"
        )

        read = slip.open_srf(path)

        pd.testing.assert_frame_equal(
            read.header, written.header, check_dtype=False, check_like=True
        )
        pd.testing.assert_frame_equal(
            read.points, written.points, check_dtype=False, check_like=True
        )
        assert read.slipt1_array.shape[0] == len(written.points)

    def test_the_planes_come_back_as_integers(self, tmp_path):
        """Reshaping a plane needs ``nstk`` and ``ndip`` to be indices, and
        xarray hands back whatever dtype it stored them as."""
        path = tmp_path / "rupture.h5"
        self.srf().to_xarray(include_slip_time_function=False).to_netcdf(
            path, engine="h5netcdf"
        )

        header = slip.open_srf(path).header

        assert header["nstk"].dtype == int
        assert header["ndip"].dtype == int

    def test_panels_read_straight_out_of_hdf5(self, tmp_path):
        path = tmp_path / "rupture.h5"
        self.srf().to_xarray(include_slip_time_function=False).to_netcdf(
            path, engine="h5netcdf"
        )

        panels = slip.read_panels(path, slip.Layout.fault)

        assert len(panels) == 2
        assert panels[0].fields["slip"].shape == (DIP_CELLS, PLANE_STRIKE_CELLS[0])


class TestUnrolling:
    def test_planes_abut_exactly(self):
        """Two planes meeting at a bend have to share an edge to the metre:
        the outlines are drawn from these corners, and a gap or an overlap
        between them reads as a feature of the rupture."""
        first, second = unrolled_panels()

        assert first.corners[0, -1, 0] == pytest.approx(second.corners[0, 0, 0])
        assert first.corners[0, -1, 0] == pytest.approx(PLANE_LENGTHS_KM[0] * 1000.0)

    def test_the_panel_is_the_size_the_header_says(self):
        first, second = unrolled_panels()

        assert second.corners[-1, -1, 0] == pytest.approx(
            sum(PLANE_LENGTHS_KM) * 1000.0
        )
        assert second.corners[-1, -1, 1] == pytest.approx(PLANE_WIDTH_KM * 1000.0)
        assert first.cell_km == pytest.approx(
            (PLANE_LENGTHS_KM[0] / PLANE_STRIKE_CELLS[0], PLANE_WIDTH_KM / DIP_CELLS)
        )

    def test_nothing_is_mirrored(self):
        """Along strike increases in the SRF's own plane order, which is the
        whole reason this layout exists."""
        first, second = unrolled_panels()

        assert first.corners[0, 0, 0] < second.corners[0, 0, 0]
        assert first.strike @ np.array([1.0, 0.0]) == pytest.approx(1.0)

    def test_the_hypocentre_lands_where_the_header_puts_it(self):
        """``shyp`` runs along strike from the middle of the plane's top edge
        and ``dhyp`` down dip from that edge."""
        first, second = unrolled_panels()

        assert first.hypocentre is None  # dhyp is negative: not this plane
        assert second.hypocentre[0] == pytest.approx(
            (PLANE_LENGTHS_KM[0] + PLANE_LENGTHS_KM[1] / 2) * 1000.0
        )
        assert second.hypocentre[1] == pytest.approx(4000.0)

    def test_the_grid_is_axis_aligned(self):
        """Which is what lets the field go down as an image."""
        assert all(panel.axis_aligned for panel in unrolled_panels())


class TestSlipDirections:
    def test_a_thrust_points_up_dip(self):
        """Rake is measured from strike towards up dip, and down dip is drawn
        downwards -- so a rake of ninety has to come out pointing at the top
        of the panel, not the bottom."""
        panel = unrolled_panels()[0]

        direction = slip.slip_directions(np.array([90.0]), panel)

        assert direction[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert direction[0, 1] == pytest.approx(-1.0)

    def test_a_rake_of_zero_points_along_strike(self):
        panel = unrolled_panels()[0]

        direction = slip.slip_directions(np.array([0.0]), panel)

        assert direction[0, 0] == pytest.approx(1.0)
        assert direction[0, 1] == pytest.approx(0.0, abs=1e-9)


class TestBlocks:
    def test_a_block_is_the_mean_of_its_subfaults(self):
        field = np.arange(16, dtype=float).reshape(4, 4)

        assert np.allclose(slip.block_mean(field, (2, 2)), [[2.5, 4.5], [10.5, 12.5]])

    def test_a_ragged_edge_averages_what_is_there(self):
        """Not what a pad would put there, and not nothing."""
        field = np.arange(6, dtype=float).reshape(2, 3)

        averaged = slip.block_mean(field, (2, 2))

        assert averaged.shape == (1, 2)
        assert averaged[0, 0] == pytest.approx(np.mean([0.0, 1.0, 3.0, 4.0]))
        assert averaged[0, 1] == pytest.approx(np.mean([2.0, 5.0]))

    def test_rake_averages_the_short_way_round(self):
        """A block slipping due west has subfaults at +179 and -179 degrees,
        whose arithmetic mean is due east."""
        angles = np.array([[179.0, -179.0]])

        averaged = slip.circular_block_mean(angles, (1, 2))

        assert abs(averaged[0, 0]) == pytest.approx(180.0)

    def test_no_block_is_left_a_sliver(self):
        """619 rows at a block of 77 leaves a last block three rows deep,
        whose arrow stands on the panel's edge and speaks for a fortieth of
        what its neighbours do."""
        cells, cell_km, spacing_km = 619, 0.2, 15.4

        size = slip.block_size(cells, cell_km, spacing_km)
        remainder = cells % size or size

        assert remainder > size / 2

    def test_a_plane_coarser_than_the_spacing_keeps_its_subfaults(self):
        assert slip.block_size(3, 20.0, 5.0) == 1


class TestArrows:
    def spacing(self, panels):
        return (panels[-1].corners[0, -1, 0] - panels[0].corners[0, 0, 0]) / 1000.0 / 8

    def test_one_arrow_per_block(self):
        """Brendon's complaint: the panel drew thirty thousand arrows on a
        grid it had subsampled, and what was countable was the moire."""
        panels = unrolled_panels()
        fig, ax = plt.subplots()
        try:
            drawn = slip.draw_arrows(ax, panels, self.spacing(panels), NATURAL)
            plotted = sum(
                len(collection.get_offsets()) for collection in ax.collections
            )
        finally:
            plt.close(fig)

        expected = sum(
            math.prod(
                math.ceil(cells / size)
                for cells, size in zip(
                    panel.centres.shape[:2],
                    slip.block_shape(panel, self.spacing(panels)),
                )
            )
            for panel in panels
        )
        assert drawn == plotted == expected

    def test_slip_does_not_reach_the_arrows(self):
        """The panel above is the slip. An arrow field that is also a slip
        field -- scaled by it, or thresholded on it -- says the second thing
        badly and leaves gaps that read as missing data."""
        panels = unrolled_panels()
        starved = [
            dataclasses.replace(
                panel,
                fields=panel.fields | {"slip": np.zeros_like(panel.fields["slip"])},
            )
            for panel in panels
        ]
        fig, ax = plt.subplots()
        try:
            drawn = slip.draw_arrows(ax, panels, self.spacing(panels), NATURAL)
            without = slip.draw_arrows(ax, starved, self.spacing(panels), NATURAL)
            lengths = {
                tuple(np.round(vector, 6))
                for collection in ax.collections
                for vector in np.column_stack([collection.U, collection.V])
            }
        finally:
            plt.close(fig)

        assert drawn == without
        assert len({round(float(np.hypot(*vector)), 6) for vector in lengths}) == 1


class TestJoiningCurves:
    """Contouring is done a plane at a time, so a curve crossing a plane
    boundary comes back as pieces. A level is not a curve and a piece is not a
    curve, and labelling either as though it were is what puts every number on
    one side of a symmetric rupture or writes the same one at every boundary
    it crosses."""

    def piece(self, start, stop, count=5):
        return (0, np.column_stack([np.linspace(start, stop, count), np.zeros(count)]))

    def test_pieces_that_meet_are_one_curve(self):
        runs = [self.piece(0.0, 10.0), self.piece(10.5, 20.0)]

        assert len(slip.join_runs(runs, reach=1.0)) == 1

    def test_pieces_that_do_not_meet_are_separate_curves(self):
        """Both branches of an isochrone spreading out from a hypocentre."""
        runs = [self.piece(0.0, 10.0), self.piece(90.0, 100.0)]

        assert len(slip.join_runs(runs, reach=1.0)) == 2

    def test_joining_is_transitive(self):
        """A front crossing three planes is three pieces in a chain."""
        runs = [self.piece(0.0, 10.0), self.piece(10.5, 20.0), self.piece(20.5, 30.0)]

        assert len(slip.join_runs(runs, reach=1.0)) == 1

    def test_a_piece_joins_whichever_end_reaches(self):
        """The pieces do not arrive in order, or pointing the same way."""
        runs = [self.piece(10.5, 20.0), self.piece(10.0, 0.0)]

        assert len(slip.join_runs(runs, reach=1.0)) == 1


class TestContourLabels:
    """The synthetic rupture times run across the panel, so their isochrones
    reach both edges -- which is where ``clabel`` left to itself slices a
    label in half and leaves the reader guessing at "00 s"."""

    def retimed(self, onset):
        """The synthetic rupture, with rupture times ``onset`` of position.

        Discretised finely enough to contour: the handful of cells a side that
        the other tests use is convenient for checking geometry by hand and
        gives isochrones made of three vertices.
        """
        return [
            dataclasses.replace(
                panel, fields=panel.fields | {"tinit": onset(panel.centres)}
            )
            for panel in unrolled_panels(strike_cells=(200, 120), dip_cells=40)
        ]

    def labelled(self, panels=None, interval=1.0):
        panels = unrolled_panels() if panels is None else panels
        fig, ax, _ = framed(panels)
        slip.draw_isochrones(ax, panels, interval=interval, display=NATURAL)
        return fig, ax, [text for text in ax.texts if text.get_text().endswith(" s")]

    def test_a_front_crossing_a_plane_boundary_is_labelled_once(self):
        """Rupture times that depend only on depth put every isochrone across
        the full length of the rupture, so each is one curve in as many pieces
        as there are planes."""
        panels = self.retimed(lambda centres: centres[..., 1] / 1000.0)
        fig, _, labels = self.labelled(panels, interval=2.0)
        try:
            counts = Counter(text.get_text() for text in labels)
        finally:
            plt.close(fig)

        assert len(counts) >= 2, "too few isochrones drawn to test anything"
        assert set(counts.values()) == {1}

    def test_both_branches_of_a_front_are_labelled(self):
        """A rupture spreading both ways from its hypocentre gives every
        isochrone two branches. Labelling the level once rather than the curve
        once puts all the numbers on whichever side happened to sort first."""
        middle = 16_000.0
        panels = self.retimed(lambda centres: np.abs(centres[..., 0] - middle) / 1000.0)
        fig, _, labels = self.labelled(panels, interval=3.0)
        try:
            sides = {}
            for text in labels:
                sides.setdefault(text.get_text(), []).append(
                    text.get_position()[0] < middle
                )
        finally:
            plt.close(fig)

        assert len(sides) >= 2, "too few isochrones drawn to test anything"
        assert all(sorted(where) == [False, True] for where in sides.values())

    def test_every_label_is_inside_the_panel(self):
        fig, ax, labels = self.labelled()
        try:
            renderer = fig.canvas.get_renderer()
            panel = ax.get_window_extent(renderer)
            escaping = [
                text.get_text()
                for text in labels
                if not panel.contains(*text.get_window_extent(renderer).p0)
                or not panel.contains(*text.get_window_extent(renderer).p1)
            ]
        finally:
            plt.close(fig)

        assert labels, "nothing was labelled, so nothing was tested"
        assert not escaping

    def test_no_two_labels_overlap(self):
        """An isochrone breaking into fragments would otherwise write its
        number over itself in the same corner."""
        fig, ax, labels = self.labelled()
        try:
            renderer = fig.canvas.get_renderer()
            boxes = [text.get_window_extent(renderer) for text in labels]
            overlaps = [
                (one, other)
                for index, one in enumerate(boxes)
                for other in boxes[index + 1 :]
                if one.overlaps(other)
            ]
        finally:
            plt.close(fig)

        assert not overlaps

    def blob(self, ax, tightness):
        """One closed contour in the middle of ``ax``, ``tightness`` across."""
        grid = np.linspace(0.0, 1.0, 300)
        x, y = np.meshgrid(grid, grid)
        return ax.contour(
            x, y, np.exp(-tightness * ((x - 0.5) ** 2 + (y - 0.5) ** 2)), levels=[0.5]
        )

    def test_a_contour_shorter_than_its_own_label_goes_unlabelled(self):
        """A closed loop a few kilometres across says nothing that the number
        written over it does not cover up."""
        fig, ax = plt.subplots(figsize=(7.2, 1.44))
        try:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            speck = slip.label_places(
                ax, [self.blob(ax, 8000.0)], slip.isochrone_text, reach=0.0
            )
            room = slip.label_places(
                ax, [self.blob(ax, 200.0)], slip.isochrone_text, reach=0.0
            )
        finally:
            plt.close(fig)

        assert speck == [[]]
        assert room[0]


class TestDownDipRule:
    def test_a_fraction_on_its_own_draws_an_unlabelled_rule(self):
        assert slip.parse_rule("0.5") == (0.5, "")

    def test_a_label_comes_after_a_colon(self):
        assert slip.parse_rule("0.5: subevent limit") == (0.5, "subevent limit")

    @pytest.mark.parametrize("value", ["half", "1.5", "-0.1", ""])
    def test_what_is_not_a_fraction_is_refused(self, value):
        with pytest.raises(typer.BadParameter):
            slip.parse_rule(value)

    def test_the_rule_lands_at_that_fraction_of_the_down_dip_extent(self):
        fig, ax, bounds = framed(unrolled_panels())
        try:
            slip.draw_rule(ax, bounds, 0.5, "subevent limit", NATURAL)
            rule = ax.lines[-1]
            drawn = rule.get_ydata()[0]
            style = rule.get_linestyle()
        finally:
            plt.close(fig)

        assert drawn == pytest.approx(PLANE_WIDTH_KM * 1000.0 / 2)
        assert style != "-", "a constraint is not a contour"

    def test_an_unlabelled_rule_writes_nothing(self):
        fig, ax, bounds = framed(unrolled_panels())
        try:
            slip.draw_rule(ax, bounds, 0.5, "", NATURAL)
            texts = list(ax.texts)
        finally:
            plt.close(fig)

        assert not texts

    def test_the_reported_box_is_where_the_label_really_is(self):
        """The box is reported by ``draw_rule`` rather than measured off the
        artist afterwards, so it has to be checked against the artist. An
        annotation offset in points gives back the *offset* from
        ``get_position``, not where it sits, and a box computed from that is
        wrong by the whole width of the label -- which is a contour label
        placed straight on top of it."""
        fig, ax, bounds = framed(unrolled_panels())
        try:
            centre, half = slip.draw_rule(
                ax, bounds, 0.5, "subevent up-dip limit", NATURAL
            )
            renderer = fig.canvas.get_renderer()
            panel = ax.get_window_extent(renderer)
            drawn = ax.texts[-1].get_window_extent(renderer)
            low = (np.array(drawn.p0) - np.array(panel.p0)) / np.array(panel.size)
            high = (np.array(drawn.p1) - np.array(panel.p0)) / np.array(panel.size)
        finally:
            plt.close(fig)

        # The box's left and bottom sit exactly on the label's, to the last
        # bit of a float, so the comparison needs a hair of slack.
        assert (centre - half <= low + 1e-9).all(), "the estimate must cover it"
        assert (centre + half >= high - 1e-9).all(), "the estimate must cover it"

    def test_contour_labels_keep_off_the_rule_label(self):
        panels = TestContourLabels().retimed(lambda centres: centres[..., 1] / 1000.0)
        fig, ax, bounds = framed(panels)
        try:
            box = slip.draw_rule(ax, bounds, 0.5, "subevent up-dip limit", NATURAL)
            slip.draw_isochrones(ax, panels, 2.0, NATURAL, [box])
            renderer = fig.canvas.get_renderer()
            boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in ax.texts]
            clashes = [
                (one, other)
                for index, (one, first) in enumerate(boxes)
                for other, second in boxes[index + 1 :]
                if first.overlaps(second)
            ]
        finally:
            plt.close(fig)

        assert len(boxes) > 1, "nothing was labelled, so nothing was tested"
        assert not clashes


class TestFaults:
    """Which planes are segments of one fault, and what that means for rake.

    A fault cut into segments is still one fault. Measuring each segment's
    rake against its own mean makes an identical rake read as a departure on
    one side of a segment boundary and not on the other, which is an artefact
    of where the file happens to have been cut.
    """

    def sited(self, *traces):
        """The synthetic planes, put where ``traces`` says on the ground."""
        return [
            dataclasses.replace(panel, trace=np.array(trace, dtype=float))
            for panel, trace in zip(unrolled_panels(), traces)
        ]

    def test_planes_that_meet_are_one_fault(self):
        panels = self.sited([[0, 0], [20_000, 0]], [[20_100, 0], [32_000, 0]])

        assert slip.fault_groups(panels) == [[0, 1]]

    def test_planes_that_step_over_are_two_faults(self):
        """Five kilometres between the end of a twenty-kilometre plane and the
        start of a twelve-kilometre one is a step-over, not a bend."""
        panels = self.sited([[0, 0], [20_000, 0]], [[25_000, 0], [37_000, 0]])

        assert slip.fault_groups(panels) == [[0], [1]]

    def test_the_threshold_scales_with_the_plane(self):
        """The trace ends are reconstructed from the subfaults, so the error
        has the whole length of the plane to accumulate over -- a gap that is
        a bend on a two-hundred-kilometre plane is a step-over on a ten."""
        gap = 0.03 * PLANE_LENGTHS_KM[1] * 1000.0

        near = self.sited([[0, 0], [20_000, 0]], [[20_000 + gap, 0], [32_000, 0]])
        assert slip.fault_groups(near) == [[0, 1]]

    def test_only_consecutive_planes_are_compared(self):
        """Two faults in a rupture can pass within a kilometre of each other
        without being the same fault."""
        panels = [
            dataclasses.replace(panel, trace=np.array(trace, dtype=float))
            for panel, trace in zip(
                unrolled_panels() + unrolled_panels()[:1],
                (
                    [[0, 0], [20_000, 0]],
                    [[80_000, 0], [92_000, 0]],
                    [[20_100, 0], [40_000, 0]],
                ),
            )
        ]

        assert slip.fault_groups(panels) == [[0], [1], [2]]

    def test_segments_of_one_fault_share_a_reference(self):
        panels = self.sited([[0, 0], [20_000, 0]], [[20_100, 0], [32_000, 0]])
        panels = [
            dataclasses.replace(
                panel,
                fields=panel.fields
                | {"rake": np.full_like(panel.fields["rake"], rake)},
            )
            for panel, rake in zip(panels, (80.0, 100.0))
        ]
        faults = slip.fault_groups(panels)

        references = slip.rake_references(panels, faults)

        assert len(set(references)) == 1
        assert references[0] == pytest.approx(
            slip.circular_mean(
                np.concatenate([p.fields["rake"].ravel() for p in panels])
            )
        )

    def test_separate_faults_keep_their_own_references(self):
        panels = self.sited([[0, 0], [20_000, 0]], [[25_000, 0], [37_000, 0]])
        panels = [
            dataclasses.replace(
                panel,
                fields=panel.fields
                | {"rake": np.full_like(panel.fields["rake"], rake)},
            )
            for panel, rake in zip(panels, (80.0, 100.0))
        ]

        references = slip.rake_references(panels, slip.fault_groups(panels))

        assert references == pytest.approx([80.0, 100.0])

    def test_departure_is_measured_the_short_way_round(self):
        """A fault slipping due west has subfaults at +179 and -179 degrees."""
        rake = np.array([[179.0, -179.0, 150.0]])

        departed = slip.rake_departure(rake, 180.0, RAKE_TOLERANCE_DEGREES)

        assert departed.tolist() == [[0.0, 0.0, 1.0]]


class TestColourbarTicks:
    def test_labels_stay_evenly_spaced(self):
        """A bar labelled 0, 400, 1000, 1600, 2000 has the reader measuring
        the gaps instead of reading the colours."""
        levels = np.arange(0.0, 2200.0, 200.0)  # eleven levels

        shown = slip.thinned(levels, limit=7)

        assert len(shown) <= 7
        assert shown[0] == levels[0]
        assert shown[-1] == levels[-1]
        assert np.allclose(np.diff(shown), shown[1] - shown[0])

    def test_the_level_range_can_be_tightened_onto_a_skewed_field(self):
        """Why `quantiles` is a parameter rather than a constant.

        A slip field is heavily right-tailed, and bands covering the default
        1st-to-99th percentile of one can put half its cells in the first band.
        The first band of the slip ramp is white, so that half comes out as
        blank paper rather than as low slip. A tighter upper quantile has to
        actually move cells out of the first band.
        """
        rng = np.random.default_rng(0)
        skewed = rng.lognormal(mean=4.5, sigma=1.0, size=20_000)

        wide = slip.discrete_levels(skewed, 10)
        tight = slip.discrete_levels(skewed, 10, quantiles=(0.0, 0.90))

        assert tight[-1] < wide[-1]
        assert (skewed < tight[1]).mean() < (skewed < wide[1]).mean()

    def test_the_default_range_is_unchanged(self):
        """The parameter is an option, not a change of behaviour."""
        values = np.linspace(0.0, 2400.0, 500)
        assert slip.discrete_levels(values, 10) == pytest.approx(
            slip.discrete_levels(values, 10, quantiles=slip.ROBUST_RANGE)
        )

    def test_a_uniform_field_still_gets_a_range(self):
        """The degenerate guard has to survive whatever quantiles it is given."""
        levels = slip.discrete_levels(np.full(100, 7.0), 10, quantiles=(0.0, 0.9))
        assert levels[-1] > levels[0]

    def test_a_short_bar_gets_fewer_labels_than_a_tall_one(self):
        assert slip.bar_label_limit(1.44) < slip.bar_label_limit(3.3)

    def test_every_band_boundary_gets_a_number(self):
        """A field cut into more bands than the bar can label leaves colours
        standing between labelled boundaries, and the reader counting bands to
        find out what any of them means."""
        for height in (1.0, 1.44, 2.2, 3.3):
            bands = slip.bands_for_bar(wanted=10, height_in=height)
            levels = slip.discrete_levels(np.linspace(0.0, 2400.0, 500), bands)

            assert len(slip.thinned(levels, slip.bar_label_limit(height))) == len(
                levels
            )

    def test_a_tall_bar_keeps_the_bands_that_were_asked_for(self):
        assert slip.bands_for_bar(wanted=10, height_in=3.3) == 10

    def test_both_ends_survive_however_short_the_bar(self):
        levels = np.arange(0.0, 2200.0, 200.0)

        assert slip.thinned(levels, limit=2) == [0.0, 2000.0]


class TestCompassPoints:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [((1.0, 0.0), "N"), ((-1.0, -1.0), "SW"), ((0.0, 1.0), "E")],
    )
    def test_the_nearest_of_eight(self, offset, expected):
        assert slip.compass_point(np.array(offset)) == expected
