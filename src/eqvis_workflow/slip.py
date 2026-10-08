"""Slip, rise time and rake on the fault, in three stacked panels.

The figure Graves and Pitarka put beside every stochastic rupture: how much
each subfault slipped, how long it took to do it, and which way it went, with
the rupture front drawn over the slip as isochrones so the three fields can be
read against the order they happened in::

    eqvis slip-panels realisation.srf -o slip_panels.png
    eqvis slip-panels sw4/R1/realisation.srf -o panels.png --levels 12

Like the other map figures it can be drawn for a size and a distance rather
than for the page -- a poster panel 30 cm tall, read from three metres away::

    eqvis slip-panels realisation.srf --display-height 30 \\
        --viewing-distance 3 -o poster_panels.png

**Every plane is laid flat rather than foreshortened.** A true surface
projection would draw a 70-degree plane at a third of its down-dip extent and a
vertical one as a bare line, which is no use at all for reading a slip
distribution -- and vertical planes are common in the New Zealand fault model.
So each plane is hinged on its top edge and rotated up into the map: its trace
stays where it belongs, its neighbours stay at their true relative positions,
and its down-dip extent is drawn at true length. The panels are therefore
equal-aspect and metrically honest -- the scale bar means what it says -- while
being a map of the fault surface rather than of the ground above it.

A rupture hundreds of kilometres long and a few tens wide comes out of that as a
diagonal ribbon across a mostly empty page, and the answer is the other layout::

    eqvis slip-panels R01.h5 --layout fault -o panels.png

which unrolls the planes into fault coordinates instead: along strike across,
down dip down, planes laid end to end in the order the SRF stores them. The
rupture then fills a landscape panel whatever its strike is, at the cost of the
figure no longer being a map -- so that layout is drawn with real axes, in
kilometres along strike and down dip, and a second scale on the right saying
what depth the down-dip distance corresponds to.
"""

import dataclasses
import enum
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from qcore import coordinates
from scipy import sparse
from source_modelling import srf as srf_module

from .console import console_warn
from .display import Display
from .geography import nice_scale_length

# Slip runs white -> pale yellow -> orange -> red -> dark red: ColorBrewer
# YlOrRd, with white put in front of its pale-yellow end so that a subfault
# which did not slip reads as blank rather than as the bottom of a ramp, and a
# darker red put after its top so the few cells carrying several times the mean
# are still tellable apart from the merely large ones.
SLIP_RAMP = (
    "#ffffff",
    "#ffffcc",
    "#ffeda0",
    "#fed976",
    "#feb24c",
    "#fd8d3c",
    "#fc4e2a",
    "#e31a1c",
    "#bd0026",
    "#800026",
    "#4d0013",
)
# Rise time gets a ramp that cannot be confused with the slip one at a glance,
# which rules out anything warm; Blues already starts near white.
RISE_CMAP = "Blues"
# Rake is a two-tone field, not a scale, so it gets two tones: white where the
# subfault slipped the way its fault did, grey where it did not. The grey is
# light enough that the arrows over it stay the darkest ink in the panel.
RAKE_TONES = ("#ffffff", "#c9c9c9")

# How far a subfault's rake may depart from its fault's mean rake before the
# panel calls it a departure. Ten degrees is about the median absolute
# departure of a stochastically perturbed rupture, so roughly half the plane
# greys and the *pattern* of the perturbation is what the panel shows; a
# threshold much larger leaves the panel uniformly white and says nothing.
# It is also about the precision to which a rake is meaningful in the first
# place.
RAKE_TOLERANCE = 10.0

# How far apart two planes' traces may be, as a fraction of the shorter one's
# length, and still be segments of one fault.
CONNECT_FRACTION = 0.05

HYPOCENTRE_COLOUR = "yellow"

# A down-dip rule marks a line the rupture generator was working to rather
# than anything the rupture itself did, so it is dashed: nothing else in the
# panels is, and a reader is owed the difference between a contour of the
# field and a constraint on it. It carries a halo for the same reason the
# contour labels do -- it crosses the whole ramp, and at the dark end black on
# its own is not there.
RULE_WIDTH = 0.9
RULE_DASHES = (5.0, 3.0)
RULE_HALO = 2.4
RULE_LABEL_OFFSET = (3.0, 2.0)  # points, up and right off the rule's left end


class Layout(str, enum.Enum):
    """Which of the two ways of drawing a rupture surface to use.

    ``map`` puts the planes where they are, each hinged flat into the map.
    ``fault`` unrolls them into fault coordinates -- along strike across, down
    dip down -- which is not a map but does fit a long, shallow subduction
    interface onto a page.
    """

    map = "map"
    fault = "fault"


# The plane outlines are the only structural ink in the panels, so they are
# hairlines: heavy enough to separate two abutting planes, light enough not to
# swamp a plane only a few subfaults across.
OUTLINE_WIDTH = 0.6
# The map layout hinges each plane up onto its trace, so of the four edges of
# the outline the trace is the one that means something on a map: it is where
# the rupture meets the ground, and which side of it the plane was folded from
# is what says the way the fault dips. Drawn over the outline at the weight the
# rupture map gives a top edge. The fault layout has no map to be oriented in
# -- there the same edge is just the top of a rectangle -- so it stays a
# hairline there.
TRACE_WIDTH = 2.2
# The isochrones sit on top of a full-strength colour field and have to be read
# against every part of it, so they are the heaviest line in the figure.
CONTOUR_WIDTH = 0.9
# Wide enough to clear the label's own strokes, so the halo reads as a gap
# in the field rather than as an outline around the text.
CONTOUR_HALO = 2.2
SCALE_BAR_WIDTH = 1.1

FONT_LABEL = 8.5
FONT_STATS = 7.0
FONT_TICK = 6.5
FONT_CONTOUR = 6.5
FONT_SCALE = 7.5

POINTS_PER_INCH = 72.0
# A character's width as a fraction of the font's size, for working out how
# much room a label needs without asking a renderer for one that has not been
# drawn yet. Digits in a proportional face run a little narrower than this;
# rounding the estimate up is what keeps the label off the edge.
LABEL_ASPECT = 0.62
# And a line of text is taller than its font's size -- ascenders, descenders
# and the leading between them. Both estimates are deliberately generous: the
# cost of one that is too wide is a label placed further from an obstacle than
# it needed to be, and of one too narrow is a label written on top of it.
LINE_HEIGHT = 1.6
# Both in units of a label's clearance -- half its own diagonal -- so that one
# number covers a short label and a long one on a panel of any shape. A run of
# contour has to be at least its own label long to be worth labelling, and two
# labels have to stand this far apart, which is a clear label's width of gap.
LABEL_MIN_REACH = 2.0
LABEL_SEPARATION = 4.0
# How much clear air to leave round text a contour label is dodging, as a
# multiple of the two half-extents that would just touch.
OBSTACLE_MARGIN = 1.15
# How far apart, in coarsened contour cells, the ends of two polylines may be
# and still be pieces of one curve. A contour runs between cell centres, so it
# stops half a cell short of a plane's edge at either end, which puts two
# pieces meeting at a plane boundary a whole cell apart; twice that is enough
# slack for the boundary not to have to be exactly where the grid says while
# staying far below the distance between two separate branches of a front.
JOIN_CELLS = 2.0

# Isochrone intervals worth labelling, in seconds. The interval is chosen from
# this list rather than computed, because a rupture-time contour reads as a
# round number of seconds or not at all -- "12 s" tells a reader nothing that
# "10 s" does not.
CONTOUR_STEPS_S = (0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
# About this many isochrones across the rupture: enough to see the front move,
# few enough that their labels do not collide on a small plane.
CONTOUR_TARGET = 6

# Slip and rise time both have long tails -- a few cells hold several times the
# mean -- and a ramp stretched to the largest of them spends most of its colour
# on values that occur nowhere. The levels cover this quantile range instead,
# and the colourbar's end triangles say that something lies outside.
ROBUST_RANGE = (0.01, 0.99)

# Roughly how many rake arrows fit across the figure. The arrows are coarsened
# to a spacing measured on the fault, not per plane, so a plane discretised
# finely does not end up carrying more of them per centimetre than a coarse one.
ARROW_COLUMNS = 22
# An arrow this fraction of the spacing leaves a clear gap between neighbours,
# so the field reads as a set of directions rather than as a texture.
ARROW_LENGTH = 0.72
ARROW_WIDTH = 0.0028  # shaft width, as a fraction of the panel width
# How much larger than that an arrow on a coarsely discretised plane may grow.
# Three times is enough for a handful of large subfaults to carry readable
# arrows and not enough for one of them to carry an arrow the size of the page.
ARROW_GROWTH_LIMIT = 3.0

# Isochrones are drawn on a coarsened copy of the rupture time rather than on
# every subfault: a plane a thousand cells across contours into a thousand tiny
# segments per line, which is slow, enormous in a vector file, and gives
# ``clabel`` a thousand places to put a label. The front itself is smooth, so
# nothing is lost by asking for no more than this many cells on a side.
CONTOUR_MAX_CELLS = 400

# The figure's design geometry, in inches. Nothing in it needs measuring at
# draw time, so the layout is placed by hand rather than left to a constrained
# solver: three panels of the rupture's own aspect ratio, a narrow colourbar
# column of exactly the same height as its panel, and margins sized for the
# text each edge carries. Placing it by hand is what keeps the three rows and
# their three bars on the same two vertical lines -- a solver given three
# panels, two colourbars and one legend puts them wherever the widest tick
# label of the moment says to.
PANEL_WIDTH = 7.2
# A rupture taller than it is wide would otherwise stack three tall panels into
# a figure no printer will take: this caps the whole thing at about 11 inches.
PANEL_HEIGHT_LIMIT = 3.3
MARGIN_TOP = 0.26  # the row label
BAR_WIDTH = 0.15
# Clear space per colourbar tick label, in label heights. Just under twice is
# about where a column of numbers stops reading as a column of numbers.
BAR_LABEL_PITCH = 1.9


@dataclasses.dataclass(frozen=True)
class Margins:
    """What each edge of the canvas has to hold, in inches.

    Attributes
    ----------
    left, right, bottom : float
        Outside the panel column, the bar column, and the bottom row.
    row_gap : float
        Between one row and the next.
    bar_gap : float
        Between a panel and its colourbar.
    pad : float
        How far the panel window is padded beyond the planes, as a fraction of
        the rupture's larger dimension.
    """

    left: float
    right: float
    bottom: float
    row_gap: float
    bar_gap: float
    pad: float


# The map layout has no tick labels and no axis titles anywhere, so its margins
# hold only the row label, the scale bar and the header over each colourbar,
# and its panels are padded off the planes so that the outlines are not clipped
# and the scale bar has somewhere to sit.
MAP_MARGINS = Margins(
    left=0.12,
    right=0.78,  # the rake key's labels; the headers centre on the bar
    bottom=0.46,  # the scale bar and its label
    row_gap=0.30,  # the next row's label
    bar_gap=0.34,
    pad=0.03,
)
# The fault layout has real axes instead, so its margins hold tick labels and
# axis titles: down dip on the left, depth on the right between the panel and
# its bar, along strike under the bottom row only -- repeating the along-strike
# axis under all three rows says the same thing three times and costs the
# panels the height to say it in.
FAULT_MARGINS = Margins(
    left=0.62,
    right=0.78,  # the colourbar's tick labels, and the rake key's wordier ones
    bottom=0.62,
    row_gap=0.30,
    bar_gap=0.52,  # the depth axis, its labels and its title
    pad=0.0,  # the axes frame is the edge; nothing to clear
)

# Tick intervals for the fault layout, in kilometres. Round distances rather
# than a locator's idea of them, and far apart down dip: the panel is a fifth
# as tall as it is wide, so a down-dip axis given as many labels as the
# along-strike one prints them on top of each other.
ALONG_STRIKE_TICK_KM = 100.0
DOWN_DIP_TICK_KM = 40.0

# How far the subfaults may sit off the one sloping surface the depth scale is
# fitted to before that scale stops being a fair summary of where they are.
DEPTH_AXIS_TOLERANCE_KM = 1.0


@dataclasses.dataclass(frozen=True)
class Panel:
    """One plane of the rupture, placed into whichever layout is being drawn.

    The positions are in metres in both layouts -- eastings and northings in
    the map one, distance along strike and down dip in the fault one -- so
    everything downstream of the two constructors is written once.

    Attributes
    ----------
    corners : np.ndarray
        ``(ndip + 1, nstk + 1, 2)`` cell-corner positions, in metres.
    centres : np.ndarray
        ``(ndip, nstk, 2)`` cell-centre positions, for contouring.
    strike : np.ndarray
        Unit vector along strike, in panel coordinates.
    down_dip : np.ndarray
        Unit vector down dip, in panel coordinates.
    cell_km : tuple[float, float]
        Cell size along strike and along dip, in kilometres.
    fields : dict[str, np.ndarray]
        The plane's ``(ndip, nstk)`` fields, keyed as in the SRF points frame.
    hypocentre : np.ndarray | None
        Where this plane's header puts the hypocentre, or ``None`` if it does
        not claim one.
    header : pd.Series
        The plane's row of the SRF header, kept for the dip and top depth the
        fault layout's depth scale is built from.
    trace : np.ndarray
        ``(2, 2)`` map positions of the two ends of the plane's top edge, in
        metres. In *map* coordinates whichever layout is being drawn, because
        what it is for is telling which planes are segments of one fault, and
        that is a fact about the ground rather than about the figure.
    """

    corners: np.ndarray
    centres: np.ndarray
    strike: np.ndarray
    down_dip: np.ndarray
    cell_km: tuple[float, float]
    fields: dict[str, np.ndarray]
    hypocentre: np.ndarray | None
    header: pd.Series
    trace: np.ndarray

    @property
    def outline(self) -> np.ndarray:
        """The plane's perimeter as a closable ``(4, 2)`` ring."""
        return np.array(
            [
                self.corners[0, 0],
                self.corners[0, -1],
                self.corners[-1, -1],
                self.corners[-1, 0],
            ]
        )

    @property
    def length_km(self) -> float:
        """How far the plane runs along strike."""
        return self.cell_km[0] * self.centres.shape[1]

    @property
    def contourable(self) -> bool:
        """Whether the plane has enough cells for a contour to run across it."""
        return min(self.centres.shape[:2]) > 1

    @property
    def axis_aligned(self) -> bool:
        """Whether the plane's cells line up with the figure's own axes.

        An unrolled plane's cells are an evenly spaced grid parallel to the
        axes, which is the one case where the field can go down as an image
        rather than as one quadrilateral per subfault -- and at two million
        subfaults that is the difference between a figure and a hung process.
        """
        return bool(self.strike[1] == 0.0 and self.down_dip[0] == 0.0)


def map_coordinates(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read a points frame into map positions and depths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(n, 2)`` NZTM easting and northing in metres, and ``(n,)`` depths in
        metres. NZTM is northing-first and a map is easting-first, so the
        columns come back swapped from what :mod:`qcore.coordinates` returns.
    """
    nztm = coordinates.wgs_depth_to_nztm(
        points[["lat", "lon", "dep"]].to_numpy() * np.array([1.0, 1.0, 1000.0])
    )
    return nztm[:, [1, 0]], nztm[:, 2]


def lay_flat(header: pd.Series, points: pd.DataFrame) -> Panel:
    """Hinge one SRF plane on its top edge and rotate it into the map.

    The plane's geometry is taken from where its subfaults actually are rather
    than from what its header says, because SRF headers are unreliable about
    strike in particular -- but a plane one subfault wide has no geometry of
    its own to read, so the header is the fallback rather than the source.

    Parameters
    ----------
    header : pd.Series
        The plane's row of the SRF header.
    points : pd.DataFrame
        The plane's subfaults, in SRF order (along strike fastest).

    Returns
    -------
    Panel
        The plane, laid flat.
    """
    nstk, ndip = int(header["nstk"]), int(header["ndip"])
    position, depth = map_coordinates(points)
    position = position.reshape(ndip, nstk, 2)
    depth = depth.reshape(ndip, nstk)

    if nstk > 1:
        span = position[0, -1] - position[0, 0]
        strike = span / np.hypot(*span)
        cell_strike = float(np.hypot(*span)) / (nstk - 1)
    else:
        # No two subfaults to draw a line between, so the header's strike is
        # all there is -- corrected from true north to the grid, because a map
        # in NZTM is a couple of degrees off true this far south.
        bearing = coordinates.great_circle_bearing_to_nztm_bearing(
            header[["elat", "elon"]].to_numpy(),
            float(header["len"]) / 2,
            float(header["stk"]),
        )
        strike = np.array([math.sin(bearing), math.cos(bearing)])
        cell_strike = float(header["len"]) * 1000.0

    # Down dip is perpendicular to strike; which of the two perpendiculars it
    # is follows from the SRF convention that a plane dips to the right of its
    # strike, and is confirmed against the subfaults themselves wherever they
    # can see it.
    down_dip = np.array([strike[1], -strike[0]])
    if ndip > 1:
        span = position[-1, 0] - position[0, 0]
        horizontal = float(np.hypot(*span)) / (ndip - 1)
        cell_dip = float(np.hypot(np.hypot(*span), depth[-1, 0] - depth[0, 0])) / (
            ndip - 1
        )
        # A near-vertical plane's subfaults are all at the same map position to
        # within rounding, and rounding has no dip direction: only a step long
        # enough to mean something is allowed to overrule the convention.
        if horizontal > 0.05 * cell_dip and span @ down_dip < 0:
            down_dip = -down_dip
    else:
        cell_dip = float(header["wid"]) * 1000.0
        horizontal = cell_dip * math.cos(math.radians(float(header["dip"])))

    # The hinge is the top edge, and the top row of subfaults sits half a cell
    # down dip of it -- but only the *horizontal* part of that half cell is
    # what the map sees, so that is what is stepped back to find the edge.
    origin = position[0, 0] - 0.5 * cell_strike * strike - 0.5 * horizontal * down_dip
    edges = (
        cell_strike * np.arange(nstk + 1)[None, :, None] * strike
        + cell_dip * np.arange(ndip + 1)[:, None, None] * down_dip
    )
    middles = (
        cell_strike * (np.arange(nstk) + 0.5)[None, :, None] * strike
        + cell_dip * (np.arange(ndip) + 0.5)[:, None, None] * down_dip
    )

    return Panel(
        corners=origin + edges,
        centres=origin + middles,
        strike=strike,
        down_dip=down_dip,
        cell_km=(cell_strike / 1000.0, cell_dip / 1000.0),
        fields=plane_fields(points, ndip, nstk),
        hypocentre=hypocentre_position(
            header, origin + 0.5 * nstk * cell_strike * strike, strike, down_dip
        ),
        header=header,
        trace=trace_ends(points, nstk),
    )


def trace_ends(points: pd.DataFrame, nstk: int) -> np.ndarray:
    """Where a plane's top edge starts and finishes, in map coordinates.

    Read off the subfaults at either end of the top row, for the same reason
    :func:`lay_flat` reads the rest of a plane's geometry there rather than
    from its header.
    """
    return map_coordinates(points.iloc[[0, nstk - 1]])[0]


def plane_fields(points: pd.DataFrame, ndip: int, nstk: int) -> dict[str, np.ndarray]:
    """A plane's subfault columns, reshaped onto its grid."""
    return {
        name: points[name].to_numpy().reshape(ndip, nstk)
        for name in ("slip", "rise", "tinit", "rake", "dep")
    }


def unroll(header: pd.Series, points: pd.DataFrame, offset: float) -> Panel:
    """Lay one SRF plane into fault coordinates, ``offset`` metres along strike.

    Where :func:`lay_flat` asks the subfaults where they are, this asks the
    header how big the plane is and puts it down as a plain rectangle: along
    strike to the right, down dip downwards, starting where the previous plane
    finished. That is what makes the panel landscape whatever the fault's
    strike is, and what makes two planes meeting at a bend abut exactly rather
    than overlap by however much their corner coordinates disagree.

    It also means the along-strike axis is the SRF's own plane order, so a
    rupture is drawn running the way its file says it runs and is never
    mirrored -- the reflection a map projection would introduce for a
    south-west striking fault is exactly what this layout is avoiding.

    Parameters
    ----------
    header : pd.Series
        The plane's row of the SRF header.
    points : pd.DataFrame
        The plane's subfaults, in SRF order (along strike fastest).
    offset : float
        Where the plane starts along strike, in metres, measured from the
        start of the first plane.

    Returns
    -------
    Panel
        The plane, unrolled.
    """
    nstk, ndip = int(header["nstk"]), int(header["ndip"])
    cell_strike = float(header["len"]) * 1000.0 / nstk
    cell_dip = float(header["wid"]) * 1000.0 / ndip
    strike = np.array([1.0, 0.0])
    down_dip = np.array([0.0, 1.0])
    origin = np.array([offset, 0.0])

    edges = (
        cell_strike * np.arange(nstk + 1)[None, :, None] * strike
        + cell_dip * np.arange(ndip + 1)[:, None, None] * down_dip
    )
    middles = (
        cell_strike * (np.arange(nstk) + 0.5)[None, :, None] * strike
        + cell_dip * (np.arange(ndip) + 0.5)[:, None, None] * down_dip
    )

    return Panel(
        corners=origin + edges,
        centres=origin + middles,
        strike=strike,
        down_dip=down_dip,
        cell_km=(cell_strike / 1000.0, cell_dip / 1000.0),
        fields=plane_fields(points, ndip, nstk),
        hypocentre=hypocentre_position(
            header, origin + 0.5 * nstk * cell_strike * strike, strike, down_dip
        ),
        header=header,
        trace=trace_ends(points, nstk),
    )


def hypocentre_position(
    header: pd.Series, top_centre: np.ndarray, strike: np.ndarray, down_dip: np.ndarray
) -> np.ndarray | None:
    """Where this plane's header puts the hypocentre, if it puts it anywhere.

    ``shyp`` is measured along strike from the middle of the top edge and
    ``dhyp`` down dip from that edge, so both together place the point. Planes
    the rupture reached rather than started on carry no hypocentre, and say so
    in two incompatible ways -- some write a large negative sentinel, others
    write plain zeros -- so what is tested for is a hypocentre that could be
    real: on the plane, and below its top edge rather than exactly on it.
    """
    shyp, dhyp = float(header["shyp"]), float(header["dhyp"])
    if not (math.isfinite(shyp) and math.isfinite(dhyp)):
        return None
    if dhyp <= 0 or abs(shyp) > float(header["len"]) / 2:
        return None
    return top_centre + 1000.0 * (shyp * strike + dhyp * down_dip)


def read_srf_hdf5(srf_path: Path) -> srf_module.SrfFile:
    """An SRF from the HDF5 form ``SrfFile.write_hdf5`` writes.

    ``SrfFile.from_hdf5`` reads this format already, but reaches for the sparse
    slip time function without first checking that the file has one -- and a
    rupture of a couple of million subfaults is normally written without,
    because the time functions are larger than everything else in the file put
    together. The panels never look at them, so a file that left them out
    becomes an SRF with none rather than a ``KeyError``.
    """
    dataset = xr.open_dataset(srf_path, engine="h5netcdf")
    names = [str(name) for name in dataset.data_vars]
    header = pd.DataFrame(
        {
            name.removeprefix("plane_"): dataset[name].values
            for name in names
            if name.startswith("plane_")
        }
    )
    header[["nstk", "ndip"]] = header[["nstk", "ndip"]].astype(int)
    points = pd.DataFrame(
        {
            name: dataset[name].values
            for name in names
            if not name.startswith("plane_")
            and name not in {"data", "indices", "indptr"}
        }
    )

    if "data" in dataset:
        values = dataset["data"].values
        indptr = np.append(dataset["indptr"].values, len(values))
        slipt1 = sparse.csr_array((values, dataset["indices"].values, indptr))
    else:
        slipt1 = sparse.csr_array((len(points), 0))

    return srf_module.SrfFile(
        version=str(dataset.attrs.get("version", "1.0")),
        header=header,
        points=points,
        slipt1_array=slipt1,
    )


def open_srf(srf_path: Path) -> srf_module.SrfFile:
    """An SRF, read from whichever of its two forms is on disk."""
    if srf_path.suffix.lower() in {".h5", ".hdf5", ".nc"}:
        return read_srf_hdf5(srf_path)
    return srf_module.read_srf(srf_path)


def read_panels(srf_path: Path, layout: Layout = Layout.map) -> list[Panel]:
    """Every plane of an SRF, placed into ``layout``."""
    srf = open_srf(srf_path)
    if layout is Layout.map:
        return [
            lay_flat(srf.header.iloc[index], segment)
            for index, segment in enumerate(srf.segments)
        ]

    panels = []
    offset = 0.0
    for index, segment in enumerate(srf.segments):
        header = srf.header.iloc[index]
        panels.append(unroll(header, segment, offset))
        offset += float(header["len"]) * 1000.0
    return panels


def gather(panels: list[Panel], field: str) -> np.ndarray:
    """Every plane's values for one field, flattened into one array."""
    values = np.concatenate([panel.fields[field].ravel() for panel in panels])
    return values[np.isfinite(values)]


def discrete_levels(
    values: np.ndarray, count: int, quantiles: tuple[float, float] = ROBUST_RANGE
) -> np.ndarray:
    """Round level boundaries covering ``values``' robust range.

    ``quantiles`` is how much of the tail the bands are asked to cover, and it
    is worth thinking about rather than taking. :data:`ROBUST_RANGE` already
    keeps a ramp off the single largest cell, but a heavily skewed field can
    still put most of its cells in the first band or two -- and the first band
    of the slip ramp is white, so "hardly moved" and "no fault here" come out
    the same colour. Tightening the upper quantile spends the bands where the
    cells actually are and leaves the rest to the colourbar's end triangle.
    """
    low, high = (float(v) for v in np.quantile(values, quantiles))
    if not high > low:
        # A uniform field -- a point source, or one segment that did not move.
        high = low + (abs(low) or 1.0) * 0.1
    return MaxNLocator(count).tick_values(low, high)


def discrete_ramp(ramp, levels: np.ndarray) -> tuple[ListedColormap, BoundaryNorm]:
    """A ``len(levels) - 1`` band colormap off ``ramp``, and its norm.

    The band colours are sampled across the whole ramp and its two ends are
    kept aside for the values the robust range left outside, so an outlier is
    drawn in the ramp's own extreme rather than in the last band's colour --
    which would quietly claim it belonged there.
    """
    colours = ramp(np.linspace(0.0, 1.0, len(levels) - 1))
    cmap = ListedColormap(colours)
    cmap.set_under(ramp(0.0))
    cmap.set_over(ramp(1.0))
    return cmap, BoundaryNorm(levels, cmap.N)


def overflow(values: np.ndarray, levels: np.ndarray) -> str:
    """Which end of a colourbar needs a triangle.

    Slip and rise time both floor at zero and their levels start there, so an
    empty triangle at the bottom would claim values that cannot exist.
    """
    under = bool(values.min() < levels[0])
    over = bool(values.max() > levels[-1])
    return {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
        (under, over), "neither"
    )


def statistics(values: np.ndarray) -> str:
    """The min / mean / max strip Graves and Pitarka put over each panel."""
    low, mean, high = values.min(), values.mean(), values.max()
    decimals = 0 if high >= 20 else 1
    return " / ".join(f"{value:.{decimals}f}" for value in (low, mean, high))


def statistics_labelled(values: np.ndarray, unit: str) -> str:
    """The same three numbers, saying which three numbers they are.

    Three numbers over a panel are three numbers over a panel: the convention
    is old enough that Graves and Pitarka leave it unwritten, and new enough
    to a reader that it gets asked about. It costs four words to say.
    """
    return f"min / mean / max = {statistics(values)} {unit}".strip()


def contour_interval(onsets: np.ndarray) -> float | None:
    """A round isochrone interval, or ``None`` if there is nothing to contour.

    The interval nearest -- in ratio, not in difference, since these steps are
    spaced multiplicatively -- to cutting the rupture into ``CONTOUR_TARGET``
    isochrones.
    """
    span = float(onsets.max() - onsets.min())
    if span <= 0:
        return None
    wanted = span / CONTOUR_TARGET
    return min(CONTOUR_STEPS_S, key=lambda step: abs(math.log(step / wanted)))


def circular_mean(angles: np.ndarray) -> float:
    """The mean of a set of angles in degrees, taken the short way round.

    Because rake is an angle: a fault slipping due west has subfaults at +179
    and -179 degrees, whose arithmetic mean is due east.
    """
    radians = np.radians(angles)
    return math.degrees(
        math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))
    )


def fault_groups(panels: list[Panel]) -> list[list[int]]:
    """The planes gathered into the faults they are segments of.

    An SRF stores a rupture as an ordered list of planes, and a fault built
    from several segments stores them one after another, so a fault is a run
    of planes whose traces meet end to end.

    What counts as meeting is measured against the plane's own length, not
    against its cell size: the trace ends are read off the subfaults at either
    extreme of the top row, and whatever the geometry disagrees with the
    ground by has had the whole length of the plane to accumulate over. On the
    ruptures to hand a bend within one fault comes to about one per cent of
    the shorter plane and the nearest step-over between two faults to sixteen,
    so the threshold sits comfortably between them.

    Only consecutive planes are compared. Two faults in a multi-fault rupture
    can pass within a kilometre of each other without being the same fault,
    and testing every plane against every other would join them.
    """
    if not panels:
        return []
    groups = [[0]]
    for index in range(1, len(panels)):
        before, plane = panels[index - 1], panels[index]
        gap = float(np.linalg.norm(before.trace[1] - plane.trace[0]))
        if gap > CONNECT_FRACTION * 1000.0 * min(before.length_km, plane.length_km):
            groups.append([])
        groups[-1].append(index)
    return groups


def rake_references(panels: list[Panel], faults: list[list[int]]) -> list[float]:
    """The rake each plane's departures are measured against.

    One mean per fault rather than one per plane. A fault cut into segments is
    still one fault, and giving each segment its own reference makes the same
    rake read as a departure on one side of a segment boundary and not on the
    other -- an artefact of where the file happens to have been cut. Separate
    faults in a rupture do get separate references, which is the point: they
    can be slipping genuinely differently.
    """
    references = [0.0] * len(panels)
    for group in faults:
        mean = circular_mean(
            np.concatenate([panels[index].fields["rake"].ravel() for index in group])
        )
        for index in group:
            references[index] = mean
    return references


def rake_departure(rake: np.ndarray, reference: float, tolerance: float) -> np.ndarray:
    """Where subfaults slipped differently from their fault as a whole."""
    departure = (rake - reference + 180.0) % 360.0 - 180.0
    return (np.abs(departure) > tolerance).astype(float)


def slip_directions(rake: np.ndarray, panel: Panel) -> np.ndarray:
    """Unit vectors along which each subfault slipped, in map coordinates.

    Rake is measured within the fault plane, from the strike direction towards
    up dip, so the slip direction is ``cos(rake)`` along strike plus
    ``sin(rake)`` up dip. Because the plane has been laid flat, both of those
    directions are drawn at true length and the arrow shows the in-plane angle
    undistorted -- which is the whole reason for laying it flat.
    """
    radians = np.radians(rake)
    return (
        np.cos(radians)[..., None] * panel.strike
        - np.sin(radians)[..., None] * panel.down_dip
    )


def block_size(cells: int, cell_km: float, spacing_km: float) -> int:
    """How many of ``cells`` subfaults go into one block, along one axis.

    Sized from the number of blocks that fits rather than straight from
    ``spacing_km``, so that a plane which does not divide evenly ends with a
    block a little short of the others rather than with a sliver: 619 rows at
    a spacing of 77 leaves a last block three rows deep, whose arrow stands on
    the very edge of the panel and speaks for a fortieth of what its
    neighbours do.

    At least one subfault: a plane already coarser than ``spacing_km`` cannot
    be coarsened, and a fractional block would silently drop it.
    """
    blocks = max(1, round(cells * cell_km / spacing_km))
    return math.ceil(cells / blocks)


def block_shape(panel: Panel, spacing_km: float) -> tuple[int, int]:
    """How many subfaults, down dip and along strike, go into one block."""
    ndip, nstk = panel.centres.shape[:2]
    return (
        block_size(ndip, panel.cell_km[1], spacing_km),
        block_size(nstk, panel.cell_km[0], spacing_km),
    )


def block_mean(field: np.ndarray, block: tuple[int, int]) -> np.ndarray:
    """``field`` averaged over ``block``-sized tiles.

    A plane whose grid does not divide evenly into blocks keeps its ragged
    edge as a partial tile rather than losing it: the array is padded with
    ``nan`` and the mean ignores the padding, so the last block along each
    axis is the average of however many subfaults are actually there. Every
    tile contains at least one real value, so no tile averages to ``nan``.
    """
    rows, columns = block
    ndip, nstk = field.shape
    padded = np.full(
        (math.ceil(ndip / rows) * rows, math.ceil(nstk / columns) * columns), np.nan
    )
    padded[:ndip, :nstk] = field
    tiled = padded.reshape(
        padded.shape[0] // rows, rows, padded.shape[1] // columns, columns
    )
    return np.nanmean(tiled, axis=(1, 3))


def circular_block_mean(angles: np.ndarray, block: tuple[int, int]) -> np.ndarray:
    """``angles`` in degrees averaged over ``block``-sized tiles, as angles.

    The same reason :func:`circular_mean` exists: a block
    holding rakes of +179 and -179 degrees is slipping due west, and its
    arithmetic mean points due east.
    """
    radians = np.radians(angles)
    return np.degrees(
        np.arctan2(
            block_mean(np.sin(radians), block), block_mean(np.cos(radians), block)
        )
    )


def block_centres(panel: Panel, block: tuple[int, int]) -> np.ndarray:
    """Where each of a plane's blocks sits, in panel coordinates."""
    return np.stack(
        [block_mean(panel.centres[..., axis], block) for axis in (0, 1)], axis=-1
    )


def draw_field(
    ax: plt.Axes,
    panels: list[Panel],
    values: list[np.ndarray],
    cmap,
    norm,
    display: Display,
    trace: bool = False,
) -> None:
    """Fill every plane with its field, and outline it.

    With ``trace``, the top edge is redrawn heavy over the outline -- see
    :data:`TRACE_WIDTH` for why that is a map-layout thing.

    A plane whose cells are axis-aligned and evenly spaced goes down as a
    rasterised image, which is both far faster than two million
    quadrilaterals and better behaved when the panel is smaller than the grid:
    ``imshow`` resamples the field with an antialiasing filter, where
    ``pcolormesh`` leaves the renderer to pick whichever subfault happens to
    land under each pixel and so turns a fine, speckled field into moire.
    """
    for panel, field in zip(panels, values):
        if panel.axis_aligned:
            ax.imshow(
                field,
                cmap=cmap,
                norm=norm,
                # The field's first row is its shallowest, which is the *top*
                # of the extent box whichever way round the axis runs.
                extent=(
                    panel.corners[0, 0, 0],
                    panel.corners[0, -1, 0],
                    panel.corners[-1, 0, 1],
                    panel.corners[0, 0, 1],
                ),
                origin="upper",
                interpolation="antialiased",
                aspect="auto",  # the axes owns the aspect ratio, not the image
                rasterized=True,
                zorder=2,
            )
        else:
            ax.pcolormesh(
                panel.corners[..., 0],
                panel.corners[..., 1],
                field,
                cmap=cmap,
                norm=norm,
                shading="flat",
                rasterized=True,
                zorder=2,
            )
        ax.add_patch(
            Polygon(
                panel.outline,
                closed=True,
                fill=False,
                edgecolor="black",
                linewidth=display.mark(OUTLINE_WIDTH),
                zorder=4,
            )
        )
        if trace:
            # Row 0 of the corners is the shallowest, i.e. the top edge.
            ax.plot(
                panel.corners[0, :, 0],
                panel.corners[0, :, 1],
                color="black",
                lw=display.mark(TRACE_WIDTH),
                solid_capstyle="round",
                zorder=5,
            )


def draw_isochrones(
    ax: plt.Axes,
    panels: list[Panel],
    interval: float,
    display: Display,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> None:
    """The rupture front, at whole multiples of ``interval`` seconds.

    The levels are the same on every plane so that one isochrone crossing two
    planes is one line, and they are labelled in place rather than in a legend
    because a contour's value belongs on the contour.

    Every plane is contoured before any of them is labelled, because a plane
    is a unit of the fault and not of the figure: an isochrone crossing three
    planes is one isochrone, and what the contouring hands back is three
    pieces of it that have to be put together again before anything is
    written on them.
    """
    onsets = gather(panels, "tinit")
    levels = interval * np.arange(
        math.ceil(onsets.min() / interval), onsets.max() / interval + 1
    )
    if not len(levels):
        return

    contour_sets = []
    reach = 0.0
    for panel in panels:
        if not panel.contourable:
            continue
        ndip, nstk = panel.centres.shape[:2]
        block = (
            max(1, math.ceil(ndip / CONTOUR_MAX_CELLS)),
            max(1, math.ceil(nstk / CONTOUR_MAX_CELLS)),
        )
        # The contour runs between cell *centres*, so it stops half a coarsened
        # cell short of the plane's edge at either end. Two pieces meeting at a
        # plane boundary are therefore a whole cell apart at their closest, and
        # that is the gap a join has to be able to see across.
        reach = max(
            reach,
            JOIN_CELLS
            * 1000.0
            * max(block[0] * panel.cell_km[1], block[1] * panel.cell_km[0]),
        )
        centres = block_centres(panel, block)
        contour_sets.append(
            ax.contour(
                centres[..., 0],
                centres[..., 1],
                block_mean(panel.fields["tinit"], block),
                levels=levels,
                colors="black",
                linewidths=display.mark(CONTOUR_WIDTH),
                zorder=5,
            )
        )

    for contours, places in zip(
        contour_sets,
        label_places(ax, contour_sets, isochrone_text, reach, obstacles),
    ):
        if not places:
            continue
        labels = ax.clabel(
            contours,
            manual=places,
            fmt=isochrone_text,
            fontsize=FONT_CONTOUR,
            inline=True,
            inline_spacing=2,
        )
        # A label sits on a full-strength colour field and has to be read
        # against every part of it; the halo is what makes it a gap in the
        # field rather than ink on top of it.
        for label in labels:
            label.set_path_effects(
                [patheffects.withStroke(linewidth=CONTOUR_HALO, foreground="white")]
            )


def isochrone_text(level: float) -> str:
    """What an isochrone is labelled with."""
    return f"{level:g} s"


def axes_inches(ax: plt.Axes) -> np.ndarray:
    """How big the panel is, in inches."""
    box = ax.get_position()
    figure = ax.get_figure()
    return np.array(
        [box.width * figure.get_figwidth(), box.height * figure.get_figheight()]
    )


def text_extent(text: str, fontsize: float) -> np.ndarray:
    """How much room a piece of text takes, in inches.

    Estimated rather than measured, because none of this has been drawn yet
    and a renderer to measure with is not always to be had. The estimate errs
    wide, which is the direction that keeps labels apart.
    """
    return (
        np.array([LABEL_ASPECT * len(text), LINE_HEIGHT]) * fontsize / POINTS_PER_INCH
    )


def label_clearance(ax: plt.Axes, text: str) -> np.ndarray:
    """How far inside the panel a contour label has to sit, in axes fractions.

    Half the label's *diagonal* in both directions rather than half its width
    one way and half its height the other: ``clabel`` rotates a label along
    its contour, so which of the two the label spends on which axis is not
    known until it is placed.
    """
    diagonal = float(np.hypot(*text_extent(text, FONT_CONTOUR)))
    return 0.5 * diagonal / axes_inches(ax)


def contour_runs(contour_sets: list) -> dict[float, list[tuple[int, np.ndarray]]]:
    """Every polyline the contouring produced, gathered by level.

    Each comes back with the index of the set it belongs to, because that is
    what ``clabel`` has to be handed the label back through.
    """
    runs: dict[float, list[tuple[int, np.ndarray]]] = {}
    for index, contours in enumerate(contour_sets):
        for path, level in zip(contours.get_paths(), contours.levels):
            for vertices in path.to_polygons(closed_only=False):
                if len(vertices) >= 2:
                    runs.setdefault(float(level), []).append((index, vertices))
    return runs


def join_runs(
    runs: list[tuple[int, np.ndarray]], reach: float
) -> list[list[tuple[int, np.ndarray]]]:
    """Polylines of one level grouped into the curves they are pieces of.

    Contouring is done a plane at a time, so a curve running across a plane
    boundary comes back as two polylines whose ends stop just short of each
    other. Two polylines belong to the same curve when an end of one is within
    ``reach`` of an end of the other, and the relation is transitive -- a
    curve crossing three planes is three pieces joined in a chain.

    This is the distinction that matters for labelling. A level is not a
    curve: a rupture front spreading both ways from its hypocentre gives every
    isochrone two branches, hundreds of kilometres apart, and they are two
    curves that each need their own number. Nor is a piece a curve, or the
    same front gets numbered again at every plane boundary it crosses.
    """
    owner = list(range(len(runs)))

    def root(index: int) -> int:
        while owner[index] != index:
            owner[index] = owner[owner[index]]
            index = owner[index]
        return index

    ends = [vertices[[0, -1]] for _, vertices in runs]
    for first in range(len(runs)):
        for second in range(first + 1, len(runs)):
            gap = np.linalg.norm(
                ends[first][:, None, :] - ends[second][None, :, :], axis=-1
            ).min()
            if gap <= reach:
                owner[root(first)] = root(second)

    curves: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, run in enumerate(runs):
        curves.setdefault(root(index), []).append(run)
    return list(curves.values())


def label_places(
    ax: plt.Axes,
    contour_sets: list,
    text_for,
    reach: float,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> list[list[tuple[float, float]]]:
    """One place per curve, chosen so that the label lands on the panel.

    ``clabel`` left to itself puts labels at intervals along a contour and
    lets them fall where they fall, which on a panel this shape means the ones
    near an edge come out sliced in half by it -- a reader sees "00 s" and has
    to guess. So the places are chosen here instead.

    The unit is the curve: every curve gets exactly one number, and no curve
    gets two. That is what keeps a symmetric rupture symmetric -- both
    branches of an isochrone are labelled, on both sides of the hypocentre --
    while a front broken into pieces at a plane boundary is still numbered
    once.

    Within a curve the label goes at the point furthest from the edges of the
    panel that is also clear of the labels already placed and of any text
    already annotated onto the panel; longer curves choose first, since they
    are the ones a reader most needs to identify. A curve with no such point
    -- one that never gets a label's width in from the edge, or that runs
    entirely under something already written -- goes unlabelled rather than
    half-labelled, as does one shorter than its own label. Every level is
    still *drawn* either way; what is chosen here is only where the numbers
    go.

    Distances are measured in units of a label's clearance -- half its own
    diagonal, since ``clabel`` rotates a label along its contour -- so that
    one rule covers a short label and a long one on a panel of any shape.

    Parameters
    ----------
    reach : float
        How far apart, in data units, the ends of two polylines may be and
        still be pieces of one curve.
    obstacles : Sequence[tuple[np.ndarray, np.ndarray]]
        Boxes already written on the panel that a label must not land on,
        each a centre and a half-extent in axes fractions.

    Returns
    -------
    list[list[tuple[float, float]]]
        One list of data-coordinate points per contour set, ready to hand
        straight to ``clabel``'s ``manual``.
    """
    to_axes = ax.transAxes.inverted().transform
    runs = contour_runs(contour_sets)
    if not runs:
        return [[] for _ in contour_sets]

    # Separations are compared against the widest label in the figure, so that
    # one number means the same thing for "5 s" as for "100 s"; the clearance
    # from the panel's own edges is each label's own, since that is about the
    # size of the label being placed rather than about crowding.
    common = np.array(
        max(
            (label_clearance(ax, text_for(level)) for level in runs),
            key=lambda clearance: clearance[0],
        )
    )

    curves = []
    for level, pieces in runs.items():
        clearance = label_clearance(ax, text_for(level))
        for curve in join_runs(pieces, reach):
            points = np.concatenate([vertices for _, vertices in curve])
            owners = np.concatenate(
                [np.full(len(vertices), index) for index, vertices in curve]
            )
            fractions = to_axes(ax.transData.transform(points))
            span = np.hypot(
                *((fractions.max(axis=0) - fractions.min(axis=0)) / clearance)
            )
            if span < LABEL_MIN_REACH:
                continue
            inset = np.minimum(
                np.minimum(fractions[:, 0], 1.0 - fractions[:, 0]) / clearance[0],
                np.minimum(fractions[:, 1], 1.0 - fractions[:, 1]) / clearance[1],
            )
            free = inset >= 1.0
            for centre, half in obstacles:
                # Two boxes miss each other when they are apart along either
                # axis, so it takes both to overlap and either to be clear.
                free &= (
                    np.abs(fractions - centre) >= OBSTACLE_MARGIN * (half + clearance)
                ).any(axis=1)
            curves.append((span, points, owners, fractions / common, inset, free))

    places: list[list[tuple[float, float]]] = [[] for _ in contour_sets]
    taken: list[np.ndarray] = []
    for _, points, owners, scaled, inset, free in sorted(curves, key=lambda c: -c[0]):
        room = free.copy()
        for other in taken:
            room &= np.hypot(*(scaled - other).T) >= LABEL_SEPARATION
        if not room.any():
            continue
        best = int(np.argmax(np.where(room, inset, -np.inf)))
        taken.append(scaled[best])
        places[owners[best]].append((float(points[best, 0]), float(points[best, 1])))
    return places


def draw_arrows(
    ax: plt.Axes, panels: list[Panel], spacing_km: float, display: Display
) -> int:
    """One slip-direction arrow per block of subfaults, on every plane.

    The plane is cut into blocks about ``spacing_km`` across and each block
    gets exactly one arrow, drawn along the block's mean slip direction. The
    block is *averaged*, not sampled: taking every nth subfault instead leaves
    the arrows carrying the roughness of the field at the scale of one
    subfault, and lays them out on a grid regular enough to beat against the
    pixel grid, which is what turns a rake panel into a moire pattern rather
    than a set of directions.

    A plane already coarser than ``spacing_km`` gets its arrows further apart
    than asked for, and they grow to suit the room they have -- shaft as well
    as length, because an arrowhead is sized off its shaft, and growing only
    the length would draw a long arrow with a head too small to show which end
    it was. The growth is capped: a one-subfault plane has room for an arrow
    the size of the whole rupture, which would say nothing about slip
    direction and a great deal about arrows.

    Each arrow is centred on its block rather than starting there, so it marks
    the ground it belongs to instead of reaching into the next block.

    Every block gets one, and every arrow is the same length. Nothing about
    the panel is weighted by how much the block slipped: the row above it is
    the slip, and an arrow field that is also a slip field says the second
    thing badly and hides gaps in the first. A reader who wants to know
    whether a direction is worth anything reads it against the slip panel,
    which is directly above and drawn in the same coordinates.

    Returns
    -------
    int
        How many arrows were drawn.
    """
    drawn = 0
    for panel in panels:
        block = block_shape(panel, spacing_km)
        achieved_km = min(block[0] * panel.cell_km[1], block[1] * panel.cell_km[0])
        centres = block_centres(panel, block)
        direction = slip_directions(
            circular_block_mean(panel.fields["rake"], block), panel
        )
        growth = min(achieved_km / spacing_km, ARROW_GROWTH_LIMIT)
        length = ARROW_LENGTH * growth * spacing_km * 1000.0
        ax.quiver(
            centres[..., 0],
            centres[..., 1],
            length * direction[..., 0],
            length * direction[..., 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            color="black",
            width=display.mark(ARROW_WIDTH) * growth,
            zorder=6,
        )
        drawn += centres[..., 0].size
    return drawn


def parse_rule(value: str) -> tuple[float, str]:
    """A ``--down-dip-line`` argument, as a position and the text for it."""
    position, _, text = value.partition(":")
    try:
        fraction = float(position)
    except ValueError:
        raise typer.BadParameter(
            f"{value!r}: expected a fraction of the down-dip extent, "
            "optionally followed by ':' and a label"
        ) from None
    if not 0.0 <= fraction <= 1.0:
        raise typer.BadParameter(f"{fraction:g} is not between 0 and 1")
    return fraction, text.strip()


def draw_rule(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    fraction: float,
    text: str,
    display: Display,
) -> tuple[np.ndarray, np.ndarray] | None:
    """A dashed rule across the panel, at ``fraction`` of the way down dip.

    For marking a line the rupture was *made* to -- how far up dip a generator
    was allowed to put its subevents, where a locking model stops -- as
    against the fields either side of it, which are what came out. The label
    goes on the rule because an unexplained horizontal line in a panel like
    this is the first thing a reader asks about.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Where the label ended up -- a centre and a half-extent in axes
        fractions -- so that whatever is placed afterwards can keep off it, or
        ``None`` if the rule carries no label. Reported rather than measured
        back off the artist: an annotation offset in points does not say where
        it is through the ordinary text interface, and guessing at that is how
        a contour label ends up on top of it anyway.
    """
    halo = [patheffects.withStroke(linewidth=RULE_HALO, foreground="white")]
    y = bounds[1] + fraction * (bounds[3] - bounds[1])
    ax.axhline(
        y,
        color="black",
        linewidth=display.mark(RULE_WIDTH),
        dashes=RULE_DASHES,
        zorder=6,
        path_effects=halo,
    )
    if not text:
        return None

    ax.annotate(
        text,
        (0.0, y),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=RULE_LABEL_OFFSET,
        ha="left",
        va="bottom",
        fontsize=FONT_STATS,
        zorder=7,
        path_effects=halo,
    )
    # Anchored at the panel's left edge, sitting up and to the right of the
    # rule, so its middle is one offset and one half-extent along from there.
    anchor = ax.transAxes.inverted().transform(ax.transData.transform((bounds[0], y)))
    half = 0.5 * text_extent(text, FONT_STATS) / axes_inches(ax)
    offset = np.array(RULE_LABEL_OFFSET) / POINTS_PER_INCH / axes_inches(ax)
    return anchor + offset + half, half


def draw_hypocentre(ax: plt.Axes, panels: list[Panel], display: Display) -> None:
    """A star wherever a plane's header claims the rupture began."""
    for panel in panels:
        if panel.hypocentre is None:
            continue
        ax.plot(
            panel.hypocentre[0],
            panel.hypocentre[1],
            marker="*",
            ls="none",
            ms=display.mark(13),
            mfc=HYPOCENTRE_COLOUR,
            mec="black",
            mew=display.mark(0.7),
            zorder=7,
        )


def draw_scale_bar(ax: plt.Axes, display: Display) -> None:
    """A round-numbered distance scale under the panel's left edge.

    The panels carry no axes at all, so the bar is the only thing in the figure
    that says how big the rupture is. It is drawn in data coordinates, which
    means it has to be drawn after the limits are set.
    """
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    length = nice_scale_length((east - west) / 1000.0) * 1000.0

    left = west + 0.01 * (east - west)
    y = south - 0.045 * (north - south)
    tick = 0.014 * (north - south)
    ax.plot(
        [left, left, left + length, left + length],
        [y + tick, y, y, y + tick],
        color="black",
        lw=display.mark(SCALE_BAR_WIDTH),
        solid_capstyle="butt",
        clip_on=False,
        zorder=8,
    )
    ax.annotate(
        f"{length / 1000:g} km",
        (left + length / 2, y),
        textcoords="offset points",
        xytext=(0, -2),
        ha="center",
        va="top",
        fontsize=FONT_SCALE,
        annotation_clip=False,
        zorder=8,
    )


def margins_for(layout: Layout) -> Margins:
    """What each edge of the canvas has to hold, for one layout."""
    return FAULT_MARGINS if layout is Layout.fault else MAP_MARGINS


def panel_bounds(
    panels: list[Panel], margins: Margins
) -> tuple[float, float, float, float]:
    """The window every panel is drawn in, padded off the planes."""
    corners = np.vstack([panel.corners.reshape(-1, 2) for panel in panels])
    west, south = corners.min(axis=0)
    east, north = corners.max(axis=0)
    pad = margins.pad * max(east - west, north - south)
    return west - pad, south - pad, east + pad, north + pad


def strip(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Set a panel's window and take away everything but the drawing."""
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")


def frame(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    bottom_row: bool,
    display: Display,
) -> None:
    """Set an unrolled panel's window and put real axes on it.

    Down dip runs downwards, which is the way a cross-section is read and the
    way the rupture actually goes. The tick intervals are round distances
    rather than a locator's idea of them, and there are deliberately few of
    them down dip: the panel is a fifth as tall as it is wide, and a label
    every ten kilometres on it is a solid line of digits.

    Only the bottom row is given an along-strike axis. All three rows show the
    same window, so repeating it says the same thing three times and takes the
    height to say it in away from the panels.
    """
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[3], bounds[1])  # down dip downwards
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    kilometres = FuncFormatter(lambda value, _: f"{value / 1000:g}")
    ax.xaxis.set_major_locator(MultipleLocator(ALONG_STRIKE_TICK_KM * 1000.0))
    ax.xaxis.set_major_formatter(kilometres)
    ax.yaxis.set_major_locator(MultipleLocator(DOWN_DIP_TICK_KM * 1000.0))
    ax.yaxis.set_major_formatter(kilometres)
    ax.tick_params(labelsize=FONT_TICK, width=display.mark(0.5), length=2.5, pad=2)
    ax.set_ylabel("Down-dip (km)", fontsize=FONT_LABEL, labelpad=2)
    if bottom_row:
        ax.set_xlabel("Along-strike (km)", fontsize=FONT_LABEL, labelpad=2)
    else:
        ax.tick_params(labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(display.mark(OUTLINE_WIDTH))


def label_ends(ax: plt.Axes, panels: list[Panel]) -> None:
    """Which compass direction each end of the along-strike axis is.

    An unrolled panel has thrown away the one thing a map gives for free --
    which way round the rupture is -- and "along strike" only says which way
    round if the reader already knows the strike. Two words at the ends of the
    axis put it back, and are the difference between a figure that could have
    been drawn mirrored and one that says it was not.
    """
    trace = np.array(
        [[float(panel.header["elat"]), float(panel.header["elon"])] for panel in panels]
    )
    if len(trace) < 2 or np.allclose(trace[0], trace[-1]):
        return
    middle = trace.mean(axis=0)
    # A degree of longitude is shorter than a degree of latitude everywhere but
    # the equator, and at forty south it is three quarters as long; unscaled,
    # a fault striking south-west comes out south-south-west.
    scale = np.array([1.0, math.cos(math.radians(float(middle[0])))])
    for x, align, end in ((0.0, "left", trace[0]), (1.0, "right", trace[-1])):
        ax.annotate(
            compass_point((end - middle) * scale),
            (x, 0.0),
            xycoords="axes fraction",
            textcoords="offset points",
            xytext=(0, -13),  # clear of the tick labels, on the axis title's line
            ha=align,
            va="top",
            fontsize=FONT_STATS,
            annotation_clip=False,
        )


def compass_point(offset: np.ndarray) -> str:
    """The nearest of the eight compass points to a north-east offset."""
    bearing = math.degrees(math.atan2(offset[1], offset[0])) % 360.0
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[round(bearing / 45.0) % len(points)]


def draw_depth_axis(ax: plt.Axes, panels: list[Panel], display: Display) -> None:
    """A second scale on the right, in depth rather than distance down dip.

    A shallow-dipping interface is a hundred kilometres wide and twenty
    kilometres deep, and a reader who takes the left-hand axis for depth is
    reading the figure five times too deep. The two are related by the dip, so
    the second axis costs nothing but the room to print it.

    Fitted to where the subfaults actually are rather than to what the header
    says the dip is, for the same reason :func:`lay_flat` reads the geometry
    off the subfaults: SRF headers are unreliable, and here the header's dip
    is what the depth scale *is*. A rupture whose planes do not all lie on one
    surface has no single depth scale, and is warned about rather than quietly
    fitted anyway.
    """
    # One column per plane is enough: on a planar fault the depth of a subfault
    # is a function of how far down dip it is and of nothing else.
    down_dip = np.concatenate([panel.centres[:, 0, 1] for panel in panels])
    depth = np.concatenate([panel.fields["dep"][:, 0] for panel in panels])
    slope, top = np.polyfit(down_dip, depth, 1)
    residual = float(np.abs(depth - (slope * down_dip + top)).max())
    if residual > DEPTH_AXIS_TOLERANCE_KM:
        console_warn(
            f"the planes' subfaults sit up to {residual:.1f} km off the single "
            "sloping surface the depth axis is drawn for; read it as "
            "approximate"
        )

    axis = ax.secondary_yaxis(
        "right",
        functions=(
            lambda metres: top + slope * metres,
            lambda km: (km - top) / slope,
        ),
    )
    axis.set_ylabel("Depth (km)", fontsize=FONT_LABEL, labelpad=2)
    axis.tick_params(labelsize=FONT_TICK, width=display.mark(0.5), length=2.5, pad=2)
    axis.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
    axis.spines["right"].set_linewidth(display.mark(OUTLINE_WIDTH))


def canvas(
    bounds: tuple[float, float, float, float], margins: Margins
) -> tuple[tuple[float, float], list[tuple[float, float, float, float]]]:
    """The design canvas, and the three panel rectangles on it.

    Every row shows the same window at equal aspect, so the rupture's own shape
    fixes the row's shape; the canvas is then built around three of them rather
    than the rows being squeezed into a canvas chosen in advance. A rupture much
    taller than it is wide would otherwise make a figure nobody can print.

    Returns
    -------
    tuple[tuple[float, float], list[tuple[float, float, float, float]]]
        The figure size in inches, and one ``(left, bottom, width, height)``
        rectangle per row in figure fractions, top row first.
    """
    aspect = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    height = min(PANEL_HEIGHT_LIMIT, PANEL_WIDTH * aspect)
    width = height / aspect

    figure = (
        margins.left + width + margins.bar_gap + BAR_WIDTH + margins.right,
        MARGIN_TOP + 3 * height + 2 * margins.row_gap + margins.bottom,
    )
    rows = [
        (
            margins.left / figure[0],
            (margins.bottom + (2 - row) * (height + margins.row_gap)) / figure[1],
            width / figure[0],
            height / figure[1],
        )
        for row in range(3)
    ]
    return figure, rows


def bar_rectangle(
    row: tuple[float, float, float, float],
    figure: tuple[float, float],
    margins: Margins,
) -> tuple[float, float, float, float]:
    """The colourbar rectangle beside ``row``, in figure fractions.

    The same offset and the same width for every row, so that the three bars
    line up with each other however different the things they scale are -- and
    exactly the height of the panel they belong to, so that each one is
    visibly *that* row's.
    """
    return (
        row[0] + row[2] + margins.bar_gap / figure[0],
        row[1],
        BAR_WIDTH / figure[0],
        row[3],
    )


def label_row(ax: plt.Axes, text: str) -> None:
    """The row's name, over its top left corner.

    Above the panel rather than inside it: a plane can reach any corner of the
    window, and a label that lands on the field it names is worse than no
    label.
    """
    ax.text(
        0.0,
        1.0,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_LABEL,
        fontweight="bold",
    )


def label_stats(ax: plt.Axes, text: str) -> None:
    """The panel's min / mean / max, over its top right corner.

    Opposite the row label, on the same line, so the two together read as one
    heading for the row. The map layout puts this over the colourbar instead,
    where it fits because it is three bare numbers; spelled out it is too wide
    for a bar an eighth of an inch across.
    """
    ax.text(
        1.0,
        1.0,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_STATS,
    )


def bar_label_limit(height_in: float) -> int:
    """How many tick labels a colourbar ``height_in`` inches tall can hold.

    A bar as tall as the page can label every level; the fault layout's bars
    are the height of a panel a fifth as tall as it is wide, and on those the
    same eleven labels are a solid column of digits -- which is the tick
    spacing complaint arriving at the other side of the figure.
    """
    return max(2, math.floor(height_in / (BAR_LABEL_PITCH * FONT_TICK / 72.0)))


def bands_for_bar(wanted: int, height_in: float) -> int:
    """How many colour bands a bar ``height_in`` inches tall should carry.

    Not more than it can label. A field cut into ten bands on a bar with room
    for six numbers leaves four of its colours standing between labelled
    boundaries, and the reader has to count bands to find out what any of them
    means; cutting the field into the number of bands that *can* be labelled
    costs some resolution in the ramp and gains a scale that can be read
    straight off. The band count is what the figure gives up, because the
    alternative is giving up the labels.
    """
    return max(2, min(wanted, bar_label_limit(height_in) - 1))


def thinned(levels: np.ndarray, limit: int) -> list[float]:
    """At most ``limit`` of ``levels``, evenly spaced and keeping both ends.

    Every ``step``th level rather than ``limit`` of them picked by index: a
    colourbar labelled 0, 400, 1000, 1600, 2000 has the reader measuring the
    gaps instead of reading the colours, and a scale is the one place a figure
    cannot afford that. Taking a step that divides the levels exactly is what
    keeps the labels evenly spaced, so the count lands on whatever the
    divisors allow rather than exactly on ``limit``.
    """
    span = len(levels) - 1
    for step in range(1, span + 1):
        if span % step == 0 and span // step + 1 <= limit:
            return [float(level) for level in levels[::step]]
    return [float(levels[0]), float(levels[-1])]


def draw_bar(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    cmap,
    norm,
    levels: np.ndarray,
    values: np.ndarray,
    display: Display,
    heading: str | None = None,
) -> None:
    """A vertical colourbar for one row, headed by the field's own extremes.

    The bar says how the colours map to numbers; the header says what numbers
    the field actually reached, which the bar cannot, because its range is the
    robust one and the extremes are exactly what that leaves out. Pass
    ``heading=None`` where the row says that above the panel instead.
    """
    cax = fig.add_axes(rect)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        extend=overflow(values, levels),
    )
    shown = thinned(
        levels,
        min(
            bar_label_limit(rect[3] * fig.get_figheight()),
            display.ticks(len(levels)),
        ),
    )
    bar.set_ticks(shown)
    bar.set_ticklabels([f"{level:g}" for level in shown], fontsize=FONT_TICK)
    bar.outline.set_linewidth(display.mark(0.5))
    if heading is not None:
        cax.set_title(heading, fontsize=FONT_STATS, pad=4.0)


def draw_legend(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    tones: tuple[str, ...],
    labels: list[str],
    display: Display,
    heading: str | None = None,
) -> None:
    """A stack of labelled swatches, in the shape and place of a colourbar.

    The rake panel is a two-tone field rather than a scale, so it has no
    colourbar -- and a row with nothing in its bar column leaves the figure
    with two of three rows reaching the right-hand margin and one stopping
    short, which reads as bad spacing rather than as an absence. It also
    leaves the two tones unexplained. Both are answered by putting the key in
    the rectangle the colourbar would have had, ticked on the right like one.
    """
    cax = fig.add_axes(rect)
    cax.imshow(
        np.arange(len(tones)).reshape(-1, 1),
        cmap=ListedColormap(list(tones)),
        extent=(0.0, 1.0, 0.0, float(len(tones))),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks(np.arange(len(tones)) + 0.5)
    cax.set_yticklabels(labels, fontsize=FONT_TICK)
    cax.tick_params(length=0, pad=2)
    for spine in cax.spines.values():
        spine.set_linewidth(display.mark(0.5))
    if heading is not None:
        cax.set_title(heading, fontsize=FONT_STATS, pad=4.0)


def slip_panels(
    srf_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            metavar="SRF",
            help="SRF file to draw, in either the text or the HDF5 format",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output image path (omit to show interactively)"
        ),
    ] = None,
    layout: Annotated[
        Layout,
        typer.Option(
            help="map: planes where they are, hinged flat into the map. "
            "fault: planes unrolled into along-strike and down-dip "
            "coordinates, which fits a long, shallow fault onto a page",
        ),
    ] = Layout.map,
    levels: Annotated[
        int,
        typer.Option(
            help="Colour bands per field, roughly. Capped at what the "
            "colourbar has room to label, so that every band boundary "
            "carries a number",
        ),
    ] = 10,
    interval: Annotated[
        float | None,
        typer.Option(
            "--contour-interval",
            help="Isochrone interval in seconds (default: a round interval "
            "giving about six isochrones)",
        ),
    ] = None,
    rake_tolerance: Annotated[
        float,
        typer.Option(
            "--rake-tolerance",
            help="Degrees a subfault's rake may depart from its fault's mean "
            "before the rake panel greys it",
        ),
    ] = RAKE_TOLERANCE,
    isochrones: Annotated[
        bool,
        typer.Option(
            "--isochrones/--no-isochrones",
            help="Draw rupture-time contours over the slip",
        ),
    ] = True,
    arrows: Annotated[
        bool,
        typer.Option("--arrows/--no-arrows", help="Draw slip-direction arrows"),
    ] = True,
    arrow_columns: Annotated[
        int,
        typer.Option(
            "--arrow-columns",
            help="Roughly how many rake arrows to fit across the figure. Each "
            "one is the mean slip direction of the block of subfaults it "
            "stands on",
        ),
    ] = ARROW_COLUMNS,
    down_dip_line: Annotated[
        list[str] | None,
        typer.Option(
            "--down-dip-line",
            metavar="FRACTION[:LABEL]",
            help="Draw a dashed rule across every panel at this fraction of "
            "the down-dip extent, for a line the rupture was made to rather "
            'than one it produced. Repeatable. Example: 0.5:"subevent limit". '
            "Fault layout only",
        ),
    ] = None,
    depth_axis: Annotated[
        bool,
        typer.Option(
            "--depth-axis/--no-depth-axis",
            help="In the fault layout, put a second scale on the right giving "
            "depth rather than distance down dip",
        ),
    ] = True,
    scale_bar: Annotated[
        bool | None,
        typer.Option(
            "--scale-bar/--no-scale-bar",
            help="Draw a distance scale (default: only in the map layout, "
            "since the fault layout has axes)",
        ),
    ] = None,
    display_height: Annotated[
        float | None,
        typer.Option(
            "--display-height",
            help="Height (cm) the figure will be displayed at, e.g. on a poster; "
            "with --viewing-distance, scales the text to suit",
        ),
    ] = None,
    viewing_distance: Annotated[
        float | None,
        typer.Option(
            "--viewing-distance",
            help="Distance (m) the figure must be readable from; "
            "needs --display-height to have any effect",
        ),
    ] = None,
    dpi: Annotated[int, typer.Option(help="Output resolution")] = 300,
) -> None:
    """Plot an SRF's slip, rise time and rake as three stacked fault panels."""
    margins = margins_for(layout)
    unrolled = layout is Layout.fault
    if scale_bar is None:
        scale_bar = not unrolled

    rules = [parse_rule(value) for value in down_dip_line or []]
    if rules and not unrolled:
        console_warn(
            "--down-dip-line needs an axis to be drawn against; the map "
            "layout has no down-dip direction on the page, so none is drawn"
        )
        rules = []

    panels = read_panels(srf_path, layout)
    bounds = panel_bounds(panels, margins)
    design, rows = canvas(bounds, margins)
    display = Display.for_figure(design, dpi, display_height, viewing_distance)
    display.report(design)

    fig = plt.figure(figsize=display.size, facecolor="white")
    axes = [fig.add_axes(rect) for rect in rows]
    written: list[tuple[np.ndarray, np.ndarray]] = []
    for index, ax in enumerate(axes):
        if unrolled:
            bottom_row = index == len(axes) - 1
            frame(ax, bounds, bottom_row, display)
            if bottom_row:
                label_ends(ax, panels)
            if depth_axis:
                draw_depth_axis(ax, panels, display)
            for fraction, text in rules:
                # Named once, on the top row. The rule is the same rule on all
                # three, and a reader who has read it once does not need it
                # again two inches further down.
                box = draw_rule(
                    ax, bounds, fraction, text if index == 0 else "", display
                )
                if box is not None:
                    written.append(box)
        else:
            strip(ax, bounds)

    bands = bands_for_bar(levels, rows[0][3] * design[1])
    if bands < levels:
        print(f"{bands} colour bands per field: as many as the colourbar can label")

    slip = gather(panels, "slip")
    slip_levels = discrete_levels(slip, bands)
    slip_cmap, slip_norm = discrete_ramp(
        LinearSegmentedColormap.from_list("slip", SLIP_RAMP), slip_levels
    )
    label_row(axes[0], "Slip (cm)")
    draw_field(
        axes[0],
        panels,
        [panel.fields["slip"] for panel in panels],
        slip_cmap,
        slip_norm,
        display,
        trace=not unrolled,
    )
    if unrolled:
        label_stats(axes[0], statistics_labelled(slip, "cm"))
    draw_bar(
        fig,
        bar_rectangle(rows[0], design, margins),
        slip_cmap,
        slip_norm,
        slip_levels,
        slip,
        display,
        heading=None if unrolled else statistics(slip),
    )

    if isochrones:
        step = interval or contour_interval(gather(panels, "tinit"))
        if step is None:
            console_warn(
                "every subfault ruptures at the same time; no isochrones drawn"
            )
        else:
            draw_isochrones(axes[0], panels, step, display, written)
            print(f"isochrones every {step:g} s")

    rise = gather(panels, "rise")
    rise_levels = discrete_levels(rise, bands)
    rise_cmap, rise_norm = discrete_ramp(plt.get_cmap(RISE_CMAP), rise_levels)
    label_row(axes[1], "Rise time (s)")
    draw_field(
        axes[1],
        panels,
        [panel.fields["rise"] for panel in panels],
        rise_cmap,
        rise_norm,
        display,
        trace=not unrolled,
    )
    if unrolled:
        label_stats(axes[1], statistics_labelled(rise, "s"))
    draw_bar(
        fig,
        bar_rectangle(rows[1], design, margins),
        rise_cmap,
        rise_norm,
        rise_levels,
        rise,
        display,
        heading=None if unrolled else statistics(rise),
    )

    rake = gather(panels, "rake")
    faults = fault_groups(panels)
    references = rake_references(panels, faults)
    print(
        f"{len(faults)} fault"
        + ("s" if len(faults) > 1 else "")
        + "; rake measured against "
        + " / ".join(f"{references[group[0]]:.1f}" for group in faults)
        + "\N{DEGREE SIGN}"
    )
    label_row(axes[2], "Rake")
    draw_field(
        axes[2],
        panels,
        [
            rake_departure(panel.fields["rake"], reference, rake_tolerance)
            for panel, reference in zip(panels, references)
        ],
        ListedColormap(RAKE_TONES),
        BoundaryNorm([-0.5, 0.5, 1.5], 2),
        display,
        trace=not unrolled,
    )
    if unrolled:
        label_stats(axes[2], statistics_labelled(rake, "\N{DEGREE SIGN}"))
    draw_legend(
        fig,
        bar_rectangle(rows[2], design, margins),
        RAKE_TONES,
        [
            f"within {rake_tolerance:g}\N{DEGREE SIGN}",
            f"beyond {rake_tolerance:g}\N{DEGREE SIGN}",
        ],
        display,
        heading=None if unrolled else statistics(rake),
    )
    if arrows:
        spacing = (bounds[2] - bounds[0]) / 1000.0 / arrow_columns
        drawn = draw_arrows(axes[2], panels, spacing, display)
        print(f"{drawn} rake arrows, one per {spacing:.1f} km block")

    # On the slip row alone. The rupture began in one place, and saying so
    # three times only puts a star over three subfaults' worth of rise time
    # and rake as well.
    draw_hypocentre(axes[0], panels, display)
    if not any(panel.hypocentre is not None for panel in panels):
        console_warn("no plane's header carries a hypocentre; no star drawn")
    if scale_bar:
        draw_scale_bar(axes[2], display)

    single = [panel for panel in panels if not panel.contourable]
    if single:
        console_warn(
            f"{len(single)} of {len(panels)} planes are a single row or column "
            "of subfaults; those carry no isochrones"
        )
    print(
        f"{len(panels)} plane"
        + ("s" if len(panels) > 1 else "")
        + f", {len(slip)} subfaults, "
        f"slip {slip.min():.1f}-{slip.max():.1f} cm"
    )

    if output is not None:
        fig.savefig(output, dpi=display.dpi, facecolor=fig.get_facecolor())
        print(f"wrote {output}")
    else:
        plt.show()
