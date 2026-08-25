"""Plan-view map of a (possibly multi-fault) realisation.

Draws each source's surface projection and top-edge trace, numbered in rupture
order and coloured by magnitude off a discrete ``magma`` ramp, over NZ CFM v1.0
traces, and marks the hypocentre::

    eqvis rupture-map sw4/R1/realisation.json
    eqvis rupture-map wellington.json -o rupture_map.png --no-cfm

Like the intensity measure maps it can be drawn for a size and a distance
rather than for the page -- a poster panel 25 cm tall, read from three metres
away::

    eqvis rupture-map sw4/R1/realisation.json --display-height 25 \\
        --viewing-distance 3 -o poster_rupture.png

The saved file has the same pixel dimensions as ever and is drawn to be
*placed* at that height; see :class:`~.display.Display` for what the scaling
does and does not stretch.
"""

import json
import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from qcore import coordinates
from source_modelling import community_fault_model, moment, rupture_propagation
from source_modelling.sources import Fault

from .console import console_warn
from .display import Display
from .geography import (
    draw_locator_map,
    draw_scale_bar,
    fill_land,
    load_coastline,
)
from .stations import corner_anchor, free_corner, place_labels

# Segments are coloured by magnitude off a discrete ``magma`` ramp. One bin per
# tenth of a magnitude unit: finer than that and neighbouring bins stop being
# tellable apart at poster distance.
MAGNITUDE_STEP = 0.1
# magma runs from near-black to near-white, and both ends disappear -- one
# against the traces, the other against the pale land -- so the ramp is
# truncated before it reaches either.
MAGMA_RANGE = (0.12, 0.88)

# Fallback for realisations that carry no magnitudes: a
# sequential-but-distinguishable palette (Okabe-Ito), cycled in rupture order.
FAULT_COLOURS = [
    "#0072b2",  # blue
    "#009e73",  # green
    "#d55e00",  # vermillion
    "#cc79a7",  # purple
    "#e69f00",  # orange
    "#56b4e9",  # sky
    "#8b4513",  # brown
]
HYPOCENTRE_COLOUR = "yellow"
CFM_GREY = "#9a9a9a"

# The rupture is the subject, so its traces are the heaviest ink on the map and
# everything else sits below the weight of a hairline.
TRACE_WIDTH = 2.6
CONTEXT_WIDTH = 0.6
# A surface projection is context for the trace on top of it, so the fill is
# only just present -- enough to say "the plane reaches under here".
PROJECTION_ALPHA = 0.22

# Where :func:`~.geography.draw_scale_bar` puts itself, as an axes-fraction
# rectangle, so the legend and the inset can be told to go somewhere else.
SCALE_BAR_CORNER = (0.66, 0.86, 0.34, 0.14)


def darken(colour, target: float = 0.42) -> tuple[float, float, float]:
    """``colour`` darkened until it is dark enough to read as text.

    A fixed factor is not enough: the pale end of the magma ramp stays pale
    after it, and a segment's number has to be legible against the white halo
    behind it whatever colour that segment came out. So the colour is scaled
    until its luminance is down to ``target``, and a colour already that dark
    is left where it is.
    """
    red, green, blue = to_rgb(colour)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance <= target:
        return (red, green, blue)
    factor = target / luminance
    return (red * factor, green * factor, blue * factor)


def magnitude_levels(magnitudes: dict[str, float]) -> np.ndarray:
    """Colour-bin boundaries spanning ``magnitudes``, on whole steps.

    Bins are whole multiples of ``MAGNITUDE_STEP``, so two maps of the same
    magnitude range coloured this way are directly comparable.
    """
    low = MAGNITUDE_STEP * math.floor(min(magnitudes.values()) / MAGNITUDE_STEP)
    high = MAGNITUDE_STEP * math.ceil(max(magnitudes.values()) / MAGNITUDE_STEP)
    if high <= low:
        # A single segment, or every segment inside one bin.
        high = low + MAGNITUDE_STEP
    count = round((high - low) / MAGNITUDE_STEP)
    return np.round(low + MAGNITUDE_STEP * np.arange(count + 1), 10)


def magnitude_ramp(levels: np.ndarray) -> tuple[ListedColormap, BoundaryNorm]:
    """The discrete ``magma`` ramp and norm for ``levels``."""
    colours = plt.get_cmap("magma")(np.linspace(*MAGMA_RANGE, len(levels) - 1))
    cmap = ListedColormap(colours)
    return cmap, BoundaryNorm(levels, cmap.N)


def worth_colouring(magnitudes: dict[str, float], order: list[str]) -> bool:
    """Whether magnitude is a difference this map can show in colour.

    It takes two segments to compare, and two bins for the comparison to be
    visible: one segment, or several that all land in the same bin, would get a
    ramp with a single colour on it and a colourbar saying nothing.
    """
    if len(order) < 2 or not all(name in magnitudes for name in order):
        return False
    return len(magnitude_levels(magnitudes)) > 2


def magnitude_colours(magnitudes: dict[str, float], cmap, norm) -> dict[str, tuple]:
    """Each fault's colour, with the top of the range kept inside the ramp."""
    return {
        name: cmap(min(int(norm(magnitude)), cmap.N - 1))
        for name, magnitude in magnitudes.items()
    }


def load_realisation(realisation_path: Path) -> tuple[dict[str, Fault], dict]:
    """Read a realisation JSON into fault sources and the raw realisation dict.

    Parameters
    ----------
    realisation_path : Path
        Path to the realisation JSON.

    Returns
    -------
    tuple[dict[str, Fault], dict]
        The named fault geometries, and the realisation as parsed JSON.
    """
    with open(realisation_path) as f:
        realisation = json.load(f)

    faults = {}
    for name, geometry in realisation["sources"]["source_geometries"].items():
        corners = np.array(
            [[c["latitude"], c["longitude"], c["depth"]] for c in geometry["corners"]]
        )
        faults[name] = Fault.from_corners(corners.reshape(-1, 4, 3))
    return faults, realisation


def surface_projection(fault: Fault) -> list[np.ndarray]:
    """The fault's down-dip extent projected onto the surface.

    Parameters
    ----------
    fault : Fault
        The fault to project.

    Returns
    -------
    list[np.ndarray]
        One (n, 2) array of (lon, lat) ring coordinates per polygon of the
        projection. Empty for a vertical fault, whose projection is its trace.
    """
    outline = shapely.transform(
        shapely.force_2d(fault.geometry),
        lambda nztm: coordinates.nztm_to_wgs_depth(nztm)[:, ::-1],
    )
    return [
        np.array(part.exterior.coords)
        for part in shapely.get_parts(outline)
        if isinstance(part, shapely.Polygon)
    ]


def trace_lon_lat(fault: Fault) -> np.ndarray:
    """The fault's top edge as an (n, 2) array of (lon, lat), in strike order."""
    corners = fault.corners.reshape(-1, 4, 3)
    trace = np.vstack([corners[:, 0, :2], corners[-1:, 1, :2]])
    return trace[:, ::-1]


def draw_strike_arrow(
    ax: plt.Axes,
    midpoint: tuple[float, float],
    strike: float,
    length_km: float,
    colour,
    display: Display,
) -> None:
    """An arrow along strike, centred on the trace's midpoint.

    Which way a segment ruptured is not something a line can say, and the
    number beside it says only when. The arrow is drawn in the segment's own
    colour and sits on top of its trace, so it reads as part of the line rather
    than as another thing on the map.
    """
    lon, lat = midpoint
    per_degree_lat = 111.32
    per_degree_lon = per_degree_lat * math.cos(math.radians(lat))
    half = length_km / 2
    dlon = half * math.sin(math.radians(strike)) / per_degree_lon
    dlat = half * math.cos(math.radians(strike)) / per_degree_lat
    ax.annotate(
        "",
        xy=(lon + dlon, lat + dlat),
        xytext=(lon - dlon, lat - dlat),
        arrowprops={
            "arrowstyle": "-|>",
            # The head is sized off the shaft rather than off the default
            # font size, so it stays in proportion to the trace it sits on
            # however the figure is scaled.
            "mutation_scale": display.mark(6 * TRACE_WIDTH),
            "color": colour,
            "linewidth": display.mark(TRACE_WIDTH),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def legend_handles(
    show_cfm: bool, show_jumps: bool, display: Display
) -> list[tuple[object, str]]:
    """Proxy artists for the map's fixed symbology, paired with their labels.

    Labels are kept short: enlarged for distance viewing, every extra character
    is several millimetres of legend.
    """
    handles: list[tuple[object, str]] = [
        (
            Patch(
                facecolor="#8a8a8a",
                alpha=PROJECTION_ALPHA,
                edgecolor="#8a8a8a",
                linestyle="--",
                linewidth=display.mark(CONTEXT_WIDTH),
            ),
            "Surface projection",
        ),
        (
            Line2D([], [], color="#404040", lw=display.mark(TRACE_WIDTH)),
            "Top edge (arrow = strike)",
        ),
    ]
    if show_jumps:
        handles.append(
            (
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="none",
                    mfc="white",
                    mec="#404040",
                    mew=display.mark(CONTEXT_WIDTH),
                    ms=display.mark(6),
                ),
                "Rupture jump",
            )
        )
    handles.append(
        (
            Line2D(
                [],
                [],
                marker="*",
                ls="none",
                mfc=HYPOCENTRE_COLOUR,
                mec="black",
                mew=display.mark(0.7),
                ms=display.mark(14),
            ),
            "Hypocentre",
        )
    )
    if show_cfm:
        handles.append(
            (
                Line2D([], [], color=CFM_GREY, lw=display.mark(CONTEXT_WIDTH)),
                "NZ CFM v1.0 trace",
            )
        )
    return handles


def rupture_map(
    realisation: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Realisation JSON file")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output image path (omit to show interactively)",
        ),
    ] = None,
    pad: Annotated[
        float, typer.Option(help="Map margin around the rupture, in degrees")
    ] = 0.18,
    cfm: Annotated[
        bool,
        typer.Option("--cfm/--no-cfm", help="Draw NZ CFM v1.0 traces underneath"),
    ] = True,
    jumps: Annotated[
        bool,
        typer.Option(
            "--jumps/--no-jumps", help="Mark where the rupture jumps between segments"
        ),
    ] = True,
    strike_arrows: Annotated[
        bool,
        typer.Option(
            "--strike-arrows/--no-strike-arrows",
            help="Draw a strike-direction arrow on each segment",
        ),
    ] = True,
    inset: Annotated[
        bool,
        typer.Option("--inset/--no-inset", help="Draw a New Zealand locator inset"),
    ] = True,
    scale_bar: Annotated[
        bool, typer.Option("--scale-bar/--no-scale-bar", help="Draw a distance scale")
    ] = True,
    coastline: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Coastline file to draw (.b64 blob or .geojson); "
            "defaults to the cached download",
        ),
    ] = None,
    title: Annotated[str | None, typer.Option(help="Override the map title")] = None,
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
    """Plot a realisation's faults, rupture order, and hypocentre in plan view."""
    faults, realisation_data = load_realisation(realisation)

    propagation = realisation_data["rupture_propagation"]
    causality_tree = propagation["rupture_causality_tree"]
    order = list(rupture_propagation.tree_nodes_in_order(causality_tree))
    # Any source missing from the tree (single-source realisations) still gets drawn.
    order += [name for name in faults if name not in order]
    magnitudes = realisation_data.get("magnitudes", {}).get("magnitudes", {})
    # Colour carries magnitude only when there is a spread of magnitudes to
    # carry; otherwise it falls back to the categorical palette, where it
    # carries nothing and says so.
    by_magnitude = worth_colouring(magnitudes, order)
    if by_magnitude:
        levels = magnitude_levels(magnitudes)
        cmap, norm = magnitude_ramp(levels)
        colours = magnitude_colours(magnitudes, cmap, norm)
    else:
        colours = {
            name: FAULT_COLOURS[i % len(FAULT_COLOURS)] for i, name in enumerate(order)
        }

    initial_fault = order[0]
    hypocentre = faults[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
        np.array([propagation["hypocentre"]["s"], propagation["hypocentre"]["d"]])
    )

    corners = np.vstack([fault.corners for fault in faults.values()])
    bounds = (
        corners[:, 1].min() - pad,
        corners[:, 0].min() - pad,
        corners[:, 1].max() + pad,
        corners[:, 0].max() + pad,
    )
    mid_lat = (bounds[1] + bounds[3]) / 2

    if title is None:
        title = realisation_data.get("metadata", {}).get("name", realisation.stem)
        if magnitudes:
            total = moment.moment_to_magnitude(
                sum(moment.magnitude_to_moment(m) for m in magnitudes.values())
            )
            title += f" | Mw {total:.2f}, {len(faults)} segment"
            title += "s" if len(faults) > 1 else ""

    # Square canvas, like the intensity measure maps: the region is padded to
    # whatever shape the rupture is, and the axes take the room they need
    # inside it.
    design = (9, 9)
    display = Display.for_figure(design, dpi, display_height, viewing_distance)
    display.report(design)
    fig, ax = plt.subplots(figsize=display.size, layout="constrained")

    coast = load_coastline(coastline)
    if coast is not None:
        fill_land(ax, coast, bounds, display)

    # Regional context: every CFM trace intersecting the map, drawn thin and pale.
    if cfm:
        traces = community_fault_model.community_fault_model_as_geodataframe().to_crs(
            "EPSG:4326"
        )
        window = shapely.box(*bounds)
        for _, fault_trace in traces[traces.intersects(window)].iterrows():
            line = np.array(fault_trace.trace.coords)
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=CFM_GREY,
                lw=display.mark(CONTEXT_WIDTH),
                zorder=2,
            )

    # Surface projections first, so the traces and labels sit on top of every fill.
    for name in order:
        for outline in surface_projection(faults[name]):
            ax.add_patch(
                Polygon(
                    outline,
                    closed=True,
                    facecolor=colours[name],
                    alpha=PROJECTION_ALPHA,
                    edgecolor=colours[name],
                    linestyle="--",
                    linewidth=display.mark(CONTEXT_WIDTH),
                    zorder=3,
                )
            )

    entries = []
    for index, name in enumerate(order):
        fault = faults[name]
        colour = colours[name]
        trace = trace_lon_lat(fault)
        ax.plot(
            trace[:, 0],
            trace[:, 1],
            color=colour,
            lw=display.mark(TRACE_WIDTH),
            solid_capstyle="round",
            zorder=5,
        )

        mid = shapely.LineString(trace).interpolate(0.5, normalized=True)
        if strike_arrows:
            draw_strike_arrow(
                ax,
                (mid.x, mid.y),
                fault.planes[len(fault.planes) // 2].strike,
                min(7.0, 0.35 * fault.length),
                colour,
                display,
            )

        # Just the rupture-order number: the names are what crowded the map,
        # and magnitude is in the colour now. One segment ruptures in no
        # order, so it goes unnumbered rather than carrying a "1" that says
        # nothing and lands on the hypocentre.
        if len(order) > 1:
            entries.append(
                {
                    "text": f"{index + 1}",
                    "x": mid.x,
                    "y": mid.y,
                    "colour": darken(colour),
                    "rank": 0,
                    "size": 9,
                    "weight": "bold",
                }
            )

    # Where the rupture steps from one segment to the next, at the surface.
    drew_jumps = False
    if jumps and propagation.get("jump_points"):
        for name, jump in propagation["jump_points"].items():
            if name not in faults:
                continue
            point = faults[name].fault_coordinates_to_wgs_depth_coordinates(
                np.array([jump["to_point"]["s"], jump["to_point"]["d"]])
            )
            ax.plot(
                point[1],
                point[0],
                marker="o",
                ls="none",
                ms=display.mark(6),
                mfc="white",
                mec="#404040",
                mew=display.mark(CONTEXT_WIDTH),
                zorder=6,
            )
            drew_jumps = True

    ax.plot(
        hypocentre[1],
        hypocentre[0],
        marker="*",
        ls="none",
        ms=display.mark(14),
        mfc=HYPOCENTRE_COLOUR,
        mec="black",
        mew=display.mark(0.7),
        zorder=7,
    )

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1 / np.cos(np.radians(mid_lat)))
    degrees = FuncFormatter(lambda v, _: f"{v:g}°")
    ax.xaxis.set_major_formatter(degrees)
    ax.yaxis.set_major_formatter(degrees)
    ax.tick_params(labelsize=9)
    if display.scale > 1.0:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=display.ticks(7)))
    for spine in ax.spines.values():
        spine.set_linewidth(display.mark(0.6))
    ax.set_title(title, fontsize=11)

    # The scale bar has the top right; the legend and the locator inset take
    # the emptiest of what is left, and are told about each other so they do
    # not both pick the same corner and stack one over the other.
    taken = [SCALE_BAR_CORNER] if scale_bar else []
    if scale_bar:
        draw_scale_bar(ax, display)

    drawn = np.vstack([trace_lon_lat(fault) for fault in faults.values()])
    if inset:
        inset_rect = free_corner(ax, drawn[:, 0], drawn[:, 1], size=0.23, taken=taken)
        taken = [*taken, inset_rect]

    if by_magnitude:
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation="horizontal",
            shrink=0.6,
            pad=0.04,
            aspect=display.mark(35),
        )
        colorbar.set_label("Segment M$_w$", fontsize=10)
        shown = display.keep(list(levels), max(3, display.ticks(len(levels))))
        colorbar.set_ticks(shown)
        colorbar.set_ticklabels([f"{level:g}" for level in shown], fontsize=8)

    handles = legend_handles(cfm, drew_jumps, display)
    loc, anchor = corner_anchor(
        free_corner(ax, drawn[:, 0], drawn[:, 1], size=0.30, taken=taken)
    )
    ax.legend(
        [handle for handle, _ in handles],
        [label for _, label in handles],
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
        borderpad=0.6,
        labelspacing=0.6,
        handlelength=1.8,
    ).set_zorder(9)

    if inset:
        # Drawn after the legend has claimed its corner, so the two never land
        # on top of each other -- a hidden locator is worse than none.
        draw_locator_map(ax.inset_axes(list(inset_rect)), coast, bounds, display)

    # Last, so the numbers are placed against the final axes size and dodge
    # each other the way station and basin labels do on the other maps.
    place_labels(fig, ax, entries)

    print(f"rupture order: {' -> '.join(order)}")
    print(
        f"hypocentre: {hypocentre[0]:.4f}, {hypocentre[1]:.4f} "
        f"at {hypocentre[2] / 1000:.2f} km depth (on {initial_fault})"
    )
    if coast is None:
        console_warn("no coastline available; the map is drawn without land")

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
