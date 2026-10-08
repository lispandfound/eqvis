"""The simulation domain: what the solver computes on, and where::

    eqvis domain sw4/R1/realisation.json
    eqvis domain sw4/R1/realisation.json --stations stations.ll -o domain.png

Two panels, because an SW4 domain is two questions. In plan, how much ground the
run covers and how much of it is *usable* -- the requested domain is the grid's
interior, and SW4 pads a supergrid sponge around it inside which it solves a
damped, coordinate-stretched equation rather than the wave equation. In depth,
which grid a given depth is resolved on, since the mesh coarsens downward and
the fault should not be sitting in the coarse part of it.

Like the other maps it can be drawn for a size and a distance rather than for
the page; see :class:`~.display.Display`.
"""

import json
import re
from pathlib import Path as FilePath
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator
from qcore import coordinates

from .console import console_warn
from .display import Display
from .geography import (
    basins_in_view,
    draw_basins,
    draw_locator_map,
    draw_scale_bar,
    fill_land,
    load_basins,
    load_coastline,
)
from .rupture import (
    HYPOCENTRE_COLOUR,
    PROJECTION_ALPHA,
    TRACE_WIDTH,
    load_realisation,
    surface_projection,
    trace_lon_lat,
)
from .stations import corner_anchor, free_corner, place_labels

# The domain is the subject here, not the context it was in the rupture map, so
# it takes the heavy ink and the fault gives way. Blue dashed is what the GMT
# figure this replaces used for it, and it is worth keeping: a reader who has
# seen one of those knows immediately which rectangle this is.
DOMAIN_COLOUR = "#1f4fd8"
DOMAIN_WIDTH = 1.8
# The sponge is a region the run does not answer for, so it is drawn as an
# exclusion -- filled, not outlined -- in the warning colour the rest of the
# toolkit uses for a value that should not be read.
SPONGE_COLOUR = "#b03030"
SPONGE_ALPHA = 0.16
GRID_WIDTH = 0.7

SOURCE_COLOUR = "#d55e00"

# Green is a recording everywhere else in the toolkit, so a site inside the
# domain -- one this run will actually have something to say about -- takes it,
# and one outside is greyed rather than coloured.
STATION_OBSERVED = "#1b5e20"
STATION_OUTSIDE = "#9a9a9a"

OBSERVATION_STATION = re.compile(r"\w{3,4}")
"""Which station names belong to real recording sites.

A workflow station file is mostly a grid: tens of thousands of interpolation
points whose names are generated, against a few hundred instruments whose names
are their GeoNet codes. Nothing in the file distinguishes them, but the codes
are three or four word characters and the generated names are not, so the name
is the discriminator available. Only these are named on the map -- labelling
ninety thousand grid points is not a figure, and the placer would spend the
afternoon finding out.
"""

# The refinement bands, shallow to deep, off a light sequential ramp: finer
# grids darker, so the eye reads the good resolution as the solid end.
REFINEMENT_RAMP = "Blues"
REFINEMENT_RANGE = (0.55, 0.18)

# SW4's own default supergrid thickness in grid points (`sw4/src/EW.C`), used
# when the realisation's `supergrid` command sets neither `gp` nor `width`,
# because that is what SW4 itself would then use. Mirrors
# `workflow.sw4.SW4_DEFAULT_SUPERGRID_GRIDPOINTS`; eqvis reads realisations as
# plain JSON rather than depending on the workflow package (see
# `animation.load_domain`), so the constant is restated rather than imported.
SW4_DEFAULT_SUPERGRID_GRIDPOINTS = 30


def sponge_width_m(realisation: dict, coarsest_resolution_m: float) -> float:
    """The supergrid sponge width SW4 will use, in metres.

    ``width=`` wins over ``gp=`` where both appear, matching SW4's own
    precedence, and the grid-point form is measured on the coarsest grid
    because a single scalar sponge serves every grid and every face.
    """
    commands = realisation.get("sw4", {}).get("commands", [])
    parameters: dict = next(
        (c.get("parameters", {}) for c in commands if c.get("name") == "supergrid"), {}
    )
    if (width := parameters.get("width")) is not None:
        return float(width)
    gridpoints = parameters.get("gp") or SW4_DEFAULT_SUPERGRID_GRIDPOINTS
    return float(gridpoints) * coarsest_resolution_m


def refinements_for_depth(realisation: dict, depth_km: float) -> list[dict]:
    """The refinement stack resolved against a domain of ``depth_km``.

    Layers below the domain are dropped and the last is truncated to it; a stack
    that does not reach the bottom is extended at the unbounded resolution. The
    last layer keeps at least two cells, so a domain ending just past a boundary
    does not produce a degenerate grid. Mirrors
    `workflow.realisations.Refinements.refinements_for_depth`.
    """
    block = realisation.get("refinements", {})
    depth_m = depth_km * 1000.0
    layers: list[dict] = []
    for refinement in block.get("refinements", []):
        layers.append(
            {
                "resolution": float(refinement["resolution"]),
                "bottom": min(float(refinement["bottom"]), depth_m),
            }
        )
        if float(refinement["bottom"]) > depth_m:
            break
    else:
        layers.append(
            {
                "resolution": float(block["unbounded_refinement_resolution"]),
                "bottom": depth_m,
            }
        )
    if len(layers) >= 2:
        layers[-1]["bottom"] = max(
            layers[-2]["bottom"] + layers[-1]["resolution"] * 2, layers[-1]["bottom"]
        )
    return layers


def pad_rectangle(corners: np.ndarray, pad_m: float) -> np.ndarray:
    """``corners`` pushed out by ``pad_m`` along the rectangle's own axes.

    The domain is a rectangle in NZTM but a rotated one in lon/lat, so padding
    it means moving along its own edges rather than along north and east --
    which is what `BoundingBox.pad` does on the workflow side, and why the
    sponge ring stays a constant width all the way round rather than pinching at
    the corners.

    Parameters
    ----------
    corners : np.ndarray
        A (4, 2) array of NZTM (northing, easting) corners, in order around the
        rectangle. Only consistency matters -- the padding is measured along the
        rectangle's own edges, not along either axis.
    pad_m : float
        How far to push each face outward, in metres.

    Returns
    -------
    np.ndarray
        The padded (4, 2) corners, in the same order.
    """
    centre = corners.mean(axis=0)
    along = corners[1] - corners[0]
    across = corners[3] - corners[0]
    unit_along = along / np.linalg.norm(along)
    unit_across = across / np.linalg.norm(across)
    half_along = np.linalg.norm(along) / 2 + pad_m
    half_across = np.linalg.norm(across) / 2 + pad_m
    return np.array(
        [
            centre - half_along * unit_along - half_across * unit_across,
            centre + half_along * unit_along - half_across * unit_across,
            centre + half_along * unit_along + half_across * unit_across,
            centre - half_along * unit_along + half_across * unit_across,
        ]
    )


def nztm_to_lon_lat(corners: np.ndarray) -> np.ndarray:
    """(n, 2) NZTM northings and eastings as (n, 2) lon/lat.

    `qcore.coordinates` orders NZTM northing-first, matching the latitude-first
    order it returns, so the axes pass through untouched and only the pair is
    reversed on the way out -- matplotlib wants x before y.
    """
    padded = np.column_stack([corners[:, 0], corners[:, 1], np.zeros(len(corners))])
    return coordinates.nztm_to_wgs_depth(padded)[:, [1, 0]]


def polygon_path(geometry: shapely.Geometry) -> Path:
    """A matplotlib path for ``geometry``, holes included.

    Every ring of every part becomes one subpath, and matplotlib's even-odd
    fill rule then leaves the holes empty. Going through the exterior alone --
    which is what a `matplotlib.patches.Polygon` per part amounts to -- fills a
    ring solid.
    """
    rings = []
    for part in shapely.get_parts(geometry):
        if not isinstance(part, shapely.Polygon) or part.is_empty:
            continue
        for ring in (part.exterior, *part.interiors):
            rings.append(Path(np.asarray(ring.coords), closed=True))
    return Path.make_compound_path(*rings) if rings else Path(np.empty((0, 2)))


def read_stations(path: FilePath) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Longitudes, latitudes and names from a whitespace `lon lat name` file."""
    longitudes, latitudes, names = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        longitudes.append(float(fields[0]))
        latitudes.append(float(fields[1]))
        names.append(fields[2] if len(fields) > 2 else "")
    return np.array(longitudes), np.array(latitudes), names


def draw_depth_panel(
    ax: plt.Axes,
    layers: list[dict],
    depth_km: float,
    sponge_km: float,
    fault_depths_km: tuple[float, float] | None,
    display: Display,
) -> None:
    """The vertical grid structure: one band per refinement, coarsening down.

    The sponge is drawn below the domain rather than inside it because SW4 pads
    the grid downward by one sponge width, exactly as it pads the sides -- so
    the requested depth is the bottom of the *usable* model, not of the grid.
    """
    cmap = plt.get_cmap(REFINEMENT_RAMP)
    shades = np.linspace(*REFINEMENT_RANGE, max(len(layers), 1))
    top = 0.0
    for layer, shade in zip(layers, shades):
        bottom = layer["bottom"] / 1000.0
        ax.axhspan(top, bottom, facecolor=cmap(shade), edgecolor="none", zorder=1)
        ax.axhline(bottom, color="white", lw=display.mark(0.8), zorder=2)
        # Inside the band where it fits, just below it where it does not: a
        # 200 m layer on a 34 km column is a couple of millimetres tall.
        centre = (top + bottom) / 2
        ax.annotate(
            f"{layer['resolution']:g} m",
            (0.5, centre),
            ha="center",
            va="center",
            fontsize=7,
            color="black" if shade < 0.4 else "white",
            zorder=4,
        )
        top = bottom

    ax.axhspan(
        depth_km,
        depth_km + sponge_km,
        facecolor=SPONGE_COLOUR,
        alpha=SPONGE_ALPHA,
        edgecolor="none",
        zorder=1,
    )
    ax.axhline(depth_km, color=DOMAIN_COLOUR, lw=display.mark(DOMAIN_WIDTH), zorder=3)
    if sponge_km:
        ax.annotate(
            "sponge",
            (0.5, depth_km + sponge_km / 2),
            ha="center",
            va="center",
            fontsize=7,
            color=SPONGE_COLOUR,
            zorder=4,
        )

    if fault_depths_km is not None:
        ax.plot(
            [0.5, 0.5],
            list(fault_depths_km),
            color=SOURCE_COLOUR,
            lw=display.mark(TRACE_WIDTH * 1.4),
            solid_capstyle="butt",
            zorder=5,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(depth_km + sponge_km, 0)
    ax.set_xticks([])
    ax.set_ylabel("Depth (km)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=display.ticks(8)))
    for spine in ax.spines.values():
        spine.set_linewidth(display.mark(0.6))
    ax.set_title("Grid", fontsize=9)


def domain_map(
    realisation: Annotated[
        FilePath,
        typer.Argument(exists=True, dir_okay=False, help="Realisation JSON file"),
    ],
    output: Annotated[
        FilePath | None,
        typer.Option(
            "--output", "-o", help="Output image path (omit to show interactively)"
        ),
    ] = None,
    stations: Annotated[
        FilePath | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Station list (`lon lat name`) to overlay, split by whether "
            "each station falls inside the domain",
        ),
    ] = None,
    label_stations: Annotated[
        bool,
        typer.Option(
            "--label-stations/--no-label-stations",
            help="Name the observation stations inside the domain. Only real "
            "recording sites are named, not the interpolation grid",
        ),
    ] = False,
    pad: Annotated[
        float, typer.Option(help="Map margin around the domain, in degrees")
    ] = 0.25,
    sources: Annotated[
        bool,
        typer.Option("--sources/--no-sources", help="Draw the realisation's faults"),
    ] = True,
    basins: Annotated[
        bool,
        typer.Option("--basins/--no-basins", help="Draw the NZCVM basin outlines"),
    ] = True,
    basin_file: Annotated[
        FilePath | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Basin outlines (GeoParquet); defaults to the cached download",
        ),
    ] = None,
    inset: Annotated[
        bool,
        typer.Option("--inset/--no-inset", help="Draw a New Zealand locator inset"),
    ] = True,
    scale_bar: Annotated[
        bool, typer.Option("--scale-bar/--no-scale-bar", help="Draw a distance scale")
    ] = True,
    coastline: Annotated[
        FilePath | None,
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
    """Plot a realisation's simulation domain, its SW4 sponge and its grid."""
    with open(realisation) as f:
        realisation_data = json.load(f)

    block = realisation_data.get("domain")
    if not block:
        raise typer.BadParameter(
            f"{realisation} has no domain: run `generate-domain` on it first",
            param_hint="REALISATION",
        )
    depth_km = float(block["depth"])
    duration_s = block.get("duration")

    corners_wgs = np.array([[c["latitude"], c["longitude"]] for c in block["domain"]])
    corners_nztm = coordinates.wgs_depth_to_nztm(
        np.column_stack([corners_wgs, np.zeros(len(corners_wgs))])
    )[:, :2]

    layers = refinements_for_depth(realisation_data, depth_km)
    coarsest = max(layer["resolution"] for layer in layers) if layers else 0.0
    sponge_m = sponge_width_m(realisation_data, coarsest) if coarsest else 0.0

    # NZTM is (northing, easting); the rectangle helpers want a consistent pair
    # and it only has to round-trip, so it is carried as given.
    grid_nztm = pad_rectangle(corners_nztm, sponge_m)
    domain_lon_lat = nztm_to_lon_lat(corners_nztm)
    grid_lon_lat = nztm_to_lon_lat(grid_nztm)

    extent_along = np.linalg.norm(corners_nztm[1] - corners_nztm[0]) / 1000.0
    extent_across = np.linalg.norm(corners_nztm[3] - corners_nztm[0]) / 1000.0

    bounds = (
        grid_lon_lat[:, 0].min() - pad,
        grid_lon_lat[:, 1].min() - pad,
        grid_lon_lat[:, 0].max() + pad,
        grid_lon_lat[:, 1].max() + pad,
    )
    mid_lat = (bounds[1] + bounds[3]) / 2

    faults, _ = load_realisation(realisation) if sources else ({}, None)

    if title is None:
        name = realisation_data.get("metadata", {}).get("name", realisation.stem)
        title = f"{name} | {extent_along:.0f} x {extent_across:.0f} x {depth_km:.0f} km"

    design = (11, 8)
    display = Display.for_figure(design, dpi, display_height, viewing_distance)
    display.report(design)
    fig, (ax, depth_ax) = plt.subplots(
        1,
        2,
        figsize=display.size,
        layout="constrained",
        width_ratios=[1, 0.09],
    )

    coast = load_coastline(coastline)
    if coast is not None:
        fill_land(ax, coast, bounds, display)

    # The basins the velocity model carries, so a reader can see which of them
    # the domain actually took in -- a domain that clips a basin is a domain
    # that will be asked about the half it does not have. Under the domain
    # furniture: they are context, not the subject.
    basin_entries = []
    if basins:
        outlines = load_basins(basin_file)
        if outlines:
            basin_entries = draw_basins(
                ax,
                basins_in_view(outlines, bounds),
                shapely.box(*bounds),
                (bounds[2] - bounds[0]) / 400,
                display=display,
            )

    # The sponge first and as a ring, so the fill says "not here" over the
    # margin without tinting the interior the run actually answers for. It has
    # to go down as a compound path rather than a `Polygon` per part: the ring
    # is a polygon *with a hole*, and a matplotlib Polygon built from an
    # exterior alone fills the hole back in -- which would shade the whole
    # domain as unusable and say the exact opposite of what the figure is for.
    sponge_ring = shapely.difference(
        shapely.Polygon(grid_lon_lat), shapely.Polygon(domain_lon_lat)
    )
    ax.add_patch(
        PathPatch(
            polygon_path(sponge_ring),
            facecolor=SPONGE_COLOUR,
            alpha=SPONGE_ALPHA,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Polygon(
            grid_lon_lat,
            closed=True,
            fill=False,
            edgecolor=SPONGE_COLOUR,
            linewidth=display.mark(GRID_WIDTH),
            zorder=3,
        )
    )
    ax.add_patch(
        Polygon(
            domain_lon_lat,
            closed=True,
            fill=False,
            edgecolor=DOMAIN_COLOUR,
            linestyle="--",
            linewidth=display.mark(DOMAIN_WIDTH),
            zorder=4,
        )
    )

    entries = []
    inside_count = 0
    observed_count = 0
    if stations is not None:
        longitudes, latitudes, names = read_stations(stations)
        interior = shapely.Polygon(domain_lon_lat)
        inside = shapely.contains_xy(interior, longitudes, latitudes)
        inside_count = int(inside.sum())
        observed = np.array([bool(OBSERVATION_STATION.fullmatch(n)) for n in names])
        observed_count = int((observed & inside).sum())
        # Only the recording sites are drawn. The rest of the file is the
        # interpolation grid -- tens of thousands of points that carpet the
        # domain and bury the coastline, the source and each other, and whose
        # individual positions say nothing a reader can act on. Their count is
        # reported instead.
        for mask, colour, size, zorder in (
            (observed & ~inside, STATION_OUTSIDE, 3.0, 4),
            (observed & inside, STATION_OBSERVED, 5.0, 6),
        ):
            ax.plot(
                longitudes[mask],
                latitudes[mask],
                marker="^",
                ls="none",
                ms=display.mark(size),
                mfc=colour,
                mec="none",
                zorder=zorder,
            )
        if label_stations:
            entries += [
                {
                    "text": name,
                    "x": lon,
                    "y": lat,
                    "colour": STATION_OBSERVED,
                    "rank": 1,
                    "size": 6,
                }
                for name, lon, lat, keep in zip(
                    names, longitudes, latitudes, inside & observed
                )
                if keep
            ]

    for name, fault in faults.items():
        for outline in surface_projection(fault):
            ax.add_patch(
                Polygon(
                    outline,
                    closed=True,
                    facecolor=SOURCE_COLOUR,
                    alpha=PROJECTION_ALPHA,
                    edgecolor=SOURCE_COLOUR,
                    linewidth=display.mark(GRID_WIDTH),
                    zorder=6,
                )
            )
        trace = trace_lon_lat(fault)
        ax.plot(
            trace[:, 0],
            trace[:, 1],
            color=SOURCE_COLOUR,
            lw=display.mark(TRACE_WIDTH),
            solid_capstyle="round",
            zorder=7,
        )

    propagation = realisation_data.get("rupture_propagation", {})
    hypocentre = None
    if faults and "hypocentre" in propagation:
        first = next(iter(faults))
        hypocentre = faults[first].fault_coordinates_to_wgs_depth_coordinates(
            np.array([propagation["hypocentre"]["s"], propagation["hypocentre"]["d"]])
        )
        ax.plot(
            hypocentre[1],
            hypocentre[0],
            marker="*",
            ls="none",
            ms=display.mark(11),
            mfc=HYPOCENTRE_COLOUR,
            mec="black",
            mew=display.mark(0.6),
            zorder=8,
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

    taken = []
    if scale_bar:
        draw_scale_bar(ax, display)
        taken.append((0.66, 0.86, 0.34, 0.14))

    if inset:
        inset_rect = free_corner(
            ax, domain_lon_lat[:, 0], domain_lon_lat[:, 1], size=0.2, taken=taken
        )
        taken = [*taken, inset_rect]

    handles = [
        (
            Line2D(
                [],
                [],
                color=DOMAIN_COLOUR,
                ls="--",
                lw=display.mark(DOMAIN_WIDTH),
            ),
            "Domain (SW4 interior)",
        ),
        (
            Patch(
                facecolor=SPONGE_COLOUR,
                alpha=SPONGE_ALPHA,
                edgecolor=SPONGE_COLOUR,
                linewidth=display.mark(GRID_WIDTH),
            ),
            f"Supergrid sponge ({sponge_m / 1000:g} km)",
        ),
    ]
    if faults:
        handles.append(
            (
                Line2D([], [], color=SOURCE_COLOUR, lw=display.mark(TRACE_WIDTH)),
                "Source",
            )
        )
    if hypocentre is not None:
        handles.append(
            (
                Line2D(
                    [],
                    [],
                    marker="*",
                    ls="none",
                    mfc=HYPOCENTRE_COLOUR,
                    mec="black",
                    mew=display.mark(0.6),
                    ms=display.mark(11),
                ),
                "Hypocentre",
            )
        )
    if stations is not None:
        handles.append(
            (
                Line2D(
                    [],
                    [],
                    marker="^",
                    ls="none",
                    mfc=STATION_OBSERVED,
                    mec="none",
                    ms=display.mark(7),
                ),
                f"Observation sites ({observed_count})",
            )
        )
    loc, anchor = corner_anchor(
        free_corner(
            ax, domain_lon_lat[:, 0], domain_lon_lat[:, 1], size=0.3, taken=taken
        )
    )
    legend = ax.legend(
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
    )
    legend.set_zorder(9)

    locator = None
    if inset:
        locator = ax.inset_axes(list(inset_rect))
        draw_locator_map(locator, coast, bounds, display)

    fault_depths = None
    if faults:
        corners = np.vstack([fault.corners for fault in faults.values()])
        fault_depths = (corners[:, 2].min() / 1000, corners[:, 2].max() / 1000)
    draw_depth_panel(
        depth_ax, layers, depth_km, sponge_m / 1000.0, fault_depths, display
    )

    place_labels(
        fig,
        ax,
        entries + basin_entries,
        avoid=[a for a in (legend, locator) if a is not None],
    )

    print(
        f"domain {extent_along:.1f} x {extent_across:.1f} km, {depth_km:g} km deep"
        + (f", {duration_s:.1f} s" if duration_s else "")
    )
    print(
        f"grid {coarsest:g} m coarsest, sponge {sponge_m / 1000:g} km "
        f"({sponge_m / coarsest:.0f} gridpoints)"
        if coarsest
        else "no refinements: grid and sponge unknown"
    )
    if fault_depths is not None:
        print(f"source {fault_depths[0]:.2f} to {fault_depths[1]:.2f} km deep")
        # The sponge is padded outside the domain, so a source properly inside
        # the domain is clear of it. Saying so is the point of the figure, so it
        # is also said in words -- and in NZTM, where a metre is a metre on both
        # axes. Measured in degrees the north-south and east-west clearances
        # would be a factor of `cos(latitude)` apart, and no single conversion
        # fixes both.
        source_nztm = shapely.force_2d(
            shapely.union_all([fault.geometry for fault in faults.values()])
        )
        domain_nztm = shapely.Polygon(corners_nztm)
        if not shapely.contains_properly(domain_nztm, source_nztm):
            console_warn(
                "the source is not properly inside the domain, so part of it "
                "sits in the supergrid sponge where SW4 does not solve the "
                "wave equation"
            )
        else:
            clearance = shapely.distance(domain_nztm.exterior, source_nztm) / 1000.0
            print(f"source clears the domain edge by {clearance:.0f} km")
    if stations is not None:
        print(
            f"{inside_count} of {len(longitudes)} stations inside the domain, "
            f"{observed_count} of them observation sites"
        )
    if coast is None:
        console_warn("no coastline available; the map is drawn without land")

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
