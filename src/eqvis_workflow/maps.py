"""``map``: an intensity measure in plan view.

Draws the interpolated field, optionally the log-difference of two runs, with
the recording stations over it coloured by their misfit against the
simulation::

    eqvis map intensity_measures.h5 PGA
    eqvis map intensity_measures.h5 pSA --period 1.0 -o psa_1s.png
    eqvis map emod3d/im.h5 PGV --diff sw4/im.h5
    eqvis map im.h5 PGA --observed 2026p530771_flatfiles.zip

Where a group of stations is too tight to label in place, it is drawn again in
a zoomed inset with a connector back to where it came from.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import (
    FuncFormatter,
    MaxNLocator,
)

from .console import console_warn
from .constants import DEFAULT_COMPONENT, HIDDEN, LOG_SCALED, UNIT_LABEL
from .data import (
    default_title,
    im_label,
    open_ims,
    print_info,
    residual_label,
    restrict_to_domain,
    run_names,
    run_title,
    select_im,
)
from .display import Display
from .flatfile import read_observed
from .geography import (
    basins_in_view,
    draw_basins,
    draw_coastline,
    draw_geometry,
    load_basins,
    load_coastline,
)
from .picks import named_mask, pick_states, read_pick_list, restrict_to_stations
from .raster import discrete_norm, fixed_symmetric_norm, rasterise, symmetric_norm
from .stations import (
    cluster_bounds,
    draw_observed,
    free_corner,
    place_labels,
    sample_simulation,
    station_clusters,
    station_labels,
)


def draw_detail_inset(
    fig: plt.Figure,
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    raster: tuple[np.ndarray, np.ndarray, np.ma.MaskedArray],
    colormap,
    norm: BoundaryNorm,
    observed: dict[str, np.ndarray],
    members: np.ndarray,
    residual: np.ndarray | None,
    station_cmap,
    station_norm: BoundaryNorm,
    coastline: shapely.MultiPolygon | None,
    basins: list[tuple[str, shapely.Geometry]] | None,
    domain: shapely.Geometry | None = None,
    taken: list[tuple[float, float, float, float]] | None = None,
    label: np.ndarray | bool = True,
) -> plt.Axes:
    """Zoomed panel over a cluster of stations, with the labels moved into it.

    Stations too close together to label on the main map get their names here
    instead, at a scale where they fit. Everything is redrawn with the same
    colormaps and norms as the main axes, so the panel is a magnifier rather
    than a second, differently-scaled plot. The magnification is worth having
    even with ``label=False`` (station naming turned off map-wide) -- it still
    separates a pile-up of triangles that overlap at the main map's scale.

    ``label`` is either one flag for the whole panel or a boolean mask over
    ``observed`` saying which stations to name. It has to be able to be a mask:
    a cluster is exactly where naming every member is too much, so naming one
    member must not drag its neighbours' names in with it.

    ``taken`` collects the axes-fraction rectangle chosen for each inset drawn
    so far on ``fig``, so that a caller placing several insets on one figure
    can pass the same list through every call and keep them from landing on
    top of each other.
    """
    x0, y0, x1, y1 = bounds
    rect = free_corner(
        ax,
        observed["lon"],
        observed["lat"],
        domain,
        target=((x0 + x1) / 2, (y0 + y1) / 2),
        taken=taken or (),
    )
    if taken is not None:
        taken.append(rect)
    axins = ax.inset_axes(rect)
    grid_lon, grid_lat, grid = raster
    axins.pcolormesh(
        grid_lon, grid_lat, grid, cmap=colormap, norm=norm, rasterized=True
    )
    if coastline is not None:
        draw_coastline(axins, coastline, bounds)
    if basins:
        # A tighter simplify tolerance: the inset resolves finer detail.
        draw_basins(axins, basins, shapely.box(*bounds), (x1 - x0) / 400)
    inside = np.zeros(len(observed["lon"]), dtype=bool)
    inside[members] = True
    draw_observed(
        axins,
        observed["lon"][inside],
        observed["lat"][inside],
        None if residual is None else residual[inside],
        station_cmap,
        station_norm,
    )
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
    ax.indicate_inset_zoom(axins, edgecolor="#333333", lw=0.7, alpha=0.9)
    naming = inside & (
        label if isinstance(label, np.ndarray) else np.full(inside.shape, bool(label))
    )
    if naming.any():
        place_labels(
            fig,
            axins,
            station_labels(
                observed["name"][naming],
                observed["lon"][naming],
                observed["lat"][naming],
                observed["value"][naming],
            ),
        )
    return axins


def map_im(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    im: Annotated[str, typer.Argument(help="Intensity measure to plot")] = "PGA",
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            exists=True,
            dir_okay=False,
            help="Second IM file: plot ln(IM_1) - ln(IM_2)",
        ),
    ] = None,
    name: Annotated[
        list[str] | None,
        typer.Option(
            "--name",
            help="Name for each simulation, in order; repeat to name the --diff run",
        ),
    ] = None,
    component: Annotated[
        str | None,
        typer.Option(help="Component of motion (default depends on the IM)"),
    ] = None,
    period: Annotated[
        float | None,
        typer.Option(help="pSA period in seconds (nearest available; default 1.0)"),
    ] = None,
    frequency: Annotated[
        float | None,
        typer.Option(help="FAS frequency in Hz (nearest available; default 1.0)"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output image path (omit to show interactively)"
        ),
    ] = None,
    levels: Annotated[int, typer.Option(help="Approximate number of colour bins")] = 10,
    cmap: Annotated[
        str | None, typer.Option(help="Matplotlib colormap name")
    ] = None,
    vmin: Annotated[float | None, typer.Option(help="Colour scale minimum")] = None,
    vmax: Annotated[float | None, typer.Option(help="Colour scale maximum")] = None,
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip: overlay in-domain recording stations, "
            "coloured by ln(observed / simulated)",
        ),
    ] = None,
    stations: Annotated[
        Path | None,
        typer.Option(
            "--stations",
            exists=True,
            dir_okay=False,
            help="Pick list from `pick`: what to draw and what to name",
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(
            "--title",
            help="Title for the plot, overriding both the pick list and the "
            "event and magnitude default; pass an empty string for none",
        ),
    ] = None,
    residual_limit: Annotated[
        float, typer.Option(help="Colour scale limit for the station residuals")
    ] = 0.7,
    info: Annotated[
        bool, typer.Option("--info", help="Describe the file contents and exit")
    ] = False,
    clip: Annotated[
        bool,
        typer.Option("--clip/--no-clip", help="Clip the raster to the NZ coastline"),
    ] = True,
    coastline: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Coastline file to clip against (.b64 blob or .geojson); "
            "defaults to the cached download",
        ),
    ] = None,
    basins: Annotated[
        bool,
        typer.Option(
            "--basins/--no-basins", help="Outline basins that intersect the domain"
        ),
    ] = True,
    label_stations: Annotated[
        bool,
        typer.Option("--label/--no-label", help="Name the observed stations"),
    ] = True,
    basin_file: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Basin outlines to draw (.parquet); defaults to the cached download",
        ),
    ] = None,
    inset: Annotated[
        bool,
        typer.Option(
            "--inset/--no-inset",
            help="Zoom clusters of stations too tight to label into a detail panel",
        ),
    ] = True,
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
):
    """Plot an intensity measure spatially, or the log-difference of two runs."""
    if info:
        print_info(im_file)
        raise typer.Exit()

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    names = run_names(name, [im_file] if diff is None else [im_file, diff])
    tree = open_ims(im_file)
    da, selection = select_im(tree, im, component, period, frequency)
    label = im_label(im, selection)
    for dim, value in selection.items():
        print(f"selected {dim}: {value:g}")

    if diff is not None:
        other, _ = select_im(open_ims(diff), im, component, period, frequency)
        da, other = xr.align(da, other, join="inner")
        if da.sizes["station"] == 0:
            raise typer.BadParameter(f"{im_file} and {diff} share no stations")
        values = np.log(da.values) - np.log(other.values)
        if vmin is not None or vmax is not None:
            # The diff scale is zero-centred, so only the magnitude survives.
            limit = max(abs(v) for v in (vmin, vmax) if v is not None)
            if vmin is not None and vmax is not None and abs(vmin) != abs(vmax):
                console_warn(f"diff scale is symmetric about zero; using +/-{limit:g}")
            boundaries = fixed_symmetric_norm(limit, levels)
        else:
            boundaries = symmetric_norm(values, levels)
        colormap = plt.get_cmap(cmap or "RdBu_r")
        colorbar_label = f"ln[{label}$_1$ / {label}$_2$]"
    else:
        values = da.values
        boundaries = discrete_norm(values, levels, im in LOG_SCALED, vmin, vmax)
        colormap = plt.get_cmap(cmap or "magma_r")
        units = UNIT_LABEL.get(da.attrs.get("units", ""), da.attrs.get("units", ""))
        colorbar_label = f"{label} ({units})" if units else label

    lon = da.longitude.values
    lat = da.latitude.values
    coast = load_coastline(coastline) if clip else None
    grid_lon, grid_lat, grid = rasterise(lon, lat, values, coast)
    norm = BoundaryNorm(boundaries, colormap.N, extend="both")

    display = Display.for_figure((9, 9), dpi, display_height, viewing_distance)
    display.report((9, 9))
    fig, ax = plt.subplots(figsize=display.size, layout="constrained")
    mesh = ax.pcolormesh(
        grid_lon, grid_lat, grid, cmap=colormap, norm=norm, rasterized=True
    )
    if coast is not None:
        draw_coastline(
            ax,
            coast,
            (grid_lon.min(), grid_lat.min(), grid_lon.max(), grid_lat.max()),
            display,
        )
    draw_geometry(ax, tree.attrs, display)

    picked = read_pick_list(stations) if stations is not None else None
    obs = residual = named = None
    if observed is not None:
        obs, resolved = read_observed(observed, im, component, selection)
        for dim, value in resolved.items():
            print(f"observed {dim}: {value:g}")
        obs = restrict_to_domain(obs, tree.attrs, grid_lon, grid_lat, observed)
        if picked is not None:
            obs = restrict_to_stations(obs, list(picked["stations"]), observed)
        named = named_mask(obs["name"], None if picked is None else picked["stations"])
        if np.isfinite(obs["value"]).any():
            simulated = sample_simulation(lon, lat, da.values, obs["lon"], obs["lat"])
            residual = np.log(simulated) - np.log(obs["value"])

    view = (grid_lon.min(), grid_lat.min(), grid_lon.max(), grid_lat.max())
    domain_shape = (
        shapely.from_wkt(tree.attrs["domain"]) if tree.attrs.get("domain") else None
    )
    basin_entries, outlines = [], None
    if basins:
        # Freeze the view first: basins run well past the domain, and drawing
        # one still grows the data limits even if it does not autoscale.
        ax.autoscale_view()
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        outlines = load_basins(basin_file)
        if outlines:
            outlines = basins_in_view(outlines, view)
            # Confine the outlines to the simulated area, and thin them to what
            # the page can actually resolve.
            clip = domain_shape if domain_shape is not None else shapely.box(*view)
            tolerance = (view[2] - view[0]) / (fig.get_figwidth() * dpi) * 2
            basin_states = pick_states(
                None if picked is None else picked["basins"] or None
            )
            if basin_states is not None:
                outlines = [
                    (name, geometry)
                    for name, geometry in outlines
                    if basin_states.get(name, HIDDEN) != HIDDEN
                ]
            basin_entries = draw_basins(
                ax,
                outlines,
                clip,
                tolerance,
                None if obs is None else (obs["lon"], obs["lat"]),
                states=basin_states,
                display=display,
            )

    stations = None
    # Same discrete levels as a simulation-vs-simulation diff, but pinned to
    # +/- residual_limit rather than derived from the data.
    residual_levels = fixed_symmetric_norm(residual_limit, levels)
    residual_cmap = plt.get_cmap("RdBu_r")
    residual_norm = BoundaryNorm(residual_levels, residual_cmap.N, extend="both")
    if obs is not None:
        stations = draw_observed(
            ax, obs["lon"], obs["lat"], residual, residual_cmap, residual_norm
        )

    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    degrees = FuncFormatter(lambda v, _: f"{v:g}°")
    ax.xaxis.set_major_formatter(degrees)
    ax.yaxis.set_major_formatter(degrees)
    ax.tick_params(labelsize=9)
    if display.scale > 1.0:
        # Enlarged text needs the room the default seven-odd ticks per axis
        # would take, and the degrees run into each other well before that.
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=display.ticks(7)))
    for spine in ax.spines.values():
        spine.set_linewidth(display.mark(0.6))

    if title is None and picked is not None:
        title = picked["title"]
    if title is None:
        title = default_title(tree.attrs, run_title(names, bool(name)))
    ax.set_title(title, fontsize=11)

    # Created first so constrained_layout stacks it below the raster's own bar.
    if stations is not None:
        residuals = fig.colorbar(
            stations,
            ax=ax,
            orientation="horizontal",
            shrink=0.6,
            pad=0.02,
            aspect=display.mark(35),
        )
        residuals.set_label(residual_label(label), fontsize=10)
        shown = display.keep(
            residual_levels, max(3, display.ticks(len(residual_levels)))
        )
        residuals.set_ticks(shown)
        residuals.set_ticklabels([f"{b:g}" for b in shown], fontsize=8)

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        shrink=0.6,
        pad=0.04,
        aspect=display.mark(35),
    )
    colorbar.set_label(colorbar_label, fontsize=10)
    shown = display.keep(boundaries, max(3, display.ticks(len(boundaries))))
    colorbar.set_ticks(shown)
    colorbar.set_ticklabels([f"{b:g}" for b in shown], fontsize=8)

    # Last, so labels are placed against the final axes size. Stations and
    # basins are laid out together, so they cannot land on top of each other.
    entries = list(basin_entries)
    if obs is not None:
        labelled = named.copy()
        # The zoom is worth having even with labelling off map-wide -- it still
        # separates a pile-up of overlapping triangles -- so it runs off
        # `inset` alone; only the text inside it follows `label_stations`.
        if inset:
            span = max(np.ptp(grid_lon), np.ptp(grid_lat))
            placed_insets: list[tuple[float, float, float, float]] = []
            for members in station_clusters(obs["lon"], obs["lat"], span):
                draw_detail_inset(
                    fig,
                    ax,
                    cluster_bounds(obs["lon"], obs["lat"], members),
                    (grid_lon, grid_lat, grid),
                    colormap,
                    norm,
                    obs,
                    members,
                    residual,
                    residual_cmap,
                    residual_norm,
                    coast,
                    outlines,
                    domain_shape,
                    taken=placed_insets,
                    label=labelled if label_stations else False,
                )
                # Their names live in the inset now; leaving them here too would
                # put the crowding straight back.
                labelled[members] = False
        if label_stations:
            entries = (
                station_labels(
                    obs["name"][labelled],
                    obs["lon"][labelled],
                    obs["lat"][labelled],
                    obs["value"][labelled],
                )
                + entries
            )
    place_labels(fig, ax, entries)

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
