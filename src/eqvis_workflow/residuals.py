"""``residual-map``: the residual against the recordings, in plan view.

Where :mod:`~.compare` and :mod:`~.heatmap` answer "how big" and "against what",
this answers "where". One panel per cell of a run dimension, the recording
stations drawn as triangles coloured by their log residual over the coastline
and the basin outlines::

    eqvis residual-map ims.duckdb --label event=2016p661400 --group-by solver
    eqvis residual-map ims.duckdb --label event=… --group-by solver \\
        --baseline solver=emod3d --period 1.0

With ``--baseline`` each panel shows the *difference* from a chosen cell station
by station, which is the map counterpart of the paired-difference panel in
``compare``: the recording cancels, so what is left is one simulation against
another at the same site.

The command refuses to put several events in one figure unless asked. A map has
one domain, one hypocentre and one extent, so panels across events are
geographically incoherent -- and a reader comparing them would be comparing
different pieces of the country, not different configurations.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from . import pairwise, store
from .compare import resolve_baseline
from .console import console_warn
from .constants import DEFAULT_COMPONENT
from .data import default_title, im_label, residual_label, restrict_to_domain
from .display import Display
from .geography import (
    basins_in_view,
    draw_basins,
    draw_coastline,
    draw_geometry,
    fill_land,
    load_basins,
    load_coastline,
)
from .picks import read_pick_list, restrict_to_stations
from .raster import fixed_symmetric_norm
from .stations import draw_observed, place_labels, station_labels

# Margin added around the recording stations, as a fraction of their span, so
# the outermost triangle is not clipped by the axes.
PADDING = 0.08


def panel_bounds(
    observed: dict[str, np.ndarray], attrs: dict
) -> tuple[float, float, float, float]:
    """The view: the recording stations, padded, but never wider than the domain.

    Bounded by the domain because a single distant recording would otherwise
    zoom the map out until the interesting part was a few pixels across.
    """
    lon, lat = observed["lon"], observed["lat"]
    span = max(np.ptp(lon), np.ptp(lat), 0.2)
    box = (
        lon.min() - PADDING * span, lat.min() - PADDING * span,
        lon.max() + PADDING * span, lat.max() + PADDING * span,
    )
    if "domain" in attrs:
        try:
            domain = shapely.from_wkt(attrs["domain"]).bounds
        except shapely.errors.GEOSException:
            return box
        return (
            max(box[0], domain[0] - 0.1), max(box[1], domain[1] - 0.1),
            min(box[2], domain[2] + 0.1), min(box[3], domain[3] + 0.1),
        )
    return box


def check_one_event(events: list[str], across_events: bool, by: list[str]) -> None:
    """Refuse a figure whose panels would not be comparable.

    A map has one domain, one hypocentre and one extent, so panels drawn for
    different earthquakes show different pieces of the country. A reader
    comparing them across a figure would be comparing geography, not
    configurations -- so it takes ``--across-events`` to say that is wanted.
    """
    if len(events) <= 1:
        return
    if not across_events and store.EVENT not in by:
        raise typer.BadParameter(
            f"the selection spans {len(events)} events ({', '.join(events[:4])}"
            f"{' ...' if len(events) > 4 else ''}), and a map has one domain and "
            "one extent. Restrict with --label event=..., or pass "
            "--across-events if incomparable panels are what you want"
        )
    console_warn(
        f"{len(events)} events in one figure: the panels have different extents "
        "and hypocentres, so they are not comparable"
    )


def draw_pair_matrix(
    ax, con, found, by, cells, period, im, component, map_norm, cmap
) -> None:
    """The pairwise matrix that goes above the map.

    The diagonal shares the map's colour scale, because it is the same quantity
    averaged: each configuration's mean residual against the recordings. Below
    the diagonal is ``mean ln(column / row)``, which needs a scale of its own --
    it is typically several times smaller, which is exactly why it is invisible
    when two configurations are drawn as two maps.
    """
    keys = [key for key, _ in cells]
    names = [pairwise.compact_label(key) for key in keys]
    data = store.read_residuals(con, found, im, component)
    if "ordinate" not in data or period is None:
        console_warn("no period axis, so no pairwise matrix")
        return
    bias, difference, resolved = pairwise.cell_statistics(data, by, keys, period)
    reach = float(np.nanmax(np.abs(difference))) if np.isfinite(difference).any() else 1.0
    pair_norm = TwoSlopeNorm(0.0, -reach, reach)
    pairwise.draw_matrix(
        ax, names, bias, difference, resolved, map_norm, pair_norm, cmap,
        f"{'/'.join(by)} at {period:g} s", show_names=True,
    )
    ax.set_xlabel(
        f"below diagonal: mean ln(column / row), scaled to ±{reach:.2f}"
        f"   ({pairwise.RESOLVED} = resolved at 95%)",
        fontsize=7,
    )


def residual_map(
    db: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Composite intensity measure database"
        ),
    ],
    im: Annotated[str, typer.Option("--im", help="Intensity measure")] = "pSA",
    component: Annotated[
        str | None, typer.Option("--component", help="Component of motion")
    ] = None,
    period: Annotated[
        float | None, typer.Option("--period", help="Period for a spectral measure")
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option("--label", help="Restrict to runs matching name=value; repeat"),
    ] = None,
    group_by: Annotated[
        list[str] | None,
        typer.Option("--group-by", help="Run dimension whose values become panels"),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            help="name=value picking the cell every panel is differenced against, "
            "station by station. The recording cancels, so the difference is "
            "between two simulations",
        ),
    ] = None,
    residual_limit: Annotated[
        float,
        typer.Option("--residual-limit", help="Colour scale limit, symmetric about zero"),
    ] = 0.7,
    basins: Annotated[
        bool, typer.Option("--basins/--no-basins", help="Draw the basin outlines")
    ] = True,
    basin_file: Annotated[
        Path | None,
        typer.Option("--basin-file", exists=True, dir_okay=False, help="Basin outlines"),
    ] = None,
    coastline: Annotated[
        Path | None,
        typer.Option("--coastline", exists=True, dir_okay=False, help="Coastline blob"),
    ] = None,
    stations: Annotated[
        Path | None,
        typer.Option(
            "--stations", exists=True, dir_okay=False,
            help="Pick list restricting which stations and basins are drawn",
        ),
    ] = None,
    name_stations: Annotated[
        bool, typer.Option("--label-stations/--no-label-stations", help="Name stations")
    ] = False,
    matrix: Annotated[
        bool,
        typer.Option(
            "--matrix/--panels",
            help="Draw one map with a pairwise matrix above it instead of one "
            "map per cell. Use this whenever the cells barely differ: several "
            "maps of nearly the same field are indistinguishable by eye, and "
            "the matrix resolves the differences the maps cannot",
        ),
    ] = False,
    across_events: Annotated[
        bool,
        typer.Option(
            "--across-events/--one-event",
            help="Allow panels spanning several events. Off by default: the "
            "panels then have different extents and are not comparable",
        ),
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead of showing")
    ] = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution")] = 300,
    display_height: Annotated[
        float | None, typer.Option("--display-height", help="Height in cm")
    ] = None,
    viewing_distance: Annotated[
        float | None, typer.Option("--viewing-distance", help="Metres")
    ] = None,
) -> None:
    """Draw the residual against the recordings in plan view, one panel per cell."""
    con = store.connect(db)
    labels = [store.parse_label(text) for text in (label or [])]
    found = store.select_runs(con, labels)

    events = sorted({run["event"] for run in found})
    by = list(group_by or store.varying(found))
    by = [name for name in by if name != store.EVENT] or by
    check_one_event(events, across_events, by)

    cells = store.group_runs(con, found, by)
    component = component or DEFAULT_COMPONENT.get(im, "geom")

    panels = []
    for key, cell_runs in cells:
        run = cell_runs[0]
        if len(cell_runs) > 1:
            console_warn(
                f"cell {store.cell_label(by, key)} has {len(cell_runs)} runs; "
                f"drawing {run['run_key']}. Add --group-by to separate them"
            )
        array, selection = store.select_im(
            con, run["run_key"], im, component, period=period
        )
        observed, _ = store.read_observed(
            con, run["run_key"], im, component, selection
        )
        attrs = store.run_attrs(con, run["run_key"])
        observed = restrict_to_domain(observed, attrs, array.longitude.values,
                                      array.latitude.values, f"{db}:{run['run_key']}")
        if stations is not None:
            picked = read_pick_list(stations)
            observed = restrict_to_stations(observed, picked, stations)
        simulated = np.full(observed["value"].shape, np.nan)
        index = {str(s): n for n, s in enumerate(array.station.values)}
        for position, station in enumerate(observed["name"]):
            found_at = index.get(str(station))
            if found_at is not None:
                simulated[position] = float(array.values[found_at])
        with np.errstate(invalid="ignore", divide="ignore"):
            residual = np.log(simulated) - np.log(observed["value"])
        panels.append(
            {
                "name": store.cell_label(by, key),
                "observed": observed,
                "residual": residual,
                "attrs": attrs,
                "selection": selection,
            }
        )

    reference = resolve_baseline(
        cells, by, [store.parse_label(t) for t in (baseline or [])]
    )
    if baseline:
        base = panels[reference]
        keyed = {str(s): v for s, v in zip(base["observed"]["name"], base["residual"])}
        drawn = []
        for position, panel in enumerate(panels):
            if position == reference:
                continue
            panel["residual"] = np.array(
                [
                    panel["residual"][n] - keyed.get(str(station), np.nan)
                    for n, station in enumerate(panel["observed"]["name"])
                ]
            )
            panel["name"] = f"{panel['name']} − {base['name']}"
            drawn.append(panel)
        panels = drawn

    # Several maps of nearly the same field cannot be told apart, so the matrix
    # layout draws one of them and puts the differences in numbers above it.
    if matrix:
        # Wider than the map alone needs, because the matrix's row labels sit
        # outside the axes and a clipped label is worse than a wide figure.
        display = Display.for_figure((6.6, 9.0), dpi, display_height, viewing_distance)
        fig, axes = plt.subplots(
            2, 1, figsize=display.size, dpi=display.dpi,
            height_ratios=[1.0, 1.55], layout="constrained",
        )
        matrix_ax, drawn_axes = axes[0], [axes[1]]
        shown = panels[:1]
    else:
        matrix_ax = None
        display = Display.for_figure(
            (3.6 * len(panels), 4.2), dpi, display_height, viewing_distance
        )
        fig, axes = plt.subplots(
            1, len(panels), figsize=display.size, dpi=display.dpi,
            layout="constrained", squeeze=False,
        )
        drawn_axes = list(axes[0])
        shown = panels

    coast = load_coastline(coastline)
    outlines = load_basins(basin_file) if basins else None
    norm = BoundaryNorm(
        fixed_symmetric_norm(residual_limit, 12), 256, extend="both"
    )
    colormap = plt.get_cmap("RdBu_r")
    mappable = None

    for ax, panel in zip(drawn_axes, shown):
        bounds = panel_bounds(panel["observed"], panel["attrs"])
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean(bounds[1::2]))))
        if coast is not None:
            fill_land(ax, coast, bounds, display)
            draw_coastline(ax, coast, bounds, display)
        rows = []
        if outlines is not None:
            # Clipped to the view, and simplified at a fraction of its width:
            # the same treatment `map` gives them, so a basin outline reads the
            # same weight whichever command drew it.
            rows = draw_basins(
                ax,
                basins_in_view(outlines, bounds),
                shapely.box(*bounds),
                (bounds[2] - bounds[0]) / 400,
                stations=(panel["observed"]["lon"], panel["observed"]["lat"]),
                display=display,
            )
        draw_geometry(ax, panel["attrs"], display)
        mappable = draw_observed(
            ax, panel["observed"]["lon"], panel["observed"]["lat"],
            panel["residual"], colormap, norm,
        ) or mappable
        # In matrix mode the title carries the row number too, so the reader can
        # tie the map to its row above without re-reading the labels.
        heading = (
            f"1  {pairwise.compact_label(cells[0][0])} — the other three differ "
            "as above"
            if matrix
            else panel["name"]
        )
        ax.set_title(heading, fontsize=10)
        ax.tick_params(labelsize=7)
        entries = list(rows)
        if name_stations:
            entries += station_labels(
                panel["observed"]["name"],
                panel["observed"]["lon"],
                panel["observed"]["lat"],
            )
        if entries and display.detailed:
            place_labels(fig, ax, entries)

    if matrix_ax is not None:
        draw_pair_matrix(
            matrix_ax, con, found, by, cells,
            panels[0]["selection"].get("period"), im, component, norm, colormap,
        )
    if mappable is not None:
        bar = fig.colorbar(
            mappable, ax=drawn_axes, ticks=display.keep(
                fixed_symmetric_norm(residual_limit, 12), 7
            ), pad=0.02, shrink=0.9,
        )
        bar.set_label(
            residual_label(im) + (" difference" if baseline else ""), fontsize=9
        )
        bar.ax.tick_params(labelsize=8)
    if display.detailed and matrix_ax is None:
        heading = default_title(panels[0]["attrs"], "")
        measure = im_label(im, panels[0]["selection"])
        fig.suptitle(f"{heading} — {measure}".strip(" —"), fontsize=11)
    con.close()

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
