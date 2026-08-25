"""``compare``: the log residual against recordings, swept across the spectrum.

The database-backed generalisation of :mod:`~.bias`. Where ``bias`` draws one or
two runs read from files, this draws the cells of whatever run dimension the
caller names -- and it takes that dimension from the database rather than from a
constant, so a tree ingested along different axes compares along *its* axes with
no change here::

    eqvis compare ims.duckdb --group-by solver --baseline solver=emod3d
    eqvis compare ims.duckdb --group-by mesh --baseline mesh=coarse

The panel below the curves is the point. The cells' own intervals overlap almost
everywhere, because most of that width is site scatter common to all of them;
only differencing them cell by cell against a baseline cancels it. And because
every configuration of an event is scored at the same stations, that pairing is
exact here in a way it cannot be for two separate files: the recording cancels
identically, so the difference is between two simulations and nothing else.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from . import store
from .bias import bias_statistics, draw_bias_curve, draw_difference_panel
from .console import console_warn
from .constants import DEFAULT_COMPONENT, SIM_ONE_BLACK, SIM_TWO_ORANGE
from .data import residual_label
from .display import Display

# Cell colours. Two cells read like a `bias` figure -- the same black and orange
# those two commands already use for a run and its comparison -- and beyond two
# the cells come off a discrete ramp, the way `rupture-map` colours its sources.
# Sorted cell order drives the choice, so a re-run reproduces the figure.
CELL_RAMP = "viridis"


def cell_colours(count: int) -> list[str]:
    """One colour per cell, in sorted cell order."""
    if count <= 2:
        return [SIM_ONE_BLACK, SIM_TWO_ORANGE][:count]
    ramp = plt.get_cmap(CELL_RAMP)
    return [ramp(position) for position in np.linspace(0.05, 0.9, count)]


def resolve_baseline(
    cells: list[tuple[tuple[str, ...], list[dict]]],
    by: list[str],
    baseline: list[tuple[str, str]],
) -> int:
    """Which cell everything else is differenced against.

    Defaults to the first in sorted order, so the figure is reproducible without
    the caller having to say. A baseline naming a dimension that is not one of
    the grouping dimensions cannot pick a cell out, and says so.
    """
    if not baseline:
        return 0
    wanted = dict(baseline)
    stray = [name for name in wanted if name not in by]
    if stray:
        raise typer.BadParameter(
            f"--baseline names {', '.join(stray)}, which is not among the "
            f"grouping dimensions ({', '.join(by)}). A baseline has to pick out "
            "one of the cells being compared; to restrict the run set instead, "
            "use --label"
        )
    for position, (key, _) in enumerate(cells):
        if all(value == wanted[name] for name, value in zip(by, key) if name in wanted):
            return position
    raise typer.BadParameter(
        f"--baseline matches none of the cells: "
        f"{', '.join(store.cell_label(by, key) for key, _ in cells)}"
    )


def compare(
    db: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Composite intensity measure database"
        ),
    ],
    im: Annotated[
        str,
        typer.Option("--im", help="Intensity measure to sweep (pSA) or score (PGA, PGV)"),
    ] = "pSA",
    component: Annotated[
        str | None,
        typer.Option("--component", help="Component of motion; defaults per measure"),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            help="Restrict to runs matching name=value; repeat. A name repeated "
            "keeps several of its values, two different names must both hold. "
            "Run `eqvis runs DB` to see the dimensions this database has",
        ),
    ] = None,
    group_by: Annotated[
        list[str] | None,
        typer.Option(
            "--group-by",
            help="Run dimension whose values become the series; repeat to cross "
            "two. Defaults to every dimension that takes more than one value",
        ),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            help="name=value picking the cell every other cell is differenced "
            "against, station by station. Defaults to the first cell in sorted "
            "order",
        ),
    ] = None,
    balanced: Annotated[
        bool,
        typer.Option(
            "--balanced/--unbalanced",
            help="Score every cell over only the (event, station) pairs all of "
            "them have. On by default: the factorial is often ragged, and a cell "
            "scored over pairs another lacks differs from it partly by those "
            "pairs' own bias rather than by the thing being compared",
        ),
    ] = True,
    interval: Annotated[
        float, typer.Option("--interval", help="Confidence level for the bands")
    ] = 0.95,
    minimum: Annotated[
        int,
        typer.Option(
            "--minimum", help="Drop ordinates scored by fewer than this many pairs"
        ),
    ] = 3,
    period_min: Annotated[
        float | None, typer.Option("--period-min", help="Shortest ordinate to draw")
    ] = None,
    period_max: Annotated[
        float | None,
        typer.Option(
            "--period-max",
            help="Longest ordinate to draw. The honest replacement for a usable "
            "band: the database holds no filter corners, so the band is the "
            "caller's to state rather than the record's to imply",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead of showing")
    ] = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution of the saved figure")] = 300,
    display_height: Annotated[
        float | None,
        typer.Option("--display-height", help="Height in cm the figure will be shown at"),
    ] = None,
    viewing_distance: Annotated[
        float | None,
        typer.Option("--viewing-distance", help="Metres the figure will be read from"),
    ] = None,
) -> None:
    """Sweep the residual against the recordings across the spectrum, by cell."""
    con = store.connect(db)
    labels = [store.parse_label(text) for text in (label or [])]
    found = store.select_runs(con, labels)

    by = list(group_by or store.varying(found))
    by = [name for name in by if name != store.EVENT] or by
    if not by:
        raise typer.BadParameter(
            "nothing to compare: every selected run shares all its dimensions. "
            "Widen the selection, or name a dimension with --group-by"
        )
    cells = store.group_runs(con, found, by)
    if len(cells) < 2:
        console_warn(
            f"only one cell ({store.cell_label(by, cells[0][0])}); there is "
            "nothing to difference against"
        )

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    data = store.read_residuals(
        con, found, im, component, ordinate_range=(period_min, period_max)
    )

    # One query for every run, then split in numpy: the join is the expensive
    # half and it does not get cheaper by being run once per cell.
    per_cell = []
    for key, cell_runs in cells:
        keys = {run["run_key"] for run in cell_runs}
        mask = np.isin(data["run_key"], list(keys))
        per_cell.append((key, {name: column[mask] for name, column in data.items()}))

    rows = None
    if balanced:
        rows = store.common_rows(
            [store.pivot_ordinates(cell)[2] for _, cell in per_cell]
        )
        if rows.size == 0:
            raise typer.BadParameter(
                "no (event, station) pair is present in every cell, so there is "
                "nothing to compare on equal terms. Re-run with --unbalanced to "
                "score each cell over whatever it has, and read the result "
                "knowing the cells cover different ground"
            )
        widest = max(len(store.pivot_ordinates(cell)[2]) for _, cell in per_cell)
        if len(rows) < widest:
            print(
                f"balanced over {len(rows)} (event, station) pairs; the fullest "
                f"cell has {widest}, so {widest - len(rows)} were set aside"
            )

    series = []
    for (key, cell), colour in zip(per_cell, cell_colours(len(per_cell))):
        matrix, ordinates, _ = store.pivot_ordinates(cell, rows)
        matrix = np.where(
            np.isfinite(matrix).sum(axis=0) >= minimum, matrix, np.nan
        )
        series.append(
            {
                "name": store.cell_label(by, key),
                "colour": colour,
                "matrix": matrix,
                "ordinates": ordinates,
                "stats": bias_statistics(matrix),
            }
        )

    reference = resolve_baseline(cells, by, [store.parse_label(t) for t in (baseline or [])])
    scalar = "ordinate" not in data
    display = Display.for_figure(
        (7.0, 3.2 + 1.5 * max(0, len(series) - 1)),
        dpi,
        display_height,
        viewing_distance,
    )

    others = [n for n in range(len(series)) if n != reference]
    heights = [3.0] + [1.4] * len(others)
    fig, axes = plt.subplots(
        len(heights),
        1,
        figsize=display.size,
        dpi=display.dpi,
        height_ratios=heights,
        sharex=not scalar,
        layout="constrained",
        squeeze=False,
    )
    panels = [row[0] for row in axes]
    ax = panels[0]

    for entry in series:
        if scalar:
            ax.errorbar(
                [entry["name"]],
                entry["stats"]["mean"],
                yerr=entry["stats"]["se"] * 1.96,
                fmt="o",
                color=entry["colour"],
                capsize=3,
            )
        else:
            draw_bias_curve(
                ax,
                entry["ordinates"],
                entry["stats"],
                entry["colour"],
                "-",
                interval,
                display,
            )
    ax.axhline(0, color="#6b6b6b", lw=display.mark(0.8), zorder=1)
    ax.set_ylabel(residual_label(im))
    if not scalar:
        ax.set_xscale("log")
        ax.legend(
            handles=[
                plt.Line2D([], [], color=e["colour"], lw=1.8, label=e["name"])
                for e in series
            ],
            fontsize=8,
            frameon=False,
        )

    # The paired difference: the recording cancels identically, so what is left
    # is one simulation against another at the same station and ordinate.
    for panel, other in zip(panels[1:], others):
        difference = series[other]["matrix"] - series[reference]["matrix"]
        draw_difference_panel(
            panel,
            series[other]["ordinates"],
            bias_statistics(difference),
            interval,
            (series[other]["name"], series[reference]["name"]),
            display,
        )
        if not scalar:
            panel.set_xscale("log")

    panels[-1].set_xlabel("period (s)" if im == "pSA" else "")
    for panel in panels:
        panel.tick_params(labelsize=9)
        panel.grid(
            True,
            which="both" if display.detailed else "major",
            lw=display.mark(0.3),
            color="#dddddd",
            zorder=0,
        )
        for spine in panel.spines.values():
            spine.set_linewidth(display.mark(0.6))
        if not scalar:
            panel.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if display.scale > 1.0:
            panel.xaxis.set_major_locator(LogLocator(numticks=display.ticks(9)))
            panel.yaxis.set_major_locator(MaxNLocator(nbins=display.ticks(6)))
            panel.tick_params(which="minor", length=0, labelbottom=False)
    for panel in panels[:-1]:
        panel.tick_params(labelbottom=False)

    scored = int(np.nanmax(series[0]["stats"]["count"]))
    events = len(set(data["event"]))
    if display.detailed:
        ax.set_title(
            f"{im} {component} -- {events} events, {scored} "
            f"(event, station) pairs, {len(series)} cells of "
            f"{'/'.join(by)}",
            fontsize=11,
        )
    con.close()

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
