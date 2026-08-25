"""``residual-heat``: the residual over period against a binned covariate.

Where :mod:`~.compare` answers "how big is the misfit at each period", this
answers "and does it depend on something" -- distance, latitude, site stiffness,
elevation, which basin the station sits in. One panel per cell of a run
dimension, period along the bottom, the covariate up the side::

    eqvis residual-heat ims.duckdb --group-by solver --bin-by rrup
    eqvis residual-heat ims.duckdb --group-by layers --bin-by basin

The covariate is not a fixed list. It is validated against
:func:`eqvis_workflow.store.available_covariates`, which describes the schema
rather than enumerating it, so every column the tables carry -- including any a
later schema adds -- is a legal ``--bin-by`` with no change here.

Numeric covariates are binned by quantile rather than by equal width, because
station distances and elevations are heavily skewed and equal-width bins would
put most of the data in one row. A text covariate is not binned at all: its
distinct values are the rows, and NULL becomes a row of its own, because "in no
basin" is a fact about a site and not a gap in the data.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.ticker import FuncFormatter, LogLocator

from . import store
from .compare import resolve_baseline
from .console import console_warn
from .constants import DEFAULT_COMPONENT
from .data import residual_label
from .display import Display
from .raster import fixed_symmetric_norm

# The row a NULL covariate goes in. Named rather than dropped: a station in no
# basin is not a station whose basin is unknown.
ABSENT = "(none)"


def numeric_bins(
    values: np.ndarray, bins: int, edges: list[float] | None
) -> tuple[np.ndarray, list[str]]:
    """Bin a numeric covariate, by explicit edges or by quantile.

    Quantile by default so every row carries a comparable number of
    observations: over these station sets an equal-width binning of ``rrup`` or
    ``elevation`` puts most of the data in the first row and leaves the rest
    drawing one station's noise.
    """
    finite = np.isfinite(values)
    if edges:
        bounds = np.array(sorted(edges), dtype=float)
    else:
        bounds = np.unique(
            np.nanquantile(values[finite], np.linspace(0, 1, bins + 1))
        )
    if bounds.size < 2:
        raise typer.BadParameter(
            "the covariate takes too few distinct values to bin; name another "
            "with --bin-by, or give --bin-edges explicitly"
        )
    index = np.digitize(values, bounds[1:-1], right=False).astype(float)
    index[~finite] = np.nan
    labels = [
        f"{bounds[n]:g}–{bounds[n + 1]:g}" for n in range(len(bounds) - 1)
    ]
    return index, labels


def text_bins(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Rows for a text covariate: its distinct values, with NULL its own row."""
    named = np.array([ABSENT if v is None else str(v) for v in values])
    labels = sorted(set(named) - {ABSENT})
    if ABSENT in named:
        labels = [*labels, ABSENT]
    lookup = {label: position for position, label in enumerate(labels)}
    return np.array([lookup[v] for v in named], dtype=float), labels


def grid_means(
    data: dict[str, np.ndarray],
    row: np.ndarray,
    n_rows: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean residual and count in every (row, ordinate) cell.

    A cell scored by fewer than ``minimum`` residuals is blanked rather than
    coloured: at one or two stations the colour is that station's noise, and a
    reader has no way to tell it from a finding.
    """
    residual = np.asarray(data["residual"], dtype=float)
    if "ordinate" in data:
        ordinates = np.unique(np.asarray(data["ordinate"], dtype=float))
        column = np.searchsorted(ordinates, np.asarray(data["ordinate"], dtype=float))
    else:
        ordinates = np.array([np.nan])
        column = np.zeros(len(residual), dtype=np.intp)

    total = np.zeros((n_rows, len(ordinates)))
    count = np.zeros((n_rows, len(ordinates)), dtype=int)
    usable = np.isfinite(residual) & np.isfinite(row)
    np.add.at(total, (row[usable].astype(np.intp), column[usable]), residual[usable])
    np.add.at(count, (row[usable].astype(np.intp), column[usable]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count >= minimum, total / np.maximum(count, 1), np.nan)
    return mean, count, ordinates


def residual_heat(
    db: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Composite intensity measure database"
        ),
    ],
    bin_by: Annotated[
        str,
        typer.Option(
            "--bin-by",
            help="Column to put up the vertical axis. Any station, geometry or "
            "source column the schema carries; run `eqvis runs DB` to list them",
        ),
    ],
    im: Annotated[str, typer.Option("--im", help="Intensity measure")] = "pSA",
    component: Annotated[
        str | None, typer.Option("--component", help="Component of motion")
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option("--label", help="Restrict to runs matching name=value; repeat"),
    ] = None,
    group_by: Annotated[
        list[str] | None,
        typer.Option(
            "--group-by",
            help="Run dimension whose values become the panels; repeat to cross two",
        ),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            help="name=value picking a cell every other panel is differenced "
            "against, cell by cell",
        ),
    ] = None,
    bins: Annotated[
        int, typer.Option("--bins", help="Quantile bins for a numeric covariate")
    ] = 6,
    bin_edges: Annotated[
        list[float] | None,
        typer.Option(
            "--bin-edges",
            help="Explicit bin boundaries; repeat. Overrides --bins, for when "
            "the quantile default is not the story you want to tell",
        ),
    ] = None,
    minimum: Annotated[
        int,
        typer.Option("--minimum", help="Blank a cell scored by fewer than this many"),
    ] = 10,
    limit: Annotated[
        float | None,
        typer.Option(
            "--limit",
            help="Colour scale limit. Taken from the data when omitted; pin it "
            "so that two figures share a scale",
        ),
    ] = None,
    counts: Annotated[
        bool, typer.Option("--counts/--no-counts", help="Annotate each cell's count")
    ] = False,
    period_min: Annotated[
        float | None, typer.Option("--period-min", help="Shortest ordinate to draw")
    ] = None,
    period_max: Annotated[
        float | None, typer.Option("--period-max", help="Longest ordinate to draw")
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
    """Draw the mean residual over period against a binned covariate."""
    con = store.connect(db)
    available = store.available_covariates(con)
    if bin_by not in available:
        raise typer.BadParameter(
            f"--bin-by {bin_by!r} is not a column this database carries. "
            f"Available: {', '.join(sorted(available))}"
        )

    labels = [store.parse_label(text) for text in (label or [])]
    found = store.select_runs(con, labels)
    by = list(group_by or store.varying(found))
    by = [name for name in by if name != store.EVENT] or by
    cells = store.group_runs(con, found, by)

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    data = store.read_residuals(
        con,
        found,
        im,
        component,
        ordinate_range=(period_min, period_max),
        covariates=(bin_by,),
    )

    # Binned once over the whole selection, not per panel, so every panel shares
    # its rows and the panels can be read against each other.
    column = data[bin_by]
    if column.dtype == object or column.dtype.kind in "US":
        row_all, row_labels = text_bins(column)
    else:
        row_all, row_labels = numeric_bins(
            np.asarray(column, dtype=float), bins, bin_edges
        )
    dropped = int(np.sum(~np.isfinite(row_all)))
    if dropped:
        console_warn(
            f"{dropped:,} residuals have no {bin_by} and are left out. A NULL "
            f"{bin_by} means the value was never derived, unlike a text "
            "covariate where it would be a row of its own"
        )

    panels_data = []
    for key, cell_runs in cells:
        keys = {run["run_key"] for run in cell_runs}
        mask = np.isin(data["run_key"], list(keys))
        mean, count, ordinates = grid_means(
            {name: col[mask] for name, col in data.items()},
            row_all[mask],
            len(row_labels),
            minimum,
        )
        panels_data.append((store.cell_label(by, key), mean, count, ordinates))

    reference = resolve_baseline(
        cells, by, [store.parse_label(t) for t in (baseline or [])]
    )
    if baseline:
        base = panels_data[reference][1]
        panels_data = [
            (f"{name} − {panels_data[reference][0]}", mean - base, count, ordinates)
            for position, (name, mean, count, ordinates) in enumerate(panels_data)
            if position != reference
        ]

    stacked = np.concatenate([mean.ravel() for _, mean, _, _ in panels_data])
    extent = limit if limit is not None else float(np.nanmax(np.abs(stacked)))
    if not np.isfinite(extent) or extent == 0:
        raise typer.BadParameter(
            f"every cell is blank at --minimum {minimum}; there are too few "
            "residuals per (row, ordinate) cell. Lower --minimum, or use fewer "
            "--bins"
        )
    boundaries = fixed_symmetric_norm(extent, 12)

    display = Display.for_figure(
        (7.2, 1.1 + 1.9 * len(panels_data)), dpi, display_height, viewing_distance
    )
    fig, axes = plt.subplots(
        len(panels_data),
        1,
        figsize=display.size,
        dpi=display.dpi,
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    drawn = [row[0] for row in axes]

    mesh = None
    spectral = len(panels_data[0][3]) > 1
    for panel, (name, mean, count, ordinates) in zip(drawn, panels_data):
        # Cell centres, not edges: `shading="nearest"` centres each quad on its
        # coordinate, which is what keeps a log period axis' cells aligned with
        # the ordinates they describe rather than offset by half a cell.
        x = ordinates if spectral else np.array([0.0])
        mesh = panel.pcolormesh(
            x,
            np.arange(len(row_labels), dtype=float),
            mean,
            cmap="RdBu_r",
            vmin=boundaries[0],
            vmax=boundaries[-1],
            shading="nearest",
            rasterized=True,
        )
        panel.set_yticks(np.arange(len(row_labels)))
        panel.set_yticklabels(row_labels, fontsize=7)
        panel.set_ylabel(name, fontsize=9)
        if spectral:
            panel.set_xscale("log")
        if counts:
            for r in range(mean.shape[0]):
                for c in range(mean.shape[1]):
                    if count[r, c]:
                        panel.text(
                            x[c], r, str(count[r, c]),
                            ha="center", va="center", fontsize=5,
                        )
        for spine in panel.spines.values():
            spine.set_linewidth(display.mark(0.6))
        panel.tick_params(labelsize=8)

    drawn[-1].set_xlabel("period (s)" if im == "pSA" else "")
    if spectral:
        drawn[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if display.scale > 1.0:
            drawn[-1].xaxis.set_major_locator(LogLocator(numticks=display.ticks(9)))
    bar = fig.colorbar(mesh, ax=drawn, ticks=display.keep(boundaries, 7), pad=0.02)
    bar.set_label(
        residual_label(im) + (" difference" if baseline else ""), fontsize=9
    )
    bar.ax.tick_params(labelsize=8)
    if display.detailed:
        drawn[0].set_title(
            f"{im} {component} by {bin_by} -- {len(set(data['event']))} events, "
            f"{len(set(zip(data['event'], data['station'])))} (event, station) pairs",
            fontsize=10,
        )
    con.close()

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
