"""``pairwise``: every pair of configurations as a lower-triangular matrix.

The problem this solves is that four bias curves drawn over one another are
indistinguishable when the configurations barely differ -- which is the usual
case, and precisely when the small differences are what a reader wants. A matrix
of numbers separates them::

    eqvis pairwise ims.duckdb --group-by solver --group-by layers
    eqvis pairwise ims.duckdb --group-by mesh --period 2 --period 5

On the **diagonal** is each configuration's own mean residual against the
recordings: red over-predicts, blue under-predicts. **Below the diagonal**, the
cell at row *A* and column *B* is

    mean ln(B / A)

over the stations both were scored at -- so red means the column configuration
runs higher than the row one. The two quantities get separate colour scales
because they are different things measured on different ranges: a configuration's
misfit against reality is typically several times larger than the difference
between two configurations, which is exactly why the difference is invisible when
the two are drawn on one map.

Each cell also carries whether the paired contrast is resolved. The pairing is
exact -- the two configurations are scored at the same station in the same
earthquake, so the recording cancels -- and cells whose 95% interval excludes
zero are marked. An unmarked cell is a difference this sample cannot see.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import TwoSlopeNorm

from . import store
from .console import console_warn
from .constants import DEFAULT_COMPONENT
from .data import residual_label

# Periods away from the hybrid crossover, where the low-frequency solution the
# solvers actually differ over dominates the response. Near 1 s the blend with
# the shared high-frequency component makes a solver comparison uninformative,
# so the default deliberately steps over it.
DEFAULT_PERIODS = (2.0, 5.0, 10.0)

# Cells whose paired interval excludes zero carry this; the mark is the claim.
RESOLVED = "•"


def compact_label(key: tuple[str, ...]) -> str:
    """A cell's values alone, for an axis that has no room for their names.

    :func:`eqvis_workflow.store.cell_label` spells out ``name=value`` so a legend
    entry stands on its own. A matrix axis cannot afford that -- four of them
    overran the figure -- and does not need it, because the title names the
    dimensions once for the whole matrix.
    """
    return "/".join(key)


def cell_statistics(
    data: dict[str, np.ndarray], by: list[str], keys: list[tuple], period: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-configuration bias, and the paired contrast between every pair.

    Returns the diagonal biases, the off-diagonal ``mean ln(column / row)``, and
    a boolean of whether each paired contrast is resolved at 95%.

    The contrast is computed from the *paired* residuals rather than from the
    difference of the two means. Those two are algebraically the same number, but
    only the pairing gives it an interval worth quoting: the site scatter and the
    recording are common to both configurations and cancel, which is what turns
    an interval of a few tenths into one of a few hundredths.
    """
    at = np.isclose(np.asarray(data["ordinate"], dtype=float), period)
    if not at.any():
        raise typer.BadParameter(
            f"no residuals at {period:g} s. The stored ordinates are "
            f"{', '.join(f'{v:g}' for v in np.unique(data['ordinate'])[:12])} ..."
        )
    cells: dict[tuple, dict[tuple, float]] = {key: {} for key in keys}
    for row in np.flatnonzero(at):
        key = tuple(str(data[name][row]) for name in by)
        if key in cells:
            cells[key][(data[store.EVENT][row], data["station"][row])] = data[
                "residual"
            ][row]

    n = len(keys)
    bias = np.array([np.mean(list(cells[key].values())) for key in keys])
    toward = np.full((n, n), np.nan)
    resolved = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            shared = sorted(set(cells[keys[i]]) & set(cells[keys[j]]))
            if len(shared) < 3:
                continue
            # ln(column / row), station by station: the recording cancels, so
            # this is one simulation against another and nothing else.
            paired = np.array(
                [cells[keys[j]][s] - cells[keys[i]][s] for s in shared]
            )
            toward[i, j] = float(paired.mean())
            half = 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))
            resolved[i, j] = abs(paired.mean()) > half
    return bias, toward, resolved


def draw_matrix(
    ax,
    labels: list[str],
    bias: np.ndarray,
    toward: np.ndarray,
    resolved: np.ndarray,
    bias_norm,
    toward_norm,
    cmap,
    title: str,
    show_names: bool,
) -> None:
    """One lower-triangular matrix: the diagonal, then the pairs below it."""
    n = len(labels)
    for i in range(n):
        for j in range(i + 1):
            value = bias[i] if i == j else toward[i, j]
            if not np.isfinite(value):
                continue
            norm = bias_norm if i == j else toward_norm
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cmap(norm(value)),
                    edgecolor="white", lw=1.2,
                )
            )
            # Ink that survives on either end of a diverging ramp.
            shade = "white" if abs(norm(value) - 0.5) > 0.34 else "0.15"
            mark = RESOLVED if (i != j and resolved[i, j]) else ""
            ax.text(
                j, i, f"{value:+.2f}{mark}",
                ha="center", va="center", fontsize=7.5, color=shade,
            )
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Rows carry the names; columns are numbered against them. In a lower
    # triangle the two axes hold the same set, so labelling both says everything
    # twice and costs the height the map below needs.
    ax.set_xticklabels(
        [f"{position + 1}" for position in range(n)] if show_names else [""] * n,
        fontsize=7.5,
    )
    ax.set_yticklabels(
        [f"{position + 1}  {label}" for position, label in enumerate(labels)]
        if show_names else [""] * n,
        fontsize=7.5,
    )
    ax.set_title(title, fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def pairwise(
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
    label: Annotated[
        list[str] | None,
        typer.Option("--label", help="Restrict to runs matching name=value; repeat"),
    ] = None,
    group_by: Annotated[
        list[str] | None,
        typer.Option(
            "--group-by",
            help="Run dimension whose values become the rows and columns; repeat "
            "to cross two",
        ),
    ] = None,
    period: Annotated[
        list[float] | None,
        typer.Option(
            "--period",
            help="Period for one matrix; repeat. Defaults to 2, 5 and 10 s, which "
            "step over the hybrid crossover near 1 s where the shared "
            "high-frequency component makes a solver comparison uninformative",
        ),
    ] = None,
    limit: Annotated[
        float | None,
        typer.Option("--limit", help="Colour limit for both scales; from the data if omitted"),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead of showing")
    ] = None,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution")] = 300,
) -> None:
    """Compare every pair of configurations as a lower-triangular matrix."""
    con = store.connect(db)
    labels_in = [store.parse_label(text) for text in (label or [])]
    found = store.select_runs(con, labels_in)
    by = list(group_by or store.varying(found))
    by = [name for name in by if name != store.EVENT] or by
    cells = store.group_runs(con, found, by)
    if len(cells) < 2:
        raise typer.BadParameter(
            "a pairwise matrix needs at least two cells; --group-by gives one"
        )
    keys = [key for key, _ in cells]
    names = [compact_label(key) for key in keys]

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    periods = sorted(period or DEFAULT_PERIODS)
    data = store.read_residuals(con, found, im, component)
    if "ordinate" not in data:
        raise typer.BadParameter(
            f"{im} has no period axis, so there is nothing to draw a matrix per. "
            "Use `eqvis compare --im " + im + "` instead"
        )

    found_stats = []
    for value in periods:
        bias, toward, resolved = cell_statistics(data, by, keys, value)
        found_stats.append((value, bias, toward, resolved))

    span = limit or max(
        float(np.nanmax(np.abs(b))) for _, b, _, _ in found_stats
    )
    reach = limit or max(
        float(np.nanmax(np.abs(t))) for _, _, t, _ in found_stats if np.isfinite(t).any()
    )
    cmap = plt.get_cmap("RdBu_r")
    bias_norm = TwoSlopeNorm(0.0, -span, span)
    toward_norm = TwoSlopeNorm(0.0, -reach, reach)

    fig, axes = plt.subplots(
        1, len(periods),
        figsize=(1.0 + 2.4 * len(periods), 3.5), dpi=dpi,
        layout="constrained", squeeze=False,
    )
    for position, (ax, (value, bias, toward, resolved)) in enumerate(
        zip(axes[0], found_stats)
    ):
        draw_matrix(
            ax, names, bias, toward, resolved, bias_norm, toward_norm, cmap,
            f"{value:g} s", show_names=(position == 0),
        )

    diagonal = fig.colorbar(
        plt.cm.ScalarMappable(bias_norm, cmap), ax=axes[0], location="bottom",
        fraction=0.05, pad=0.04, aspect=40,
    )
    diagonal.set_label(f"diagonal: {residual_label(im)}", fontsize=8)
    diagonal.ax.tick_params(labelsize=7)
    offset = fig.colorbar(
        plt.cm.ScalarMappable(toward_norm, cmap), ax=axes[0], location="bottom",
        fraction=0.05, pad=0.08, aspect=40,
    )
    offset.set_label(
        f"below diagonal: mean ln(column / row)  ({RESOLVED} = resolved at 95%)",
        fontsize=8,
    )
    offset.ax.tick_params(labelsize=7)

    events = len(set(data[store.EVENT]))
    pairs = len(set(zip(data[store.EVENT], data["station"])))
    unresolved = sum(
        int((~r[np.tril_indices_from(r, -1)]).sum()) for _, _, _, r in found_stats
    )
    print(
        f"{len(keys)} configurations of {'/'.join(by)} over {events} events and "
        f"{pairs} (event, station) pairs; {unresolved} of "
        f"{len(periods) * len(keys) * (len(keys) - 1) // 2} contrasts unresolved"
    )
    if unresolved:
        console_warn(
            "an unmarked cell is a difference this sample cannot see, not a "
            "difference of zero"
        )
    con.close()

    if output is not None:
        fig.savefig(output, dpi=dpi)
        print(f"wrote {output}")
    else:
        plt.show()
