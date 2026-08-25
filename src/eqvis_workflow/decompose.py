"""``decompose``: the mixed-effects residual decomposition, swept over period.

The residual against the recordings, split into what is systematic to an
earthquake, what is systematic to a site, and what is left::

    ln(sim/obs) = a + dB_e + dS2S_s + dW_es

after Lee et al. (2022), whose notation is Al Atik et al. (2010) and whose
estimation follows Stafford (2014). One independent fit per period, per cell of
whatever run dimension the caller names -- and the dimensions come from the
database, so this works on any ingested tree::

    eqvis decompose ims.duckdb --group-by solver
    eqvis decompose ims.duckdb --group-by mesh --fixed vs30

Three things this command insists on, because each is a way the answer goes
quietly wrong.

*Per cell, not pooled.* Fitting all configurations at once would let the site
term absorb something it should not: the several configurations of one event at
one station share the *same recording*, so their residuals are correlated by
construction and that correlation is not site response. Each cell is fitted on
its own, where every (event, station) contributes exactly one row.

*The contrast comes from the pairing, not from the fit.* ``--pair`` differences
two cells station by station before fitting, which cancels the recording
identically -- so the difference is between two simulations and nothing else,
and it is exact rather than modelled.

*The identification diagnostic is printed every run.* A site term needs stations
that record more than one earthquake. Where they are rare the split of the
within-event variance is barely constrained, so the total is quoted as the
headline and the split is shown with an interval and a boundary flag rather than
as a number.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from matplotlib.ticker import FuncFormatter

from . import mixed, store
from .compare import cell_colours, resolve_baseline
from .console import console_warn
from .constants import DEFAULT_COMPONENT, DIFFERENCE_INK
from .data import residual_label
from .display import Display

# The random effects, in the order the papers name them: the earthquake first,
# then the site. Defaults only -- `--random` replaces them.
DEFAULT_RANDOM = (store.EVENT, "station")


def codes_for(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes and their level labels, with unused levels dropped.

    Dropped because an empty level is not harmless: it survives into the BLUP
    arrays, the level counts and the identification diagnostic, and would report
    a site that contributed nothing as though it had been measured.
    """
    labels, codes = np.unique(np.asarray(values, dtype=object).astype(str),
                              return_inverse=True)
    return codes, labels


def build_fixed(
    data: dict[str, np.ndarray], mask: np.ndarray, fixed: list[str], baseline: dict
) -> tuple[np.ndarray, list[str]]:
    """The fixed-effect design: an intercept, plus each named term.

    A numeric term enters as itself, centred so the intercept stays
    interpretable as the bias at the mean rather than at zero -- a vs30 of zero
    is not a site. A text term enters as treatment contrasts against a reference
    level, which is printed, because an unstated reference makes every
    coefficient unreadable.
    """
    columns = [np.ones(int(mask.sum()))]
    names = ["intercept"]
    for term in fixed:
        values = data[term][mask]
        if values.dtype == object or values.dtype.kind in "US":
            labels = sorted({str(v) for v in values})
            reference = baseline.get(term, labels[0])
            for label in labels:
                if label == reference:
                    continue
                columns.append((values.astype(str) == label).astype(float))
                names.append(f"{term}={label}")
        else:
            numeric = np.asarray(values, dtype=float)
            columns.append(numeric - np.nanmean(numeric))
            names.append(f"{term} (centred)")
    return np.column_stack(columns), names


def fit_one(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    random: list[str],
    fixed: list[str],
    baseline: dict,
) -> mixed.MixedFit | None:
    """One period of one cell, or None if there is too little to fit."""
    usable = mask & np.isfinite(np.asarray(data["residual"], dtype=float))
    if usable.sum() <= len(fixed) + 2:
        return None
    codes, levels = [], []
    for name in random:
        code, label = codes_for(data[name][usable])
        codes.append(code)
        levels.append(label)
    design, terms = build_fixed(data, usable, fixed, baseline)
    try:
        return mixed.fit_mixed(
            np.asarray(data["residual"], dtype=float)[usable],
            design,
            codes,
            [len(label) for label in levels],
            groups=random,
            terms=terms,
            levels=levels,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        console_warn(f"a fit failed and is left out: {error}")
        return None


def paired(data: dict[str, np.ndarray], by: list[str], a: tuple, b: tuple) -> dict:
    """Two cells differenced at matched (event, station, ordinate).

    The recording cancels identically here, so the difference carries no
    observation error and no site response -- which is what makes it the sharper
    instrument on the configuration contrast, and why it gets sharper rather
    than weaker as the sample shrinks.
    """
    keys = [store.EVENT, "station"] + (["ordinate"] if "ordinate" in data else [])

    def index(cell):
        chosen = np.ones(len(data["residual"]), dtype=bool)
        for name, value in zip(by, cell):
            chosen &= np.asarray(data[name]).astype(str) == value
        return {
            tuple(str(data[k][n]) for k in keys): n for n in np.flatnonzero(chosen)
        }

    left, right = index(a), index(b)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise typer.BadParameter(
            "the two cells share no (event, station, ordinate), so there is "
            "nothing to pair"
        )
    out = {
        "residual": np.array(
            [data["residual"][left[k]] - data["residual"][right[k]] for k in shared]
        )
    }
    for name in keys:
        out[name] = np.array([data[name][left[k]] for k in shared])
    return out


def decompose(
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
            help="Run dimension whose values are fitted separately; repeat to "
            "cross two. Separate fits rather than one pooled fit, because the "
            "configurations of an event share their recording",
        ),
    ] = None,
    pair: Annotated[
        list[str] | None,
        typer.Option(
            "--pair",
            help="name=value naming the cell to difference every other cell "
            "against before fitting. The recording cancels identically, so this "
            "is the sharpest test of the contrast",
        ),
    ] = None,
    random: Annotated[
        list[str] | None,
        typer.Option(
            "--random",
            help="Crossed grouping factor; repeat. Defaults to event and station",
        ),
    ] = None,
    fixed: Annotated[
        list[str] | None,
        typer.Option("--fixed", help="Extra fixed effect, by column name; repeat"),
    ] = None,
    baseline: Annotated[
        list[str] | None,
        typer.Option(
            "--baseline",
            help="name=value fixing the reference level of a text fixed effect",
        ),
    ] = None,
    interval: Annotated[
        float, typer.Option("--interval", help="Confidence level")
    ] = 0.95,
    minimum: Annotated[
        int, typer.Option("--minimum", help="Skip a period with fewer rows than this")
    ] = 20,
    every: Annotated[
        int,
        typer.Option("--every", help="Fit every Nth stored ordinate, to go faster"),
    ] = 1,
    period_min: Annotated[
        float | None, typer.Option("--period-min", help="Shortest ordinate to fit")
    ] = None,
    period_max: Annotated[
        float | None, typer.Option("--period-max", help="Longest ordinate to fit")
    ] = None,
    table: Annotated[
        Path | None,
        typer.Option(
            "--table",
            help="Write the numbers here as .parquet or .csv. A report quotes "
            "numbers, and reading them off a figure is not quoting them",
        ),
    ] = None,
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
    """Decompose the residual into event, site and remaining variance."""
    con = store.connect(db)
    labels = [store.parse_label(text) for text in (label or [])]
    found = store.select_runs(con, labels)
    by = list(group_by or store.varying(found))
    by = [name for name in by if name != store.EVENT] or by
    cells = store.group_runs(con, found, by)

    random = list(random or DEFAULT_RANDOM)
    fixed = list(fixed or [])
    store.check_names(con, [n for n in random if n != "station"], "--random")
    bases = dict(store.parse_label(text) for text in (baseline or []))

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    data = store.read_residuals(
        con,
        found,
        im,
        component,
        ordinate_range=(period_min, period_max),
        covariates=tuple(dict.fromkeys(n for n in fixed if n not in data_names(con))),
    )

    ordinates = (
        np.unique(np.asarray(data["ordinate"], dtype=float))[:: max(1, every)]
        if "ordinate" in data
        else np.array([np.nan])
    )

    reference = resolve_baseline(
        cells, by, [store.parse_label(t) for t in (pair or [])]
    )
    series = []
    for position, (key, cell_runs) in enumerate(cells):
        if pair and position == reference:
            continue
        keys = {run["run_key"] for run in cell_runs}
        subset = {
            name: column[np.isin(data["run_key"], list(keys))]
            for name, column in data.items()
        }
        if pair:
            subset = paired(data, by, key, cells[reference][0])
            name = (
                f"{store.cell_label(by, key)} − "
                f"{store.cell_label(by, cells[reference][0])}"
            )
            use_random = [n for n in random if n in subset]
            use_fixed = []
        else:
            name = store.cell_label(by, key)
            use_random, use_fixed = random, fixed

        if not pair:
            warn_shared_observation(subset, name)
        fits = {}
        for ordinate in ordinates:
            mask = (
                np.isclose(np.asarray(subset["ordinate"], dtype=float), ordinate)
                if "ordinate" in subset
                else np.ones(len(subset["residual"]), dtype=bool)
            )
            if mask.sum() < minimum:
                continue
            got = fit_one(subset, mask, use_random, use_fixed, bases)
            if got is not None:
                fits[float(ordinate)] = got
        if fits:
            series.append({"name": name, "fits": fits})

    if not series:
        raise typer.BadParameter(
            f"nothing could be fitted at --minimum {minimum}; there are too few "
            "residuals per ordinate"
        )
    report_identification(series)
    figure(series, im, component, by, bool(pair), interval, dpi, display_height,
           viewing_distance, output)
    if table is not None:
        write_table(series, table)
    con.close()


def warn_shared_observation(cell: dict, name: str) -> None:
    """Refuse to be quiet about several runs of one event at one station.

    The configurations of an event at a given station are scored against the
    *same recording*, so their residuals are correlated by construction. Fitted
    together, the site term absorbs that shared recording rather than site
    response, and the site variance comes out inflated -- measurably so: on a
    simulation with four configurations per cell it doubled. The fix is to group
    by every dimension, so each cell holds one run per event, or to use --pair,
    where the recording cancels instead.
    """
    if "ordinate" not in cell:
        keys = list(zip(cell[store.EVENT], cell["station"]))
    else:
        first = np.asarray(cell["ordinate"], dtype=float)
        pick = first == first[0]
        keys = list(zip(cell[store.EVENT][pick], cell["station"][pick]))
    repeats = len(keys) - len(set(keys))
    if repeats:
        console_warn(
            f"cell {name!r} holds {repeats} extra rows per ordinate from several "
            "runs of the same event at the same station, which share one "
            "recording. The site variance will absorb that rather than site "
            "response. Group by every run dimension, or use --pair"
        )


def data_names(con) -> set[str]:
    """Columns :func:`store.read_residuals` returns without being asked."""
    return {"run_key", store.EVENT, "station", "ordinate", "residual"} | set(
        store.dimensions(con)
    )


def report_identification(series: list[dict]) -> None:
    """Print why each component is or is not resolvable, every run.

    Unconditional, because the number that matters most here -- how far the
    within-event variance can be split -- depends entirely on how often a
    station records more than one earthquake, and that is invisible in the
    fitted values themselves.
    """
    print("\nidentification")
    for entry in series:
        first = next(iter(entry["fits"].values()))
        print(f"  {entry['name']}")
        for diagnostic in first.diagnostics.values():
            print(f"    {diagnostic.summary()}")
        at_boundary = sum(1 for f in entry["fits"].values() if f.boundary)
        if at_boundary:
            print(
                f"    a variance was driven to zero at {at_boundary} of "
                f"{len(entry['fits'])} ordinates -- the estimator reporting no "
                "evidence, not a failure"
            )


def curve(entry: dict, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """One quantity against ordinate, for plotting."""
    ordinates = np.array(sorted(entry["fits"]))
    values = []
    for ordinate in ordinates:
        fit = entry["fits"][ordinate]
        if quantity == "a":
            values.append(fit.a)
        elif quantity == "phi":
            values.append(fit.phi)
        elif quantity == "sigma":
            values.append(fit.sigma)
        else:
            values.append(fit.sd.get(quantity, np.nan))
    return ordinates, np.array(values)


def figure(
    series, im, component, by, is_paired, interval, dpi, height, distance, output
) -> None:
    """Bias against ordinate, then the variance components against ordinate."""
    display = Display.for_figure((7.0, 6.4), dpi, height, distance)
    fig, axes = plt.subplots(
        2, 1, figsize=display.size, dpi=display.dpi, sharex=True,
        height_ratios=[1.0, 1.15], layout="constrained",
    )
    top, bottom = axes
    colours = cell_colours(len(series))
    scalar = np.isnan(next(iter(series[0]["fits"])))

    for entry, colour in zip(series, colours):
        ordinates, bias = curve(entry, "a")
        half = np.array(
            [
                entry["fits"][o].confint(interval)[0]
                for o in sorted(entry["fits"])
            ]
        )
        colour = DIFFERENCE_INK if is_paired and len(series) == 1 else colour
        if scalar:
            top.errorbar([entry["name"]], bias, yerr=(half[:, 1] - bias), fmt="o",
                         color=colour, capsize=3)
        else:
            top.fill_between(ordinates, half[:, 0], half[:, 1], color=colour,
                             alpha=0.22, lw=0, zorder=3)
            top.plot(ordinates, bias, color=colour, lw=display.mark(1.8), zorder=4,
                     label=entry["name"])
        # phi heavy, its split light: the total is well determined where the
        # split may not be.
        if not scalar:
            bottom.plot(*curve(entry, "phi"), color=colour, lw=display.mark(2.0),
                        zorder=4, label=f"{entry['name']}  φ")
            for quantity, style in (("station", (0, (4, 2))), ("residual", (0, (1, 1.6))),
                                    (store.EVENT, (0, (6, 2, 1, 2)))):
                ords, values = curve(entry, quantity)
                if np.all(~np.isfinite(values)) or np.all(values == 0):
                    continue
                bottom.plot(ords, values, color=colour, lw=display.mark(1.0),
                            ls=style, zorder=3)

    top.axhline(0, color="#6b6b6b", lw=display.mark(0.8), zorder=1)
    top.set_ylabel(
        (residual_label(im) + " difference") if is_paired else residual_label(im)
    )
    bottom.set_ylabel("standard deviation (ln units)")
    bottom.set_ylim(bottom=0)
    if not scalar:
        for panel in axes:
            panel.set_xscale("log")
            panel.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        bottom.set_xlabel("period (s)" if im == "pSA" else "")
        top.legend(fontsize=7.5, frameon=False)
        bottom.legend(
            fontsize=6.8, frameon=False, ncol=2,
            title="heavy φ; dashed φ$_{S2S}$, dotted φ$_{ss}$, dash-dot τ",
            title_fontsize=6.5,
        )
    for panel in axes:
        panel.tick_params(labelsize=9)
        panel.grid(True, which="both" if display.detailed else "major",
                   lw=display.mark(0.3), color="#dddddd", zorder=0)
        for spine in panel.spines.values():
            spine.set_linewidth(display.mark(0.6))
    if display.detailed:
        top.set_title(f"{im} {component} decomposed by {'/'.join(by)}", fontsize=11)

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()


def write_table(series: list[dict], path: Path) -> None:
    """One long, tidy table of every number the fits produced."""
    rows: dict[str, list] = {
        k: [] for k in ("cell", "ordinate", "kind", "term", "estimate", "se",
                        "low", "high", "n", "boundary")
    }
    for entry in series:
        for ordinate, fit in sorted(entry["fits"].items()):
            bounds = fit.confint()
            for position, term in enumerate(fit.terms):
                rows["cell"].append(entry["name"])
                rows["ordinate"].append(ordinate)
                rows["kind"].append("coefficient")
                rows["term"].append(term)
                rows["estimate"].append(float(fit.beta[position]))
                rows["se"].append(float(fit.se()[position]))
                rows["low"].append(float(bounds[position, 0]))
                rows["high"].append(float(bounds[position, 1]))
                rows["n"].append(fit.n)
                rows["boundary"].append(False)
            for name, value in {**fit.sd, "phi": fit.phi, "sigma": fit.sigma}.items():
                rows["cell"].append(entry["name"])
                rows["ordinate"].append(ordinate)
                rows["kind"].append("component")
                rows["term"].append(name)
                rows["estimate"].append(float(value))
                for key in ("se", "low", "high"):
                    rows[key].append(float("nan"))
                rows["n"].append(fit.n)
                rows["boundary"].append(name in fit.boundary)
    written = pa.table(rows)
    if path.suffix == ".csv":
        import pyarrow.csv as csv

        csv.write_csv(written, path)
    else:
        pq.write_table(written, path)
    print(f"wrote {path} ({written.num_rows:,} rows)")
