"""``bias``: the log residual against recordings, swept across the spectrum.

At each pSA period the residual is averaged over the stations, so a run that
runs low at long period shows as a curve below zero there; the band is the
confidence interval on that mean. With two runs a panel below differences them
station by station, which is where a real difference between the runs shows up
-- their own intervals overlap almost everywhere, because most of that width is
site scatter common to both and only the pairing cancels it::

    eqvis bias sw4/im.h5 --observed flatfiles.zip --diff emod3d/im.h5 \\
        --empirical NSHM2022 --name SW4 --name EMOD3D
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
)
from scipy.stats import t as student

from .console import console_warn
from .constants import (
    DEFAULT_COMPONENT,
    DIFFERENCE_INK,
    EMPIRICAL_BLUE,
    FAS_PREFIX,
    SIM_ONE_BLACK,
    SIM_TWO_ORANGE,
)
from .data import (
    Screen,
    comparison_labels,
    default_title,
    open_ims,
    residual_label,
    restrict_to_domain,
    run_names,
    run_title,
    select_empirical,
    supergrid,
)
from .display import NATURAL, Display
from .flatfile import read_observed_spectra
from .picks import read_pick_list, restrict_to_stations
from .stations import nearest_stations


def match_columns(
    values: np.ndarray,
    periods: np.ndarray,
    reference: np.ndarray,
    tolerance: float = 0.05,
) -> np.ndarray:
    """Columns of a (station, period) table lined up with a reference period grid.

    Recordings and simulations are tabulated on period sets that usually match
    but need not, so each column is taken from the nearest period available and
    dropped when the nearest is not near enough to be the same period.
    """
    nearest = np.abs(periods[:, None] - reference[None, :]).argmin(axis=0)
    matched = values[:, nearest]
    matched[:, np.abs(periods[nearest] - reference) / reference > tolerance] = np.nan
    return matched


def bias_statistics(residual: np.ndarray) -> dict[str, np.ndarray]:
    """Per-period mean, spread and count over the stations of a residual table.

    ``sd`` is how much the misfit varies from station to station and ``se`` how
    well the mean itself is pinned down; they answer different questions and the
    plot shows both, since a large bias over three stations is not the same
    finding as the same bias over thirty.
    """
    known = np.isfinite(residual)
    count = known.sum(axis=0)
    safe = np.where(known, residual, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, safe.sum(axis=0) / count, np.nan)
        spread = np.where(known, (residual - mean) ** 2, 0.0).sum(axis=0)
        sd = np.where(count > 1, np.sqrt(spread / np.maximum(count - 1, 1)), np.nan)
    return {"count": count, "mean": mean, "sd": sd, "se": sd / np.sqrt(count)}


def interval_half_width(stats: dict[str, np.ndarray], interval: float) -> np.ndarray:
    """Half the confidence interval on the mean, from Student's t."""
    with np.errstate(invalid="ignore"):
        return student.ppf(0.5 + interval / 2, stats["count"] - 1) * stats["se"]


def draw_bias_curve(
    ax: plt.Axes,
    periods: np.ndarray,
    stats: dict[str, np.ndarray],
    colour: str,
    style: str,
    interval: float,
    display: Display | None = None,
) -> None:
    """One series' bias against period: the mean and the interval on it.

    The band is the confidence interval of the mean itself, so zero falling
    outside it is the claim that the run really is biased at that period. It is
    the only thing shaded here -- how far individual stations scatter about the
    mean is a different question, and shading both at once turned three series
    into six overlapping washes that hid the curves they described. The scatter
    gets its own panel, as lines.

    The interval is Student's rather than normal: at long period only a handful
    of records are still usable, and over three stations the normal interval is
    less than half the width it should be.
    """
    display = display or NATURAL
    half = interval_half_width(stats, interval)
    ax.fill_between(
        periods,
        stats["mean"] - half,
        stats["mean"] + half,
        color=colour,
        alpha=0.22,
        lw=0,
        zorder=3,
    )
    ax.plot(
        periods,
        stats["mean"],
        color=colour,
        lw=display.mark(1.8),
        ls=style,
        zorder=4,
    )


def draw_difference_panel(
    ax: plt.Axes,
    periods: np.ndarray,
    stats: dict[str, np.ndarray],
    interval: float,
    labels: tuple[str, str],
    display: Display | None = None,
) -> None:
    """Whether the two runs really differ, period by period.

    Reading that off the panel above -- do the two intervals overlap? -- is the
    wrong test and a badly conservative one: both runs are scored at the same
    stations, so most of the width of those intervals is site-to-site
    variability common to both, which cancels when the runs are differenced
    station by station. What is plotted here is that paired difference, whose
    mean is exactly the gap between the two curves above but whose interval is
    far tighter. Where the shading clears zero, the runs differ.

    The band is laid down whole and the significant stretches darkened over the
    top of it, rather than drawn as two complementary ``where`` fills: a fill
    takes only the segments with both ends inside its mask, so two of them leave
    the changeover segment to neither and the band comes apart at every
    transition.
    """
    display = display or NATURAL
    half = interval_half_width(stats, interval)
    real = np.abs(stats["mean"]) > half
    low, high = stats["mean"] - half, stats["mean"] + half
    ax.axhline(0, color="#6b6b6b", lw=display.mark(0.8), zorder=1)
    ax.fill_between(
        periods, low, high, color=DIFFERENCE_INK, alpha=0.15, lw=0, zorder=2
    )
    ax.fill_between(
        periods,
        low,
        high,
        where=real,
        color=DIFFERENCE_INK,
        alpha=0.35,
        lw=0,
        zorder=3,
    )
    line = ax.plot(
        periods, stats["mean"], color=DIFFERENCE_INK, lw=display.mark(1.6), zorder=4
    )[0]
    ax.set_ylabel(
        f"{labels[0]} − {labels[1]}\nbias difference"
        if display.detailed
        # Two lines of rotated label cost twice the width, and enlarged that is
        # width the panel needs; the sign convention is the half worth keeping.
        else f"{labels[0]} − {labels[1]}",
        fontsize=9,
    )

    known = np.isfinite(stats["mean"])
    share = real[known].mean() if known.any() else 0.0
    if not display.detailed:
        # Enlarged, this legend is wider than the panel it sits in, and a
        # legend that wide does not just overflow -- constrained_layout gives
        # it the room out of the axes, until the curve is a sliver. The
        # shading is a caption's sentence, so the caption takes it.
        console_warn(
            f"the runs differ over {share:.0%} of periods; enlarged, the panel "
            "drops the legend that says so"
        )
        return
    # The darkened patch is the base plus the overlay, so its swatch has to be
    # too -- a single patch at either alpha would misname one of the two states.
    darkened = (
        plt.Rectangle((0, 0), 1, 1, fc=DIFFERENCE_INK, alpha=0.15, lw=0),
        plt.Rectangle((0, 0), 1, 1, fc=DIFFERENCE_INK, alpha=0.35, lw=0),
    )
    ax.legend(
        handles=[
            line,
            darkened,
        ],
        labels=[
            "paired mean",
            f"{interval:.0%} interval",
        ],
        # ndivide=1 so the pair share one swatch and composite to the colour
        # they make on the plot, instead of sitting side by side as two.
        handler_map={darkened: HandlerTuple(ndivide=1, pad=0)},
        fontsize=7,
        frameon=False,
        ncols=2,
        loc="upper left",
        columnspacing=1.2,
        handlelength=1.6,
    )
    # Headroom, so the legend sits over blank axes rather than over the band.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.38 * (top - bottom))


def bias(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    observed: Annotated[
        Path,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip holding the recordings to score against",
        ),
    ],
    im: Annotated[
        str, typer.Argument(help="Intensity measure to sweep: pSA (period) or FAS (frequency)")
    ] = "pSA",
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff", exists=True, dir_okay=False, help="Second IM file to compare"
        ),
    ] = None,
    empirical: Annotated[
        str | None,
        typer.Option(
            "--empirical",
            help="Empirical model to score alongside the runs, e.g. NSHM2022",
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
    stations: Annotated[
        Path | None,
        typer.Option(
            "--stations",
            exists=True,
            dir_okay=False,
            help="Pick list from `pick`: what to draw and what to name",
        ),
    ] = None,
    interval: Annotated[
        float,
        typer.Option(help="Confidence level for the interval on the mean, 0-1"),
    ] = 0.95,
    minimum: Annotated[
        int,
        typer.Option(
            "--minimum",
            help="Drop periods scored by fewer than this many recordings",
        ),
    ] = 3,
    usable: Annotated[
        bool,
        typer.Option(
            "--usable/--no-usable",
            help="Ignore periods beyond a record's high-pass corner, where the "
            "recording is filter rather than ground motion",
        ),
    ] = True,
    screen: Annotated[
        Screen,
        typer.Option(
            "--supergrid",
            help="What to do with stations inside the SW4 supergrid absorbing "
            "layer, where the simulation is a damped, coordinate-stretched "
            "solution rather than ground motion",
        ),
    ] = Screen.exclude,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output image path (omit to show interactively)"
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
):
    """Plot the bias of one or two simulations against recordings, swept across a spectrum.

    The single number a distance plot gives at one period or frequency, swept
    across the whole spectrum: at each point the log residual ln(sim/obs) is
    averaged over every station that recorded, so a run that is systematically
    low at long period (pSA) or low frequency (FAS) shows up as a curve sitting
    below zero there rather than as a cloud that has to be read one point at a
    time. ``im`` is pSA (period, s) or FAS (frequency, Hz); FAS carries no
    empirical predictions, so ``--empirical`` only applies to pSA.

    Two kinds of contaminated data are dropped by default, for the same reason:
    ``--usable`` drops periods beyond a recording's high-pass corner, where the
    recording is filter rather than ground motion, and ``--supergrid`` drops
    stations inside SW4's absorbing layer, where the simulation is a damped,
    coordinate-stretched solution rather than ground motion. Both are reported
    as counts, and the second is named on the figure's count panel.
    """
    if not 0 < interval < 1:
        raise typer.BadParameter("--interval is a confidence level, strictly 0 to 1")
    if im not in ("pSA", "FAS"):
        raise typer.BadParameter("bias sweeps a spectrum, so im must be pSA or FAS")
    # pSA is swept over period (s), FAS over frequency (Hz); everything below
    # keys off this one flag rather than repeating the pSA/FAS branch.
    dim = "period" if im == "pSA" else "frequency"
    prefix = "pSA_" if im == "pSA" else FAS_PREFIX
    component = component or DEFAULT_COMPONENT[im]
    files = [im_file] if diff is None else [im_file, diff]
    names = run_names(name, files)
    legend_names = (
        [names[0] if name else "simulation"]
        if len(files) == 1
        else comparison_labels(name, ("sim 1", "sim 2"))
    )

    runs = []
    for index, (path, legend, colour) in enumerate(
        zip(files, legend_names, (SIM_ONE_BLACK, SIM_TWO_ORANGE))
    ):
        tree = open_ims(path)
        if im not in tree.children:
            raise typer.BadParameter(f"{path} has no {im}")
        node = tree[im]
        if component not in node.data_vars:
            raise typer.BadParameter(
                f"component {component!r} not in {im}. "
                f"Available: {[str(c) for c in node.data_vars]}"
            )
        da = node[component]
        if dim not in da.dims:
            raise typer.BadParameter(f"{path} {im} has no {dim} dimension")
        runs.append(
            {
                "name": legend,
                "colour": colour,
                "tree": tree,
                "da": da.transpose("station", dim),
                "periods": da[dim].values,
                "first": index == 0,
            }
        )

    obs, obs_periods = read_observed_spectra(
        observed, component, metric=None, prefix=prefix
    )
    first = runs[0]
    obs = restrict_to_domain(
        obs,
        first["tree"].attrs,
        first["da"].longitude.values,
        first["da"].latitude.values,
        observed,
    )
    if stations is not None:
        obs = restrict_to_stations(
            obs, list(read_pick_list(stations)["stations"]), observed
        )
    if obs["name"].size == 0:
        raise typer.BadParameter(f"no {observed} stations inside the domain")

    # The first run's period/frequency grid is the axis; everything else is
    # matched onto it, so the runs are compared at the same points rather than
    # each on its own.
    periods = first["periods"]
    recorded = np.log(match_columns(obs["spectrum"], obs_periods, periods))
    if usable:
        # High-pass filtering contaminates long periods / low frequencies, not
        # the ground motion above its corner: scoring a simulation against it
        # would report the filter instead.
        if im == "pSA":
            longest = np.where(
                np.isfinite(obs["high_pass"]), 1.0 / obs["high_pass"], np.inf
            )
            filtered = periods[None, :] > longest[:, None]
        else:
            lowest = np.where(np.isfinite(obs["high_pass"]), obs["high_pass"], 0.0)
            filtered = periods[None, :] < lowest[:, None]
        dropped = int((filtered & np.isfinite(recorded)).sum())
        recorded = np.where(filtered, np.nan, recorded)
        if dropped:
            plural = "periods" if im == "pSA" else "frequencies"
            print(f"ignored {dropped} station-{plural} beyond the high-pass corner")

    series = []
    modelled = None
    # What the supergrid screen cost, so the figure can say so: the worst of
    # the runs, since the count panel already warns where the runs disagree and
    # what matters on the label is that stations were dropped at all.
    screened = 0
    for run in runs:
        nearest, reached = nearest_stations(
            run["da"].longitude.values,
            run["da"].latitude.values,
            obs["lon"],
            obs["lat"],
        )
        # A station inside the SW4 supergrid absorbing layer is not a ground
        # motion prediction -- the solver integrates a damped, coordinate-
        # stretched equation in there -- so scoring a recording against one
        # reports the absorbing layer, exactly as scoring against a
        # high-passed recording beyond its corner reports the filter. That is
        # why this defaults to excluding, as --usable does.
        inside = supergrid(run["tree"], run["da"]).flagged[nearest] & reached
        if (count := int(inside.sum())) and screen is not Screen.keep:
            print(
                f"ignored {count} recordings whose nearest {run['name']} station "
                "is inside the supergrid absorbing layer"
                if screen is Screen.exclude
                else f"{count} recordings pair with a {run['name']} station inside "
                "the supergrid absorbing layer, and are scored anyway"
            )
        if screen is Screen.exclude:
            reached &= ~inside
            screened = max(screened, count)
            if count and not reached.any():
                # The one message the original investigation needed: a whole
                # run's worth of stations sat in the layer, and every figure
                # drawn from it was of the layer rather than of the ground.
                raise typer.BadParameter(
                    f"every recording pairs with a {run['name']} station inside "
                    "the supergrid absorbing layer, so there is nothing left to "
                    "score; --supergrid keep scores them anyway"
                )
        simulated = match_columns(run["da"].values[nearest], run["periods"], periods)
        simulated[~reached] = np.nan
        residual = np.log(simulated) - recorded
        series.append(
            {
                "name": run["name"],
                "colour": run["colour"],
                "style": "-",
                "residual": residual,
                "stats": bias_statistics(residual),
            }
        )
        if empirical is not None and run["first"]:
            mean, _ = select_empirical(run["tree"], im, empirical, {})
            predicted = match_columns(
                mean.sel(station=run["da"].station)
                .transpose("station", dim)
                .values[nearest],
                mean[dim].values,
                periods,
            )
            predicted[~reached] = np.nan
            # Held back so the runs, which are the subject, lead the legend.
            modelled = {
                "name": empirical,
                "colour": EMPIRICAL_BLUE,
                "style": "--",
                "stats": bias_statistics(recorded - predicted),
            }
    # Paired station by station, so everything the two runs have in common --
    # the site, the path, the recording itself -- cancels instead of being
    # carried into the comparison as noise.
    paired = None
    if len(runs) > 1:
        paired = {
            "stats": bias_statistics(series[0]["residual"] - series[1]["residual"])
        }
    if modelled is not None:
        series.append(modelled)

    # A period only a couple of stations reach says more about those stations
    # than about the run, so it is dropped rather than drawn with a band wide
    # enough to mean nothing.
    scored = np.zeros(periods.shape, bool)
    for entry in series:
        scored |= entry["stats"]["count"] >= minimum
    if not scored.any():
        raise typer.BadParameter(
            f"no period is scored by {minimum} or more recordings; lower --minimum"
        )
    periods = periods[scored]
    for entry in series + ([paired] if paired is not None else []):
        entry["stats"] = {key: value[scored] for key, value in entry["stats"].items()}

    # One question per panel: where each series sits, whether the two runs
    # really differ, how much their stations disagree, and how much there was to
    # average in the first place. Only the first two are shaded, and never over
    # each other.
    design = (8.0, 9.5 if paired is not None else 8.0)
    display = Display.for_figure(design, dpi, display_height, viewing_distance)
    display.report(design)
    # Enlarged, the last two panels go. They qualify the curves above rather
    # than carry them -- how far the stations scatter about the mean, and how
    # many there were to average -- and at poster scale the two inches they
    # cost are inches the bias curve and its interval need to stay readable.
    stack = ["bias"] + (["difference"] if paired is not None else [])
    if display.detailed:
        stack += ["spread", "count"]
    fig, drawn = plt.subplots(
        len(stack),
        1,
        figsize=display.size,
        height_ratios=[
            {"bias": 3.0, "difference": 1.2, "spread": 1.0, "count": 0.6}[panel]
            for panel in stack
        ],
        sharex=True,
        layout="constrained",
        # A lone bias panel would otherwise come back as a bare Axes.
        squeeze=False,
    )
    drawn = drawn[:, 0]
    axes = dict(zip(stack, drawn))
    ax = axes["bias"]
    ax.axhline(0, color="#6b6b6b", lw=display.mark(0.8), zorder=1)
    for entry in series:
        draw_bias_curve(
            ax,
            periods,
            entry["stats"],
            entry["colour"],
            entry["style"],
            interval,
            display,
        )
    ax.set_ylabel(residual_label(im) if display.detailed else "ln[sim / obs]")
    ax.set_xscale("log")

    handles = [
        plt.Line2D(
            [],
            [],
            color=entry["colour"],
            ls=entry["style"],
            lw=display.mark(1.8),
            label=entry["name"],
        )
        for entry in series
    ]
    if display.detailed:
        # Enlarged, this entry is the longest line in the legend and the one
        # a caption can carry instead: the shading is the interval, and which
        # interval it is has to be said in the caption at that size anyway.
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc="#6b6b6b",
                alpha=0.22,
                lw=0,
                label=f"{interval:.0%} interval on the mean",
            )
        )
    ax.legend(
        handles=handles,
        fontsize=8,
        frameon=False,
        # Two columns of enlarged text reach past the axes, and the layout
        # engine buys that width back out of the panel itself.
        ncols=2 if display.detailed else 1,
        loc="best" if display.detailed else "upper left",
    )

    if paired is not None:
        draw_difference_panel(
            axes["difference"],
            periods,
            paired["stats"],
            interval,
            (series[0]["name"], series[1]["name"]),
            display,
        )

    if (ax_spread := axes.get("spread")) is not None:
        for entry in series:
            ax_spread.plot(
                periods,
                entry["stats"]["sd"],
                color=entry["colour"],
                ls=entry["style"],
                lw=display.mark(1.3),
            )
        ax_spread.set_ylabel("s.d. between\nstations", fontsize=9)
        ax_spread.set_ylim(bottom=0)

    # One line: the series agree on the count wherever they cover the same
    # stations, which is the usual case, and stacking identical lines only
    # obscured them. Where they do disagree the difference is said in words
    # rather than drawn, since it is a caveat and not a curve.
    counts = np.vstack([entry["stats"]["count"] for entry in series])
    if (ax_count := axes.get("count")) is not None:
        ax_count.plot(
            periods,
            counts[0],
            color="#6b6b6b",
            lw=display.mark(1.4),
            drawstyle="steps-mid",
        )
        # Never a bare "recordings" once any were dropped: a figure that
        # silently lost stations is exactly the failure this whole flag exists
        # to make visible.
        ax_count.set_ylabel(
            f"recordings\n({screened} in supergrid)" if screened else "recordings",
            fontsize=9,
        )
        ax_count.set_ylim(bottom=0)
    # Said whether or not the panel is there to say it against: how much there
    # was to average is a caveat on the curves, not on the panel.
    apart = int(np.abs(counts - counts[0]).max())
    if apart:
        console_warn(
            f"series differ by up to {apart} recordings at some periods"
            + (
                f"; the count shown is {series[0]['name']}'s"
                if ax_count is not None
                else ""
            )
        )
    drawn[-1].set_xlabel("period (s)" if im == "pSA" else "frequency (Hz)")

    for panel in drawn:
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
        panel.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if display.scale > 1.0:
            # Enlarged text takes the room the default ticks were using, and a
            # log decade's minor ticks merge into a comb well before that.
            panel.xaxis.set_major_locator(LogLocator(numticks=display.ticks(9)))
            panel.yaxis.set_major_locator(MaxNLocator(nbins=display.ticks(6)))
            panel.tick_params(which="minor", length=0, labelbottom=False)
    for panel in drawn[:-1]:
        panel.tick_params(labelbottom=False)

    # Enlarged, a title longer than the canvas is clipped at both ends rather
    # than shrunk, and the run names are the part of it the legend has already
    # said; what is left -- the event and its magnitude -- fits.
    heading = default_title(
        first["tree"].attrs,
        run_title(names, bool(name)) if display.detailed else "",
    )
    if heading:
        ax.set_title(heading, fontsize=11)

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
