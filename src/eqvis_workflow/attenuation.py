"""``distance``: an intensity measure against a source distance metric.

The same values ``map`` draws spatially, plotted against rrup, rjb, rx, ry, epi
or hyp -- whichever the file carries::

    eqvis distance im.h5 PGA --metric rrup
    eqvis distance im.h5 PGA --observed flatfiles.zip --view broad

Against observations it offers two views: ``broad`` puts the whole simulation
cloud behind the recordings, ``focussed`` drops the cloud and plots the log
residual alone. Combining ``--diff`` with the recordings or an empirical model
turns it into a stack of panels, one question per panel: how the two runs
differ over the whole field, which sits closer to the model, and which sits
closer to what was recorded.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
from matplotlib.colors import BoundaryNorm

from .console import console_warn
from .constants import (
    DEFAULT_COMPONENT,
    DIFFERENCE_INK,
    DISTANCE_COLUMN,
    DISTANCE_LABEL,
    EMPIRICAL_BLUE,
    LOG_SCALED,
    OBSERVED_GREEN,
    SIGNED_DISTANCES,
    SIM_CLOUD_GREY,
    SIM_ONE_BLACK,
    SIM_TWO_CLOUD,
    SIM_TWO_ORANGE,
    UNIT_LABEL,
)
from .data import (
    comparison_labels,
    default_title,
    empirical_loess,
    im_label,
    open_ims,
    residual_label,
    restrict_to_domain,
    run_names,
    run_title,
    select_empirical,
    select_im,
)
from .flatfile import read_observed
from .picks import read_pick_list, restrict_to_stations
from .raster import fixed_symmetric_norm
from .stations import place_labels, sample_simulation, station_labels


class View(str, Enum):
    """How to show observations against the simulation on a distance plot."""

    broad = "broad"
    focussed = "focussed"


def select_distance(da: xr.DataArray, metric: str) -> np.ndarray:
    """The named distance, which rides along as a station coordinate of the IM."""
    if metric not in DISTANCE_COLUMN:
        raise typer.BadParameter(
            f"unknown metric {metric!r}. Choose from: {list(DISTANCE_COLUMN)}"
        )
    if metric not in da.coords:
        available = [m for m in DISTANCE_COLUMN if m in da.coords]
        raise typer.BadParameter(
            f"{metric!r} is not a coordinate in this file. Available: {available}"
        )
    return da[metric].values


def run_series(
    path: Path,
    name: str,
    colour: str,
    cloud: str,
    im: str,
    component: str,
    period: float | None,
    frequency: float | None,
    metric: str,
    empirical: str | None,
    strict: bool = True,
) -> dict:
    """Everything the distance panels need about one simulation, in one dict.

    Each run resolves its own nearest period/frequency and carries its own
    empirical prediction, since two runs need not share a station set, a period
    set or -- the reason the prediction is not simply borrowed from the first
    run -- their site terms. A run that carries no prediction at all is only
    tolerated when ``strict`` is false, and then falls back to another run's;
    see :func:`attach_recordings`.
    """
    tree = open_ims(path)
    da, selection = select_im(tree, im, component, period, frequency)
    series = {
        "name": name,
        "colour": colour,
        "cloud": cloud,
        "tree": tree,
        "da": da,
        "selection": selection,
        "distance": select_distance(da, metric),
        "mean": None,
        "sigma": None,
    }
    if empirical is not None:
        try:
            mean, sigma = select_empirical(tree, im, empirical, selection)
        except typer.BadParameter:
            if strict:
                raise
        else:
            series["mean"] = mean.sel(station=da.station).values
            series["sigma"] = sigma.sel(station=da.station).values
    return series


def attach_recordings(
    series: dict, observed: dict[str, np.ndarray], fallback: dict | None = None
) -> None:
    """Sample a run, and its prediction, at the observed stations.

    Every panel that pairs a run against a recording works from these, so they
    are computed once here. A run without a prediction of its own is scored
    against ``fallback``'s -- the same model, so the only thing borrowed is the
    site terms it was conditioned on.
    """
    at = (series["da"].longitude.values, series["da"].latitude.values)
    where = (observed["lon"], observed["lat"])
    series["at_recordings"] = sample_simulation(*at, series["da"].values, *where)
    if series["mean"] is not None:
        series["mean_here"] = sample_simulation(*at, series["mean"], *where)
        series["sigma_here"] = sample_simulation(*at, series["sigma"], *where)
    elif fallback is not None and "mean_here" in fallback:
        series["mean_here"] = fallback["mean_here"]
        series["sigma_here"] = fallback["sigma_here"]


def connector_span(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where to start and end the dashed connector at each recording.

    The connector spans everything plotted at that station, so its length reads
    as the spread between them. A station needs at least two finite values to
    have a span at all, which for a single run means both the recording and the
    simulation -- the rule the one-run plot has always used.
    """
    stack = np.vstack(rows)
    known = np.isfinite(stack)
    keep = known.sum(axis=0) >= 2
    low = np.min(np.where(known, stack, np.inf), axis=0)
    high = np.max(np.where(known, stack, -np.inf), axis=0)
    return keep, low, high


def annotate_skill(ax: plt.Axes, entries: list[tuple[str, str, np.ndarray]]) -> None:
    """Bias and scatter of each run's residuals, stated rather than eyeballed.

    ``mu`` is the mean residual (which run sits high or low) and ``sigma`` its
    standard deviation (how much of the misfit is scatter rather than offset),
    both over the stations where the residual exists. Sigma is the sample
    standard deviation, so it is the same quantity ``bias`` reports.
    """
    line = 0
    for label, colour, values in entries:
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            continue
        ax.annotate(
            f"{label}  μ {np.mean(finite):+.2f}  σ {np.std(finite, ddof=1):.2f}",
            xy=(0.012, 0.95 - 0.14 * line),
            xycoords="axes fraction",
            fontsize=7,
            color=colour,
            ha="left",
            va="top",
            zorder=8,
            bbox={"fc": "white", "ec": "none", "alpha": 0.65, "pad": 1.0},
        )
        line += 1


def draw_attenuation_panel(
    ax: plt.Axes,
    series: list[dict],
    obs: dict[str, np.ndarray] | None,
    simulated: list[np.ndarray],
    im_name: str,
    units: str,
    log_im: bool,
    log_x: bool,
    empirical: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The IM itself against distance: every run, the recordings and the model.

    Returns the anchors for the station labels -- the recording, or its
    simulated counterpart when the flatfile carries no value to label.
    """
    # Later runs go down first so the first run's cloud reads on top of them;
    # with two clouds overlapping both need to stay translucent to be seen.
    alpha = None if len(series) == 1 else 0.55
    for run in reversed(series):
        ax.scatter(
            run["distance"],
            run["da"].values,
            s=3,
            color=run["cloud"],
            alpha=alpha,
            lw=0,
            rasterized=True,
            zorder=1,
        )
    if log_im:
        ax.set_yscale("log")
    ax.set_ylabel(f"{im_name} ({units})" if units else im_name)

    # Over the simulation cloud, under everything at the recordings. Only the
    # first run's prediction is drawn: a second near-identical band would cost
    # more clutter than it explains, and the panels below score each run
    # against its own.
    banded = False
    if series[0].get("mean") is not None:
        band = empirical_loess(series[0]["distance"], series[0]["mean"], log_x)
        if band is not None:
            centre, fitted, spread = band
            ax.fill_between(
                centre,
                np.exp(fitted - spread),
                np.exp(fitted + spread),
                color=EMPIRICAL_BLUE,
                alpha=0.20,
                lw=0,
                zorder=2,
            )
            ax.plot(
                centre,
                np.exp(fitted),
                color=EMPIRICAL_BLUE,
                lw=1.4,
                ls="--",
                alpha=0.95,
                zorder=3,
            )
            banded = True

    if obs is not None:
        # Every member of a group sits at the recording's own distance, so the
        # dashed connector between them is vertical and its length reads
        # directly as the misfit.
        keep, low, high = connector_span([obs["value"], *simulated])
        ax.vlines(
            obs["distance"][keep],
            low[keep],
            high[keep],
            colors="#555555",
            lw=0.7,
            ls="--",
            zorder=4,
        )
        for run, values in zip(series, simulated):
            ax.scatter(
                obs["distance"],
                values,
                marker="o",
                s=28,
                color=run["colour"],
                lw=0,
                zorder=5,
            )
        ax.scatter(
            obs["distance"],
            obs["value"],
            marker="^",
            s=90,
            fc=OBSERVED_GREEN,
            ec="black",
            lw=0.6,
            zorder=6,
        )

    # A lone unnamed run needs no legend -- the axis label already says what the
    # cloud is -- but a second run does, since nothing else distinguishes the
    # two clouds.
    if obs is None and len(series) == 1:
        return None
    handles = []
    for run in series:
        handles.append(
            plt.Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4,
                color=run["cloud"],
                label=f"{run['name']} (all stations)",
            )
        )
        if obs is not None:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=5,
                    color=run["colour"],
                    label=f"{run['name']} (at recordings)",
                )
            )
    if obs is not None:
        handles.append(
            plt.Line2D(
                [],
                [],
                ls="none",
                marker="^",
                ms=8,
                mfc=OBSERVED_GREEN,
                mec="black",
                mew=0.6,
                label="observed",
            )
        )
    if banded:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=EMPIRICAL_BLUE,
                ls="--",
                lw=1.2,
                label=f"{empirical} (LOESS ±1 s.d.)",
            )
        )
    ax.legend(handles=handles, fontsize=8, frameon=False)

    if obs is None:
        return None
    return obs["distance"], np.where(
        np.isfinite(obs["value"]), obs["value"], simulated[0]
    )


def draw_ratio_panel(
    ax: plt.Axes,
    series: list[dict],
    metric: str,
    im_name: str,
    files: list[Path],
    annotate: bool,
) -> None:
    """Log ratio of the two runs, over every station they share.

    The only panel that sees the whole field rather than the handful of
    stations that recorded, so it is where a systematic difference between the
    runs shows up.
    """
    first, second = xr.align(series[0]["da"], series[1]["da"], join="inner")
    if first.sizes["station"] == 0:
        raise typer.BadParameter(f"{files[0]} and {files[1]} share no stations")
    ratio = np.log(first.values) - np.log(second.values)
    ax.axhline(0, color="#6b6b6b", lw=0.8, zorder=2)
    ax.scatter(
        select_distance(first, metric),
        ratio,
        s=3,
        color=DIFFERENCE_INK,
        alpha=0.35,
        lw=0,
        rasterized=True,
    )
    ax.set_ylabel(f"ln[{im_name}$_1$ / {im_name}$_2$]")
    if annotate:
        annotate_skill(ax, [("all stations", DIFFERENCE_INK, ratio)])


def draw_zscore_panel(
    ax: plt.Axes,
    series: list[dict],
    obs: dict[str, np.ndarray],
    empirical: str,
) -> None:
    """Every run and the recordings on one axis, in units of the model's sigma.

    Scored against the empirical model rather than against each other, so zero
    is the prediction and +/-1 is one standard deviation of it. A run closer to
    the recordings' own scores is the run that reproduces what was recorded.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z_observed = (np.log(obs["value"]) - series[0]["mean_here"]) / series[0][
            "sigma_here"
        ]
        z_simulated = [
            (np.log(run["at_recordings"]) - run["mean_here"]) / run["sigma_here"]
            for run in series
        ]
    ax.axhline(0, color=EMPIRICAL_BLUE, lw=1.0, ls="--", zorder=3)
    for level in (-2, -1, 1, 2):
        ax.axhline(level, color="#c8c8c8", lw=0.5, ls=":", zorder=2)
    keep, low, high = connector_span([z_observed, *z_simulated])
    ax.vlines(
        obs["distance"][keep],
        low[keep],
        high[keep],
        colors="#555555",
        lw=0.7,
        ls="--",
        zorder=4,
    )
    for run, values in zip(series, z_simulated):
        ax.scatter(
            obs["distance"],
            values,
            marker="o",
            s=28,
            color=run["colour"],
            lw=0,
            zorder=5,
        )
    ax.scatter(
        obs["distance"],
        z_observed,
        marker="^",
        s=90,
        fc=OBSERVED_GREEN,
        ec="black",
        lw=0.6,
        zorder=6,
    )
    ax.set_ylabel(f"z-score vs {empirical}", fontsize=9)
    for label, value in (("+1σ", 1), ("−1σ", -1)):
        ax.annotate(
            label,
            xy=(0.998, value),
            xycoords=("axes fraction", "data"),
            fontsize=6,
            color="#8a8a8a",
            ha="right",
            va="bottom",
        )
    if len(series) > 1:
        annotate_skill(
            ax,
            [
                ("observed", OBSERVED_GREEN, z_observed),
                *(
                    (run["name"], run["colour"], values)
                    for run, values in zip(series, z_simulated)
                ),
            ],
        )


def draw_residual_panel(
    ax: plt.Axes,
    series: list[dict],
    obs: dict[str, np.ndarray],
    simulated: list[np.ndarray],
    im_name: str,
) -> None:
    """Each run's misfit against the recordings, the quantity the map colours.

    A circle is a simulated value throughout, so these stay circles in their
    run's colour even though they sit at recording sites; zero is agreement.
    """
    ax.axhline(0, color="#6b6b6b", lw=0.8, zorder=2)
    residuals = [np.log(values) - np.log(obs["value"]) for values in simulated]
    for run, residual in zip(series, residuals):
        ax.scatter(
            obs["distance"],
            residual,
            marker="o",
            s=45,
            fc=run["colour"],
            ec="black",
            lw=0.4,
            zorder=5,
        )
    ax.set_ylabel(residual_label(im_name), fontsize=9)
    annotate_skill(
        ax,
        [
            (run["name"], run["colour"], residual)
            for run, residual in zip(series, residuals)
        ],
    )


def distance(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    im: Annotated[str, typer.Argument(help="Intensity measure to plot")] = "PGA",
    metric: Annotated[
        str, typer.Option(help="Distance metric: rrup, rjb, rx, ry, epi or hyp")
    ] = "rrup",
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            exists=True,
            dir_okay=False,
            help="Second IM file: plot ln(IM_1) - ln(IM_2), or with --observed "
            "/ --empirical compare both runs panel by panel",
        ),
    ] = None,
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip: overlay in-domain recording stations",
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
    view: Annotated[
        View,
        typer.Option(
            help="With --observed: 'broad' keeps the simulation cloud behind the "
            "recordings, 'focussed' plots the log residual alone"
        ),
    ] = View.broad,
    name: Annotated[
        list[str] | None,
        typer.Option(
            "--name",
            help="Name for each simulation, in order; repeat to name the --diff run",
        ),
    ] = None,
    empirical: Annotated[
        str | None,
        typer.Option(
            "--empirical",
            help="Empirical model to predict with, e.g. NSHM2022; adds its median "
            "+/- sigma and a z-score panel",
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
    residual_limit: Annotated[
        float, typer.Option(help="Colour scale limit for the station residuals")
    ] = 0.7,
    label: Annotated[
        bool,
        typer.Option("--label/--no-label", help="Name the observed stations"),
    ] = True,
    dpi: Annotated[int, typer.Option(help="Output resolution")] = 300,
):
    """Plot an intensity measure against source distance.

    A second run given as ``--diff`` alone is plotted as a log ratio, as before.
    Bringing recordings or an empirical model in beside it splits the figure
    into panels instead, so each comparison gets its own axis rather than
    crowding one: the runs against each other, each run against the model, and
    each run against what was recorded.
    """
    multi = diff is not None and (observed is not None or empirical is not None)
    if view is View.focussed and (empirical is not None or multi):
        raise typer.BadParameter(
            "the residual-only view is for one run against its recordings; "
            "an empirical model or a second run needs --view broad"
        )

    component = component or DEFAULT_COMPONENT.get(im, "geom")
    files = [im_file] if diff is None else [im_file, diff]
    names = run_names(name, files)
    # In-plot labels, unlike the title, cannot carry a full path, so an unnamed
    # run gets a generic word instead.
    legend_names = (
        [names[0] if name else "simulation"]
        if len(files) == 1
        else comparison_labels(name, ("sim 1", "sim 2"))
    )
    series = [
        run_series(
            path,
            legend,
            colour,
            cloud,
            im,
            component,
            period,
            frequency,
            metric,
            empirical,
            # The first run has to carry the prediction, since it is the one
            # drawn as a band; a second run may borrow it.
            strict=index == 0,
        )
        for index, (path, legend, colour, cloud) in enumerate(
            zip(
                files,
                legend_names,
                (SIM_ONE_BLACK, SIM_TWO_ORANGE),
                (SIM_CLOUD_GREY, SIM_TWO_CLOUD),
            )
        )
    ]
    tree, da, selection = series[0]["tree"], series[0]["da"], series[0]["selection"]
    im_name = im_label(im, selection)
    for dim, value in selection.items():
        print(f"selected {dim}: {value:g}")

    units = UNIT_LABEL.get(da.attrs.get("units", ""), da.attrs.get("units", ""))
    distance_label = f"{DISTANCE_LABEL[metric]} (km)"
    log_x = metric not in SIGNED_DISTANCES

    obs = None
    simulated = []
    if observed is not None:
        obs, resolved = read_observed(observed, im, component, selection, metric)
        for dim, value in resolved.items():
            print(f"observed {dim}: {value:g}")
        obs = restrict_to_domain(
            obs, tree.attrs, da.longitude.values, da.latitude.values, observed
        )
        if stations is not None:
            obs = restrict_to_stations(
                obs, list(read_pick_list(stations)["stations"]), observed
            )
        for run in series:
            attach_recordings(run, obs, fallback=series[0])
            if run["mean"] is None and empirical is not None:
                console_warn(
                    f"{run['name']} carries no {empirical} prediction; scoring it "
                    f"against {series[0]['name']}'s"
                )
        simulated = [run["at_recordings"] for run in series]

    # The z-score panel needs recordings to score, so it only appears with both.
    scored = empirical is not None and observed is not None
    if multi:
        # One question per panel: how the runs differ (ratio), which sits closer
        # to the model (z-score), and which sits closer to what was recorded
        # (residual).
        panels = ["attenuation", "ratio"]
        if scored:
            panels.append("zscore")
        if observed is not None:
            panels.append("residual")
    elif diff is not None:
        panels = ["ratio"]
    elif observed is not None and view is View.focussed:
        panels = ["focussed"]
    else:
        panels = ["attenuation"] + (["zscore"] if scored else [])

    if len(panels) == 1:
        fig, only = plt.subplots(figsize=(8, 6), layout="constrained")
        drawn = [only]
    else:
        fig, drawn = plt.subplots(
            len(panels),
            1,
            figsize=(8, {2: 7.5, 3: 10.0, 4: 12.5}[len(panels)]),
            height_ratios=[2.2 if panel == "attenuation" else 1.0 for panel in panels],
            sharex=True,
            layout="constrained",
        )
    axes = dict(zip(panels, drawn))
    plot_x = plot_y = None

    if "attenuation" in panels:
        anchors = draw_attenuation_panel(
            axes["attenuation"],
            series,
            obs,
            simulated,
            im_name,
            units,
            im in LOG_SCALED,
            log_x,
            empirical,
        )
        if anchors is not None:
            plot_x, plot_y = anchors
    if "focussed" in panels:
        # The residual is the y axis, so the simulation cloud has no place here;
        # the colour just echoes the height for continuity with --view broad and
        # the map.
        residual = np.log(simulated[0]) - np.log(obs["value"])
        if not np.isfinite(residual).any():
            raise typer.BadParameter(
                f"{observed} has no observed {im}, so there is no residual to "
                f"plot; use --view broad"
            )
        residual_cmap = plt.get_cmap("RdBu_r")
        residual_levels = fixed_symmetric_norm(residual_limit, levels)
        residual_norm = BoundaryNorm(residual_levels, residual_cmap.N, extend="both")
        ax = axes["focussed"]
        ax.axhline(0, color="#6b6b6b", lw=0.8, zorder=2)
        known = np.isfinite(residual)
        # No colorbar for this one: the y axis already is the residual scale, so
        # the colour only flags which points saturate it.
        ax.scatter(
            obs["distance"][known],
            residual[known],
            c=residual[known],
            cmap=residual_cmap,
            norm=residual_norm,
            marker="^",
            s=90,
            ec="black",
            lw=0.6,
            zorder=6,
        )
        ax.set_ylabel(residual_label(im_name))
        plot_x, plot_y = obs["distance"], residual
    if "ratio" in panels:
        draw_ratio_panel(axes["ratio"], series, metric, im_name, files, multi)
    if "zscore" in panels:
        draw_zscore_panel(axes["zscore"], series, obs, empirical)
    if "residual" in panels:
        draw_residual_panel(axes["residual"], series, obs, simulated, im_name)

    if log_x:
        # A log axis silently drops non-positive values, and rjb is legitimately
        # zero directly over the rupture, so say so rather than lose the points.
        for run in series:
            if (run["distance"] <= 0).any():
                zeros = int((run["distance"] <= 0).sum())
                where = f" in {run['name']}" if len(series) > 1 else ""
                console_warn(
                    f"{zeros} stations have {metric} <= 0{where}; omitted from log axis"
                )
        for ax in drawn:
            ax.set_xscale("log")
    for ax in drawn:
        ax.tick_params(labelsize=9)
        ax.grid(True, which="both", lw=0.3, color="#dddddd", zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    # The distance axis belongs to the bottom panel; the rest share it.
    for ax in drawn[:-1]:
        ax.tick_params(labelbottom=False)
    drawn[-1].set_xlabel(distance_label)

    if heading := default_title(tree.attrs, run_title(names, bool(name))):
        drawn[0].set_title(heading, fontsize=11)

    # Last, so labels are placed against the final axes size.
    if plot_x is not None and label:
        anchored = axes.get("attenuation", axes.get("focussed"))
        place_labels(fig, anchored, station_labels(obs["name"], plot_x, plot_y))

    if output is not None:
        fig.savefig(output, dpi=dpi)
        print(f"wrote {output}")
    else:
        plt.show()
