"""``psa-spectrum``: pSA against period at one station.

The response-spectrum counterpart of :mod:`~.spectra`. Where the file carries
the full rotd180 set, an inset shows how the motion at that station depends on
azimuth -- which is where directivity shows itself.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
from matplotlib.ticker import (
    LogLocator,
    MaxNLocator,
)

from .console import console_warn
from .constants import (
    COMPARISON_BLUE,
    DEFAULT_COMPONENT,
    DIRECTIVITY_COLOURS,
    EMPIRICAL_BLUE,
    OBSERVED_GREEN,
)
from .data import open_ims, select_empirical
from .display import Display
from .flatfile import read_observed_spectra
from .geography import draw_inset_map, load_coastline
from .spectra import resample_spectrum


def decode_rotd180(values: np.ndarray, ln_step: float) -> np.ndarray:
    """Invert the ln-quantized, angle-axis delta encoding of a `rotd180` variable.

    The encoding (``workflow.psa_compression.encode_psa_rotd180`` in the
    workflow repo, applied when IM calculation is run with
    ``--full-rotd180``) stores ``round(ln(pSA) / ln_step)`` at angle 0 and its
    first differences along the angle axis thereafter, so a cumulative sum
    undoes the differencing and ``exp`` undoes the log-quantization. Kept
    inline here (rather than importing ``workflow``) since this script is a
    standalone ``uv run --script`` with its own dependency block.
    """
    return np.exp(np.cumsum(values.astype(np.int64), axis=-1) * ln_step)


def read_directivity(
    node: xr.DataTree, station: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Full 0-359 degree pSA RotD curve at one station, for every period.

    Returns ``(periods, curve)`` where ``curve`` has shape
    ``(n_periods, 360)``, mirrored from the stored 0-179 degree sweep: for a
    linearly rotated seismogram the peak absolute amplitude at angle theta
    equals the peak at theta + 180 exactly (rotating a sign-symmetric peak
    detector by half a turn changes nothing it measures), so the stored half
    already determines the whole circle. Returns None if the file carries no
    `rotd180` variable, i.e. it was produced without ``--full-rotd180``.
    """
    if "rotd180" not in node.data_vars:
        return None
    da = node["rotd180"].sel(station=station)
    half = decode_rotd180(da.values, da.attrs["ln_step"])  # (n_periods, 180)
    full = np.concatenate([half, half], axis=-1)  # (n_periods, 360)
    return node.period.values, full


def draw_directivity_inset(
    ax: plt.Axes,
    periods: np.ndarray,
    curve: np.ndarray,
    requested_periods: Sequence[float],
) -> plt.Axes:
    """Square polar inset of pSA directivity (RotD amplitude by azimuth).

    Sits over the top-right corner of ``ax``. pSA attenuates with period, so
    on a log-log spectrum the curve (and any comparison band) runs through
    the bottom-right -- long period, low amplitude -- while top-right (long
    period, high amplitude) is the one corner attenuation guarantees stays
    empty, whatever the station.

    One curve per requested period, each mirrored to the full circle (see
    :func:`read_directivity`) and normalised to its own RotD100 -- periods
    here span more than an order of magnitude in absolute amplitude, and a
    directivity plot is about the *shape* of the radiation pattern, not its
    level, so one shared radial scale is what makes the four comparable at a
    glance. The actual RotD100 (g) is given alongside each period in the
    legend so the normalisation does not throw that number away.

    Angle 0 is the simulation's 000 component axis, increasing counter-
    clockwise towards 090 -- the same convention `_core._psa_rotd180` (in
    IM_calculation) rotates through, not compass bearing.
    """
    axins = ax.inset_axes((0.68, 0.68, 0.30, 0.30), projection="polar")
    angles = np.deg2rad(np.append(np.arange(360), 360))
    handles = []
    for period, colour in zip(requested_periods, DIRECTIVITY_COLOURS):
        index = int(np.argmin(np.abs(periods - period)))
        resolved, values = periods[index], curve[index]
        peak = float(values.max())
        wrapped = np.append(values, values[0]) / peak
        axins.plot(angles, wrapped, color=colour, lw=1.1, zorder=3)
        handles.append(
            plt.Line2D(
                [], [], color=colour, lw=1.6, label=f"{resolved:g} s ({peak:.3g} g)"
            )
        )
    axins.set_theta_zero_location("E")
    axins.set_theta_direction(1)
    axins.set_ylim(0, 1.05)
    axins.set_yticks([0.5, 1.0])
    axins.set_yticklabels(["0.5", "1.0"], fontsize=5)
    axins.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    axins.set_xticklabels(["000", "", "090", "", "", "", "", ""])
    axins.tick_params(labelsize=5, pad=1)
    axins.grid(lw=0.3, alpha=0.6, color="#9a9a9a")
    axins.set_facecolor("white")
    axins.patch.set_alpha(0.88)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
    axins.set_title("directivity (RotD, normalised)", fontsize=5.5, pad=2)
    axins.legend(
        handles=handles,
        fontsize=5,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.3, -0.06),
        labelspacing=0.3,
    )
    return axins


def psa_spectrum(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    station: Annotated[str, typer.Argument(help="Station to review")],
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            exists=True,
            dir_okay=False,
            help="Second IM file: compare two simulations at this station",
        ),
    ] = None,
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip: overlay this station's recorded pSA",
        ),
    ] = None,
    empirical: Annotated[
        str | None,
        typer.Option(
            "--empirical", help="Empirical model to compare against, e.g. NSHM2022"
        ),
    ] = None,
    name: Annotated[
        list[str] | None,
        typer.Option(
            "--name", help="Name for each series, in order; repeat for the second"
        ),
    ] = None,
    component: Annotated[
        str | None,
        typer.Option(help="Component of motion (default rotd50)"),
    ] = None,
    directivity_period: Annotated[
        list[float],
        typer.Option(
            "--directivity-period",
            help="Periods (s) to draw in the directivity inset; repeat up to 4",
        ),
    ] = [0.5, 1.0, 5.0, 10.0],  # noqa: B006
    coastline: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Coastline file for the inset map (.b64 blob or .geojson)",
        ),
    ] = None,
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
    """pSA analogue of `spectra`: pSA vs period at one station.

    Structured like `spectra` (an EAS Fourier-spectrum comparison) -- an info
    panel, a locator map, the spectrum itself, and a log-ratio panel beneath
    it -- but on the period axis. The simulation can be compared against a
    second simulation (``--diff``, on its own), or against ``--observed`` and
    ``--empirical`` together, since a recording and a GMM prediction are the
    two things worth showing side by side at one station. When the file
    carries the full-angle RotD curve (``im-calc --full-rotd180``), a square
    directivity inset sits in the spectrum panel's top-right corner -- pSA
    attenuates fast with period, so the top-right is otherwise the emptiest
    part of the panel, unlike the bottom-right which the long-period tail
    runs through.
    """
    if diff is not None and (observed is not None or empirical is not None):
        raise typer.BadParameter(
            "--diff is mutually exclusive with --observed/--empirical "
            "(which may be combined with each other)"
        )
    if len(directivity_period) > 4:
        raise typer.BadParameter("at most 4 --directivity-period values fit the inset")

    component = component or DEFAULT_COMPONENT["pSA"]
    tree = open_ims(im_file)
    if "pSA" not in tree.children:
        raise typer.BadParameter(f"{im_file} has no pSA group")
    node = tree["pSA"]
    if station not in set(node.station.values):
        raise typer.BadParameter(f"{station!r} is not a station in {im_file}")
    # rotd180 is the full-angle directivity curve (int16-encoded, an "angle"
    # dim instead of a plain per-station scalar), not a selectable component.
    if component == "rotd180" or component not in node.data_vars:
        raise typer.BadParameter(
            f"component {component!r} not in pSA. "
            f"Available: {[str(c) for c in node.data_vars if c != 'rotd180']}"
        )

    sim_da = node[component].sel(station=station)
    sim_period, sim_psa = sim_da.period.values, sim_da.values

    # First --name overrides the simulation's own label; any more are taken,
    # in order, by whichever of --diff/--observed/--empirical were passed.
    names = list(name) if name else []
    sim_label = names[0] if names else "simulation"
    extra_names = iter(names[1:])

    def next_name(default: str) -> str:
        return next(extra_names, default)

    # Each entry compares the simulation against one reference series: a
    # second run, a recording, or a GMM. --observed and --empirical can both
    # be given, since a recording and a model prediction are the two things
    # worth showing at once; --diff (a second simulation) stands alone.
    comparisons = []
    if diff is not None:
        other_tree = open_ims(diff)
        if "pSA" not in other_tree.children:
            raise typer.BadParameter(f"{diff} has no pSA group")
        other_node = other_tree["pSA"]
        if station not in set(other_node.station.values):
            raise typer.BadParameter(f"{station!r} is not a station in {diff}")
        other_da = other_node[component].sel(station=station)
        comparisons.append(
            {
                "name": next_name("sim 2"),
                "colour": COMPARISON_BLUE,
                "style": "-",
                "period": other_da.period.values,
                "psa": other_da.values,
                "band": None,
            }
        )
    if observed is not None:
        recordings, obs_periods_all = read_observed_spectra(
            observed, component, metric=None
        )
        matches = np.flatnonzero(recordings["name"] == station)
        if matches.size == 0:
            raise typer.BadParameter(f"{station!r} has no record in {observed}")
        comparisons.append(
            {
                "name": next_name("observed"),
                "colour": OBSERVED_GREEN,
                "style": "-",
                "period": obs_periods_all,
                "psa": recordings["spectrum"][matches[0]],
                "band": None,
            }
        )
    if empirical is not None:
        mean, sigma = select_empirical(tree, "pSA", empirical, {})
        mean, sigma = mean.sel(station=station), sigma.sel(station=station)
        comparisons.append(
            {
                "name": next_name(empirical),
                "colour": EMPIRICAL_BLUE,
                "style": "--",
                "period": mean.period.values,
                "psa": np.exp(mean.values),
                "band": (
                    np.exp(mean.values - sigma.values),
                    np.exp(mean.values + sigma.values),
                ),
            }
        )

    for entry in comparisons:
        entry["on_grid"] = resample_spectrum(sim_period, sim_psa, entry["period"])
        with np.errstate(divide="ignore", invalid="ignore"):
            entry["residual"] = np.log(entry["psa"]) - np.log(entry["on_grid"])

    display = Display.for_figure((11, 8.5), dpi, display_height, viewing_distance)
    display.report((11, 8.5))
    # The info block and the locator map are for reading up close. Enlarged
    # for distance they would take half the canvas to say what a caption can
    # say instead, so the mosaic drops to the two panels carrying the result.
    detailed = display.detailed
    fig, axd = plt.subplot_mosaic(
        [["info", "map"], ["spectrum", "spectrum"], ["ratio", "ratio"]]
        if detailed
        else [["spectrum"], ["ratio"]],
        figsize=display.size,
        height_ratios=[0.7, 1.7, 1] if detailed else [1.7, 1],
        width_ratios=[4.5, 1] if detailed else None,
        layout="constrained",
    )
    spectrum, ratio = axd["spectrum"], axd["ratio"]
    ratio.sharex(spectrum)
    spectrum.tick_params(labelbottom=False)

    lat, lon = float(sim_da.latitude), float(sim_da.longitude)
    if "map" in axd:
        draw_inset_map(axd["map"], lon, lat, load_coastline(coastline))

    # A single plain (bandless) comparison keeps the sign-shaded misfit fill;
    # with two comparisons at once (observed + empirical) that shading would
    # overlap into a muddle, so it is skipped and the lines are left to speak
    # for themselves -- the band already carries the empirical comparison's
    # spread, and colour ties each curve to its panel-2 residual.
    if len(comparisons) == 1 and comparisons[0]["band"] is None:
        entry = comparisons[0]
        for where, colour in (
            (entry["psa"] > entry["on_grid"], entry["colour"]),
            (entry["psa"] <= entry["on_grid"], "#4a4a4a"),
        ):
            spectrum.fill_between(
                entry["period"], entry["on_grid"], entry["psa"], where=where,
                color=colour, alpha=0.18, lw=0, zorder=1,
            )
    for entry in comparisons:
        if entry["band"] is not None:
            spectrum.fill_between(
                entry["period"], entry["band"][0], entry["band"][1],
                color=entry["colour"], alpha=0.20, lw=0, zorder=1,
            )
    spectrum.plot(sim_period, sim_psa, color="black", lw=1.0, zorder=3, label=sim_label)
    for entry in comparisons:
        spectrum.plot(
            entry["period"], entry["psa"], color=entry["colour"], lw=1.0,
            ls=entry["style"], zorder=4, label=entry["name"],
        )

    spectrum.set_xscale("log")
    spectrum.set_yscale("log")
    # A label longer than the axis beside it runs off the canvas once enlarged.
    spectrum.set_ylabel(f"pSA, {component} (g)" if detailed else "pSA (g)")
    if not detailed:
        # Nothing else names the station once the info block is gone.
        spectrum.set_title(station)
    spectrum.grid(
        True,
        which="both" if detailed else "major",
        lw=display.mark(0.3),
        color="#e8e8e8",
    )
    spectrum.legend(fontsize=8, frameon=False, loc="lower left")

    directivity = read_directivity(node, station)
    if directivity is not None:
        draw_directivity_inset(spectrum, *directivity, directivity_period)
    else:
        console_warn(
            f"{im_file} carries no rotd180 variable (run im-calc with "
            "--full-rotd180 to get the directivity inset)"
        )

    ratio.axhline(0, color="#6b6b6b", lw=0.8, zorder=3)
    for guide, mark in ((np.log(2), "×2"), (-np.log(2), "÷2")):
        ratio.axhline(guide, color="#9a9a9a", lw=0.6, ls=":", zorder=3)
        ratio.annotate(
            mark,
            xy=(0.998, guide),
            xycoords=("axes fraction", "data"),
            fontsize=6,
            color="#6b6b6b",
            ha="right",
            va="bottom",
        )
    if len(comparisons) == 1:
        entry = comparisons[0]
        for where, colour in (
            (entry["residual"] > 0, entry["colour"]),
            (entry["residual"] <= 0, "#4a4a4a"),
        ):
            ratio.fill_between(
                entry["period"], 0, entry["residual"], where=where, color=colour,
                alpha=0.35, lw=0, zorder=1,
            )
        ratio.plot(entry["period"], entry["residual"], color="black", lw=0.5, zorder=2)
    else:
        for entry in comparisons:
            ratio.plot(
                entry["period"], entry["residual"], color=entry["colour"],
                ls=entry["style"], lw=1.2, zorder=2,
            )
    ratio.set_xscale("log")
    ratio.set_xlabel("period (s)")
    if not detailed:
        ratio.set_ylabel("ln ratio")
    elif len(comparisons) == 1:
        ratio.set_ylabel(f"ln[pSA({comparisons[0]['name']}) / pSA({sim_label})]")
    else:
        ratio.set_ylabel(f"ln[pSA(reference) / pSA({sim_label})]")
    ratio.grid(
        True,
        which="both" if detailed else "major",
        lw=display.mark(0.3),
        color="#e8e8e8",
    )
    for axis in (spectrum, ratio):
        for spine in axis.spines.values():
            spine.set_linewidth(display.mark(0.6))
        axis.tick_params(labelsize=8)
        if not detailed:
            # Enlarged, the minor ticks of a log decade merge into a comb and
            # the ratio panel's y labels run into one another.
            axis.tick_params(which="minor", length=0)
            axis.yaxis.set_major_locator(
                LogLocator(numticks=display.ticks(6))
                if axis.get_yscale() == "log"
                else MaxNLocator(nbins=display.ticks(6))
            )

    summary = [f"{station}    {lat:.3f}, {lon:.3f}"]
    details = []
    for key, text, fmt in (
        ("rrup", "Rrup", "{:.0f} km"),
        ("vs30", "Vs30", "{:.0f} m/s"),
    ):
        if key in sim_da.coords:
            details.append(f"{text} {fmt.format(float(sim_da[key]))}")
    if details:
        summary.append("    ".join(details))
    if event := tree.attrs.get("event"):
        summary.append(f"{event}  M{float(tree.attrs.get('magnitude', float('nan'))):.2f}")
    if "info" in axd:
        axd["info"].axis("off")
        axd["info"].text(
            0,
            1,
            "\n".join(summary),
            family="monospace",
            fontsize=8,
            va="top",
            ha="left",
            transform=axd["info"].transAxes,
        )

    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()
