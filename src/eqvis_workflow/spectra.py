"""``spectra``: Fourier spectra (EAS) at one station.

Against a recording, another run, or both, with the bias reported band by band
so a run that is right in the middle of the spectrum and wrong at the edges
cannot read as right overall.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr

from .console import console_warn
from .constants import COMPARISON_BLUE, OBSERVED_GREEN, SPECTRAL_BANDS
from .data import comparison_labels, open_ims
from .flatfile import read_observed_spectrum
from .geography import draw_inset_map, load_coastline


def resample_spectrum(
    frequency: np.ndarray, amplitude: np.ndarray, onto: np.ndarray
) -> np.ndarray:
    """Put a spectrum on another frequency grid, interpolating in log-log space."""
    if len(frequency) == len(onto) and np.allclose(frequency, onto):
        return amplitude
    with np.errstate(divide="ignore"):
        return np.exp(np.interp(np.log(onto), np.log(frequency), np.log(amplitude)))


def band_bias(
    frequency: np.ndarray,
    residual: np.ndarray,
    high_pass: float | None = None,
    low_pass: float | None = None,
) -> list[tuple[float, float, float]]:
    """Mean log residual within each decade, as (low, high, mean) triples.

    The reported bounds are the decade clipped to the usable band, so a decade
    only half covered by the recording is not labelled as if it were whole.
    """
    summary = []
    for low, high in SPECTRAL_BANDS:
        inside = (frequency >= low) & (frequency < high) & np.isfinite(residual)
        if inside.any():
            summary.append(
                (
                    max(low, high_pass or low),
                    min(high, low_pass or high),
                    float(residual[inside].mean()),
                )
            )
    return summary


def usable_band(
    row: dict[str, str], fmax: float | None
) -> tuple[float | None, float | None]:
    """Frequency range over which the recording is signal rather than filter.

    The low corner is the record's horizontal high-pass. GeoNet flatfiles carry
    no upper corner (``fmax_*`` and ``LPF_*`` are empty), so the top of the band
    has to come from the caller; without it the recorded spectrum runs into its
    own anti-alias roll-off.
    """
    corners: list[float | None] = []
    for key in ("HPF_h", "LPF_h"):
        try:
            corners.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            corners.append(None)
    return corners[0], fmax if fmax is not None else corners[1]


def spectra(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    station: Annotated[str, typer.Argument(help="Station to review")],
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip: overlay the recorded EAS",
        ),
    ] = None,
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            exists=True,
            dir_okay=False,
            help="Second IM file: compare two simulations at this station",
        ),
    ] = None,
    name: Annotated[
        list[str] | None,
        typer.Option(
            "--name", help="Name for each series, in order; repeat for the second"
        ),
    ] = None,
    fmax: Annotated[
        float | None,
        typer.Option(
            help="Upper usable frequency of the recording in Hz; above this the "
            "record is anti-alias roll-off, and the flatfile does not say where "
            "it starts",
        ),
    ] = None,
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
    dpi: Annotated[int, typer.Option(help="Output resolution")] = 300,
    transition: Annotated[float, typer.Option(help="Mark transition frequency")] = 1.0,
):
    """Compare Fourier spectra (EAS) at one station, against a record or another run."""
    if diff is not None and observed is not None:
        raise typer.BadParameter("--diff and --observed are mutually exclusive")

    def read_simulated_eas(path: Path) -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
        tree = open_ims(path)
        if "FAS" not in tree.children:
            raise typer.BadParameter(f"{path} has no FAS group")
        node = tree["FAS"]
        if station not in set(node.station.values):
            raise typer.BadParameter(f"{station!r} is not a station in {path}")
        if "eas" not in node.data_vars:
            raise typer.BadParameter(
                f"{path} FAS has no eas component. "
                f"Available: {[str(c) for c in node.data_vars]}"
            )
        eas = node["eas"].sel(station=station)
        return node.frequency.values, eas.values, eas

    sim_frequency, sim_amplitude, sim_eas = read_simulated_eas(im_file)
    labels = comparison_labels(
        name, ("sim 1", "sim 2") if diff is not None else ("simulation", "observed")
    )
    compare_colour = COMPARISON_BLUE if diff is not None else OBSERVED_GREEN

    obs_frequency = obs_amplitude = residual = None
    row: dict[str, str] = {}
    if diff is not None:
        obs_frequency, obs_amplitude, _ = read_simulated_eas(diff)
    elif observed is not None:
        record = read_observed_spectrum(observed, station)
        if record is None:
            console_warn(f"{station} has no eas record in {observed}")
        else:
            obs_frequency, obs_amplitude, row = record
    if obs_amplitude is not None:
        on_grid = resample_spectrum(sim_frequency, sim_amplitude, obs_frequency)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = np.log(obs_amplitude) - np.log(on_grid)

    fig, axd = plt.subplot_mosaic(
        [["info", "map"], ["spectrum", "spectrum"], ["ratio", "ratio"]],
        figsize=(11, 8.5),
        height_ratios=[0.7, 1.7, 1],
        width_ratios=[4.5, 1],
        layout="constrained",
    )
    spectrum, ratio = axd["spectrum"], axd["ratio"]
    ratio.sharex(spectrum)
    spectrum.tick_params(labelbottom=False)

    lat = float(sim_eas.latitude)
    lon = float(sim_eas.longitude)
    draw_inset_map(axd["map"], lon, lat, load_coastline(coastline))

    # The two spectra, with the gap between them shaded by who is larger: the
    # colour tells you the sign of the misfit without reading off two curves.
    if obs_amplitude is not None:
        on_grid = resample_spectrum(sim_frequency, sim_amplitude, obs_frequency)
        for where, colour in (
            (obs_amplitude > on_grid, compare_colour),
            (obs_amplitude <= on_grid, "#4a4a4a"),
        ):
            spectrum.fill_between(
                obs_frequency,
                on_grid,
                obs_amplitude,
                where=where,
                color=colour,
                alpha=0.18,
                lw=0,
                zorder=1,
            )
    spectrum.plot(
        sim_frequency, sim_amplitude, color="black", lw=1.0, zorder=3, label=labels[0]
    )
    if obs_amplitude is not None:
        spectrum.plot(
            obs_frequency,
            obs_amplitude,
            color=compare_colour,
            lw=1.0,
            zorder=4,
            label=labels[1],
        )

    # Outside the record's usable band the recorded spectrum is filter roll-off
    # and noise, not signal, so grey it out rather than inviting comparison.
    high_pass, low_pass = usable_band(row, fmax)
    if observed is not None and obs_amplitude is not None and low_pass is None:
        console_warn(
            "eas table has no upper usable corner, so the top decade includes the "
            "recording's anti-alias roll-off; pass --fmax to trim it"
        )
    # Statistics only count frequencies where the recording carries signal.
    scored = residual
    if residual is not None:
        inside = np.isfinite(residual)
        if high_pass:
            inside &= obs_frequency >= high_pass
        if low_pass:
            inside &= obs_frequency <= low_pass
        scored = np.where(inside, residual, np.nan)
    for axis in (spectrum, ratio):
        if high_pass:
            axis.axvspan(sim_frequency.min(), high_pass, color="#f0f0f0", zorder=0)
        if low_pass:
            axis.axvspan(low_pass, sim_frequency.max(), color="#f0f0f0", zorder=0)
        # On both panels, but labelled once so the legend has a single entry.
        axis.axvline(
            transition,
            color="red",
            zorder=2,
            linestyle="--",
            label="transition frequency" if axis is spectrum else None,
        )
    spectrum.set_xscale("log")
    spectrum.set_yscale("log")
    spectrum.set_ylabel("EAS (g·s)")
    spectrum.grid(True, which="both", lw=0.3, color="#e8e8e8")
    spectrum.legend(fontsize=8, frameon=False, loc="lower left")

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
    if residual is not None:
        for where, colour in (
            (residual > 0, compare_colour),
            (residual <= 0, "#4a4a4a"),
        ):
            ratio.fill_between(
                obs_frequency,
                0,
                residual,
                where=where,
                color=colour,
                alpha=0.35,
                lw=0,
                zorder=1,
            )
        ratio.plot(obs_frequency, residual, color="black", lw=0.5, zorder=2)
        # Per-decade means: the trend, separated from the wiggle.
        for low, high, mean in band_bias(obs_frequency, scored, high_pass, low_pass):
            ratio.plot(
                [low, high],
                [mean, mean],
                color="#c1272d",
                lw=2.0,
                solid_capstyle="butt",
                zorder=5,
            )
    ratio.set_xscale("log")
    ratio.set_xlabel("frequency (Hz)")
    ratio.set_ylabel(f"ln[EAS({labels[1]}) / EAS({labels[0]})]")
    ratio.grid(True, which="both", lw=0.3, color="#e8e8e8")
    for axis in (spectrum, ratio):
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
        axis.tick_params(labelsize=8)

    summary = [f"{station}    {lat:.3f}, {lon:.3f}"]
    details = []
    for key, text, fmt in (
        ("r_rup", "Rrup", "{:.0f} km"),
        ("Vs30", "Vs30", "{:.0f} m/s"),
    ):
        try:
            details.append(f"{text} {fmt.format(float(row[key]))}")
        except (KeyError, TypeError, ValueError):
            continue
    if not details and "rrup" in sim_eas.coords:
        details.append(f"Rrup {float(sim_eas.rrup):.0f} km")
    if details:
        summary.append("    ".join(details))
    if high_pass or low_pass:
        summary.append(
            f"usable {high_pass or sim_frequency.min():g}"
            f"-{low_pass or sim_frequency.max():g} Hz"
        )
    if residual is not None:
        summary.append("")
        summary.append(f"mean ln[{labels[1]}/{labels[0]}] by decade")
        for low, high, mean in band_bias(obs_frequency, scored, high_pass, low_pass):
            summary.append(f"  {low:g}-{high:g} Hz {mean:+.2f}  (x{np.exp(mean):.2f})")
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
        fig.savefig(output, dpi=dpi)
        print(f"wrote {output}")
    else:
        plt.show()
