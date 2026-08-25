"""``waveform``: one station's time series.

Acceleration, velocity and the Husid plot, against a recording or a second
simulation. The traces are offset rather than overlaid, since what matters is
usually the arrival and the envelope rather than a cycle-by-cycle match.
"""

import io
import zipfile
from pathlib import Path
from typing import Annotated

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
)
from scipy.integrate import cumulative_trapezoid

from .console import console_warn
from .constants import (
    ARIAS_LEVELS,
    COMPARISON_BLUE,
    GRAVITY_CM_S2,
    OBSERVED_GREEN,
    WAVEFORM_COMPONENTS,
)
from .data import comparison_labels
from .geography import draw_inset_map, load_coastline


def open_waveforms(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="h5netcdf", mask_and_scale=False)


def read_simulated_waveform(
    path: Path, station: str, lag: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Acceleration traces (g) for one station, on the observed time base.

    The broadband file's own time coordinate runs from before the origin; the
    caller's ``lag`` shifts it onto the recordings' clock, per the convention
    that simulation time minus the lag is observed time.
    """
    bb = open_waveforms(path)
    if station not in set(bb.station.values):
        raise typer.BadParameter(f"{station!r} is not a station in {path}")
    traces = bb.waveform.sel(station=station)
    return bb.time.values - lag, {
        component: traces.sel(component=component).values
        for component in WAVEFORM_COMPONENTS
    }


def read_observed_waveform(
    archive: Path, station: str
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Acceleration traces (g) for one station from a GeoNet waveform zip.

    Returns None when the station was not recorded. Stations with more than one
    instrument keep the first record in sorted order, which is named so the
    caller can say which one it drew.
    """
    with zipfile.ZipFile(archive) as zf:
        prefixes = sorted(
            {
                name.rsplit(".", 1)[0]
                for name in zf.namelist()
                if name.startswith(f"{station}/")
                and name.rsplit(".", 1)[-1] in set(WAVEFORM_COMPONENTS.values())
            }
        )
        if not prefixes:
            return None
        if len(prefixes) > 1:
            console_warn(
                f"{station} has {len(prefixes)} records; plotting "
                f"{Path(prefixes[0]).name}"
            )
        traces, time = {}, None
        for component, suffix in WAVEFORM_COMPONENTS.items():
            with zf.open(f"{prefixes[0]}.{suffix}") as raw:
                # Two header lines: a title, then "npts dt ..."; then the samples.
                lines = io.TextIOWrapper(raw, encoding="utf-8").read().split("\n")
            npts, dt = int(lines[1].split()[0]), float(lines[1].split()[1])
            traces[component] = np.fromstring(" ".join(lines[2:]), sep=" ")[:npts]
            time = np.arange(npts) * dt
    return time, traces


def arias_markers(acc: np.ndarray, time: np.ndarray) -> np.ndarray | None:
    """Times at which Arias intensity reaches 5%, 75% and 95% of its total."""
    arias = np.cumsum(acc.astype(float) ** 2)
    if arias[-1] <= 0:
        return None
    return np.interp(ARIAS_LEVELS, arias / arias[-1], time)


def husid(acc: np.ndarray, time: np.ndarray, onto: np.ndarray) -> np.ndarray | None:
    """Normalised cumulative Arias intensity, resampled onto a shared time base.

    Monotonic and smooth, so unlike the raw traces two of these can be drawn on
    one axis and the area between them read directly as the difference in how
    the energy arrived. Clamped outside the record rather than extrapolated.
    """
    arias = np.cumsum(acc.astype(float) ** 2)
    if arias[-1] <= 0:
        return None
    return np.interp(onto, time, arias / arias[-1], left=0.0, right=1.0)


def nice_amplitude(scale: float) -> float:
    """Largest round number no bigger than ``scale``, for an amplitude tick.

    Stacked traces have no natural y axis of their own, so each baseline gets
    ticks at +/- this value: a round number is far easier to read a wiggle
    against than the peak amplitude itself.
    """
    candidates = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, scale)
    inside = [value for value in candidates if 0 < value <= scale]
    return inside[-1] if inside else scale


def draw_offset_trace(
    ax: plt.Axes,
    time: np.ndarray,
    acc: np.ndarray,
    baseline: float,
    colour: str,
    scale: float,
) -> np.ndarray | None:
    """Draw one filled trace on its own baseline, with its peak and Arias marks.

    Simulated and recorded traces get separate baselines on a shared amplitude
    scale: overlaying two dense oscillatory traces makes neither readable, while
    stacking keeps them comparable sample for sample.
    """
    ax.axhline(baseline, color="#d0d0d0", lw=0.5, zorder=1)
    ax.fill_between(
        time, baseline, baseline + acc, color=colour, alpha=0.45, lw=0, zorder=2
    )
    ax.plot(time, baseline + acc, color=colour, lw=0.3, zorder=3)
    peak = int(np.argmax(np.abs(acc)))
    ax.plot(time[peak], baseline + acc[peak], marker="o", ms=4, color=colour, zorder=5)
    marks = arias_markers(acc, time)
    if marks is not None:
        ax.vlines(
            marks,
            baseline - scale,
            baseline + scale,
            colors=colour,
            lw=0.7,
            ls="--",
            alpha=0.8,
            zorder=4,
        )
    return marks


def to_velocity(acc: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Integrate acceleration in g to velocity in cm/s, about a zero baseline."""
    dt = float(np.median(np.diff(time)))
    return cumulative_trapezoid((acc - acc.mean()) * GRAVITY_CM_S2, dx=dt, initial=0.0)


def fourier_spectrum(
    acc: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fourier amplitude spectrum (g*s) against frequency (Hz)."""
    dt = float(np.median(np.diff(time)))
    amplitude = np.abs(np.fft.rfft(acc - acc.mean())) * dt
    return np.fft.rfftfreq(len(acc), dt), amplitude


def waveform(
    bb_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Broadband waveform file"),
    ],
    station: Annotated[str, typer.Argument(help="Station to review")],
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet waveform zip: overlay the recording, if this station has one",
        ),
    ] = None,
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            exists=True,
            dir_okay=False,
            help="Second broadband file: compare two simulations at this station",
        ),
    ] = None,
    name: Annotated[
        list[str] | None,
        typer.Option(
            "--name", help="Name for each series, in order; repeat for the second"
        ),
    ] = None,
    lag: Annotated[
        float,
        typer.Option(
            help="Phase lag in seconds: simulation time minus this is observed time"
        ),
    ] = 3.0,
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
):
    """Review one station's waveforms against a recording or another simulation."""
    if diff is not None and observed is not None:
        raise typer.BadParameter("--diff and --observed are mutually exclusive")
    bb = open_waveforms(bb_file)
    sim_time, sim = read_simulated_waveform(bb_file, station, lag)
    lon = float(bb.longitude.sel(station=station))
    lat = float(bb.latitude.sel(station=station))

    labels = comparison_labels(
        name, ("sim 1", "sim 2") if diff is not None else ("sim", "obs")
    )
    compare_colour = COMPARISON_BLUE if diff is not None else OBSERVED_GREEN
    obs_time, obs = None, None
    if diff is not None:
        # The same lag on both, so the shared axis keeps its meaning and the
        # comparison between the two runs is untouched.
        obs_time, obs = read_simulated_waveform(diff, station, lag)
    elif observed is not None:
        record = read_observed_waveform(observed, station)
        if record is None:
            console_warn(f"{station} has no recording in {observed}")
        else:
            obs_time, obs = record

    # The energy panel only says something when there are two records to compare.
    energy = obs is not None
    mosaic = [["info", "map"], *([[c, c] for c in WAVEFORM_COMPONENTS])]
    heights = [0.75, 1, 1, 1]
    if energy:
        mosaic.append(["husid", "husid"])
        heights.append(1.2)
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(11, 9.5 if energy else 8),
        height_ratios=heights,
        width_ratios=[4.5, 1],
        layout="constrained",
    )
    panels = [*WAVEFORM_COMPONENTS, *(["husid"] if energy else [])]
    for panel in panels[1:]:
        axd[panel].sharex(axd["x"])
    # sharex links the limits but leaves every panel labelled; only the bottom
    # one needs ticks.
    for panel in panels[:-1]:
        axd[panel].tick_params(labelbottom=False)
    draw_inset_map(axd["map"], lon, lat, load_coastline(coastline))

    summary = [f"{station}    {lat:.3f}, {lon:.3f}", ""]
    summary.append(f"{'':10} {labels[0]:>32}  {labels[1]:>32}")
    header = f"{'PGA (g)':>10} {'PGV (cm/s)':>10} {'Ds5-95 (s)':>10}"
    summary.append(f"{'':10} {header}  {header}")

    windows = []
    for component, suffix in WAVEFORM_COMPONENTS.items():
        ax = axd[component]
        row = [f"{component} ({suffix}){'':<2}"]
        # One amplitude scale per component, so the two traces stay comparable.
        scale = max(
            float(np.abs(sim[component]).max()),
            float(np.abs(obs[component]).max()) if obs is not None else 0.0,
        )
        # With something to compare against, the two traces sit on separate
        # baselines: overlaying two dense oscillatory traces makes neither
        # readable. Alone, there is no second baseline to leave room for, so the
        # trace is centred and takes the whole panel -- which is the difference
        # between a trace a third of the panel high and one filling it.
        offset = 1.35 * scale if obs is not None else 0.0
        baselines = ((offset, labels[0]), (-offset, labels[1])) if obs is not None \
            else ((0.0, labels[0]),)
        marks = {}
        for time, traces, colour, baseline, key in (
            (sim_time, sim, "black", offset, "sim"),
            (obs_time, obs, compare_colour, -offset, "obs"),
        ):
            if traces is None:
                row.append(f"{'-':>10} {'-':>10} {'-':>10}")
                continue
            acc = traces[component]
            found = draw_offset_trace(ax, time, acc, baseline, colour, scale)
            marks[key] = found
            if found is not None:
                windows.append(found)
            duration = f"{found[2] - found[0]:.1f}" if found is not None else "-"
            row.append(
                f"{np.abs(acc).max():>10.4g} "
                f"{np.abs(to_velocity(acc, time)).max():>10.4g} "
                f"{duration:>10}"
            )
        summary.append("  ".join(row))

        ax.set_ylim(-(offset + 1.1 * scale), offset + 1.1 * scale)
        ax.set_ylabel(f"{component} ({suffix})", fontsize=9)
        # Each baseline carries its own zero and a round +/- amplitude tick,
        # so any wiggle can be read in g off the nearest gridline.
        tick = nice_amplitude(scale)
        positions, tick_labels = [], []
        for baseline, _ in baselines:
            for delta in (tick, 0.0, -tick):
                positions.append(baseline + delta)
                tick_labels.append("0" if delta == 0 else f"{delta:+g}")
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.grid(True, axis="y", lw=0.3, color="#eeeeee")
        for baseline, text in baselines:
            ax.annotate(
                text,
                xy=(0.997, baseline),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                ha="right",
                va="bottom",
                color="#333333",
                path_effects=[
                    patheffects.withStroke(linewidth=2.5, foreground="white")
                ],
            )
        # The gap between the two baselines is free space by construction, so
        # the timing arrows live there instead of crowding the traces.
        if marks.get("sim") is not None and marks.get("obs") is not None:
            for index, text, level in ((0, "5%", 0.42), (2, "95%", -0.42)):
                start, end = marks["sim"][index], marks["obs"][index]
                ax.annotate(
                    "",
                    xy=(end, level * scale),
                    xytext=(start, level * scale),
                    arrowprops=dict(arrowstyle="->", color="#c1272d", lw=1.0),
                )
                ax.text(
                    (start + end) / 2,
                    level * scale,
                    f"{text} {end - start:+.1f}s",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    color="#c1272d",
                )
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="x", lw=0.3, color="#e8e8e8")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    if energy:
        # Cumulative Arias: smooth enough to overlay, so the shaded area between
        # them shows directly whether the recording gained its energy earlier
        # (green) or later (grey) than the simulation.
        ax = axd["husid"]
        grid = np.linspace(
            min(sim_time[0], obs_time[0]), max(sim_time[-1], obs_time[-1]), 4000
        )
        ticks = []
        for index, component in enumerate(WAVEFORM_COMPONENTS):
            base = (len(WAVEFORM_COMPONENTS) - 1 - index) * 1.25
            simulated = husid(sim[component], sim_time, grid)
            recorded = husid(obs[component], obs_time, grid)
            if simulated is None or recorded is None:
                continue
            for where, colour in (
                (recorded >= simulated, compare_colour),
                (recorded < simulated, "#4a4a4a"),
            ):
                ax.fill_between(
                    grid,
                    base + simulated,
                    base + recorded,
                    where=where,
                    color=colour,
                    alpha=0.35,
                    lw=0,
                    zorder=1,
                )
            ax.plot(grid, base + simulated, color="black", lw=0.9, zorder=3)
            ax.plot(grid, base + recorded, color=compare_colour, lw=0.9, zorder=3)
            for level, text in ((0.05, "5%"), (0.95, "95%")):
                ax.axhline(
                    base + level, color="#8c8c8c", lw=0.9, ls=(0, (4, 3)), zorder=2
                )
                ax.annotate(
                    text,
                    xy=(0.997, base + level),
                    xycoords=("axes fraction", "data"),
                    fontsize=7,
                    color="#5a5a5a",
                    ha="right",
                    va="bottom",
                    path_effects=[
                        patheffects.withStroke(linewidth=2.5, foreground="white")
                    ],
                )
            ticks.append((base + 0.5, component))
        ax.margins(y=0.06)
        ax.set_yticks([position for position, _ in ticks])
        ax.set_yticklabels([name for _, name in ticks], fontsize=8)
        ax.set_ylabel("cumulative Arias", fontsize=9)
        ax.grid(True, axis="x", lw=0.3, color="#e8e8e8")
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # Frame the shaking rather than the whole record, so the duration markers
    # and arrows are readable.
    if windows:
        span = np.array(windows)
        axd["x"].set_xlim(span[:, 0].min() - 10, span[:, 2].max() + 20)
    # Reading a time off the plot matters here, so label generously and carry
    # minor ticks between the labels.
    for panel in panels:
        axis = axd[panel]
        axis.xaxis.set_major_locator(MaxNLocator(nbins=18, steps=[1, 2, 5, 10]))
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.grid(True, which="minor", axis="x", lw=0.2, color="#f4f4f4")
        axis.tick_params(which="minor", length=2)
        axis.tick_params(which="major", length=4)
    axd[panels[-1]].set_xlabel("time (s)")
    fig.supylabel("acceleration (g)", fontsize=10)

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
    handles = [plt.Line2D([], [], color="black", lw=1, label=labels[0])]
    if obs is not None:
        handles.append(plt.Line2D([], [], color=compare_colour, lw=1, label=labels[1]))
    # Drawn in each trace's own colour, so the key shows it uncoloured.
    handles.append(
        plt.Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=5,
            color="#333333",
            label="peak acceleration",
        )
    )
    axd["info"].legend(
        handles=handles, fontsize=8, frameon=False, loc="lower left", ncols=len(handles)
    )

    if output is not None:
        fig.savefig(output, dpi=dpi)
        print(f"wrote {output}")
    else:
        plt.show()
