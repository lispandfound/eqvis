"""``radiation``: the residual against source azimuth, and the mechanism it implies.

The question this answers is whether the moment tensor a simulation was built
from points its lobes in the right directions. An empirical model has no
mechanism -- it is azimuthally symmetric apart from its own hanging-wall and
directivity terms -- so the recordings differenced against one carry whatever
azimuthal structure the earthquake actually radiated::

    eqvis radiation sw4/im.h5 --observed flatfiles.zip --empirical NSHM2022

Three things are drawn from that, and they are three different claims:

* **by azimuth** -- where each station sits, against the double-couple pattern
  the mechanism predicts. The eye's test, and the one that shows the gaps.
* **the sweep** -- the same correlation with the strike rotated through a full
  turn, the mechanism's dip and rake held. If the tensor is right the curve
  peaks near the tensor's own strike; if it peaks somewhere else, that is the
  strike the amplitudes prefer.
* **against period** -- the correlation period by period. It has to *grow* with
  period to be believed: the radiation pattern lives in the deterministic
  low-frequency solution, and above the hybrid crossover the stochastic high
  frequency deliberately randomises it.

Three limits are drawn rather than argued, because each of them can make this
figure say something it cannot support:

* **The two nodal planes are one pattern.** A double couple radiates identically
  from either plane, so this can never choose between them. The subtitle says so
  every time, and ``--strike/--dip/--rake`` is the way to ask about a *different*
  tensor rather than about the other plane of this one.
* **Azimuthal coverage.** The largest gap in station azimuth is measured and
  shaded. Half a focal sphere of stations cannot resolve a four-lobed pattern,
  and a peak in the sweep means much less when the lobes fall in the gap.
* **The takeoff angle is assumed, not known.** At regional distance the ray that
  carries the S wave leaves the source at an angle this command does not solve
  for, and the pattern moves with it -- so ``--takeoff`` is swept as a band
  rather than taken on faith at one value.
"""

import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
from matplotlib.colors import BoundaryNorm

from .bias import match_columns
from .console import console_warn
from .constants import DEFAULT_COMPONENT, EMPIRICAL_BLUE, OBSERVED_GREEN, SIM_ONE_BLACK
from .data import (
    Screen,
    open_ims,
    restrict_to_domain,
    select_empirical,
    supergrid,
)
from .display import Display
from .flatfile import read_observed_spectra
from .picks import read_pick_list, restrict_to_stations
from .stations import nearest_stations

# Below the hybrid crossover the ground motion is the deterministic solution and
# carries the mechanism; above it the stochastic high frequency has replaced the
# radiation pattern with a random one on purpose. Scoring a mechanism at 0.2 s
# therefore measures the high-frequency generator's random numbers, not the
# tensor -- so the default sits well clear of it and a shorter period warns.
DETERMINISTIC_PERIOD = 1.0

# Regional S from a shallow crustal source leaves the focal sphere well below
# the horizontal, and the pattern is not stationary under that angle -- so the
# figure carries a band over plausible angles rather than one curve. 50 degrees
# is the centre of it, not a solved value.
DEFAULT_TAKEOFF = 50.0
TAKEOFF_BAND = (40.0, 70.0)

# |F| never quite reaches zero on a real ray, and a node put through a log would
# take the whole axis with it. This floor is the amplitude a node is treated as
# having, well below anything the data can distinguish.
NODE_FLOOR = 1e-3

# The fixed-strike null is the same distribution whichever strike it is built
# at -- shuffling breaks the link to azimuth either way -- so one is named
# rather than the tensor's own passed through, which would read as though the
# null knew about the tensor.
DEFAULT_NULL_STRIKE = 0.0

# Enough draws to put a couple of significant figures on a p-value near 0.05
# without making the command slow enough that nobody runs it.
NULL_DRAWS = 2000

# The panels, in the order they are drawn when all are asked for. Named here so
# --panels can be validated against the set rather than against a string.
PANELS = ("polar", "azimuth", "sweep", "period")


def radiation_amplitude(
    strike: float, dip: float, rake: float, azimuth: np.ndarray, takeoff: float
) -> np.ndarray:
    """Far-field S amplitude of a double couple, as Aki & Richards write it.

    ``azimuth`` is degrees east of north from the source to the station and
    ``takeoff`` the angle the ray leaves the source at, measured from the
    downward vertical. The return is ``sqrt(F_SV^2 + F_SH^2)`` -- the total S
    amplitude rather than either component alone, because the measure being
    scored against it is RotD50, which is a horizontal amplitude with no
    fixed relation to the SV/SH split.
    """
    s, d, l = np.radians([strike, dip, rake])
    phi = np.radians(azimuth) - s
    i = math.radians(takeoff)
    sv = (
        np.sin(l) * np.cos(2 * d) * np.cos(2 * i) * np.sin(phi)
        - np.cos(l) * np.cos(d) * np.cos(2 * i) * np.cos(phi)
        + 0.5 * np.cos(l) * np.sin(d) * np.sin(2 * i) * np.sin(2 * phi)
        - 0.5 * np.sin(l) * np.sin(2 * d) * np.sin(2 * i) * (1 + np.sin(phi) ** 2)
    )
    sh = (
        np.cos(l) * np.cos(d) * np.cos(i) * np.sin(phi)
        + np.cos(l) * np.sin(d) * np.sin(i) * np.cos(2 * phi)
        + np.sin(l) * np.cos(2 * d) * np.cos(i) * np.cos(phi)
        - 0.5 * np.sin(l) * np.sin(2 * d) * np.sin(i) * np.sin(2 * phi)
    )
    return np.sqrt(sv**2 + sh**2)


def log_amplitude(
    strike: float, dip: float, rake: float, azimuth: np.ndarray, takeoff: float
) -> np.ndarray:
    """:func:`radiation_amplitude` in log space, where the residual lives.

    The residual is a log ratio, so the pattern has to be one too for a
    correlation between them to be a statement about amplitude rather than
    about the shape of the exponential.
    """
    return np.log(radiation_amplitude(strike, dip, rake, azimuth, takeoff) + NODE_FLOOR)


def source_azimuth(
    hypo_lon: float, hypo_lat: float, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Initial great-circle bearing from the hypocentre to each station, degrees."""
    dlon = np.radians(lon - hypo_lon)
    origin, station = math.radians(hypo_lat), np.radians(lat)
    return (
        np.degrees(
            np.arctan2(
                np.sin(dlon) * np.cos(station),
                math.cos(origin) * np.sin(station)
                - math.sin(origin) * np.cos(station) * np.cos(dlon),
            )
        )
        % 360
    )


def largest_gap(azimuth: np.ndarray) -> tuple[float, float, float]:
    """The widest unsampled wedge: its width, and the bearings it runs between.

    Reported because it is the figure's own limit. A four-lobed pattern has
    lobes 90 degrees apart, so a gap approaching that can hide a whole lobe --
    and a sweep that peaks confidently on the stations that remain is then
    fitting the half of the focal sphere that was sampled.
    """
    if azimuth.size < 2:
        return 360.0, 0.0, 360.0
    order = np.sort(azimuth)
    gaps = np.diff(np.concatenate([order, [order[0] + 360.0]]))
    widest = int(gaps.argmax())
    return float(gaps[widest]), float(order[widest]), float(
        (order[widest] + gaps[widest]) % 360
    )


def mechanism_from_attributes(attrs: dict) -> tuple[float, float, float]:
    """Strike, dip and rake of the source the IM file was written from.

    Dip and rake are attributes outright. Strike is not: what the file carries
    is the top edge as a line, whose *direction* is a property of how the
    geometry happened to be written down rather than a convention. It is
    recovered the way :func:`eqvis_workflow.rupture.orient_to_strike` recovers a
    CFM trace's -- the sense that leaves the dip 90 degrees clockwise, with the
    dip direction taken from the side the surface projection falls on.
    """
    trace = shapely.from_wkt(str(attrs["trace"]))
    (x0, y0), (x1, y1) = trace.coords[0], trace.coords[-1]
    scale = math.cos(math.radians((y0 + y1) / 2))
    bearing = math.degrees(math.atan2((x1 - x0) * scale, y1 - y0)) % 360

    # Which side the plane dips to: the surface projection extends down-dip from
    # the trace, so its centroid is on the down-dip side of the trace midpoint.
    source = shapely.from_wkt(str(attrs["source"]))
    centre = source.centroid
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    dip_direction = (
        math.degrees(
            math.atan2((centre.x - mid_x) * scale, centre.y - mid_y)
        )
        % 360
    )
    error = ((dip_direction - (bearing + 90)) + 180) % 360 - 180
    strike = bearing if abs(error) <= 90 else (bearing + 180) % 360
    return strike % 360, float(attrs["dip"]), float(attrs["rake"])


def correlation(residual: np.ndarray, pattern: np.ndarray) -> float:
    """Pearson correlation over the stations both are finite at."""
    known = np.isfinite(residual) & np.isfinite(pattern)
    if known.sum() < 3 or np.ptp(pattern[known]) == 0:
        return float("nan")
    return float(np.corrcoef(residual[known], pattern[known])[0, 1])


def strike_sweep(
    residual: np.ndarray,
    azimuth: np.ndarray,
    dip: float,
    rake: float,
    takeoff: float,
    step: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation against assumed strike, the dip and rake held fixed.

    The sweep is over strike alone because that is the parameter the station
    geometry constrains: dip and rake move the lobes' amplitudes, strike moves
    where they point, and only the second is what an azimuthal distribution of
    amplitudes can see.
    """
    strikes = np.arange(0.0, 360.0, step)
    return strikes, np.array(
        [correlation(residual, log_amplitude(s, dip, rake, azimuth, takeoff))
         for s in strikes]
    )


def sweep_null(
    residual: np.ndarray,
    azimuth: np.ndarray,
    dip: float,
    rake: float,
    takeoff: float,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """What a sweep peak and a fixed-strike correlation are worth under no signal.

    The residuals are shuffled between stations, which destroys any relation to
    azimuth while keeping their distribution and the station geometry exactly as
    they are. Two null distributions come back: the peak of a whole sweep, and
    the correlation at one strike fixed in advance.

    They answer different questions and the difference is the point. Sweeping
    180 strikes and reporting the best is a search, and a search over a smooth
    curve finds a high correlation in pure noise most of the time -- so a peak
    has to beat the *peak* null, not the fixed one. Reading the correlation at
    the tensor's own strike is a single hypothesis stated before looking, and it
    is scored against the fixed null.
    """
    known = np.isfinite(residual)
    values = residual[known]
    rng = np.random.default_rng(seed)
    peaks = np.empty(draws)
    fixed = np.empty(draws)
    reference = log_amplitude(DEFAULT_NULL_STRIKE, dip, rake, azimuth, takeoff)
    for draw in range(draws):
        shuffled = np.full(residual.shape, np.nan)
        shuffled[known] = rng.permutation(values)
        peaks[draw] = np.nanmax(
            strike_sweep(shuffled, azimuth, dip, rake, takeoff)[1]
        )
        fixed[draw] = correlation(shuffled, reference)
    return peaks, fixed


def angular_offset(a: float, b: float) -> float:
    """Separation of two strikes, modulo the 180-degree ambiguity of a plane."""
    raw = abs(a - b) % 360
    raw = min(raw, 360 - raw)
    return min(raw, abs(180 - raw))


def radiation(
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
    empirical: Annotated[
        str,
        typer.Option(
            "--empirical",
            help="Azimuthally symmetric model to difference the recordings "
            "against, so what is left is the mechanism's own structure",
        ),
    ] = "NSHM2022",
    period: Annotated[
        float,
        typer.Option(help="pSA period in seconds to score the mechanism at"),
    ] = 3.0,
    takeoff: Annotated[
        float,
        typer.Option(
            "--takeoff",
            help="Angle the S ray leaves the source at, degrees from the "
            "downward vertical. Assumed, not solved: the panels carry a band "
            "over plausible angles beside the value chosen here",
        ),
    ] = DEFAULT_TAKEOFF,
    strike: Annotated[
        float | None,
        typer.Option(help="Strike to test, overriding the file's own source"),
    ] = None,
    dip: Annotated[
        float | None, typer.Option(help="Dip to test, overriding the file's source")
    ] = None,
    rake: Annotated[
        float | None, typer.Option(help="Rake to test, overriding the file's source")
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
    usable: Annotated[
        bool,
        typer.Option(
            "--usable/--no-usable",
            help="Ignore recordings beyond their high-pass corner, where the "
            "record is filter rather than ground motion",
        ),
    ] = True,
    screen: Annotated[
        Screen,
        typer.Option(
            "--supergrid",
            help="What to do with stations inside the SW4 supergrid absorbing layer",
        ),
    ] = Screen.exclude,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output image path (omit to show interactively)"
        ),
    ] = None,
    panels: Annotated[
        str,
        typer.Option(
            "--panels",
            help="Which panels to draw, comma separated, in order: "
            "polar, azimuth, sweep, period. All four by default -- one claim "
            "each is what a slide wants, and all four is what an analyst does",
        ),
    ] = "polar,azimuth,sweep,period",
    seed: Annotated[
        int,
        typer.Option(help="Seed for the permutation null under the strike sweep"),
    ] = 20260906,
    table: Annotated[
        Path | None,
        typer.Option("--table", help="Also write the strike sweep as CSV"),
    ] = None,
    display_height: Annotated[
        float | None,
        typer.Option("--display-height", help="Height (cm) the figure is shown at"),
    ] = None,
    viewing_distance: Annotated[
        float | None,
        typer.Option("--viewing-distance", help="Distance (m) it must be read from"),
    ] = None,
    dpi: Annotated[int, typer.Option(help="Output resolution")] = 300,
):
    """Score a moment tensor's radiation pattern against what the stations recorded.

    The recordings are differenced against an empirical model, which has no
    mechanism, so what remains at each station is the azimuthal structure the
    earthquake radiated plus its own site and path terms. That is correlated
    against the double-couple pattern of the tensor the simulation was built
    from, and against every other strike, so the figure says both how well the
    tensor does and whether anything else would do better.

    This cannot distinguish a tensor's two nodal planes -- they radiate the same
    pattern -- and it is only meaningful below the hybrid crossover, where the
    ground motion is still the deterministic solution that carries a mechanism.
    """
    component = component or DEFAULT_COMPONENT["pSA"]
    tree = open_ims(im_file)
    if "pSA" not in tree.children:
        raise typer.BadParameter(f"{im_file} has no pSA")
    node = tree["pSA"]
    if component not in node.data_vars:
        raise typer.BadParameter(
            f"component {component!r} not in pSA. "
            f"Available: {[str(c) for c in node.data_vars]}"
        )
    if period < DETERMINISTIC_PERIOD:
        console_warn(
            f"{period:g} s is above the hybrid crossover at "
            f"{DETERMINISTIC_PERIOD:g} s, where the stochastic high frequency has "
            "replaced the radiation pattern with a random one: this scores the "
            "high-frequency generator rather than the mechanism"
        )

    da = node[component].transpose("station", "period")
    periods = da["period"].values
    obs, obs_periods = read_observed_spectra(
        observed, component, metric=None, prefix="pSA_"
    )
    obs = restrict_to_domain(
        obs, tree.attrs, da.longitude.values, da.latitude.values, observed
    )
    if stations is not None:
        obs = restrict_to_stations(
            obs, list(read_pick_list(stations)["stations"]), observed
        )
    if obs["name"].size == 0:
        raise typer.BadParameter(f"no {observed} stations inside the domain")

    nearest, reached = nearest_stations(
        da.longitude.values, da.latitude.values, obs["lon"], obs["lat"]
    )
    inside = supergrid(tree, da).flagged[nearest] & reached
    if (count := int(inside.sum())) and screen is Screen.exclude:
        print(f"ignored {count} recordings inside the supergrid absorbing layer")
        reached &= ~inside

    index = int(np.abs(periods - period).argmin())
    resolved = float(periods[index])
    recorded_all = np.log(match_columns(obs["spectrum"], obs_periods, periods))
    if usable:
        # The same screen the bias sweep applies, and for the same reason: past
        # its high-pass corner a record is the instrument's filter, and at 3 s a
        # good many of these are. Correlating a filter response against a
        # radiation pattern would be scoring the mechanism on the one part of
        # the record that cannot carry it.
        longest = np.where(np.isfinite(obs["high_pass"]), 1.0 / obs["high_pass"], np.inf)
        filtered = periods[None, :] > longest[:, None]
        dropped = int((filtered & np.isfinite(recorded_all)).sum())
        recorded_all = np.where(filtered, np.nan, recorded_all)
        if dropped:
            print(f"ignored {dropped} station-periods beyond the high-pass corner")
    recorded = recorded_all[:, index]
    mean, _ = select_empirical(tree, "pSA", empirical, {})
    predicted = match_columns(
        mean.sel(station=da.station).transpose("station", "period").values[nearest],
        mean["period"].values,
        periods,
    )[:, index]
    simulated = np.log(match_columns(da.values[nearest], periods, periods))[:, index]

    # The empirical model is the flat reference the mechanism is measured
    # against: what is left of a recording once an azimuthally symmetric
    # prediction is removed is the structure a symmetric model cannot make.
    residual = np.where(reached, recorded - predicted, np.nan)
    simulated_residual = np.where(reached, simulated - predicted, np.nan)
    azimuth = source_azimuth(
        float(tree.attrs["hypo_lon"]), float(tree.attrs["hypo_lat"]),
        obs["lon"], obs["lat"],
    )

    file_strike, file_dip, file_rake = mechanism_from_attributes(tree.attrs)
    strike = file_strike if strike is None else strike
    dip = file_dip if dip is None else dip
    rake = file_rake if rake is None else rake

    gap, gap_from, gap_to = largest_gap(azimuth[np.isfinite(residual)])
    pattern = log_amplitude(strike, dip, rake, azimuth, takeoff)
    r_tensor = correlation(residual, pattern)
    r_simulated = correlation(residual, simulated_residual)
    strikes, sweep = strike_sweep(residual, azimuth, dip, rake, takeoff)
    peak = float(strikes[np.nanargmax(sweep)])
    offset = angular_offset(peak, strike)

    print(
        f"pSA {resolved:g} s, {int(np.isfinite(residual).sum())} recordings; "
        f"largest azimuthal gap {gap:.0f} deg ({gap_from:.0f}-{gap_to:.0f})"
    )
    null_peaks, null_fixed = sweep_null(
        residual, azimuth, dip, rake, takeoff, NULL_DRAWS, seed
    )
    p_peak = float((null_peaks >= np.nanmax(sweep)).mean())
    p_tensor = float((null_fixed >= r_tensor).mean())
    print(
        f"  tensor {strike:.1f}/{dip:.1f}/{rake:.1f} at takeoff {takeoff:g}: "
        f"r = {r_tensor:+.3f} (p = {p_tensor:.3f}, one strike stated in advance)"
    )
    print(f"  sweep peaks at strike {peak:.0f} (r = {np.nanmax(sweep):+.3f}), "
          f"{offset:.0f} deg from the tensor "
          f"-- p = {p_peak:.3f} against the peak null, so "
          f"{'not ' if p_peak > 0.05 else ''}better than a search of 180 strikes "
          "finds in noise")
    print(f"  simulation reproduces the recorded pattern at r = {r_simulated:+.3f}")

    if table is not None:
        with table.open("w") as handle:
            handle.write("strike,correlation\n")
            for value, score in zip(strikes, sweep):
                handle.write(f"{value:.1f},{score:.6f}\n")
        print(f"wrote {table}")

    wanted = [name.strip() for name in panels.split(",") if name.strip()]
    if unknown := [name for name in wanted if name not in PANELS]:
        raise typer.BadParameter(
            f"unknown panel(s) {unknown}; choose from {list(PANELS)}"
        )
    if not wanted:
        raise typer.BadParameter("--panels needs at least one panel")

    # One row per pair, so two panels come out side by side and four as the
    # square the analyst reads. A figure asked for one panel is drawn square
    # rather than as a strip a slide would have to shrink to fit.
    columns = 1 if len(wanted) == 1 else 2
    rows = -(-len(wanted) // columns)
    design = (4.5 * columns, 4.5 * rows)
    display = Display.for_figure(design, dpi, display_height, viewing_distance)
    display.report(design)
    fig = plt.figure(figsize=display.size, layout="constrained")
    grid = fig.add_gridspec(rows, columns)

    for position, name in enumerate(wanted):
        cell = grid[position // columns, position % columns]
        ax = fig.add_subplot(cell, projection="polar" if name == "polar" else None)
        if name == "polar":
            draw_polar(ax, azimuth, residual, strike, dip, rake, takeoff,
                       gap_from, gap_to, gap, display)
        elif name == "azimuth":
            draw_azimuth(ax, azimuth, residual, simulated_residual, strike, dip,
                         rake, takeoff, empirical, display)
        elif name == "sweep":
            draw_sweep(ax, strikes, sweep, strike, peak, azimuth, residual,
                       dip, rake, null_peaks, p_peak, display)
        else:
            draw_period(ax, da, periods, recorded_all, nearest, reached,
                        mean, azimuth, strike, dip, rake, takeoff, resolved,
                        display)

    # Damped by the display scale: enlarging the text for a projector enlarges
    # the title too, and a title is the one thing on a figure that cannot wrap
    # itself out of trouble -- it simply runs off the canvas.
    fig.suptitle(
        f"{tree.attrs.get('event', im_file.stem)} — radiation pattern at "
        f"pSA {resolved:g} s\n"
        f"tensor {strike:.0f}°/{dip:.0f}°/{rake:.0f}°, takeoff {takeoff:g}° "
        "assumed; both nodal planes radiate this",
        fontsize=display.mark(10),
    )
    if output is not None:
        fig.savefig(output, dpi=display.dpi)
        print(f"wrote {output}")
    else:
        plt.show()


def draw_polar(
    ax, azimuth, residual, strike, dip, rake, takeoff, gap_from, gap_to, gap, display
):
    """Stations on the focal sphere, over the lobes the tensor predicts.

    Compass convention, not the mathematical one: north up and bearings running
    clockwise, so a station's place on this plot is where it is on a map.
    """
    fine = np.linspace(0, 360, 721)
    amplitude = radiation_amplitude(strike, dip, rake, fine, takeoff)
    amplitude = amplitude / max(amplitude.max(), 1e-9)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.fill_between(
        np.radians(fine), 0, amplitude, color=EMPIRICAL_BLUE, alpha=0.20, lw=0
    )
    ax.plot(np.radians(fine), amplitude, color=EMPIRICAL_BLUE, lw=display.mark(1.4))

    # The gap is the figure's limit, so it is drawn rather than left to be
    # inferred from where the markers thin out.
    span = np.linspace(gap_from, gap_from + gap, 90)
    ax.fill_between(np.radians(span), 0, 1.28, color="#bbbbbb", alpha=0.35, lw=0)
    ax.text(
        np.radians(gap_from + gap / 2), 0.72, f"{gap:.0f}° gap",
        ha="center", va="center", fontsize=7, color="#555555", rotation=0,
    )

    known = np.isfinite(residual)
    # A single dead record would otherwise take the whole scale: the limit is a
    # high percentile, not the maximum.
    limit = float(np.nanpercentile(np.abs(residual[known]), 90)) if known.any() else 1.0
    limit = max(limit, 0.1)
    ax.scatter(
        np.radians(azimuth[known]),
        np.full(known.sum(), 1.12),
        c=residual[known],
        cmap="RdBu_r",
        norm=BoundaryNorm(np.linspace(-limit, limit, 11), 256),
        s=display.mark(30),
        ec="black",
        lw=0.4,
        zorder=5,
    )
    ax.set_ylim(0, 1.28)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_title("stations by azimuth, over |F$_S$|", fontsize=8, pad=12)


def draw_azimuth(
    ax, azimuth, residual, simulated, strike, dip, rake, takeoff, empirical, display
):
    """The same thing unrolled, where a trend is easier to judge than on a circle.

    The pattern is drawn as a band over the plausible takeoff angles rather than
    as a line at one, because the angle is assumed and the lobes move with it --
    a single curve would claim a precision the command does not have.
    """
    fine = np.linspace(0, 360, 721)
    low, high = TAKEOFF_BAND
    band = np.array(
        [log_amplitude(strike, dip, rake, fine, angle)
         for angle in np.linspace(low, high, 7)]
    )
    known = np.isfinite(residual)
    # The pattern is a shape, not a level: it is offset onto the residuals it is
    # being compared with so the two can be looked at on one axis at all.
    centre = log_amplitude(strike, dip, rake, fine, takeoff)
    shift = np.nanmean(residual[known]) - centre.mean()
    ax.fill_between(
        fine, band.min(axis=0) + shift, band.max(axis=0) + shift,
        color=EMPIRICAL_BLUE, alpha=0.20, lw=0,
        label=f"|F$_S$|, takeoff {low:g}–{high:g}°",
    )
    ax.plot(fine, centre + shift, color=EMPIRICAL_BLUE, lw=display.mark(1.3))
    ax.scatter(
        azimuth[known], residual[known], s=display.mark(28), color=OBSERVED_GREEN,
        ec="black", lw=0.4, zorder=5, label=f"recording / {empirical}",
    )
    finite_sim = np.isfinite(simulated)
    ax.scatter(
        azimuth[finite_sim], simulated[finite_sim], s=display.mark(16),
        color=SIM_ONE_BLACK, marker="x", lw=display.mark(1.0), zorder=4,
        label=f"simulation / {empirical}",
    )
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("azimuth from hypocentre (°)", fontsize=9)
    ax.set_ylabel("ln residual", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, color="#dddddd")


def draw_sweep(
    ax, strikes, sweep, strike, peak, azimuth, residual, dip, rake,
    null_peaks, p_peak, display,
):
    """Correlation against assumed strike, and what a rotated tensor is worth.

    The band is the same sweep over the plausible takeoff angles. A peak that
    survives it means the strike the amplitudes prefer, rather than the strike
    one assumed angle prefers.
    """
    low, high = TAKEOFF_BAND
    band = np.array(
        [strike_sweep(residual, azimuth, dip, rake, angle)[1]
         for angle in np.linspace(low, high, 7)]
    )
    ax.fill_between(
        strikes, np.nanmin(band, axis=0), np.nanmax(band, axis=0),
        color=EMPIRICAL_BLUE, alpha=0.20, lw=0,
    )
    ax.plot(strikes, sweep, color=EMPIRICAL_BLUE, lw=display.mark(1.4))
    ax.axhline(0, color="#999999", lw=0.8)
    # The level a sweep over shuffled residuals clears one time in twenty. A
    # peak below this line is what searching 180 strikes finds in noise, and
    # drawing it stops the tallest bump on the curve being read as a result.
    level = float(np.percentile(null_peaks, 95))
    ax.axhline(level, color="#b03030", lw=display.mark(1.0), ls=(0, (4, 2)))
    ax.annotate(
        f"95% of sweeps over shuffled residuals (p = {p_peak:.2f})",
        (4, level), fontsize=display.mark(6.5), color="#b03030",
        va="top", ha="left", xytext=(0, -3), textcoords="offset points",
    )
    # Offset the two annotations when the peak lands on the tensor, which is
    # the outcome the panel exists to show and the one that overprints them.
    close = angular_offset(peak, strike) < 30
    for value, colour, label, shift in (
        (strike, SIM_ONE_BLACK, f"tensor {strike:.0f}°", -1 if close else 0),
        (peak, OBSERVED_GREEN, f"peak {peak:.0f}°", 1 if close else 0),
    ):
        ax.axvline(value, color=colour, lw=display.mark(1.2), ls="--")
        ax.annotate(
            label, (value, ax.get_ylim()[1]), fontsize=7, color=colour,
            ha="center", va="bottom",
            xytext=(shift * 26, 2 + (8 if shift > 0 else 0)),
            textcoords="offset points",
        )
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("assumed strike (°)", fontsize=9)
    ax.set_ylabel("correlation with recorded residual", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, color="#dddddd")


def draw_period(
    ax, da, periods, recorded, nearest, reached, mean, azimuth,
    strike, dip, rake, takeoff, resolved, display,
):
    """The correlation period by period, which is the check on the whole idea.

    A radiation pattern recovered from ground motion has to appear at long
    period and fade towards the hybrid crossover, because that is where the
    deterministic solution that carries it gives way to a stochastic one that
    does not. A correlation flat across the spectrum would be measuring
    something else -- geometry, site, or the empirical model's own distance
    term -- and this panel is what tells the two apart.
    """
    predicted = match_columns(
        mean.sel(station=da.station).transpose("station", "period").values[nearest],
        mean["period"].values,
        periods,
    )
    simulated = np.log(match_columns(da.values[nearest], periods, periods))
    pattern = log_amplitude(strike, dip, rake, azimuth, takeoff)
    against_tensor, against_sim = [], []
    for index in range(periods.size):
        residual = np.where(reached, recorded[:, index] - predicted[:, index], np.nan)
        against_tensor.append(correlation(residual, pattern))
        against_sim.append(
            correlation(residual, np.where(reached, simulated[:, index]
                                           - predicted[:, index], np.nan))
        )
    ax.axvspan(
        periods.min(), DETERMINISTIC_PERIOD, color="#dddddd", alpha=0.55, lw=0
    )
    ax.text(
        periods.min() * 1.15, 0.92, "stochastic HF\n(no mechanism)",
        fontsize=6.5, color="#666666", va="top",
    )
    ax.plot(periods, against_tensor, color=EMPIRICAL_BLUE,
            lw=display.mark(1.4), label="vs tensor |F$_S$|")
    ax.plot(periods, against_sim, color=SIM_ONE_BLACK, lw=display.mark(1.4),
            label="vs simulation")
    ax.axvline(resolved, color="#999999", lw=0.8, ls=":")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("period (s)", fontsize=9)
    ax.set_ylabel("correlation with recorded residual", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.3, color="#dddddd")
