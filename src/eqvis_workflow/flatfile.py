"""Reading recordings out of a GeoNet flatfile archive.

The archive holds one ground motion table per component of motion, in the same
units as the simulation. Only the scalar measures and the spectra are
tabulated; CAV, Arias intensity and the durations are simulation-only. Beside
the values it carries what the network knows about each record -- how the
instrument was mounted, how clean the recording is, the band over which it is
signal rather than filter -- which is what lets a crowded map be thinned on
something better than position.
"""

import csv
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import typer

from .console import console_warn
from .constants import DISTANCE_COLUMN, FAS_PREFIX

# GeoNet flatfile archives hold one ground motion table per component, in the
# same units as the simulation (g for PGA/pSA, cm/s for PGV). Only these IMs
# are tabulated -- CAV, AI, the durations and FAS are simulation-only.
# The table suffix varies between archives ("_flat.csv" and "_flatfile.csv"
# both occur), so the component is matched rather than the whole name.
FLATFILE_TABLE_PATTERN = re.compile(
    r"ground_motion_im_table_(?P<component>.+?)_flat(?:file)?\.csv$", re.IGNORECASE
)


FLATFILE_SCALAR_IMS = {"PGA", "PGV", "PGD"}


def flatfile_member(zf: zipfile.ZipFile, component: str, archive: Path) -> str:
    """Locate a component's ground motion table, wherever it sits in the archive.

    Matched on basename, since archives come both flat and nested under a
    directory named after the event, and on a pattern rather than an exact
    name, since the table suffix differs between archives.
    """
    available = []
    for member in zf.namelist():
        match = FLATFILE_TABLE_PATTERN.fullmatch(PurePosixPath(member).name)
        if match is None:
            continue
        if match["component"].casefold() == component.casefold():
            return member
        available.append(match["component"])
    raise typer.BadParameter(
        f"{archive} has no observed {component!r} component. Available: {available}"
    )


def flatfile_column(
    fields: list[str], im: str, selection: dict[str, float]
) -> tuple[str | None, dict[str, float]]:
    """Find the flatfile column holding ``im``, and what it actually resolved to.

    pSA is stored as one ``pSA_<period>`` column per period, on a period set
    that need not match the simulation's, so the nearest period is taken and
    reported back. Returns ``(None, {})`` for IMs the flatfile does not carry.
    """
    if "period" in selection:
        periods = {}
        for field in fields:
            if field.startswith("pSA_"):
                try:
                    periods[float(field.removeprefix("pSA_"))] = field
                except ValueError:
                    continue
        if not periods:
            return None, {}
        nearest = min(periods, key=lambda p: abs(p - selection["period"]))
        return periods[nearest], {"period": nearest}
    if im in FLATFILE_SCALAR_IMS and im in fields:
        return im, {}
    return None, {}


def read_observed(
    archive: Path,
    im: str,
    component: str,
    selection: dict[str, float],
    metric: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read observed station IMs from a GeoNet flatfile zip, without extracting.

    Returns ``{"name", "lon", "lat", "value"}`` (plus ``"distance"`` when a
    ``metric`` is asked for) and the resolved selection -- the flatfile's own
    period for pSA. ``value`` is all-NaN when the flatfile has no column for
    this IM, leaving the caller to plot bare station markers.
    """
    with zipfile.ZipFile(archive) as zf:
        table = flatfile_member(zf, component, archive)
        with zf.open(table) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            fields = reader.fieldnames or []
            column, resolved = flatfile_column(fields, im, selection)
            if column is None:
                console_warn(f"{archive} has no observed {im}; plotting stations only")
            distance_column = DISTANCE_COLUMN.get(metric) if metric else None
            if metric and distance_column not in fields:
                available = [m for m, c in DISTANCE_COLUMN.items() if c in fields]
                raise typer.BadParameter(
                    f"{archive} has no observed {metric!r} distance. "
                    f"Available: {available}"
                )
            # One row per station/location pair; the first row for a station wins.
            stations: dict[str, tuple[float, ...]] = {}
            for row in reader:
                try:
                    lon, lat = float(row["sta_lon"]), float(row["sta_lat"])
                except (TypeError, ValueError):
                    continue
                try:
                    value = float(row[column]) if column else np.nan
                except (TypeError, ValueError):
                    value = np.nan
                try:
                    distance = (
                        float(row[distance_column]) if distance_column else np.nan
                    )
                except (TypeError, ValueError):
                    continue
                stations.setdefault(row["sta"], (lon, lat, value, distance))

    lon, lat, value, distance = np.array(list(stations.values())).T
    observed = {
        "name": np.array(list(stations)),
        "lon": lon,
        "lat": lat,
        "value": value,
    }
    if metric:
        observed["distance"] = distance
    return observed, resolved


# Flatfile fields the automatic thinning judges a recording on. The
# signal-to-noise score is stored per component; the weakest of the three is
# what decides whether the record is worth plotting.
SNR_SCORE_COLUMNS = ("score_X", "score_Y", "score_Z")


def read_station_quality(archive: Path, component: str) -> dict[str, np.ndarray]:
    """Per-station record metadata, for deciding which stations are worth keeping.

    Read from the same table as :func:`read_observed` and deduplicated the same
    way -- one row per station, the first wins -- but keyed by name rather than
    positional, since the two readers drop rows on different grounds.

    Returns ``{"name", "free_field", "snr", "distance", "vs30", "high_pass",
    "low_pass"}``: whether the instrument sits at ground level rather than up a
    building, the weakest of its three component signal-to-noise scores, its
    rupture distance and site stiffness, and the corners of the band over which
    the record is signal rather than filter.
    """
    with zipfile.ZipFile(archive) as zf:
        table = flatfile_member(zf, component, archive)
        with zf.open(table) as raw:
            rows: dict[str, dict[str, str]] = {}
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
                rows.setdefault(row["sta"], row)

    def column(field: str) -> np.ndarray:
        return np.array([number_or_nan(row.get(field)) for row in rows.values()])

    scores = np.vstack([column(field) for field in SNR_SCORE_COLUMNS])
    known = np.isfinite(scores)
    return {
        "name": np.array(list(rows)),
        # An absent or unparsable mounting reads as free-field: not knowing
        # where an instrument sits is no grounds for dropping its record.
        "free_field": np.array(
            [row.get("is_ground_level", "True") != "False" for row in rows.values()]
        ),
        # np.nanmin over a station with no scores at all would warn and return
        # NaN; spelling it out keeps the all-unknown case quiet and explicit.
        "snr": np.where(
            known.any(axis=0), np.min(np.where(known, scores, np.inf), axis=0), np.nan
        ),
        "distance": column("r_rup"),
        "vs30": column("Vs30"),
        "high_pass": column("HPF_h"),
        "low_pass": column("LPF_h"),
    }


def align_quality(
    quality: dict[str, np.ndarray], names: np.ndarray
) -> dict[str, np.ndarray]:
    """Reorder record metadata onto ``names``, the stations actually plotted.

    Stations the metadata has no row for come back with everything unknown,
    which every filter reads as "no grounds to drop this".
    """
    index = {name: position for position, name in enumerate(quality["name"])}
    take = np.array([index.get(name, -1) for name in names], dtype=int)
    found = take >= 0
    aligned = {}
    for key, value in quality.items():
        blank = "" if value.dtype.kind in "US" else (True if key == "free_field" else np.nan)
        aligned[key] = np.where(found, value[take], blank)
    aligned["name"] = np.asarray(names)
    return aligned


def usable_at(
    high_pass: np.ndarray,
    low_pass: np.ndarray,
    selection: dict[str, float],
) -> np.ndarray:
    """Whether each record resolves the measure being plotted.

    A pSA period longer than ``1 / high_pass`` -- or a Fourier frequency
    outside the corners altogether -- is the filter's doing rather than the
    ground's, so the residual there says more about the instrument than the
    simulation. A scalar measure spans the whole band and excludes nothing,
    and so does a record whose corners are unknown.
    """
    if period := selection.get("period"):
        inside = (period <= 1 / high_pass) & (period >= 1 / low_pass)
    elif frequency := selection.get("frequency"):
        inside = (frequency >= high_pass) & (frequency <= low_pass)
    else:
        inside = np.ones(high_pass.shape, dtype=bool)
    return inside | ~(np.isfinite(high_pass) & np.isfinite(low_pass))


def read_observed_spectra(
    archive: Path, component: str, metric: str | None = None, prefix: str = "pSA_"
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Every recorded pSA period (or FAS frequency) at once, as a (station, x) table.

    Reading the whole spectrum in one pass is what makes a bias curve tractable:
    the alternative is reopening the archive and rescanning the table once per
    period, and there are of order a hundred of them. ``prefix`` selects which
    columns hold the spectrum -- ``pSA_`` (period, s) by default, or
    :data:`FAS_PREFIX` (frequency, Hz) for a FAS bias sweep.

    ``high_pass`` is the record's horizontal high-pass corner (Hz), carried
    along because a period beyond ``1 / high_pass`` -- or, for FAS, a frequency
    below it -- is the filter's doing rather than the ground's -- see
    :func:`usable_band`, which reads the same field for one station at a time.
    """
    with zipfile.ZipFile(archive) as zf:
        table = flatfile_member(zf, component, archive)
        with zf.open(table) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            fields = reader.fieldnames or []
            columns = {}
            for field in fields:
                if field.startswith(prefix):
                    try:
                        columns[float(field.removeprefix(prefix))] = field
                    except ValueError:
                        continue
            if not columns:
                raise typer.BadParameter(f"{archive} {component} table has no {prefix}*")
            periods = np.array(sorted(columns))
            ordered = [columns[period] for period in periods]
            distance_column = DISTANCE_COLUMN.get(metric) if metric else None
            if metric and distance_column not in fields:
                available = [m for m, c in DISTANCE_COLUMN.items() if c in fields]
                raise typer.BadParameter(
                    f"{archive} has no observed {metric!r} distance. "
                    f"Available: {available}"
                )
            # One row per station/location pair; the first row for a station wins.
            stations: dict[str, tuple] = {}
            for row in reader:
                try:
                    lon, lat = float(row["sta_lon"]), float(row["sta_lat"])
                except (TypeError, ValueError):
                    continue
                if row["sta"] in stations:
                    continue
                spectrum = np.array(
                    [number_or_nan(row.get(field)) for field in ordered]
                )
                stations[row["sta"]] = (
                    lon,
                    lat,
                    number_or_nan(row.get(distance_column) if metric else None),
                    number_or_nan(row.get("HPF_h")),
                    spectrum,
                )

    if not stations:
        raise typer.BadParameter(f"{archive} {component} table has no stations")
    lon, lat, distance, high_pass, spectra = zip(*stations.values())
    observed = {
        "name": np.array(list(stations)),
        "lon": np.array(lon),
        "lat": np.array(lat),
        "high_pass": np.array(high_pass),
        "spectrum": np.vstack(spectra),
    }
    if metric:
        observed["distance"] = np.array(distance)
    return observed, periods


def number_or_nan(text: str | None) -> float:
    """A flatfile cell as a number, or NaN when it is blank or not one."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def read_observed_spectrum(
    archive: Path, station: str
) -> tuple[np.ndarray, np.ndarray, dict[str, str]] | None:
    """Observed EAS (g*s) against frequency (Hz) for one station, plus its row.

    Returns None when the station has no record in the eas table. The row is
    handed back so the caller can read site metadata and the usable frequency
    band off it.
    """
    with zipfile.ZipFile(archive) as zf:
        with zf.open(flatfile_member(zf, "eas", archive)) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            columns = [f for f in reader.fieldnames or [] if f.startswith(FAS_PREFIX)]
            if not columns:
                raise typer.BadParameter(
                    f"{archive} eas table has no {FAS_PREFIX}* columns"
                )
            frequency = np.array([float(f.removeprefix(FAS_PREFIX)) for f in columns])
            for row in reader:
                if row["sta"] != station:
                    continue
                amplitude = np.array(
                    [float(row[f]) if row[f] else np.nan for f in columns]
                )
                return frequency, amplitude, row
    return None
