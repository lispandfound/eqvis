"""The land the maps are drawn on.

Coastline and basin outlines, fetched once and cached, plus the helpers that
put them on an axes: filled land, stroked shore, dashed basin boundaries, a
distance scale, and a locator inset of New Zealand.
"""

import base64
import gzip
import math
import urllib.request
from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import shapely
from matplotlib.patches import Rectangle

from .console import console_warn
from .constants import BASIN_LINE, BASIN_TEXT, HIDDEN, NAMED, OBSERVED_GREEN
from .display import NATURAL, Display

# Simplified NZ coastline (base64 of gzip'd ISO-WKB MultiPolygon, ~0.002deg /
# 200 m tolerance) used to clip the interpolated raster to land. The blob is
# hosted rather than embedded to keep this file small; it is fetched once and
# cached. Decode with: from_wkb(gunzip(b64decode(blob))).
COASTLINE_URL = "https://www.dropbox.com/scl/fi/6c6p1345xpx8lvv1850fa/coastline_nz.b64?rlkey=a80zzwvaz5ucgqlpkaw4m9vee&st=4vmd4zus&dl=1"  # raw link to coastline_nz.b64


COASTLINE_CACHE = Path.home() / ".cache" / "eqvis" / "coastline_nz.b64"


# Basin outlines as GeoParquet (basin_name, layer, part, priority, geometry),
# fetched and cached the same way as the coastline.
BASINS_URL = "https://www.dropbox.com/scl/fi/5h81ep32rnu3km9h7kgd7/basins.parquet?rlkey=27mhbl8x3d9jicx6kzi1jtgv1&st=lt6ootwg&dl=1"  # raw link to basins.parquet


BASINS_CACHE = Path.home() / ".cache" / "eqvis" / "basins.parquet"


def load_coastline(source: Path | None = None) -> shapely.MultiPolygon | None:
    """Return the NZ coastline as a shapely geometry, or None if unavailable.

    Reads an explicit ``source`` (a ``.b64`` blob or a GeoJSON file) if given;
    otherwise the cached download, fetching it from ``COASTLINE_URL`` on a cache
    miss. Any failure (no URL configured, offline, bad file) returns None so the
    caller can fall back to the station-distance mask.
    """
    if source is not None:
        if source.suffix == ".geojson":
            return shapely.from_geojson(source.read_text())
        return _decode_coastline(source.read_text())

    if COASTLINE_CACHE.exists():
        return _decode_coastline(COASTLINE_CACHE.read_text())

    if not COASTLINE_URL or COASTLINE_URL == "REPLACE_ME":
        console_warn("no coastline URL configured; skipping coastline clip")
        return None
    try:
        with urllib.request.urlopen(COASTLINE_URL, timeout=30) as response:
            blob = response.read().decode()
    except OSError as exc:
        console_warn(f"could not fetch coastline ({exc}); skipping coastline clip")
        return None
    COASTLINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COASTLINE_CACHE.write_text(blob)
    return _decode_coastline(blob)


def _decode_coastline(blob: str) -> shapely.MultiPolygon:
    return shapely.from_wkb(gzip.decompress(base64.b64decode(blob)))


def load_basins(
    source: Path | None = None,
) -> list[tuple[str, shapely.Geometry]] | None:
    """Return ``(name, geometry)`` for each basin, or None if unavailable.

    Reads an explicit ``source`` if given; otherwise the cached download,
    fetching it from ``BASINS_URL`` on a cache miss. Any failure (no URL
    configured, offline, bad file) returns None so the caller can carry on
    without basin outlines.
    """
    if source is not None:
        return read_basins(source)
    if BASINS_CACHE.exists():
        return read_basins(BASINS_CACHE)
    if not BASINS_URL or BASINS_URL == "REPLACE_ME":
        console_warn("no basins URL configured; skipping basin outlines")
        return None
    try:
        with urllib.request.urlopen(BASINS_URL, timeout=30) as response:
            blob = response.read()
    except OSError as exc:
        console_warn(f"could not fetch basins ({exc}); skipping basin outlines")
        return None
    BASINS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASINS_CACHE.write_bytes(blob)
    return read_basins(BASINS_CACHE)


def read_basins(path: Path) -> list[tuple[str, shapely.Geometry]]:
    """Top-level basin outlines from a GeoParquet file, with overlaps resolved.

    A basin may be split into ``part``s, which are separate pieces of the same
    outline, and may be stacked into geological ``layer``s, which are not: for
    each piece only the shallowest layer is wanted, and ``priority`` orders
    them. Where two basins overlap the lower ``priority`` wins, so each is
    clipped against everything ahead of it and the outlines never double up.
    """
    table = pq.read_table(path)
    names = table.column("basin_name").to_pylist()
    parts = table.column("part").to_pylist()
    priorities = table.column("priority").to_pylist()
    geometries = shapely.from_wkb(table.column("geometry").to_pylist())

    # One row per piece: the layer that sits on top of it.
    shallowest: dict[tuple[str, float | None], int] = {}
    for index, key in enumerate(zip(names, parts)):
        if key not in shallowest or priorities[index] < priorities[shallowest[key]]:
            shallowest[key] = index

    basins, covered = [], None
    for index in sorted(shallowest.values(), key=lambda i: priorities[i]):
        geometry = geometries[index]
        if covered is not None:
            geometry = shapely.difference(geometry, covered)
        covered = (
            geometries[index]
            if covered is None
            else shapely.union(covered, geometries[index])
        )
        if not geometry.is_empty:
            basins.append((names[index], geometry))

    # Parts are pieces of one basin, so they merge back into a single named
    # outline: otherwise a basin in six parts is labelled six times.
    merged: dict[str, list] = {}
    for name, geometry in basins:
        merged.setdefault(name, []).append(geometry)
    return [(name, shapely.union_all(pieces)) for name, pieces in merged.items()]


def basins_in_view(
    basins: list[tuple[str, shapely.Geometry]],
    bounds: tuple[float, float, float, float],
) -> list[tuple[str, shapely.Geometry]]:
    """Basins that reach into the plotted area, kept whole."""
    box = shapely.box(*bounds)
    return [
        (name, geometry)
        for name, geometry in basins
        if shapely.intersects(geometry, box)
    ]


MIN_LABELLED_BASIN_AREA = 0.1  # square degrees


def draw_basins(
    ax: plt.Axes,
    basins: list[tuple[str, shapely.Geometry]],
    clip: shapely.Geometry,
    tolerance: float,
    stations: tuple[np.ndarray, np.ndarray] | None = None,
    states: dict[str, int] | None = None,
    display: Display | None = None,
) -> list[dict]:
    """Outline each basin and return label entries anchored on its boundary.

    Drawn like the domain but neutral and hairline, so the outlines read as
    background rather than competing with the station residuals; a light white
    halo keeps them legible where the raster goes near-black. The *boundary* is
    clipped to ``clip`` rather than the polygon -- clipping the polygon would
    turn the edge of the clip region into what looks like basin boundary -- and
    simplified at ``tolerance``, which at plot resolution drops most of a
    coastline-scale vertex count with no visible change.

    Only basins worth naming are labelled: those containing a station, or, with
    nothing observed to anchor to, those big enough to read. Labels rank behind
    the stations, which get first claim on the free space.

    ``states`` overrides that rule with a name -> state mapping from the
    picker, where a basin is drawn unless it is :data:`HIDDEN` and named only
    when it is :data:`NAMED`. Chosen basins are exactly the ones asked for,
    whether or not they hold a station or are big enough to have earned it.
    """
    display = display or NATURAL
    halo = [
        patheffects.withStroke(
            linewidth=display.mark(1.5), foreground="white", alpha=0.75
        )
    ]
    entries = []
    for name, geometry in basins:
        if states is not None and states.get(name, HIDDEN) == HIDDEN:
            continue
        outline = shapely.intersection(geometry.boundary, clip)
        if outline.is_empty:
            continue
        for line in shapely.get_parts(shapely.simplify(outline, tolerance)):
            if line.is_empty:
                continue
            ax.plot(
                *line.xy,
                color=BASIN_LINE,
                lw=display.mark(0.55),
                # Matplotlib scales a dash pattern by the line width, so
                # damping both leaves the dashes exactly as designed -- they
                # are texture that says "approximate", not something that has
                # to be resolved from across a room.
                ls=(0, (display.mark(4), display.mark(3))),
                path_effects=halo,
                zorder=4,
            )
        # Enlarged for distance viewing there is no room for basin names
        # beside the labels that carry the figure, so the outlines stay and
        # the names go.
        if not display.detailed:
            continue
        visible = [
            part
            for part in shapely.get_parts(shapely.intersection(geometry, clip))
            if part.geom_type == "Polygon"
        ]
        if not visible:
            continue
        if states is not None:
            if states.get(name, HIDDEN) != NAMED:
                continue
        elif stations is not None:
            if not shapely.contains_xy(geometry, *stations).any():
                continue
        elif sum(part.area for part in visible) < MIN_LABELLED_BASIN_AREA:
            continue
        # Anchor on the outline itself, at the top of the largest visible piece.
        largest = max(visible, key=lambda polygon: polygon.area)
        x, y = np.asarray(largest.exterior.coords).T
        top = int(np.argmax(y))
        entries.append(
            {
                "text": name,
                "x": x[top],
                "y": y[top],
                "colour": BASIN_TEXT,
                "rank": 1,
                "size": 5.5,
                "style": "italic",
            }
        )
    return entries


def land_mask(
    mesh_lon: np.ndarray, mesh_lat: np.ndarray, coastline: shapely.MultiPolygon
) -> np.ndarray:
    """Boolean mask (True = off land) for grid cells outside the coastline.

    The coastline is clipped to the grid's bounding box first so containment
    tests run against a handful of local polygons rather than all of NZ.
    """
    bbox = shapely.box(mesh_lon.min(), mesh_lat.min(), mesh_lon.max(), mesh_lat.max())
    local = shapely.intersection(coastline, bbox)
    if local.is_empty:
        return np.zeros(mesh_lon.shape, dtype=bool)
    on_land = shapely.contains_xy(local, mesh_lon.ravel(), mesh_lat.ravel())
    return ~on_land.reshape(mesh_lon.shape)


def draw_coastline(
    ax: plt.Axes,
    coastline: shapely.MultiPolygon,
    bounds: tuple[float, float, float, float],
    display: Display | None = None,
) -> None:
    """Outline the coastline (clipped to the plot bounds) in a soft grey."""
    display = display or NATURAL
    local = shapely.intersection(coastline, shapely.box(*bounds))
    for poly in shapely.get_parts(local):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            x, y = ring.xy
            ax.plot(x, y, color="#6b6b6b", lw=display.mark(0.5), zorder=3)


def draw_geometry(
    ax: plt.Axes, attrs: dict, display: Display | None = None
) -> None:
    """Overlay domain boundary, source outline and hypocentre if present."""
    display = display or NATURAL
    if domain := attrs.get("domain"):
        x, y = shapely.from_wkt(domain).exterior.xy
        ax.plot(x, y, color="black", lw=display.mark(0.8), ls="--", zorder=4)
    if source := attrs.get("source"):
        geometry = shapely.from_wkt(source)
        polygons = (
            geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
        )
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color="black", lw=display.mark(1.0), zorder=4)
    if "hypo_lon" in attrs and "hypo_lat" in attrs:
        ax.plot(
            float(attrs["hypo_lon"]),
            float(attrs["hypo_lat"]),
            marker="*",
            ms=display.mark(14),
            mfc="yellow",
            mec="black",
            mew=0.7,
            ls="none",
            zorder=5,
        )


def fill_land(
    ax: plt.Axes,
    coastline: shapely.MultiPolygon,
    bounds: tuple[float, float, float, float],
    display: Display | None = None,
) -> None:
    """Shade the land inside ``bounds``, so the sea reads as sea.

    The companion to :func:`draw_coastline`, which only strokes the shore. A
    map whose subject is the land itself (a raster of ground motion, say)
    wants the stroke alone; one whose subject sits *on* the land wants the
    fill underneath it for the coastline to be legible as a shape.
    """
    display = display or NATURAL
    local = shapely.intersection(coastline, shapely.box(*bounds))
    for poly in shapely.get_parts(local):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        ax.fill(
            *poly.exterior.xy, fc="#e2e2e2", ec="none", zorder=0, rasterized=False
        )
        for hole in poly.interiors:
            ax.fill(*hole.xy, fc="white", ec="none", zorder=0)
    draw_coastline(ax, coastline, bounds, display)


def draw_nz_outline(ax: plt.Axes, coastline: shapely.MultiPolygon | None) -> None:
    """Fill a bare outline of New Zealand on ``ax``, for a locator inset."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    if coastline is None:
        return
    for poly in shapely.get_parts(coastline):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        ax.fill(*poly.exterior.xy, fc="#ececec", ec="#8a8a8a", lw=0.3, zorder=1)


def draw_inset_map(
    ax: plt.Axes, lon: float, lat: float, coastline: shapely.MultiPolygon | None
) -> None:
    """Locate the station on a small outline of New Zealand."""
    draw_nz_outline(ax, coastline)
    ax.plot(
        lon, lat, marker="^", ms=7, mfc=OBSERVED_GREEN, mec="black", mew=0.6, zorder=3
    )
    ax.set_aspect(1 / np.cos(np.radians(lat)))


def draw_locator_map(
    ax: plt.Axes,
    coastline: shapely.MultiPolygon | None,
    bounds: tuple[float, float, float, float],
    display: Display | None = None,
) -> None:
    """Locate a whole map region on a small outline of New Zealand.

    The region is usually a few tens of kilometres across against 1500 km of
    country, so it is drawn as a red box with a minimum size: at true scale it
    would be a dot too small to find.
    """
    display = display or NATURAL
    draw_nz_outline(ax, coastline)
    west, south, east, north = bounds
    # A box under about a degree across vanishes at inset scale, so it is
    # grown about its centre rather than drawn faithfully and lost.
    floor = 0.9
    if east - west < floor:
        west, east = (west + east - floor) / 2, (west + east + floor) / 2
    if north - south < floor:
        south, north = (south + north - floor) / 2, (south + north + floor) / 2
    ax.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            fc="none",
            ec="#d62728",
            lw=display.mark(1.2),
            zorder=3,
        )
    )
    ax.set_xlim(166, 179)
    ax.set_ylim(-47.5, -34)
    ax.set_aspect(1 / np.cos(np.radians((south + north) / 2)))


def nice_scale_length(span_km: float) -> int:
    """A round scale-bar length, in km, spanning about a quarter of a map."""
    quarter = max(span_km / 4, 1e-6)
    decade = 10 ** math.floor(math.log10(quarter))
    return int(max(m for m in (1, 2, 5) if m * decade <= quarter) * decade)


def draw_scale_bar(
    ax: plt.Axes, display: Display | None = None, corner: str = "upper right"
) -> None:
    """A round-numbered distance scale in a corner of a lon/lat map.

    Degrees on the axes say where, not how far, and how far is what a reader
    wants of a rupture map. The bar is drawn in data coordinates so it stays
    true to the axes' own scale, which means it has to be redrawn if the limits
    move -- call this after they are set.
    """
    display = display or NATURAL
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    km_per_degree = 111.32 * np.cos(np.radians((south + north) / 2))
    length = nice_scale_length((east - west) * km_per_degree)
    width = length / km_per_degree

    inset = 0.045
    y = north - inset * (north - south)
    if corner.endswith("right"):
        right = east - inset * (east - west)
        left = right - width
    else:
        left = west + inset * (east - west)
    if corner.startswith("lower"):
        y = south + inset * (north - south)

    tick = 0.012 * (north - south)
    ax.plot(
        [left, left, left + width, left + width],
        [y + tick, y, y, y + tick],
        color="black",
        lw=display.mark(1.0),
        solid_capstyle="butt",
        zorder=8,
        clip_on=False,
    )
    ax.annotate(
        f"{length:g} km",
        ((left + left + width) / 2, y),
        textcoords="offset points",
        xytext=(0, -3),
        ha="center",
        va="top",
        fontsize=8,
        path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
        zorder=8,
    )
