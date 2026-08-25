"""Drawing stations, and finding room for their labels.

The label placer is the interesting part: labels are laid out in pixels against
the settled axes, each taking the first offset that clears every anchor and
every label already placed, lowest rank first. Basin names, station names and
rupture-order numbers all go through it, so nothing lands on anything else.
"""

from collections.abc import Sequence

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import BoundaryNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree


def station_clusters(
    lon: np.ndarray,
    lat: np.ndarray,
    span: float,
    link_frac: float = 0.03,
    min_stations: int = 4,
    max_extent_frac: float = 0.08,
    limit: int = 2,
) -> list[np.ndarray]:
    """Indices of station groups too tight to label in place.

    Complete-linkage clustering at ``link_frac`` of the map span, keeping
    groups of at least ``min_stations`` that still fit inside
    ``max_extent_frac`` of it. Complete linkage (cluster diameter, not
    nearest-neighbour distance) matters here: single linkage chains a tight
    pileup together with everything else within ``link_frac`` of any of its
    members, one hop at a time, until the "cluster" spans a good part of the
    map and fails the extent check -- silently dropping a genuine pileup
    because it once touched a scattered station. The thresholds are meant to
    fire only on genuine pile-ups -- a handful of stations sitting within a
    few percent of the map width of each other -- not on stations that merely
    look regional, which the label placer already handles. Densest first,
    capped at ``limit``.
    """
    if len(lon) < min_stations:
        return []
    aspect = np.cos(np.radians(lat.mean()))
    points = np.column_stack([lon * aspect, lat])
    labels = fcluster(linkage(points, "complete"), span * link_frac, "distance")

    clusters = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < min_stations:
            continue
        extent = max(np.ptp(lon[members]), np.ptp(lat[members]))
        if extent < max_extent_frac * span:
            clusters.append(members)
    return sorted(clusters, key=len, reverse=True)[:limit]


def cluster_bounds(
    lon: np.ndarray, lat: np.ndarray, members: np.ndarray, pad: float = 0.35
) -> tuple[float, float, float, float]:
    """Padded lon/lat bounds around a cluster, never narrower than a floor.

    The floor keeps a cluster that is nearly a point from zooming to an absurd
    scale where the raster is a single flat colour.
    """
    x, y = lon[members], lat[members]
    width = max(np.ptp(x), np.ptp(y), 0.05)
    margin = width * pad
    centre_x, centre_y = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = width / 2 + margin
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)


def free_corner(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    domain: shapely.Geometry | None = None,
    target: tuple[float, float] | None = None,
    size: float = 0.28,
    taken: Sequence[tuple[float, float, float, float]] = (),
) -> tuple[float, float, float, float]:
    """Axes-fraction rectangle for an inset, in whichever corner is emptiest.

    Scored by the stations a corner would cover plus how much of the simulated
    domain it would sit on, so the panel lands on blank map rather than over
    either the stations or the result. Among corners that are equally free, the
    one nearest ``target`` wins, which keeps the connector to the zoomed area
    short rather than dragging it across the whole map.

    ``taken`` is the axes-fraction rectangles of insets already placed on this
    figure. A corner that overlaps one is penalised in proportion to the
    overlap, so a second or third inset lands beside the others instead of
    stacking on top of one -- which would silently hide it.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    diagonal = np.hypot(x1 - x0, y1 - y0)
    best, lowest = (0.02, 0.02), None
    for cx in (0.02, 1 - size - 0.02):
        for cy in (0.02, 1 - size - 0.02):
            left, right = x0 + cx * (x1 - x0), x0 + (cx + size) * (x1 - x0)
            bottom, top = y0 + cy * (y1 - y0), y0 + (cy + size) * (y1 - y0)
            covered = (
                (lon >= left) & (lon <= right) & (lat >= bottom) & (lat <= top)
            ).sum()
            score = float(covered)
            if domain is not None:
                rectangle = shapely.box(left, bottom, right, top)
                score += (
                    3
                    * shapely.area(shapely.intersection(rectangle, domain))
                    / shapely.area(rectangle)
                )
            if target is not None:
                score += 0.9 * float(
                    np.hypot(
                        (left + right) / 2 - target[0], (bottom + top) / 2 - target[1]
                    )
                    / diagonal
                )
            for tcx, tcy, tw, th in taken:
                overlap_frac = max(0.0, min(cx + size, tcx + tw) - max(cx, tcx)) * max(
                    0.0, min(cy + size, tcy + th) - max(cy, tcy)
                )
                score += 1e4 * overlap_frac / (size * size)
            if lowest is None or score < lowest:
                best, lowest = (cx, cy), score
    return (best[0], best[1], size, size)


def nearest_stations(
    lon: np.ndarray, lat: np.ndarray, at_lon: np.ndarray, at_lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Index of the simulation station nearest each point, and whether it counts.

    A point further from the simulation grid than a few cells -- outside the
    domain, or in a sea gap the land-masked grid does not cover -- has no
    nearest station worth the name, and comes back unreached.

    Split out from :func:`sample_simulation` because a whole spectrum reuses one
    set of indices across every period, and rebuilding the tree per period would
    dominate the cost.
    """
    aspect = np.cos(np.radians(lat.mean()))
    pts = np.column_stack([lon * aspect, lat])
    tree = cKDTree(pts)
    spacing = float(np.median(tree.query(pts, k=2)[0][:, 1]))
    distance, nearest = tree.query(np.column_stack([at_lon * aspect, at_lat]))
    return nearest, distance <= 3 * spacing


def sample_simulation(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    at_lon: np.ndarray,
    at_lat: np.ndarray,
) -> np.ndarray:
    """Simulated value at each observed station: its nearest simulation station.

    Stations the simulation does not reach come back as NaN.
    """
    nearest, reached = nearest_stations(lon, lat, at_lon, at_lat)
    return np.where(reached, values[nearest], np.nan)


def draw_observed(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    residual: np.ndarray | None,
    colormap: plt.cm.ScalarMappable,
    norm: BoundaryNorm,
) -> plt.cm.ScalarMappable | None:
    """Plot observed stations as triangles, coloured by their log residual.

    Stations whose residual is unavailable (no observed value for this IM, or
    no simulation coverage) are drawn hollow so they stay on the map without
    claiming a colour they do not have.
    """
    scatter = None
    known = np.isfinite(residual) if residual is not None else np.zeros(lon.shape, bool)
    if known.any():
        scatter = ax.scatter(
            lon[known],
            lat[known],
            c=residual[known],
            cmap=colormap,
            norm=norm,
            marker="^",
            s=90,
            ec="black",
            lw=0.6,
            zorder=6,
        )
    ax.scatter(
        lon[~known],
        lat[~known],
        marker="^",
        s=90,
        fc="none",
        ec="black",
        lw=0.6,
        zorder=6,
    )
    return scatter


# Where a station label may sit relative to its marker, in points, best first:
# straight above, then the other cardinals and diagonals, then a wider ring for
# the crowded clusters. (dx, dy, horizontal alignment, vertical alignment).
LABEL_OFFSETS = [
    (0, 9, "center", "bottom"),
    (0, -9, "center", "top"),
    (10, 0, "left", "center"),
    (-10, 0, "right", "center"),
    (9, 7, "left", "bottom"),
    (-9, 7, "right", "bottom"),
    (9, -7, "left", "top"),
    (-9, -7, "right", "top"),
    (0, 26, "center", "bottom"),
    (0, -26, "center", "top"),
    (28, 0, "left", "center"),
    (-28, 0, "right", "center"),
    (24, 20, "left", "bottom"),
    (-24, 20, "right", "bottom"),
    (24, -20, "left", "top"),
    (-24, -20, "right", "top"),
]


def _label_box(
    x: float, y: float, width: float, height: float, ha: str, va: str
) -> tuple[float, float, float, float]:
    """Pixel bounding box of a label anchored at (x, y) with the given alignment."""
    left = {"left": 0.0, "center": width / 2, "right": width}[ha]
    bottom = {"bottom": 0.0, "center": height / 2, "top": height}[va]
    return (x - left, y - bottom, x - left + width, y - bottom + height)


def _overlaps(a: tuple, b: tuple, pad: float = 1.0) -> bool:
    return (
        a[0] < b[2] + pad
        and b[0] < a[2] + pad
        and a[1] < b[3] + pad
        and b[1] < a[3] + pad
    )


def place_labels(fig: plt.Figure, ax: plt.Axes, entries: list[dict]) -> None:
    """Lay out labels so none overlap another label or an anchor point.

    Each entry is ``{"text", "x", "y", "colour", "rank"}``, optionally with
    ``size``, ``style`` and ``weight``. Labels are placed
    greedily -- lowest ``rank`` first, then least crowded -- each taking the
    first offset in ``LABEL_OFFSETS`` whose box clears every anchor and every
    label already placed, so a lower rank claims the space it wants and later
    ones work around it. Anything pushed past the inner ring gets a leader line
    back to its anchor. Placement happens in pixels, so the figure is laid out
    once first to settle the axes size, and works against whatever the axes are
    showing -- lon/lat on the map, distance/IM on the scatter.
    """
    if not entries:
        return
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72  # points -> pixels
    xy = ax.transData.transform(
        np.column_stack([[e["x"] for e in entries], [e["y"] for e in entries]])
    )
    # Anchors with nothing to point at: no observed value for this IM, or a
    # non-positive value that a log axis cannot place.
    on_axes = np.isfinite(xy).all(axis=1)
    entries = [entry for entry, keep in zip(entries, on_axes) if keep]
    xy = xy[on_axes]
    if not entries:
        return

    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    texts = [
        ax.annotate(
            entry["text"],
            (entry["x"], entry["y"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            va="bottom",
            fontsize=entry.get("size", 6),
            style=entry.get("style", "normal"),
            weight=entry.get("weight", "normal"),
            linespacing=0.9,
            color=entry.get("colour", "black"),
            path_effects=halo,
            zorder=7,
        )
        for entry in entries
    ]
    sizes = [
        (bbox.width, bbox.height)
        for bbox in (text.get_window_extent(renderer) for text in texts)
    ]

    # Anchors are obstacles too, so labels never land on a neighbouring marker.
    occupied = [(x - 6, y - 6, x + 6, y + 6) for x, y in xy]
    crowding = cKDTree(xy).query_ball_point(xy, r=60, return_length=True)
    ranks = np.array([entry.get("rank", 0) for entry in entries])

    for index in np.lexsort((crowding, ranks)):
        (x, y), (width, height) = xy[index], sizes[index]
        for dx, dy, ha, va in LABEL_OFFSETS:
            box = _label_box(x + dx * scale, y + dy * scale, width, height, ha, va)
            if not any(_overlaps(box, other) for other in occupied):
                break
        texts[index].set_position((dx, dy))
        texts[index].set_ha(ha)
        texts[index].set_va(va)
        occupied.append(box)
        if abs(dx) > 12 or abs(dy) > 12:
            edge = ax.transData.inverted().transform(
                [(x, y), (x + dx * scale, y + dy * scale)]
            )
            ax.plot(
                *edge.T,
                color=entries[index].get("colour", "#444444"),
                lw=0.4,
                zorder=5,
                solid_capstyle="butt",
            )


def station_labels(
    names: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    values: np.ndarray | None = None,
) -> list[dict]:
    """Label entries for stations: name over value, and first claim on space."""
    if values is None:
        values = np.full(len(names), np.nan)
    return [
        {
            "text": name if not np.isfinite(value) else f"{name}\n{value:.3g}",
            "x": x,
            "y": y,
            "colour": "black",
            "rank": 0,
        }
        for name, x, y, value in zip(names, x_data, y_data, values)
    ]


def corner_anchor(rect: tuple[float, float, float, float]) -> tuple[str, tuple]:
    """A matplotlib ``loc`` and ``bbox_to_anchor`` filling a :func:`free_corner`.

    ``free_corner`` answers "which corner is emptiest" with a rectangle; a
    legend wants to be told which of its own corners to pin and where. Pinning
    the corner of the legend that faces the middle of the axes is what keeps it
    growing inwards, into the free space, rather than off the edge.
    """
    cx, cy, width, height = rect
    vertical, y = ("lower", cy) if cy < 0.5 else ("upper", cy + height)
    horizontal, x = ("left", cx) if cx < 0.5 else ("right", cx + width)
    return f"{vertical} {horizontal}", (x, y)
