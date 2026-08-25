"""Site terms derived from the coordinates the database already holds.

Two properties of a site that the IM files do not carry and that a residual
analysis wants: how high the station sits, and which sedimentary basin it sits
in. Both are functions of latitude and longitude alone, and ``run_stations``
already holds those for every station of every run -- so neither needs the HDF5
tree reopened. That is the whole point of deriving them here rather than staging
them with the rest: filling these two columns costs a re-assembly, seconds of
work over a finished database, instead of reconverting tens of gigabytes.

Neither input is required, and a missing one leaves its column NULL rather than
guessing. A zero elevation would read as sea level, and a station's basin is not
``none`` merely because nobody tested it; NULL is the only honest value for a
covariate whose source was not supplied.

Two facts about the inputs, both checked here rather than trusted, because
getting either wrong would be invisible in a plot:

* The NZCVM DEM stores **depth, positive down** -- Aoraki reads about -3500 and
  Christchurch about -5 -- so the elevation is its negation. Read the sign
  backwards and every station goes below sea level, which is obvious, but the
  result would still correlate almost perfectly with the truth, which is not.
* The DEM's ``x``/``y`` are NZTM metres on a grid that is genuinely curvilinear
  in projection, yet the underlying graticule is regular in longitude and
  latitude. Recovering those axes turns sampling into index arithmetic; assuming
  it without checking would sample a rotated DEM at confidently wrong indices.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import shapely
import xarray as xr
from pyproj import Transformer

from .console import console_warn
from .geography import load_basins

# The DEM is NZTM2000. The zarr records no CRS of its own, so it is stated here;
# nzcvm names the same code in its own coordinate handling.
DEM_CRS = "EPSG:2193"

# How far apart two nodes' recovered longitudes may sit and still be called the
# same meridian. The measured disagreement on the NZCVM DEM is about 5e-6
# degrees against a 0.005 degree step, so this is generous by two orders of
# magnitude and still refuses a rotated grid outright.
GRATICULE_TOLERANCE = 1e-4


def dem_axes(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """The DEM's longitude and latitude axes, recovered from its projected grid.

    The zarr carries ``x`` and ``y`` in NZTM metres over dims ``(i, j)`` and no
    geographic coordinate at all, so the axes have to be recovered: the first
    column is inverse-projected to get the longitude of each ``i``, the first
    row to get the latitude of each ``j``.

    Recovered rather than assumed, because the projected grid really is
    curvilinear -- predict the fourth corner of the NZCVM DEM from the other
    three and you miss by 284 km -- which is exactly what a regular graticule
    looks like once it has been projected. That the grid separates this way is
    then checked against every node, so a DEM that is rotated or irregular fails
    here with a message instead of being sampled at wrong indices.

    Longitudes are unwrapped onto one continuous strip: the last node of the
    NZCVM DEM sits on the antimeridian and comes back as -180.
    """
    inverse = Transformer.from_crs(DEM_CRS, "EPSG:4326", always_xy=True)
    x = dataset.x.values
    y = dataset.y.values

    # Every node, not just the first row and column: the check is the point.
    lon, lat = inverse.transform(x, y)
    lon = np.unwrap(lon, period=360.0, axis=0)
    lon = np.unwrap(lon, period=360.0, axis=1)

    spread_lon = float(np.nanmax(np.ptp(lon, axis=1)))
    spread_lat = float(np.nanmax(np.ptp(lat, axis=0)))
    if spread_lon > GRATICULE_TOLERANCE or spread_lat > GRATICULE_TOLERANCE:
        raise ValueError(
            f"{DEM_CRS} inverse of this DEM is not a regular graticule: "
            f"longitude varies by up to {spread_lon:.2e} deg along a column and "
            f"latitude by {spread_lat:.2e} deg along a row, against a tolerance "
            f"of {GRATICULE_TOLERANCE:.0e}. Sampling it by index would be wrong; "
            "it needs a nearest-neighbour search instead"
        )
    return lon.mean(axis=1), lat.mean(axis=0)


@dataclass(frozen=True)
class Dem:
    """A DEM resolved onto its graticule, ready to sample by index.

    Held as an object because recovering the axes inverse-projects every node --
    three seconds, once -- and the depths are 36 MB, so a caller sampling twice
    should not pay for either twice.
    """

    longitude: np.ndarray   # one value per i
    latitude: np.ndarray    # one value per j
    depth: np.ndarray       # (i, j), positive down


def read_dem(dem: Path) -> Dem:
    """Load a DEM and recover the graticule it is sampled on.

    The whole of ``z`` is read at once -- 36 MB, a fraction of a second -- rather
    than the chunks being visited station by station: at a hundred thousand
    scattered coordinates every chunk is touched anyway, so chunk-wise
    cleverness buys a slower version of the same read.
    """
    with xr.open_zarr(dem) as dataset:
        longitude, latitude = dem_axes(dataset)
        return Dem(longitude, latitude, dataset.z.values)


def sample_dem(dem: Dem, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Elevation in metres above sea level at each coordinate, nearest DEM node.

    Nearest node rather than bilinear. The grid is 0.005 degrees, about 550 m,
    and a station's own coordinate differs between solvers by up to 0.0023
    degrees, so interpolating would claim a precision the input does not have.

    **The sign is flipped**, because the DEM stores depth positive down. See the
    module docstring for why that is worth a paragraph.

    The sea is clamped to zero in the DEM itself, so a station over water reads
    0.0 m rather than NULL -- that is the DEM's answer, not a missing value, and
    it is left alone. A coordinate off the grid comes back NaN, which the caller
    writes as SQL NULL.
    """
    axis_lon, axis_lat, depth = dem.longitude, dem.latitude, dem.depth
    step_lon = float(np.mean(np.diff(axis_lon)))
    step_lat = float(np.mean(np.diff(axis_lat)))

    # Onto the DEM's own strip, so a station east of the antimeridian -- stored
    # as a negative longitude -- is looked up where it is rather than off the
    # western edge.
    shifted = np.where(lon < axis_lon[0] - 180.0, lon + 360.0, lon)

    i = np.rint((shifted - axis_lon[0]) / step_lon)
    j = np.rint((lat - axis_lat[0]) / step_lat)
    inside = (
        np.isfinite(i) & np.isfinite(j)
        & (i >= 0) & (i < len(axis_lon))
        & (j >= 0) & (j < len(axis_lat))
    )

    elevation = np.full(lon.shape, np.nan)
    rows = i[inside].astype(np.intp)
    columns = j[inside].astype(np.intp)
    elevation[inside] = -depth[rows, columns]
    return elevation


def assign_basins(
    lon: np.ndarray,
    lat: np.ndarray,
    basins: list[tuple[str, shapely.Geometry]],
) -> tuple[np.ndarray, int]:
    """The basin containing each coordinate, and how many matched more than one.

    The basin is None where the coordinate is in none of them.

    :func:`eqvis_workflow.geography.read_basins` has already done the hard part:
    one merged outline per basin, each clipped against everything of higher
    priority, so the outlines do not overlap and a point falls in at most one.
    That is checked rather than assumed -- the count of coordinates matching two
    outlines is returned to the caller to report -- and where it does happen the
    lowest-priority basin wins, which is the rule ``read_basins`` used to build
    the outlines in the first place.

    One vectorised :func:`shapely.contains_xy` per basin over the whole
    coordinate array, rather than an ``STRtree`` query. Measured on this data,
    46 containment tests over 103,056 points take 0.16 s where building and
    querying the tree took 0.95 s: the tree is the right structure for a few
    points against many polygons and the wrong one for many points against 46.
    """
    assigned = np.full(lon.shape, None, dtype=object)
    hits = np.zeros(lon.shape, dtype=np.int32)
    for name, geometry in basins:
        inside = shapely.contains_xy(geometry, lon, lat)
        # First outline wins, so the earlier (higher priority) name is kept.
        assigned = np.where(inside & (hits == 0), name, assigned)
        hits += inside
    return assigned, int((hits > 1).sum())


def station_coordinates(con) -> pa.Table:
    """One coordinate per station, and how much the runs disagree about it.

    ``stations`` is keyed on the station name but ``run_stations`` is keyed on
    ``(run_id, station)``, and the solvers snap stations to their own grids -- on
    this data 102,537 of 103,056 stations have coordinates that differ between
    runs, by up to 0.0023 degrees. So a single coordinate has to be chosen, and
    the mean is chosen. The extremes come back too, so the caller can say how
    many stations would have landed on a different DEM cell or in a different
    basin had a corner been picked instead. Stating the size of the choice beats
    pretending there was not one.
    """
    return con.execute(
        """
        SELECT station,
               avg(longitude) AS longitude, avg(latitude) AS latitude,
               min(longitude) AS lon_lo, max(longitude) AS lon_hi,
               min(latitude)  AS lat_lo, max(latitude)  AS lat_hi
        FROM run_stations
        GROUP BY station
        ORDER BY station
        """
    ).to_arrow_table()


def enrich_stations(
    con,
    dem: Path | None = None,
    basins: Path | None = None,
    with_basins: bool = True,
) -> dict[str, int]:
    """Fill ``stations.elevation`` and ``stations.basin`` from stored coordinates.

    The last assembly step of :func:`eqvis_workflow.database.build`, and not a
    command of its own: the columns then exist -- NULL when their input was not
    supplied -- after every build, so the schema never varies with how the
    database was made.

    Idempotent: it recomputes and overwrites rather than filling only the NULLs,
    so running it again with a newer DEM does what the caller meant.

    Returns the counts it wrote, for the build report.
    """
    counts: dict[str, int] = {}
    if dem is None and not with_basins:
        return counts

    table = station_coordinates(con)
    station = table.column("station").to_numpy(zero_copy_only=False)
    lon = table.column("longitude").to_numpy()
    lat = table.column("latitude").to_numpy()
    counts["stations"] = len(station)

    columns: dict[str, pa.Array] = {"station": pa.array(station)}

    if dem is not None:
        grid = read_dem(dem)
        elevation = sample_dem(grid, lon, lat)
        counts["elevation"] = int(np.isfinite(elevation).sum())
        off_grid = len(elevation) - counts["elevation"]
        if off_grid:
            console_warn(
                f"{off_grid} stations fall outside {dem}; their elevation is NULL"
            )
        # How much the coordinate choice mattered: the corners of the per-run
        # spread, against the mean that was actually used.
        corner = sample_dem(
            grid,
            table.column("lon_lo").to_numpy(),
            table.column("lat_lo").to_numpy(),
        )
        moved = int(np.sum(np.abs(corner - elevation) > 1e-6))
        counts["elevation_coordinate_sensitive"] = moved
        # NaN reaches the database as SQL NULL, never as NaN: one NaN in the
        # column makes AVG over it NaN, where one NULL is skipped. Same move as
        # :func:`eqvis_workflow.ingest.float_array`, inlined because `ingest`
        # imports `database`, which imports this module.
        columns["elevation"] = pa.array(
            elevation, type=pa.float32(), mask=np.isnan(elevation)
        )

    if with_basins:
        outlines = load_basins(basins)
        if outlines is None:
            console_warn("no basin outlines available; stations.basin stays NULL")
        else:
            assigned, overlapping = assign_basins(lon, lat, outlines)
            counts["basin"] = int(sum(name is not None for name in assigned))
            counts["basin_none"] = len(assigned) - counts["basin"]
            if overlapping:
                console_warn(
                    f"{overlapping} stations fall in more than one basin outline; "
                    "taking the highest-priority one. The outlines are meant to "
                    "be clipped against each other, so this is worth looking into"
                )
            columns["basin"] = pa.array(
                [None if name is None else str(name) for name in assigned]
            )

    if len(columns) == 1:  # nothing but the key: nothing to write
        return counts

    assignments = ", ".join(f"{name} = t.{name}" for name in columns if name != "station")
    con.register("station_terms", pa.table(columns))
    con.execute(
        f"""
        UPDATE stations s SET {assignments}
        FROM station_terms t
        WHERE t.station = s.station
        """
    )
    con.unregister("station_terms")
    return counts
