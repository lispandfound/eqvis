"""Tests for the site terms derived from stored coordinates.

The DEM fixture is a real zarr on a real graticule, projected to NZTM the same
way the NZCVM DEM is, so the axis recovery is exercised rather than mocked. It
carries a known elevation field, which is what makes the sign convention -- the
one thing here that would be invisible in a plot if it were wrong -- assertable.
"""

import duckdb
import numpy as np
import pytest
import shapely
import xarray as xr
from pyproj import Transformer

from eqvis_workflow import enrich

# A graticule over the South Island, coarse enough to be quick and fine enough
# that nearest-node lookup is unambiguous.
LON = np.arange(170.0, 172.0001, 0.01)
LAT = np.arange(-44.0, -42.9999, 0.01)


def elevation_field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A smooth, invertible test surface: 1000 m per degree east, 500 m north."""
    return 1000.0 * (lon - 170.0) + 500.0 * (lat + 44.0)


@pytest.fixture(scope="module")
def dem(tmp_path_factory):
    """A DEM zarr in the NZCVM's own shape: NZTM x/y over (i, j), depth in z."""
    path = tmp_path_factory.mktemp("dem") / "dem.zarr"
    grid_lon, grid_lat = np.meshgrid(LON, LAT, indexing="ij")
    forward = Transformer.from_crs("EPSG:4326", enrich.DEM_CRS, always_xy=True)
    x, y = forward.transform(grid_lon, grid_lat)
    xr.Dataset(
        {
            "x": (("i", "j"), x.astype(np.float32)),
            "y": (("i", "j"), y.astype(np.float32)),
            # Depth, positive down: the negation of the elevation.
            "z": (("i", "j"), -elevation_field(grid_lon, grid_lat).astype(np.float32)),
        },
        coords={"i": np.arange(len(LON)), "j": np.arange(len(LAT))},
    ).to_zarr(path)
    return path


@pytest.fixture(scope="module")
def basins():
    """Two disjoint square basins, and one overlapping a third, for the tie rule."""
    return [
        ("Alpha", shapely.box(170.2, -43.8, 170.6, -43.4)),
        ("Beta", shapely.box(171.2, -43.8, 171.6, -43.4)),
    ]


class TestAxisRecovery:
    def test_the_graticule_is_recovered_from_the_projected_grid(self, dem):
        """The zarr carries only NZTM metres, so the axes have to be inverted out."""
        with xr.open_zarr(dem) as dataset:
            axis_lon, axis_lat = enrich.dem_axes(dataset)
        assert axis_lon == pytest.approx(LON, abs=1e-6)
        assert axis_lat == pytest.approx(LAT, abs=1e-6)

    def test_a_rotated_grid_is_refused_rather_than_sampled(self, tmp_path):
        """Sampling a rotated DEM by index would give confidently wrong answers,
        so the separability of the graticule is checked, not assumed."""
        grid_lon, grid_lat = np.meshgrid(LON, LAT, indexing="ij")
        angle = np.deg2rad(20.0)
        turned_lon = 170.0 + (grid_lon - 170.0) * np.cos(angle) - (
            grid_lat + 43.5
        ) * np.sin(angle)
        turned_lat = -43.5 + (grid_lon - 170.0) * np.sin(angle) + (
            grid_lat + 43.5
        ) * np.cos(angle)
        forward = Transformer.from_crs("EPSG:4326", enrich.DEM_CRS, always_xy=True)
        x, y = forward.transform(turned_lon, turned_lat)
        path = tmp_path / "rotated.zarr"
        xr.Dataset(
            {
                "x": (("i", "j"), x.astype(np.float32)),
                "y": (("i", "j"), y.astype(np.float32)),
                "z": (("i", "j"), np.zeros_like(x, dtype=np.float32)),
            },
            coords={"i": np.arange(len(LON)), "j": np.arange(len(LAT))},
        ).to_zarr(path)
        with xr.open_zarr(path) as dataset:
            with pytest.raises(ValueError, match="not a regular graticule"):
                enrich.dem_axes(dataset)


class TestSampling:
    def test_the_sign_is_flipped_from_depth_to_elevation(self, dem):
        """The DEM stores depth positive down. Getting this backwards would still
        correlate almost perfectly with the truth, so it needs its own test."""
        lon = np.array([170.5, 171.0, 171.5])
        lat = np.array([-43.5, -43.5, -43.2])
        got = enrich.sample_dem(enrich.read_dem(dem), lon, lat)
        assert got == pytest.approx(elevation_field(lon, lat), abs=6.0)
        assert np.all(got > 0)

    def test_a_coordinate_off_the_grid_is_null_not_clamped(self, dem):
        """An off-grid station must not silently take the nearest edge value."""
        got = enrich.sample_dem(
            enrich.read_dem(dem), np.array([151.2, 170.5]), np.array([-33.9, -43.5])
        )
        assert np.isnan(got[0])
        assert np.isfinite(got[1])

    def test_sampling_is_nearest_node_not_interpolated(self, dem):
        """Two coordinates inside one cell must give the same answer: the grid is
        coarser than the disagreement between two solvers' coordinates."""
        grid = enrich.read_dem(dem)
        a = enrich.sample_dem(grid, np.array([170.5001]), np.array([-43.5001]))
        b = enrich.sample_dem(grid, np.array([170.5004]), np.array([-43.5004]))
        assert a[0] == b[0]


class TestBasinAssignment:
    def test_a_station_gets_the_basin_containing_it(self, basins):
        lon = np.array([170.4, 171.4, 170.9])
        lat = np.array([-43.6, -43.6, -43.6])
        assigned, overlapping = enrich.assign_basins(lon, lat, basins)
        assert list(assigned) == ["Alpha", "Beta", None]
        assert overlapping == 0

    def test_being_in_no_basin_is_none_and_is_not_an_error(self, basins):
        """NULL here means "in no basin", which is a fact about the site."""
        assigned, _ = enrich.assign_basins(
            np.array([160.0]), np.array([-40.0]), basins
        )
        assert assigned[0] is None

    def test_overlapping_outlines_are_counted_and_the_first_wins(self):
        """read_basins clips the outlines so this should not happen; if it does,
        the count is reported rather than the ambiguity being hidden."""
        overlapping_pair = [
            ("First", shapely.box(170.0, -44.0, 171.0, -43.0)),
            ("Second", shapely.box(170.5, -43.5, 171.5, -42.5)),
        ]
        assigned, count = enrich.assign_basins(
            np.array([170.7]), np.array([-43.2]), overlapping_pair
        )
        assert assigned[0] == "First"
        assert count == 1


@pytest.fixture
def database():
    """The two tables the enrichment reads and writes, and nothing else."""
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE stations (
            station VARCHAR PRIMARY KEY, vs30 FLOAT, z1pt0 FLOAT, z2pt5 FLOAT,
            elevation FLOAT, basin VARCHAR
        );
        CREATE TABLE run_stations (
            run_id INTEGER, station VARCHAR, latitude DOUBLE, longitude DOUBLE
        );
        """
    )
    con.executemany(
        "INSERT INTO stations VALUES (?, 400, 0.1, 1.0, NULL, NULL)",
        [("INBASIN",), ("OUTSIDE",), ("OFFGRID",)],
    )
    con.executemany(
        "INSERT INTO run_stations VALUES (?, ?, ?, ?)",
        [
            # Two runs disagree slightly about each site, as the solvers do.
            (1, "INBASIN", -43.60, 170.40), (2, "INBASIN", -43.601, 170.401),
            (1, "OUTSIDE", -43.20, 170.90), (2, "OUTSIDE", -43.201, 170.901),
            (1, "OFFGRID", -33.90, 151.20), (2, "OFFGRID", -33.901, 151.201),
        ],
    )
    yield con
    con.close()


class TestEnrichStations:
    def test_both_columns_are_filled_from_stored_coordinates(
        self, database, dem, basins, monkeypatch
    ):
        """The whole point: no IM file is opened, only run_stations is read."""
        monkeypatch.setattr(enrich, "load_basins", lambda source: basins)
        counts = enrich.enrich_stations(database, dem=dem)
        assert counts["stations"] == 3
        assert counts["elevation"] == 2  # OFFGRID is off the DEM
        assert counts["basin"] == 1
        assert counts["basin_none"] == 2
        rows = dict(
            database.execute(
                "SELECT station, basin FROM stations ORDER BY station"
            ).fetchall()
        )
        assert rows == {"INBASIN": "Alpha", "OUTSIDE": None, "OFFGRID": None}

    def test_no_dem_leaves_elevation_null_rather_than_zero(
        self, database, basins, monkeypatch
    ):
        """A zero would read as sea level. NULL is the only honest missing value."""
        monkeypatch.setattr(enrich, "load_basins", lambda source: basins)
        enrich.enrich_stations(database, dem=None)
        assert database.execute("SELECT count(elevation) FROM stations").fetchone() == (
            0,
        )
        assert database.execute("SELECT count(basin) FROM stations").fetchone() == (1,)

    def test_no_basins_leaves_basin_null(self, database, dem):
        enrich.enrich_stations(database, dem=dem, with_basins=False)
        assert database.execute("SELECT count(basin) FROM stations").fetchone() == (0,)
        assert database.execute("SELECT count(elevation) FROM stations").fetchone() == (
            2,
        )

    def test_it_is_idempotent(self, database, dem, basins, monkeypatch):
        """Run again with a newer DEM and it recomputes, rather than filling only
        the NULLs and leaving a mix of two vintages."""
        monkeypatch.setattr(enrich, "load_basins", lambda source: basins)
        first = enrich.enrich_stations(database, dem=dem)
        before = database.execute(
            "SELECT station, elevation, basin FROM stations ORDER BY station"
        ).fetchall()
        second = enrich.enrich_stations(database, dem=dem)
        after = database.execute(
            "SELECT station, elevation, basin FROM stations ORDER BY station"
        ).fetchall()
        assert first == second
        assert before == after

    def test_nothing_asked_for_does_nothing(self, database):
        enrich.enrich_stations(database, dem=None, with_basins=False)
        assert database.execute(
            "SELECT count(elevation) + count(basin) FROM stations"
        ).fetchone() == (0,)
