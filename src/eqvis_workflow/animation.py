"""Animate an EMOD3D XYTS ground-motion wavefield over New Zealand topography.

This renders a 3D scene of the simulation domain: land topography (from the
NZCVM DEM), the fault (its surface footprint and epicentre, from the SRF), the
domain outline, and the ground-velocity wavefield (from the merged XYTS file)
as a glowing overlay sheet above the terrain, animated through time::

    eqvis animate realisation.xyts --srf realisation.srf -o waveform.mp4
    eqvis animate realisation.xyts --preview      # one frame to PNG, for iterating

The XYTS file is the quantised, merged output of ``merge-ts``; its ``waveform``
variable is the magnitude of the ground velocity (cm/s). Its stored lat/lon are
unreliable, so the grid is rebuilt from the model's spherical projection
(mlon/mlat/mrot + dx) and reprojected to NZTM to line up with the DEM and the
fault.

Everything is drawn in NZTM metres (easting=X, northing=Y) with up being
positive Z. Depths/elevations are multiplied by a uniform vertical
exaggeration so the topography and fault read in a wide (~500 km) domain.

This is the one command that is not a matplotlib figure -- a shaded terrain
seen in perspective is not something matplotlib's 3D axes can draw -- but its
colours still come from matplotlib, so the wavefield reads the same here as it
does in the maps.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import shapely
import typer
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from qcore import coordinates
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from source_modelling import srf
from tqdm import tqdm

from .console import console_warn
from .constants import GRAVITY_CM_S2
from .geography import draw_coastline, load_coastline

DEFAULT_DEM = Path("/home/jake/src/nzcvm/resources/dem.zarr")

# The sea, and the background it sits against -- the latter slightly darker to
# account for the effects of lighting.
BLUE = "#000e5c"
BACKGROUND = "#000e4e"


def _pyvista():
    """Import PyVista on demand.

    It pulls VTK in behind it, which takes a second or so to load and is needed
    by this one command alone; importing it at module scope would put that
    second on the front of every other command in the CLI.
    """
    import pyvista

    return pyvista


def load_dem(dem_path: Path, bounds, margin_m, max_side, vertical_exaggeration):
    """Load and crop the DEM to the domain bounds, returning a StructuredGrid.

    The DEM ``z`` is stored depth-positive-down (land is negative), so surface
    elevation is ``-z``. The grid is curvilinear in NZTM, so we crop the
    rectangular index block that covers the bounding box and stride it down to
    at most ``max_side`` points per axis.
    """
    pv = _pyvista()
    dem = xr.open_zarr(dem_path)
    east = dem.x.values  # (i, j) NZTM easting
    north = dem.y.values  # (i, j) NZTM northing
    depth = dem.z.values  # (i, j) depth (+down)

    emin, emax, nmin, nmax = bounds
    inside = (
        (east >= emin - margin_m)
        & (east <= emax + margin_m)
        & (north >= nmin - margin_m)
        & (north <= nmax + margin_m)
    )
    if not inside.any():
        raise ValueError("DEM does not cover the simulation domain.")
    irows = np.where(inside.any(axis=1))[0]
    icols = np.where(inside.any(axis=0))[0]
    i0, i1 = irows[0], irows[-1] + 1
    j0, j1 = icols[0], icols[-1] + 1

    si = max(1, (i1 - i0) // max_side)
    sj = max(1, (j1 - j0) // max_side)
    east = east[i0:i1:si, j0:j1:sj]
    north = north[i0:i1:si, j0:j1:sj]
    elev = -depth[i0:i1:si, j0:j1:sj]  # up-positive metres

    # Land height threshold for the colormap, off the 99th percentile so one
    # summit does not flatten the rest of the range.
    valid_land = elev[elev > 0]
    dem_height = (
        float(max(1.0, np.percentile(valid_land, 99))) if valid_land.size > 0 else 1.0
    )

    # gist_earth opens on a blue that would read as more sea, so the ramp is
    # cut to its land half and the sea painted separately underneath.
    land_colors = mpl.colormaps["gist_earth"].resampled(256)(np.linspace(0.5, 1.0, 256))
    land_cmap = mpl.colors.LinearSegmentedColormap.from_list("land", land_colors)
    norm = mpl.colors.Normalize(vmin=0.001, vmax=dem_height, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=land_cmap)
    rgb_colors = mapper.to_rgba(elev)[:, :, :3]
    rgb_colors[elev <= 0.0] = mpl.colors.to_rgb(BLUE)

    grid = pv.StructuredGrid(
        east[..., None], north[..., None], (elev * vertical_exaggeration)[..., None]
    )
    # PyVista reads a 3-component float array as direct colours.
    grid.point_data["colors"] = rgb_colors.reshape(-1, 3, order="F")

    return grid, east, north, elev


def load_fault(srf_path: Path):
    """Return the fault's surface-projected footprint corners and epicentre.

    A fault can be small and deep (tens of kilometres down), so instead of
    drawing it far below the surface we project its outline (and the epicentre)
    up onto the map. Both are returned as NZTM (easting, northing); the caller
    drapes them onto the terrain.
    """
    plane = srf.read_srf(srf_path).planes[0]
    corners = plane.bounds  # (4, 3) NZTM -> [northing, easting, depth]
    corners_en = corners[:, [1, 0]]  # -> easting, northing
    epicentre_en = corners_en.mean(axis=0)
    return corners_en, epicentre_en


def load_domain(realisation_path: Path):
    """Return the simulation domain outline corners in NZTM (easting, northing).

    The 4 corners are stored in lat/lon in the realisation JSON; they are
    reprojected to NZTM for rendering in the 3D scene.
    """
    with open(realisation_path) as f:
        data = json.load(f)
    corners = data["domain"]["domain"]
    lats = [c["latitude"] for c in corners]
    lons = [c["longitude"] for c in corners]
    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    east, north = to_nztm.transform(lons, lats)
    return np.column_stack([east, north])


def build_dashed_polyline(vertices, dash_len=5000, gap_len=3000):
    """Build a dashed polyline from a closed polygon using on/off segments.

    This produces actual disconnected line segments rather than relying on
    render-backend-dependent stipple patterns (which are unreliable in modern
    VTK OpenGL2). Each edge is walked in (dash + gap) steps; only the dash
    portions are kept as individual line segments in the returned PolyData.

    Parameters
    ----------
    vertices : (N, 3) array of xyz points forming a closed polygon.
    dash_len : length of each visible dash (metres, in scene units).
    gap_len  : length of each hidden gap (metres).

    Returns
    -------
    pv.PolyData with each dash as a disconnected 2-point line.
    """
    pv = _pyvista()
    vertices = np.asarray(vertices)
    ring = np.vstack([vertices, vertices[0]])
    segments = []
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        edge = b - a
        edge_len = np.linalg.norm(edge)
        if edge_len < 1:
            continue
        direction = edge / edge_len
        step = dash_len + gap_len
        t = 0.0
        while t < edge_len:
            end = min(t + dash_len, edge_len)
            segments.append([a + direction * t, a + direction * end])
            t += step
    if not segments:
        return pv.PolyData()
    points = np.vstack(segments)
    n = len(segments)
    lines = np.column_stack(
        [
            np.full(n, 2, dtype=np.intp),
            np.arange(0, 2 * n, 2, dtype=np.intp),
            np.arange(1, 2 * n, 2, dtype=np.intp),
        ]
    ).ravel()
    return pv.PolyData(points, lines=lines)


def wave_colormap():
    """Transparent-black -> hot wavefield colormap."""
    return LinearSegmentedColormap.from_list(
        "wave", ["#101020", "#3b0f70", "#b5367a", "#f9762a", "#fcffa4"]
    )


def get_nice_vmax(vmax: float) -> float:
    """Return the nearest clean upper bound for a scale ranging from 0 to vmax."""
    if vmax <= 0:
        return 1.0

    exponent = np.floor(np.log10(vmax))
    fraction = vmax / (10**exponent)

    # Matplotlib's standard nice steps: 1, 2, 2.5, 5, 10.
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0

    return nice_fraction * (10**exponent)


def is_broadband(path: Path) -> bool:
    """Whether this is a broadband station file rather than an XYTS wavefield.

    Told apart by shape, not by name. An XYTS file carries the wavefield on the
    simulation grid -- ``waveform(nt, ny, nx)`` with the grid's origin and
    rotation as root attributes -- while a broadband file carries one trace per
    station, so it has a ``station`` dimension and the XYTS one does not.
    """
    with xr.open_dataset(path, engine="h5netcdf", mask_and_scale=False) as data:
        return "station" in data.dims


def land_stations(
    latitude: np.ndarray, longitude: np.ndarray, coast
) -> np.ndarray:
    """Which stations are on land.

    A simulation domain is mostly sea, and a sea of markers shaking is both
    slower to draw and harder to read than the land alone -- there is nothing
    there to shake. Without a coastline every station is kept, because dropping
    them all would be worse than drawing too many.
    """
    if coast is None:
        console_warn("no coastline available; animating every station, sea included")
        return np.ones(latitude.shape, dtype=bool)
    return shapely.contains_xy(coast, longitude, latitude)


def integrate_snapshots(
    acceleration, dt: float, stride: int, block: int = 2000
) -> np.ndarray:
    """Velocity at every ``stride``-th sample, in cm/s, without holding it all.

    Integration needs every sample that came before, but the animation needs
    only one in ``stride`` of them. Carrying a running total across blocks of
    time gives both: the peak memory is one block rather than the whole trace,
    which for five thousand stations at thirty thousand samples is the
    difference between a few megabytes and a few gigabytes.
    """
    total = np.zeros(acceleration.shape[0])
    frames = []
    for start in range(0, acceleration.shape[1], block):
        chunk = np.asarray(
            acceleration[:, start : start + block], dtype=np.float64
        )
        running = total[:, None] + np.cumsum(chunk, axis=1) * dt * GRAVITY_CM_S2
        # Offset within this block of the samples the frames land on.
        first = (-start) % stride
        frames.append(running[:, first::stride])
        total = running[:, -1]
    return np.concatenate(frames, axis=1)


def animate_broadband(
    bb_file: Path,
    output: Path,
    coastline: Path | None,
    time_stride: int,
    fps: int,
    clip_percentile: float,
    min_scale: float,
    title: str | None,
    preview: bool,
    size: tuple[int, int],
) -> None:
    """Animate station ground velocity in plan view -- the poor man's wavefield.

    The XYTS path renders the wavefield itself over topography, which is the
    honest picture and needs the wavefield to have been kept. This draws what the
    broadband output does keep: one trace per station, shown as a shaken map of
    markers. It is coarse where the station spacing is coarse and says nothing
    between stations, but it is enough to see whether a disturbance travels like a
    wave -- outward, decaying -- or does something else.
    """
    with xr.open_dataset(bb_file, engine="h5netcdf", mask_and_scale=False) as data:
        latitude = data.latitude.values.astype(float)
        longitude = data.longitude.values.astype(float)
        time = data.time.values.astype(float)
        dt = float(data.attrs.get("dt", time[1] - time[0]))
        coast = load_coastline(coastline)
        on_land = land_stations(latitude, longitude, coast)
        if not on_land.any():
            raise typer.BadParameter(
                "no station is on land, so there is nothing to animate. Pass "
                "--coastline to point at a coastline that covers this domain"
            )
        print(
            f"{on_land.sum():,} of {len(latitude):,} stations on land; "
            f"{len(time):,} samples at {dt:g} s"
        )
        # Horizontal only: the vertical is a different quantity and averaging
        # the three would show neither.
        speed = None
        for component in ("x", "y"):
            trace = data.waveform.sel(component=component).values[on_land]
            velocity = integrate_snapshots(trace, dt, time_stride)
            speed = velocity**2 if speed is None else speed + velocity**2
        speed = np.sqrt(speed)

    lon, lat = longitude[on_land], latitude[on_land]
    stamps = time[:: time_stride][: speed.shape[1]]
    limit = get_nice_vmax(float(np.percentile(np.abs(speed), clip_percentile)))
    print(f"{speed.shape[1]:,} frames; colour scale to {limit:g} cm/s")

    figure, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    figure.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    bounds = (lon.min() - 0.3, lat.min() - 0.3, lon.max() + 0.3, lat.max() + 0.3)
    if coast is not None:
        draw_coastline(ax, coast, bounds)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(lat)))))
    ax.set_axis_off()
    scatter = ax.scatter(
        lon, lat, c=np.zeros_like(lon), s=6, cmap=wave_colormap(),
        vmin=0.0, vmax=limit, linewidths=0, zorder=3,
    )
    bar = figure.colorbar(scatter, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("ground speed (cm/s)", color="white")
    bar.ax.tick_params(colors="white")
    caption = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=11, family="monospace",
    )

    def draw(index: int) -> None:
        values = speed[:, index]
        # Below the floor the marker is left dark rather than tinted, so a quiet
        # map reads as quiet instead of as a uniform wash.
        scatter.set_array(np.where(values < min_scale, 0.0, values))
        caption.set_text(
            f"{(title + chr(10)) if title else ''}t = {stamps[index]:6.1f} s"
        )

    if preview:
        draw(int(np.argmax(np.abs(speed).max(axis=0))))
        target = output.with_suffix(".png")
        figure.savefig(target, dpi=100, facecolor=BACKGROUND)
        print(f"wrote {target} (the loudest frame)")
        return

    import imageio.v2 as imageio

    with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
        for index in range(speed.shape[1]):
            draw(index)
            figure.canvas.draw()
            writer.append_data(
                np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            )
    print(f"wrote {output} ({speed.shape[1]:,} frames)")


def animate(
    xyts: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Merged XYTS wavefield, or a broadband station file. Which one "
            "it is is detected from the file's own structure",
        ),
    ],
    srf_path: Annotated[
        Path | None,
        typer.Option(
            "--srf",
            exists=True,
            dir_okay=False,
            help="SRF file: draws the fault footprint and epicentre",
        ),
    ] = None,
    realisation: Annotated[
        Path | None,
        typer.Option(
            "--realisation",
            exists=True,
            dir_okay=False,
            help="Realisation JSON: draws the simulation domain outline",
        ),
    ] = None,
    dem: Annotated[
        Path, typer.Option(exists=True, help="DEM zarr store")
    ] = DEFAULT_DEM,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output mp4 (default: waveform.mp4 beside the XYTS)"
        ),
    ] = None,
    title: Annotated[
        str | None, typer.Option(help="Caption drawn in the corner of every frame")
    ] = None,
    vertical_exaggeration: Annotated[
        float, typer.Option("--ve", help="Vertical exaggeration")
    ] = 4.0,
    smooth_sigma: Annotated[
        float,
        typer.Option(help="Gaussian smoothing (grid cells) applied per frame"),
    ] = 1.1,
    time_stride: Annotated[int, typer.Option(help="Use every Nth timestep")] = 1,
    fps: Annotated[int, typer.Option(help="Movie framerate")] = 25,
    dem_max_side: Annotated[
        int, typer.Option(help="Max DEM points per axis after cropping")
    ] = 500,
    clip_percentile: Annotated[
        float, typer.Option(help="Velocity colour-scale saturation percentile")
    ] = 99.5,
    min_scale: Annotated[
        float, typer.Option(help="Hide ground velocity below this (cm/s)")
    ] = 0.01,
    preview: Annotated[
        bool, typer.Option("--preview", help="Render one frame to PNG and exit")
    ] = False,
    azim: Annotated[
        float, typer.Option(help="Camera azimuth, degrees east of north")
    ] = 35.0,
    window_size: Annotated[
        tuple[int, int], typer.Option(help="Render size, in pixels")
    ] = (1600, 1008),
) -> None:
    """Render a wavefield as a movie, from XYTS or from broadband stations."""
    if is_broadband(xyts):
        animate_broadband(
            xyts,
            output or xyts.with_name("waveform.mp4"),
            None,
            time_stride,
            fps,
            clip_percentile,
            min_scale,
            title,
            preview,
            window_size,
        )
        return

    pv = _pyvista()
    output = output or xyts.with_name("waveform.mp4")

    print(f"Loading XYTS {xyts} ...")
    # NOTE: xarray's CF mask-and-scale decoding of the full waveform array
    # segfaults for these files, so we read the raw uint16 values and apply the
    # scale factor / fill value ourselves, one frame at a time.
    ds = xr.open_dataset(xyts, engine="h5netcdf", mask_and_scale=False)
    times = ds.time.values
    raw = ds.waveform.values  # (nt, ny, nx) uint16, undecoded
    scale = float(ds.waveform.attrs.get("scale_factor", 1.0))
    fill = int(ds.waveform.attrs.get("_FillValue", np.iinfo(np.uint16).max))
    _, ny, nx = raw.shape

    def decode(f):
        """Decode frame f: raw uint16 -> cm/s, fill -> 0, optionally smoothed.

        The data is coarsely quantised (values are multiples of the 0.1 cm/s
        scale) and spatially sparse, so a light Gaussian blur turns the sprinkle
        of lit cells into a coherent, propagating wavefront.
        """
        frame = raw[f].astype(np.float32) * scale
        frame[raw[f] == fill] = 0.0
        if smooth_sigma > 0:
            frame = gaussian_filter(frame, smooth_sigma, mode="nearest")
        return frame

    # The stored latitude/longitude in the XYTS file are unreliable, so we
    # rebuild the grid from the model's spherical projection (mlon/mlat/mrot and
    # the XYTS spacing dx) and reproject to NZTM.
    proj_sph = coordinates.SphericalProjection(
        float(ds.attrs["mlon"]), float(ds.attrs["mlat"]), float(ds.attrs["mrot"])
    )
    dx = float(ds.attrs["dx"])
    y_bounds = np.linspace(-0.5, 0.5, num=ny) * ny * dx
    x_bounds = np.linspace(-0.5, 0.5, num=nx) * nx * dx
    y_sim, x_sim = np.meshgrid(y_bounds, x_bounds, indexing="ij")
    latlon = proj_sph.inverse(x_sim.ravel(order="F"), y_sim.ravel(order="F"))
    to_nztm = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    east_flat, north_flat = to_nztm.transform(latlon[:, 1], latlon[:, 0])
    east = east_flat.reshape(ny, nx, order="F")
    north = north_flat.reshape(ny, nx, order="F")
    bounds = (east.min(), east.max(), north.min(), north.max())

    print("Loading DEM ...")
    domain_km = (east.max() - east.min()) / 1000.0
    terrain, dem_e, dem_n, dem_elev = load_dem(
        dem,
        bounds,
        margin_m=10_000.0,
        max_side=dem_max_side,
        vertical_exaggeration=vertical_exaggeration,
    )

    # A KD-tree over the terrain lets us drape the fault footprint onto it.
    tree = cKDTree(np.column_stack([dem_e.ravel(), dem_n.ravel()]))
    lift_m = 250.0

    # The wavefield is drawn as a flat sheet just above the highest terrain so it
    # always reads as an overlay "on top" and is never occluded by peaks.
    z_flat = (float(dem_elev.max()) + 500.0) * vertical_exaggeration
    wave_up = np.full((ny, nx), z_flat, dtype=np.float64)
    wave_grid = pv.StructuredGrid(east[..., None], north[..., None], wave_up[..., None])

    def frame_scalars(f):
        """Smoothed velocity for frame f, with sub-threshold cells -> NaN."""
        v = decode(f)
        v[v < min_scale] = np.nan
        return v.ravel(order="F")

    wave_grid["velocity"] = frame_scalars(0)

    # Colour scale from the smoothed peak frame (smoothing lowers peak values).
    peak_per_frame = np.where(raw == fill, 0, raw).reshape(len(times), -1).max(axis=1)
    peak_frame = int(np.argmax(peak_per_frame))
    vmax = get_nice_vmax(float(np.percentile(decode(peak_frame), clip_percentile)))
    vmax = max(vmax, 1e-3)
    print(f"Domain ~{domain_km:.0f} km wide; velocity clim [0, {vmax:.3g}] cm/s")

    def drape(en):
        """Drape (E, N) points onto the terrain surface (plot Z)."""
        _, i = tree.query(np.atleast_2d(en))
        up = (dem_elev.ravel()[i] + lift_m) * vertical_exaggeration
        return np.column_stack([np.atleast_2d(en), up])

    # Fault footprint + epicentre, projected onto the terrain surface.
    fault_ring = None
    epicentre_xyz = None
    if srf_path is not None:
        print("Loading fault (SRF) ...")
        corners_en, epicentre_en = load_fault(srf_path)
        ring = drape(corners_en)
        fault_ring = pv.lines_from_points(np.vstack([ring, ring[0]]), close=False)
        epicentre_xyz = drape(epicentre_en)

    # Domain outline, projected onto the terrain surface.
    domain_ring = None
    if realisation is not None:
        print("Loading domain outline ...")
        ring = drape(load_domain(realisation))
        domain_ring = pv.lines_from_points(np.vstack([ring, ring[0]]), close=False)

    pv.set_plot_theme("dark")
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(BACKGROUND)

    plotter.add_mesh(
        terrain,
        scalars="colors",
        rgb=True,  # read the RGB values directly instead of through a colormap
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.25,
        diffuse=0.7,
        specular=0.1,
    )
    if fault_ring is not None:
        plotter.add_mesh(fault_ring, color="#ff3b3b", line_width=4)
        plotter.add_points(
            epicentre_xyz,
            color="cyan",
            render_points_as_spheres=True,
            point_size=22,
        )

    if domain_ring is not None:
        domain_dashed = build_dashed_polyline(
            domain_ring.points, dash_len=5000, gap_len=3000
        )
        plotter.add_mesh(domain_dashed, color="#ffffff", line_width=3)

    plotter.add_mesh(
        wave_grid,
        scalars="velocity",
        cmap=wave_colormap(),
        clim=[0.0, vmax],
        opacity=0.92,
        nan_opacity=0.0,
        show_scalar_bar=True,
        scalar_bar_args=dict(
            title="Ground motion (cm/s)",
            title_font_size=22,
            label_font_size=18,
            position_x=0.30,
            position_y=0.05,
            width=0.4,
            height=0.05,
            n_colors=5,
            fmt="{:.2f}",
        ),
    )

    # Camera: a tilted view from the given azimuth, showing the full domain.
    cx = 0.5 * (east.min() + east.max())
    cy = 0.5 * (north.min() + north.max())
    extent = max(east.max() - east.min(), north.max() - north.min())

    cam_r = 1.05 * extent
    focal_z = float(np.percentile(dem_elev, 90)) * vertical_exaggeration
    azim_rad = np.radians(azim)
    plotter.camera.position = (
        cx + cam_r * np.sin(azim_rad),
        cy - cam_r * np.cos(azim_rad),
        0.8 * extent,
    )
    plotter.camera.focal_point = (cx, cy, focal_z)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.reset_clipping_range()
    plotter.camera.zoom(1.1)

    def set_time_text(t):
        caption = f"{title}\n" if title else ""
        plotter.add_text(
            f"{caption}t = {t:6.1f} s",
            name="hud",
            position="upper_left",
            font_size=16,
            color="white",
        )

    if preview:
        # Show a lively frame rather than t=0: the frame with the largest peak.
        wave_grid.point_data["velocity"][:] = frame_scalars(peak_frame)
        set_time_text(times[peak_frame])
        png = output.with_suffix(".png")
        plotter.screenshot(str(png))
        print(f"Preview frame {peak_frame} (t={times[peak_frame]:.1f}s) -> {png}")
        return

    frames = list(range(0, len(times), time_stride))
    print(f"Rendering {len(frames)} frames -> {output}")
    plotter.open_movie(str(output), framerate=fps, quality=8)
    for f in tqdm(frames, unit="frame"):
        wave_grid.point_data["velocity"][:] = frame_scalars(f)
        set_time_text(times[f])
        plotter.write_frame()
    plotter.close()
    print(f"Wrote {output}")
