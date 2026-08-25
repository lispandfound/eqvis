"""``pick``: the way out of a map with more on it than it can carry.

Opens the same map beside a table of the recording stations and a table of the
basins, each row cycling between three states -- left off, drawn, or drawn with
its name -- so a station whose residual is worth showing need not also spend
the space its label would cost. Stations cycle from the map too: click the
marker. Above the station table are filters that thin the set by what the
flatfile knows about each record.

What it writes is a :mod:`pick list <.picks>`, which the other commands read.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import shapely
import typer
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FuncFormatter,
)

# Tk backs the ``pick`` window and nothing else, so a Python built without it
# should still be able to draw every other figure; ``pick`` reports the lack.
try:
    import tkinter as tk
    from tkinter import filedialog, ttk

    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
except ImportError:  # pragma: no cover - a Python built without tkinter
    tk = None
from .constants import (
    DEFAULT_COMPONENT,
    HIDDEN,
    KM_PER_DEGREE,
    LOG_SCALED,
    NAMED,
    SHOWN,
    UNIT_LABEL,
)
from .data import (
    default_title,
    im_label,
    open_ims,
    residual_label,
    restrict_to_domain,
    select_im,
)
from .flatfile import align_quality, read_observed, read_station_quality, usable_at
from .geography import (
    basins_in_view,
    draw_basins,
    draw_coastline,
    draw_geometry,
    load_basins,
    load_coastline,
)
from .maps import draw_detail_inset
from .picks import TITLE_SECTION, read_pick_list, write_pick_list
from .raster import discrete_norm, fixed_symmetric_norm, rasterise
from .stations import (
    cluster_bounds,
    draw_observed,
    nearest_stations,
    place_labels,
    sample_simulation,
    station_clusters,
    station_labels,
)

# How close to a marker, in pixels, a click has to land to count as picking it.
MAP_CLICK_RADIUS = 14


# Stations the picker is currently hiding: present enough to click back on,
# quiet enough not to read as data.
HIDDEN_STATION_STYLE = {
    "marker": "x",
    "s": 26,
    "c": "#9c9c9c",
    "linewidths": 0.9,
    "zorder": 5,
}


# The three states in the tick column. ASCII only: Tk prints an escape for any
# glyph the system font lacks, and a station list is read by people too.
STATE_GLYPH = {HIDDEN: "", SHOWN: "x", NAMED: "x*"}


def thin_by_separation(
    lon: np.ndarray,
    lat: np.ndarray,
    keep: np.ndarray,
    order: np.ndarray,
    separation: float,
) -> np.ndarray:
    """Thin a station set so no two survivors sit within ``separation`` km.

    Greedy and best-first: stations are offered in ``order`` (an index array,
    best first) and one is taken whenever it clears every station already
    taken. This is what a map wants from thinning -- one station per
    neighbourhood, spread over the domain, the best record in each -- rather
    than the arbitrary subset that cutting on name or index would leave.
    """
    if separation <= 0:
        return keep.copy()
    aspect = np.cos(np.radians(lat.mean()))
    points = np.column_stack([lon * aspect, lat]) * KM_PER_DEGREE
    taken: list[int] = []
    for index in order:
        if not keep[index]:
            continue
        if taken and np.hypot(*(points[taken] - points[index]).T).min() < separation:
            continue
        taken.append(int(index))
    thinned = np.zeros(keep.shape, dtype=bool)
    thinned[taken] = True
    return thinned


def thinning_order(quality: dict[str, np.ndarray]) -> np.ndarray:
    """Stations ranked by how much a map wants to keep them, best first.

    The cleanest recording wins its neighbourhood, and the nearer of two
    equally clean ones breaks the tie: close-in records are the ones that
    constrain a simulation, and the far field has stations to spare.
    """
    snr = np.nan_to_num(quality["snr"], nan=-1.0)
    distance = np.nan_to_num(quality["distance"], nan=np.inf)
    return np.lexsort((distance, -snr))


def _axes_contents(ax: plt.Axes) -> list:
    """Every artist and inset the axes owns, in one flat list.

    Used to diff the axes either side of drawing the overlay, so the overlay
    can be torn off again without the base map having to know what it was
    made of.
    """
    return [
        *ax.collections,
        *ax.lines,
        *ax.texts,
        *ax.patches,
        *ax.artists,
        *ax.child_axes,
    ]


def basin_table(
    basins: list[tuple[str, shapely.Geometry]],
    clip: shapely.Geometry,
    lon: np.ndarray,
    lat: np.ndarray,
    mean_lat: float,
) -> tuple[list[tuple[str, shapely.Geometry]], dict[str, tuple[bool, float]]]:
    """The basins that will actually draw, and what the picker shows about each.

    A basin whose boundary falls entirely outside the clip region has nothing
    to draw and is left out of the table rather than listed as an entry that
    does nothing. The rest are described by whether they hold a recording
    station -- which is what decides their default state -- and their visible
    area, as a way of sorting the small ones down the list.
    """
    drawable, rows = [], {}
    for name, geometry in basins:
        if shapely.intersection(geometry.boundary, clip).is_empty:
            continue
        visible = shapely.intersection(geometry, clip)
        holds = bool(shapely.contains_xy(geometry, lon, lat).any())
        area = visible.area * KM_PER_DEGREE**2 * np.cos(np.radians(mean_lat))
        drawable.append((name, geometry))
        rows[name] = (holds, area)
    return drawable, rows


def build_base_map(
    im_file: Path,
    im: str,
    component: str,
    period: float | None,
    frequency: float | None,
    observed: Path,
    levels: int,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    residual_limit: float,
    clip: bool,
    coastline: Path | None,
    basins: bool,
    basin_file: Path | None,
) -> dict:
    """Draw everything about the map that the picking cannot change.

    The raster, coastline, source geometry and both colour bars go on once
    here; the returned scene carries what the picker needs to dress that base
    with markers, outlines and labels and to strip them off again. Basins are
    prepared but not drawn, since they are picked too. The residual bar is fed
    from a standalone mappable rather than the station scatter, so it survives
    every scatter the picker throws away.
    """
    tree = open_ims(im_file)
    da, selection = select_im(tree, im, component, period, frequency)
    label = im_label(im, selection)
    for dim, value in selection.items():
        print(f"selected {dim}: {value:g}")

    lon, lat = da.longitude.values, da.latitude.values
    values = da.values
    boundaries = discrete_norm(values, levels, im in LOG_SCALED, vmin, vmax)
    colormap = plt.get_cmap(cmap or "magma_r")
    norm = BoundaryNorm(boundaries, colormap.N, extend="both")
    units = UNIT_LABEL.get(da.attrs.get("units", ""), da.attrs.get("units", ""))

    coast = load_coastline(coastline) if clip else None
    grid_lon, grid_lat, grid = rasterise(lon, lat, values, coast)
    view = (grid_lon.min(), grid_lat.min(), grid_lon.max(), grid_lat.max())

    figure = Figure(figsize=(9, 9), layout="constrained")
    ax = figure.add_subplot()
    mesh = ax.pcolormesh(
        grid_lon, grid_lat, grid, cmap=colormap, norm=norm, rasterized=True
    )
    if coast is not None:
        draw_coastline(ax, coast, view)
    draw_geometry(ax, tree.attrs)
    # Freeze the view before anything pickable goes on: basins run well past
    # the domain, and the markers, outlines and leader lines would otherwise
    # shift the limits under the pointer on every redraw.
    ax.autoscale_view()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    obs, resolved = read_observed(observed, im, component, selection)
    for dim, value in resolved.items():
        print(f"observed {dim}: {value:g}")
    obs = restrict_to_domain(obs, tree.attrs, grid_lon, grid_lat, observed)
    if obs["name"].size == 0:
        raise typer.BadParameter(f"no {observed} stations inside the domain")
    quality = align_quality(read_station_quality(observed, component), obs["name"])

    residual = None
    if np.isfinite(obs["value"]).any():
        simulated = sample_simulation(lon, lat, values, obs["lon"], obs["lat"])
        residual = np.log(simulated) - np.log(obs["value"])
    _, reached = nearest_stations(lon, lat, obs["lon"], obs["lat"])

    domain_shape = (
        shapely.from_wkt(tree.attrs["domain"]) if tree.attrs.get("domain") else None
    )
    basin_clip = domain_shape if domain_shape is not None else shapely.box(*view)
    outlines, basin_rows = [], {}
    if basins:
        outlines, basin_rows = basin_table(
            basins_in_view(load_basins(basin_file) or [], view),
            basin_clip,
            obs["lon"],
            obs["lat"],
            float(lat.mean()),
        )

    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    degrees = FuncFormatter(lambda v, _: f"{v:g}°")
    ax.xaxis.set_major_formatter(degrees)
    ax.yaxis.set_major_formatter(degrees)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # The same default the command line would draw, so opening the picker,
    # changing nothing and saving reproduces it exactly.
    heading = default_title(tree.attrs, "")
    ax.set_title(heading, fontsize=11)

    residual_levels = fixed_symmetric_norm(residual_limit, levels)
    residual_cmap = plt.get_cmap("RdBu_r")
    residual_norm = BoundaryNorm(residual_levels, residual_cmap.N, extend="both")
    residuals = figure.colorbar(
        plt.cm.ScalarMappable(norm=residual_norm, cmap=residual_cmap),
        ax=ax,
        orientation="horizontal",
        shrink=0.6,
        pad=0.02,
        aspect=35,
    )
    residuals.set_label(residual_label(label), fontsize=10)
    residuals.set_ticks(residual_levels)
    residuals.set_ticklabels([f"{b:g}" for b in residual_levels], fontsize=8)

    colorbar = figure.colorbar(
        mesh, ax=ax, orientation="horizontal", shrink=0.6, pad=0.04, aspect=35
    )
    colorbar.set_label(f"{label} ({units})" if units else label, fontsize=10)
    colorbar.set_ticks(boundaries)
    colorbar.set_ticklabels([f"{b:g}" for b in boundaries], fontsize=8)

    return {
        "figure": figure,
        "axes": ax,
        "raster": (grid_lon, grid_lat, grid),
        "colormap": colormap,
        "norm": norm,
        "residual_cmap": residual_cmap,
        "residual_norm": residual_norm,
        "coastline": coast,
        "basins": outlines,
        "basin_rows": basin_rows,
        "basin_clip": basin_clip,
        "basin_tolerance": (view[2] - view[0]) / (figure.get_figwidth() * 100) * 2,
        "domain": domain_shape,
        "span": max(np.ptp(grid_lon), np.ptp(grid_lat)),
        "observed": obs,
        "quality": quality,
        "residual": residual,
        "reached": reached,
        "usable": usable_at(quality["high_pass"], quality["low_pass"], selection),
        "title": heading,
        "label": label,
        "im": im,
        "component": component,
        "selection": selection,
        "archive": observed,
        "im_file": im_file,
    }


class PickTable:
    """A table of items each cycling hidden -> drawn -> drawn with its name.

    Stations and basins are picked the same way, so the table, its sorting, its
    click handling and the buttons under it are written once here. The owner
    supplies the rows and a baseline state for each, and hears back through
    ``on_change`` whenever a pick moves.

    Picks are held as exceptions to that baseline rather than as the state
    itself, which is what lets a filter shift underneath the table without
    discarding what was chosen by hand -- and lets *Reset* drop the hand-picking
    without disturbing the filters.
    """

    # A cell the row has no value for. ASCII, like everything else bound for a
    # widget: Tk prints an escape for a glyph the system font lacks.
    MISSING = "-"

    def __init__(
        self,
        parent,
        columns: Sequence[tuple[str, str, int, str, str]],
        baseline,
        on_change,
        hint: str,
        noun: str,
    ):
        self.columns = columns
        self.baseline = baseline
        self.on_change = on_change
        self.noun = noun
        self.override: dict[str, int] = {}
        self.names: list[str] = []
        self.sort_key: str | None = None
        self.sort_reverse = False

        self.frame = ttk.Frame(parent, padding=(6, 6))
        self.body = ttk.Frame(self.frame)
        self.body.pack(fill=tk.BOTH, expand=True)
        keys = [key for key, *_ in columns]
        self.table = ttk.Treeview(
            self.body, columns=keys, show="tree headings", selectmode="extended"
        )
        self.table.column("#0", width=34, minwidth=34, stretch=False, anchor="center")
        self.table.heading("#0", text="on")
        for key, title, width, anchor, _ in columns:
            self.table.heading(key, text=title, command=lambda k=key: self.sort_by(k))
            self.table.column(key, width=width, anchor=anchor, stretch=False)
        bar = ttk.Scrollbar(self.body, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.tag_configure("hidden", foreground="#9c9c9c")
        self.table.tag_configure("plain", foreground="#444444")
        self.table.bind("<Button-1>", self.on_click)
        self.table.bind("<space>", self.on_key)
        self.table.bind("<Return>", self.on_key)

        ttk.Label(
            self.frame, text=hint, foreground="#666666", wraplength=340
        ).pack(anchor="w", pady=(4, 0))
        self.status = ttk.Label(self.frame, text="")
        self.status.pack(anchor="w", pady=(4, 2))
        buttons = ttk.Frame(self.frame)
        buttons.pack(fill=tk.X)
        for text, command in (
            ("Reset", self.reset),
            ("Name all", lambda: self.set_all(NAMED)),
            ("Draw all", lambda: self.set_all(SHOWN)),
            ("Hide all", lambda: self.set_all(HIDDEN)),
        ):
            ttk.Button(buttons, text=text, command=command).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=1
            )

    # ------------------------------------------------------------------ state

    def fill(self, rows: dict[str, tuple]) -> None:
        """Lay the rows in, once; only their state changes after this."""
        self.names = list(rows)
        for name, values in rows.items():
            self.table.insert("", "end", iid=name, values=values)

    def states(self) -> dict[str, int]:
        base = self.baseline()
        return {
            name: self.override.get(name, base.get(name, HIDDEN))
            for name in self.names
        }

    def by_hand(self) -> int:
        base = self.baseline()
        return sum(
            1
            for name, state in self.override.items()
            if state != base.get(name, HIDDEN)
        )

    def cycle(self, names: Sequence[str]) -> None:
        """Step each named item on to its next state, and tell the owner."""
        states = self.states()
        for name in names:
            self.override[name] = (states[name] + 1) % len(STATE_GLYPH)
        self.on_change()

    def reset(self) -> None:
        self.override.clear()
        self.on_change()

    def set_all(self, state: int) -> None:
        self.override = dict.fromkeys(self.names, state)
        self.on_change()

    def seed(self, picked: dict[str, bool]) -> None:
        """Start from a saved selection: anything it omits is hidden."""
        self.override = {
            name: (NAMED if picked[name] else SHOWN) if name in picked else HIDDEN
            for name in self.names
        }

    # ----------------------------------------------------------------- events

    def on_click(self, event) -> str | None:
        if self.table.identify_region(event.x, event.y) != "tree":
            return None
        if row := self.table.identify_row(event.y):
            self.cycle([row])
        return "break"

    def on_key(self, _event) -> str:
        if selected := self.table.selection():
            self.cycle(selected)
        return "break"

    def sort_by(self, key: str) -> None:
        """Reorder on a column, descending first for the numeric ones."""
        kinds = {column: kind for column, _, _, _, kind in self.columns}
        self.sort_reverse = (
            not self.sort_reverse if self.sort_key == key else kinds[key] != "text"
        )
        self.sort_key = key
        column = [column for column, *_ in self.columns].index(key)

        def sortable(name: str):
            text = self.table.item(name, "values")[column]
            if kinds[key] == "text":
                return text
            return -np.inf if text == self.MISSING else float(text)

        for position, name in enumerate(
            sorted(self.names, key=sortable, reverse=self.sort_reverse)
        ):
            self.table.move(name, "", position)

    # ---------------------------------------------------------------- display

    def update(self, states: dict[str, int]) -> None:
        for name, state in states.items():
            self.table.item(
                name,
                text=STATE_GLYPH[state],
                tags=("hidden",)
                if state == HIDDEN
                else ("plain",)
                if state == SHOWN
                else (),
            )
        drawn = sum(1 for state in states.values() if state != HIDDEN)
        named = sum(1 for state in states.values() if state == NAMED)
        hand = f", {count} by hand" if (count := self.by_hand()) else ""
        self.status.configure(
            text=f"{drawn} of {len(states)} {self.noun} drawn, {named} named{hand}"
        )


class StationPicker:
    """A Tk window pairing the map with tables of what it may draw.

    The map is expensive to build and cheap to re-dress, so
    :func:`build_base_map` draws the raster, coastline and geometry once and
    this class only tears off and rebuilds what is picked -- station markers,
    basin outlines, labels and detail insets -- as the selection changes.

    Two :class:`PickTable`\\ s sit behind the notebook, one for the recording
    stations and one for the basins, each cycling its rows between hidden,
    drawn, and drawn with a name. The stations additionally have a baseline
    computed from the record metadata by the filters above them; the basins
    default to what the map would have chosen for itself, which is an outline
    for everything in view and a name for whatever holds a station.
    """

    STATION_COLUMNS = (
        ("station", "Station", 76, "w", "text"),
        ("residual", "ln sim/obs", 76, "e", "number"),
        ("distance", "r_rup", 58, "e", "number"),
        ("vs30", "Vs30", 54, "e", "number"),
        ("snr", "SNR", 48, "e", "number"),
    )

    BASIN_COLUMNS = (
        ("basin", "Basin", 150, "w", "text"),
        ("stations", "stations", 66, "e", "number"),
        ("area", "area km2", 80, "e", "number"),
    )

    def __init__(self, scene: dict, seed: dict[str, dict[str, bool]] | None = None):
        self.scene = scene
        self.names = list(scene["observed"]["name"])
        self.order = thinning_order(scene["quality"])
        self.overlay: list = []
        # Which stations each detail inset drew, so a click in one can be
        # matched back to them; rebuilt with the overlay on every redraw.
        self.insets: list[tuple[plt.Axes, np.ndarray]] = []
        self.pending: str | None = None
        # Kept outside Tk: the variables behind them die with the window, and
        # the point of the session is to still have the picks afterwards.
        self.picks: dict = {"stations": {}, "basins": {}, "title": ""}
        self.summary = ""
        # A saved title wins over the default the map would have drawn.
        self.title = scene["title"]
        if seed is not None and seed.get("title"):
            self.title = seed["title"]
        self.title_pending: str | None = None

        self.root = tk.Tk()
        self.root.title(f"pick - {scene['im_file'].name}")
        self.root.geometry("1500x950")
        self.switches: dict[str, tk.BooleanVar] = {}
        self.sliders: dict[str, tk.DoubleVar] = {}
        self.captions: dict[str, ttk.Widget] = {}
        self.wording: dict[str, str] = {}
        self.caps: dict[str, float] = {}
        self._build()
        if seed is not None:
            self.stations.seed(seed["stations"])
            if seed["basins"]:
                self.basins.seed(seed["basins"])
        self.apply_title()
        self.refresh()
        # That first draw settled the constrained layout. Freezing it there
        # stops the map shifting under the pointer as labels come and go, and
        # takes a third off every redraw after it.
        scene["figure"].set_layout_engine("none")

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(panes, padding=(8, 8))
        right = ttk.Frame(panes)
        panes.add(left, weight=0)
        panes.add(right, weight=1)

        heading = ttk.LabelFrame(left, text="Title", padding=(6, 4))
        heading.pack(fill=tk.X)
        self.title_var = tk.StringVar(value=self.title)
        entry = ttk.Entry(heading, textvariable=self.title_var)
        entry.pack(fill=tk.X)
        entry.bind("<KeyRelease>", lambda _: self.schedule_title())
        entry.bind("<Return>", lambda _: self.apply_title())
        ttk.Label(
            heading,
            text="matplotlib mathtext works here: M$_w$, $\\alpha$",
            foreground="#666666",
        ).pack(anchor="w", pady=(2, 0))

        view = ttk.LabelFrame(left, text="Map", padding=(6, 4))
        view.pack(fill=tk.X, pady=(8, 0))
        for key, text, value in (
            ("insets", "zoom crowded clusters into insets", False),
            ("hidden", "mark the hidden stations", True),
        ):
            variable = tk.BooleanVar(value=value)
            self.switches[key] = variable
            ttk.Checkbutton(
                view, text=text, variable=variable, command=self.schedule
            ).pack(anchor="w")

        book = ttk.Notebook(left)
        book.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        stations = ttk.Frame(book)
        book.add(stations, text="Stations")
        self._build_filters(stations)
        self.stations = PickTable(
            stations,
            self.STATION_COLUMNS,
            self.station_baseline,
            self.schedule_now,
            "click the 'on' column, or a marker on the map or in a detail "
            "inset, to cycle a station between hidden, drawn (x) and drawn "
            "with its name (x*); space cycles every selected row",
            "stations",
        )
        self.stations.frame.pack(fill=tk.BOTH, expand=True)
        self.stations.fill(
            {
                name: self.station_row(position)
                for position, name in enumerate(self.names)
            }
        )

        basins = ttk.Frame(book)
        book.add(basins, text="Basins")
        self.basins = PickTable(
            basins,
            self.BASIN_COLUMNS,
            self.basin_baseline,
            self.schedule_now,
            "click the 'on' column to cycle a basin between hidden, outlined "
            "(x) and outlined with its name (x*)",
            "basins",
        )
        self.basins.frame.pack(fill=tk.BOTH, expand=True)
        self.basins.fill(
            {
                name: (name, "yes" if holds else "no", f"{area:.0f}")
                for name, (holds, area) in self.scene["basin_rows"].items()
            }
        )

        actions = ttk.Frame(left)
        actions.pack(fill=tk.X, pady=(8, 0))
        for text, command in (
            ("Save picks...", self.save_picks),
            ("Save figure...", self.save_figure),
        ):
            ttk.Button(actions, text=text, command=command).pack(
                side=tk.LEFT, expand=True, fill=tk.X, padx=1
            )

        self.canvas = FigureCanvasTkAgg(self.scene["figure"], master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_map_click)

    def _switch(self, parent, key: str, text: str, value: bool = True) -> None:
        variable = tk.BooleanVar(value=value)
        self.switches[key] = variable
        self.wording[key] = text
        button = ttk.Checkbutton(
            parent, text=text, variable=variable, command=self.schedule
        )
        button.pack(anchor="w")
        self.captions[key] = button

    def _slider(
        self, parent, key: str, text: str, low: float, high: float, start: float
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(6, 0))
        caption = ttk.Label(frame, text=text)
        caption.pack(anchor="w")
        variable = tk.DoubleVar(value=start)
        self.sliders[key] = variable
        self.caps[key] = high
        self.wording[key] = text
        self.captions[key] = caption
        ttk.Scale(
            frame,
            from_=low,
            to=high,
            variable=variable,
            command=lambda _: self.schedule(),
        ).pack(fill=tk.X)

    def _build_filters(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Automatic thinning", padding=(6, 4))
        box.pack(fill=tk.X, padx=6, pady=(6, 0))
        self._switch(box, "free_field", "free-field instruments only")
        self._switch(box, "usable", f"{self.scene['im']} inside the usable band")
        self._switch(box, "reached", "covered by the simulation grid")
        self._switch(box, "residual", "has a residual to colour", value=False)
        furthest = float(np.nanmax(self.scene["quality"]["distance"]))
        cap = 10 * np.ceil(furthest / 10) if np.isfinite(furthest) else 500.0
        self._slider(box, "snr", "signal-to-noise floor", 0.0, 1.0, 0.0)
        self._slider(box, "distance", "rupture distance cap", 0.0, cap, cap)
        self._slider(box, "separation", "minimum station separation", 0.0, 80.0, 0.0)

    # ------------------------------------------------------------- selection

    def filtered(self, without: str | None = None) -> np.ndarray:
        """The stations the filters alone would keep, optionally sparing one.

        ``without`` names a filter to leave out, which is how the panel prices
        each one: what a filter costs is what it removes on top of all the
        others, not what it would remove on its own.
        """
        quality = self.scene["quality"]
        keep = np.ones(len(self.names), dtype=bool)
        if self.enabled("free_field", without):
            keep &= quality["free_field"].astype(bool)
        if self.enabled("usable", without):
            keep &= self.scene["usable"]
        if self.enabled("reached", without):
            keep &= self.scene["reached"]
        if self.enabled("residual", without) and self.scene["residual"] is not None:
            keep &= np.isfinite(self.scene["residual"])
        # An unknown score or distance is not evidence against a record, so
        # negating the comparison is deliberate: only the known ones are held
        # to the threshold.
        if without != "snr" and (floor := self.sliders["snr"].get()) > 0:
            keep &= ~(quality["snr"] < floor)
        if without != "distance":
            keep &= ~(quality["distance"] > self.sliders["distance"].get())
        separation = 0.0 if without == "separation" else self.sliders["separation"].get()
        return thin_by_separation(
            self.scene["observed"]["lon"],
            self.scene["observed"]["lat"],
            keep,
            self.order,
            separation,
        )

    def enabled(self, key: str, without: str | None) -> bool:
        return key != without and self.switches[key].get()

    def station_baseline(self) -> dict[str, int]:
        """What the filters would draw: everything they keep, named."""
        return {
            name: NAMED if keep else HIDDEN
            for name, keep in zip(self.names, self.filtered())
        }

    def basin_baseline(self) -> dict[str, int]:
        """What the map would have chosen: outline everything, name the useful.

        Naming only the basins that hold a recording station is the rule
        :func:`draw_basins` applies on its own, kept here so that opening the
        picker and changing nothing draws the map it would have drawn anyway.
        """
        return {
            name: NAMED if holds else SHOWN
            for name, (holds, _) in self.scene["basin_rows"].items()
        }

    def station_row(self, position: int) -> tuple:
        quality = self.scene["quality"]
        residual = self.scene["residual"]
        missing = PickTable.MISSING
        return (
            self.names[position],
            missing
            if residual is None or not np.isfinite(residual[position])
            else f"{residual[position]:+.2f}",
            *(
                missing if not np.isfinite(value) else f"{value:.0f}"
                for value in (quality["distance"][position], quality["vs30"][position])
            ),
            missing
            if not np.isfinite(quality["snr"][position])
            else f"{quality['snr'][position]:.2f}",
        )

    # ---------------------------------------------------------------- events

    def schedule(self) -> None:
        """Coalesce a burst of slider events into one redraw."""
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(150, self.refresh)

    def schedule_title(self) -> None:
        """Let the typing settle before redrawing.

        On its own timer, so an edit in the entry and a drag on a slider cannot
        cancel one another.
        """
        if self.title_pending is not None:
            self.root.after_cancel(self.title_pending)
        self.title_pending = self.root.after(400, self.apply_title)

    def apply_title(self) -> None:
        """Put the edited title on the map, without rebuilding the overlay.

        The title is the one thing on the figure that owes nothing to the
        picking, so it is drawn straight rather than through :meth:`refresh`.
        """
        self.title_pending = None
        self.title = self.title_var.get()
        self.picks[TITLE_SECTION] = self.title
        self.scene["axes"].set_title(self.title, fontsize=11)
        self.canvas.draw_idle()

    def schedule_now(self) -> None:
        """A click is a single event and should not wait on the slider timer."""
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.refresh()

    def clicked_axes(self, event) -> tuple[plt.Axes, np.ndarray] | None:
        """The axes a click landed in, and which stations are pickable there.

        A detail inset is pickable too, and is the only place a station in a
        pile-up can be aimed at: at the main map's scale its members sit a
        fraction of a pixel apart, which is what the inset exists to spread
        out. An inset only offers its own members, so a click there cannot
        reach a station from the other side of the map.
        """
        if event.inaxes is self.scene["axes"]:
            return self.scene["axes"], np.arange(len(self.names))
        for axes, members in self.insets:
            if event.inaxes is axes:
                return axes, members
        return None

    def on_map_click(self, event) -> None:
        """Cycle the station nearest a click on the map, if one is close enough.

        Where several stations are within reach of one click, a drawn one wins
        over a hidden one: what is under the pointer is the marker the eye
        picked out, and turning on the hidden neighbour it happens to be
        sitting on is never what was meant. A hidden station is still reachable
        wherever it is not buried -- and in the inset, where nothing is.
        """
        if event.button != 1 or self.toolbar.mode:  # panning or zooming
            return
        where = self.clicked_axes(event)
        if where is None:
            return
        axes, candidates = where
        observed = self.scene["observed"]
        xy = axes.transData.transform(
            np.column_stack(
                [observed["lon"][candidates], observed["lat"][candidates]]
            )
        )
        gap = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        within = gap <= MAP_CLICK_RADIUS
        if not within.any():
            return
        states = self.stations.states()
        drawn = np.array(
            [states[self.names[index]] != HIDDEN for index in candidates]
        )
        reachable = within & drawn if (within & drawn).any() else within
        nearest = candidates[np.flatnonzero(reachable)[np.argmin(gap[reachable])]]
        self.stations.cycle([self.names[nearest]])

    # ----------------------------------------------------------------- files

    def provenance(self) -> str:
        return (
            f"{len(self.picks['stations'])} of {len(self.names)} "
            f"{self.scene['archive'].name} stations and "
            f"{len(self.picks['basins'])} of {len(self.scene['basin_rows'])} basins "
            f"picked against {self.scene['im_file'].name}, "
            f"{self.scene['label']} {self.scene['component']}"
        )

    def save_picks(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save picks",
            defaultextension=".stations",
            initialfile=f"{self.scene['im_file'].stem}_{self.scene['im']}.stations",
        )
        if not path:
            return
        write_pick_list(Path(path), self.picks, self.summary)
        print(f"wrote {path}")

    def save_figure(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save figure",
            defaultextension=".png",
            initialfile=f"{self.scene['im_file'].stem}_{self.scene['im']}.png",
        )
        if path:
            self.scene["figure"].savefig(path, dpi=300)
            print(f"wrote {path}")

    # ---------------------------------------------------------------- drawing

    def refresh(self) -> None:
        self.pending = None
        stations = self.stations.states()
        basins = self.basins.states()
        self.picks = {
            section: {
                name: state == NAMED
                for name, state in states.items()
                if state != HIDDEN
            }
            for section, states in (("stations", stations), ("basins", basins))
        }
        self.picks[TITLE_SECTION] = self.title
        self.summary = self.provenance()
        self.redraw(stations, basins)
        self.stations.update(stations)
        self.basins.update(basins)
        self.update_captions()

    def update_captions(self) -> None:
        """Price each filter by what it removes on top of all the others."""
        standing = int(self.filtered().sum())
        for key, wording in self.wording.items():
            cost = int(self.filtered(without=key).sum()) - standing
            price = f"  (-{cost})" if cost else ""
            if key in self.sliders:
                value = self.sliders[key].get()
                unit = "" if key == "snr" else " km"
                off = value >= self.caps[key] if key == "distance" else value <= 0
                setting = "off" if off else f"{value:.3g}{unit}"
                self.captions[key].configure(text=f"{wording}: {setting}{price}")
            else:
                self.captions[key].configure(text=f"{wording}{price}")

    def redraw(self, stations: dict[str, int], basins: dict[str, int]) -> None:
        """Re-dress the base map with the current picks."""
        scene, ax, figure = self.scene, self.scene["axes"], self.scene["figure"]
        for artist in reversed(self.overlay):
            artist.remove()
        before = {id(artist) for artist in _axes_contents(ax)}

        observed, residual = scene["observed"], scene["residual"]
        drawn = np.array([stations[name] != HIDDEN for name in self.names])
        named = np.array([stations[name] == NAMED for name in self.names])
        chosen = {key: value[drawn] for key, value in observed.items()}
        chosen_residual = None if residual is None else residual[drawn]

        outlines = [
            (name, geometry)
            for name, geometry in scene["basins"]
            if basins.get(name, HIDDEN) != HIDDEN
        ]
        basin_entries = draw_basins(
            ax,
            scene["basins"],
            scene["basin_clip"],
            scene["basin_tolerance"],
            states=basins,
        )

        if self.switches["hidden"].get() and (~drawn).any():
            ax.scatter(
                observed["lon"][~drawn], observed["lat"][~drawn], **HIDDEN_STATION_STYLE
            )
        draw_observed(
            ax,
            chosen["lon"],
            chosen["lat"],
            chosen_residual,
            scene["residual_cmap"],
            scene["residual_norm"],
        )

        labelled = named[drawn]
        self.insets = []
        if self.switches["insets"].get():
            placed: list[tuple[float, float, float, float]] = []
            # Cluster members index into the drawn subset; a click needs them
            # back in the order the tables and the observations are held in.
            positions = np.flatnonzero(drawn)
            for members in station_clusters(chosen["lon"], chosen["lat"], scene["span"]):
                axins = draw_detail_inset(
                    figure,
                    ax,
                    cluster_bounds(chosen["lon"], chosen["lat"], members),
                    scene["raster"],
                    scene["colormap"],
                    scene["norm"],
                    chosen,
                    members,
                    chosen_residual,
                    scene["residual_cmap"],
                    scene["residual_norm"],
                    scene["coastline"],
                    outlines,
                    scene["domain"],
                    taken=placed,
                    label=labelled,
                )
                self.insets.append((axins, positions[members]))
                # Their names live in the inset now; leaving them here too
                # would put the crowding straight back.
                labelled[members] = False
        entries = (
            station_labels(
                chosen["name"][labelled],
                chosen["lon"][labelled],
                chosen["lat"][labelled],
                chosen["value"][labelled],
            )
            + basin_entries
        )
        place_labels(figure, ax, entries)

        self.overlay = [
            artist for artist in _axes_contents(ax) if id(artist) not in before
        ]
        self.canvas.draw_idle()

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()


def pick(
    im_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Intensity measure file")
    ],
    im: Annotated[str, typer.Argument(help="Intensity measure to plot")] = "PGA",
    observed: Annotated[
        Path | None,
        typer.Option(
            "--observed",
            exists=True,
            dir_okay=False,
            help="GeoNet flatfile zip holding the recordings to pick from",
        ),
    ] = None,
    stations: Annotated[
        Path | None,
        typer.Option(
            "--stations",
            exists=True,
            dir_okay=False,
            help="Pick list to open with, as written by --output",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the picks here when the window closes, without asking",
        ),
    ] = None,
    component: Annotated[
        str | None,
        typer.Option(help="Component of motion (default depends on the IM)"),
    ] = None,
    period: Annotated[
        float | None,
        typer.Option(help="pSA period in seconds (nearest available; default 1.0)"),
    ] = None,
    frequency: Annotated[
        float | None,
        typer.Option(help="FAS frequency in Hz (nearest available; default 1.0)"),
    ] = None,
    levels: Annotated[int, typer.Option(help="Approximate number of colour bins")] = 10,
    cmap: Annotated[
        str | None, typer.Option(help="Matplotlib colormap name")
    ] = None,
    vmin: Annotated[float | None, typer.Option(help="Colour scale minimum")] = None,
    vmax: Annotated[float | None, typer.Option(help="Colour scale maximum")] = None,
    residual_limit: Annotated[
        float, typer.Option(help="Colour scale limit for the station residuals")
    ] = 0.7,
    clip: Annotated[
        bool,
        typer.Option("--clip/--no-clip", help="Clip the raster to the NZ coastline"),
    ] = True,
    coastline: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Coastline file to clip against"),
    ] = None,
    basins: Annotated[
        bool,
        typer.Option(
            "--basins/--no-basins", help="Offer the basins that intersect the domain"
        ),
    ] = True,
    basin_file: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Basin outlines to draw"),
    ] = None,
):
    """Choose what a map draws, with the map redrawing as you go.

    Opens a window with the map on the right and, on the left, a table of the
    recording stations and a table of the basins. Every row cycles between
    three states -- left off, drawn, or drawn with its name -- so a station
    whose residual is worth showing need not also spend the space its label
    would cost. Stations cycle from the map too: click the marker.

    Above the station table are filters that thin the set by what the flatfile
    knows about each record: how the instrument was mounted, how clean the
    recording is, whether the measure being plotted falls inside the record's
    usable band, how far it sits from the rupture, and how close it sits to its
    neighbours. Each is priced by what it removes, and each is a starting
    point that clicking overrides rather than a decision that overrules the
    clicking.

    What comes out is a pick list, which ``map``, ``distance`` and ``bias``
    all read through ``--stations``, so the figure that ends up in the paper
    comes off the command line rather than out of a window::

        eqvis pick taumarunui_sw4.h5 PGA \\
            --observed 2026p553250_flatfiles.zip -o taumarunui.stations

        eqvis map taumarunui_sw4.h5 PGA \\
            --observed 2026p553250_flatfiles.zip \\
            --stations taumarunui.stations -o pga.png
    """
    if observed is None:
        raise typer.BadParameter("--observed is required: there is nothing to pick from")
    if tk is None:
        raise typer.BadParameter(
            "this Python has no tkinter, so there is no window to open; "
            "use `map --stations` with a hand-written list instead"
        )
    component = component or DEFAULT_COMPONENT.get(im, "geom")
    scene = build_base_map(
        im_file,
        im,
        component,
        period,
        frequency,
        observed,
        levels,
        cmap,
        vmin,
        vmax,
        residual_limit,
        clip,
        coastline,
        basins,
        basin_file,
    )
    picker = StationPicker(
        scene, read_pick_list(stations) if stations is not None else None
    )
    picker.run()
    if output is not None:
        write_pick_list(output, picker.picks, picker.summary)
        print(f"wrote {output}")
