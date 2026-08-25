# eqvis-workflow

Figures for earthquake simulation workflows: intensity measure maps, attenuation
and bias plots, spectra, waveforms, rupture maps, and wavefield animations.

Everything is drawn with matplotlib, from one shared vocabulary. The same
colours mean the same things across the commands — green is a recording, blue a
comparison run or an empirical model, grey a simulation drawn in bulk and black
the same simulation drawn at a point. The same `Display` sizes a figure for the
wall it will hang on. The same coastline, basin outlines and label placer
provide the geography.

## Install

```sh
uv sync
uv run eqvis --help
```

Python 3.14 or newer. `source-modelling` and `qcore-utils` come from the
[ucgmsim](https://github.com/ucgmsim) toolchain; if you are working against a
checkout of either, point `[tool.uv.sources]` at it.

## Commands

### Intensity measures

These read a simulation IM file: a netCDF/HDF5 datatree with one group per
intensity measure (PGA, PGV, CAV, AI, Ds575, Ds595, pSA, FAS), each holding one
variable per component of motion and carrying the station coordinates, the
source distances (rrup, rjb, rx, ry, epi, hyp) and the site terms (vs30, z1pt0,
z2pt5). The source geometry, domain boundary, event name and magnitude are
attributes on the root.

| Command | What it draws |
| --- | --- |
| `map` | An intensity measure spatially, or the log-difference of two runs |
| `distance` | The same values against a source distance metric |
| `bias` | The log residual against recordings, swept across the spectrum |
| `spectra` | Fourier and response spectra at a station |
| `psa-spectrum` | A pSA spectrum at a station, with directivity |
| `waveform` | Simulated and recorded time series at a station |
| `pick` | An interactive window for choosing what a crowded map carries |
| `convert` | A flat IM dataset reshaped into the datatree the rest read |
| `ingest` | Many runs' IM files gathered into one queryable DuckDB file |

```sh
# What's in the file
eqvis map intensity_measures.h5 --info

# PGA, interactively (a window opens when -o is omitted)
eqvis map intensity_measures.h5 PGA

# pSA at 1 s, saved
eqvis map intensity_measures.h5 pSA --period 1.0 -o psa_1s.png

# Two simulations compared: ln(IM_1) - ln(IM_2)
eqvis map emod3d/im.h5 PGV --diff sw4/im.h5

# Against recordings, coloured by misfit
eqvis map im.h5 PGA --observed 2026p530771_flatfiles.zip

# Attenuation, and bias across the spectrum
eqvis distance im.h5 PGA --metric rrup --observed flatfiles.zip --view broad
eqvis bias sw4/im.h5 --observed flatfiles.zip --diff emod3d/im.h5 \
    --empirical NSHM2022 --name SW4 --name EMOD3D
```

The raster is clipped to the NZ coastline by default (fetched once and cached);
`--no-clip` disables it and `--coastline PATH` clips against a local `.b64` blob
or `.geojson` instead.

`pick` is the way out of a map with more on it than it can carry. It opens the
map beside a table of the recording stations and a table of the basins, each row
cycling between three states — left off, drawn, or drawn with its name. What it
writes is a pick list, which `map`, `distance` and `bias` all read through
`--stations`, so the figure that ends up in the paper still comes off the
command line:

```sh
eqvis pick taumarunui_sw4.h5 PGA --observed flatfiles.zip -o taumarunui.stations
eqvis map taumarunui_sw4.h5 PGA --observed flatfiles.zip \
    --stations taumarunui.stations -o pga.png
```

### Sources

`rupture-map` draws a realisation's faults in plan view: each source's surface
projection and top-edge trace, numbered in rupture order and coloured by
magnitude off a discrete `magma` ramp, over NZ CFM v1.0 traces, with the
hypocentre marked.

```sh
eqvis rupture-map sw4/R1/realisation.json -o rupture_map.png
```

### Wavefields

`animate` renders an EMOD3D XYTS wavefield over NZCVM topography as a movie —
the one command that is not a matplotlib figure, since a shaded terrain in
perspective is not something matplotlib's 3D axes can draw. Its colours still
come from matplotlib, so the wavefield reads the same here as in the maps.

```sh
eqvis animate realisation.xyts --srf realisation.srf --preview   # one frame, fast
eqvis animate realisation.xyts --srf realisation.srf -o waveform.mp4
```

Given a broadband station file instead, it animates the stations themselves in
plan view — the poor man's wavefield. Which mode you get is detected from the
file's own structure, not from its name: an XYTS file carries the wavefield on
the simulation grid, a broadband file carries one trace per station and so has a
`station` dimension.

```sh
eqvis animate realisation.bb --preview            # the loudest frame
eqvis animate realisation.bb --time-stride 25 -o waveform.mp4
```

It is coarse where the station spacing is coarse and says nothing between
stations, but it needs only the output the workflow already keeps, and it is
enough to see whether a disturbance travels like a wave — outward, decaying — or
does something else. Stations in the sea are dropped: there is nothing there to
shake, and a sea of markers is slower to draw and harder to read than the land
alone.

## The composite database

`ingest` turns a tree of simulation output into a single DuckDB file that can be
queried across runs — which the individual HDF5 files cannot be. It exists to
feed a mixed-effects residual decomposition, so it holds tidy normalised tables
and no pre-computed statistics: no comparison views, no residuals table.

```sh
eqvis ingest validation_results_24-08/ ims.duckdb \
    --observed ../simulation_validation/metadata/ \
    --observed-component rotd50
```

Simulation input is `<event>/[R<n>/]<solver>_<layers>/intensity_measures.h5`, the
cylc `<realisation, sim, layers>` factorial. Each run is converted to parquet
under a staging directory keyed by the source file's size and mtime, so
re-running after adding one event converts one run rather than all of them; the
database itself is reassembled from the whole staged set every time. `--fresh`
reconverts everything, `--jobs` converts in parallel, `--no-fas` skips the
Fourier spectra.

`--observed` is the directory of raw CSVs the ground motion database exports,
which is what the recordings are read from — the earliest form they take, so
nothing sits between them and the database that could already have made a choice
about them:

| File | Shape |
| --- | --- |
| `im_obs.csv` | One row per recorded ground motion: `gm_id`, `event_id`, `stat_id`, the scalars (`PGA`, `PGV`, `CAV`, `AI`, `Ds575`, `Ds595` — no `PGD`), then one wide column per period (`pSA_0.010000000000` …) and per frequency (`EAS_0.100000015334` …) |
| `events.csv` | `event_id` → `event_name`, the dictionary `im_obs.event_id` points into |
| `stations.csv` | `stat_id` → `stat_name`, likewise, and the list of sites that get every spectral component staged |

The ids are resolved to names on the way in and not kept: the database is keyed
on `event` and `station` throughout. CSVs carry no schema, so what a schema
would have refused is checked instead — the files are there and carry their key
columns before any conversion starts, and once the records are loaded, a row
naming an id no dictionary defines, or a second row repeating an
`(event, station)` pair, is reported and dropped. A repeat matters more than it
looks: it would give the observed run two values per
`(station, component, period)` and every simulated row joined to it would pair
with both.

### Shape

| Table | Grain |
| --- | --- |
| `runs` | One row per run, simulated or observed, with the source parameters (magnitude, ztor, dip, rake, tect_type) off the file's root attributes |
| `stations` | `vs30`, `z1pt0`, `z2pt5` — verified identical across an event's configurations, so held once per station |
| `run_stations` | Coordinates and source distances **per run**: EMOD3D and SW4 snap stations to their own grids |
| `scalars` | PGA/PGV/PGD/CAV/AI/Ds575/Ds595, every station, every component |
| `psa`, `fas` | `rotd50`/`eas` at every station; every component at stations that record |
| `periods`, `frequencies` | The ordinate grids, the latter tagged `sim`/`obs` |
| `im_units`, `notes` | Units, and the traps a reader cannot see from the schema |

Read the `notes` table first. It records the things that will otherwise bite:

- **The recordings are one run per event, not one run overall.** A station
  records many earthquakes, so pairing on station alone compares an event
  against every other event's ground motion. Always pair through the run.
- **The spectral tiers.** A `GROUP BY component` over `psa` sees very different
  station counts per component; that is the tier rule, not missing data.
  `run_stations.is_observation_site` says which stations carry all components.
- **pSA periods match between simulation and recording; Fourier frequencies do
  not.** The simulations carry 100 frequencies to 100 Hz, the recordings 240 to
  24.5 Hz, and no value is shared exactly. A join on frequency returns nothing.
  Resampling is interpolation, so it is left outside the database.
- **The observed component is an assertion, not a measurement.** `im_obs.csv`
  has no component column and does not document its convention, which is why
  `--observed-component` is required and has no default.

### Analysis from the database

The commands above all read an `.h5` file or a flatfile. These five read the
database instead, which is the only way to ask a question across runs:

| Command | What it does |
| --- | --- |
| `runs` | What a database holds: its dimensions, its coverage, its site terms |
| `compare` | The residual swept across the spectrum, one series per cell, with a paired-difference panel |
| `residual-map` | The residual in plan view, one panel per cell |
| `residual-heat` | The mean residual over period against a binned covariate |
| `decompose` | The mixed-effects decomposition into event, site and remaining variance |

Two options carry all of them, and they are the reason these are reusable rather
than a report:

```sh
eqvis runs ims.duckdb                       # what dimensions are there?
eqvis compare ims.duckdb --group-by solver --baseline solver=emod3d
eqvis residual-heat ims.duckdb --bin-by rrup --group-by solver
eqvis decompose ims.duckdb --group-by solver --group-by layers --table out.parquet
```

`--label name=value` restricts the run set and `--group-by name` chooses the
comparison axis. Both take their names from the `run_labels` table rather than
from a constant, so a tree ingested with a different `--extract` groups by *its*
dimensions with no change in the code — the same argument the schema makes for
keeping the dimensions as rows. `--bin-by` and `--fixed` are validated against
the columns the schema actually has, so the two enrichment columns became
options without being mentioned anywhere.

The paired-difference panel is the point of `compare`, and it is sharper here
than `bias` can be from two files: every configuration of an event is scored at
the same stations, so differencing them cancels the recording *identically*.
What is left is one simulation against another.

Three things the database cannot support, and the commands say so rather than
guessing: Fourier residuals (the simulated and observed frequency grids share no
value, so the join is empty), a usable-band cutoff (there are no filter corners
in the database, so `--period-max` is the caller's to state), and empirical
model comparisons (no GMM predictions are stored).

### The regression feed

Predictors beyond the source parameters live in `events.duckdb`; attach it.

```sql
ATTACH 'events.duckdb' AS ev (READ_ONLY);
SELECT rs.event, rs.solver, rs.layers, s.station, s.period,
       ln(s.pSA) - ln(o.pSA) AS ln_resid,
       rs.magnitude, rs.ztor, rs.dip, g.rrup, st.vs30, st.z1pt0,
       e.faulting_style, e.basin_area_frac
FROM psa s
JOIN runs rs ON rs.run_id = s.run_id AND rs.kind = 'simulated'
JOIN runs ro ON ro.kind = 'observed' AND ro.event = rs.event
JOIN psa  o  ON o.run_id = ro.run_id AND o.station = s.station
            AND o.period = s.period AND o.component = s.component
JOIN run_stations g ON (g.run_id, g.station) = (s.run_id, s.station)
JOIN stations st ON st.station = s.station
JOIN ev.events e ON e.event = rs.event
WHERE s.component = 'rotd50';
```

## Drawing for a poster

`map`, `rupture-map`, `psa-spectrum` and `bias` can be drawn for a size and a
viewing distance rather than for the page — a panel 25 cm tall, read from three
metres away:

```sh
eqvis map im.h5 PGV --display-height 25 --viewing-distance 3 -o poster_pgv.png
```

Text has to grow about fivefold relative to the figure to stay legible that far
back, which it cannot do without taking the room something else was using: the
tick labels thin out, basin names and the various info panels are dropped
entirely, and the axes end up a smaller part of the image. The saved file has
the same pixel dimensions as ever and is drawn to be *placed* at that height —
scale it to 25 cm and it will be readable; leave it at some other size and the
numbers no longer hold. Either option works alone, each falling back to what the
figure is already drawn for: its natural height, read at arm's length.

## Layout

| Module | Holds |
| --- | --- |
| `cli` | The subcommand register, and nothing else |
| `display` | `Display`: the geometry a figure is drawn for, and how its ink scales |
| `constants` | The shared vocabulary — units, distance metrics, colours |
| `geography` | Coastline and basin outlines; land, scale bars, locator insets |
| `data` | Opening IM files, selecting measures, naming runs |
| `flatfile` | Reading GeoNet flatfile archives |
| `picks` | Reading and writing pick lists |
| `stations` | Station markers, and the collision-avoiding label placer |
| `raster` | Interpolating stations onto a grid, and colour level boundaries |
| `ingest` | Finding runs and staging them to parquet |
| `database` | The database schema, and assembling it from the staged set and the recording CSVs |
| `enrich` | Station elevation and basin, derived from the coordinates already stored |
| `store` | The database as a data source: the run vocabulary, and the file readers' twins |
| `mixed` | Crossed random-effects REML. Knows nothing about earthquakes |
| `compare`, `residuals`, `heatmap`, `decompose` | One database-backed command each |
| `maps`, `picker`, `attenuation`, `bias`, `spectra`, `psa`, `waveforms`, `convert`, `rupture`, `animation` | One command each, with its own drawing code |
