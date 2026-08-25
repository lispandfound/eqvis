"""``convert``: a flat IM dataset reshaped into the datatree the commands read.

Some IM files come flat -- every measure side by side, the components of motion
stacked into a dimension. This turns one into the group-per-measure datatree
the rest of the package expects. A file for the same event supplies what the
flat form leaves out (source geometry, site terms, empirical predictions)::

    eqvis convert flat.h5 intensity_measures.h5 --like validation/im.h5
"""

from pathlib import Path, PurePosixPath
from typing import Annotated

import numpy as np
import typer
import xarray as xr

from .console import console_warn
from .constants import BORROWED_TERMS, DEFAULT_COMPONENT, PINT_UNITS
from .data import open_ims


def short_units(attrs: dict) -> dict:
    """``attrs`` with a spelled-out unit name replaced by its short form."""
    units = attrs.get("units")
    if units is None:
        return {}
    return {"units": PINT_UNITS.get(str(units), str(units))}


def defined_components(da: xr.DataArray, sample: int = 500) -> list[str]:
    """The components a measure is actually defined for.

    A flat dataset gives every measure the whole component axis and fills the
    combinations that do not exist -- rotd on the durations, on CAV, on the
    Fourier spectra -- with NaN, so which ones are real is found by looking
    rather than hardcoded. The first ``sample`` stations decide it: checking
    every station of every component would mean reading the entire file just to
    learn its shape.
    """
    probe = da.isel(station=slice(0, sample)).notnull()
    spare = [str(dim) for dim in probe.dims if dim != "component"]
    defined = probe.any(dim=spare)
    return [str(c) for c in da.component.values if bool(defined.sel(component=c))]


def donor_node(reference: xr.DataTree) -> xr.Dataset | None:
    """A group of ``reference`` carrying the per-station coordinates."""
    for name in reference.children:
        node = reference[name]
        if "station" in node.dims:
            return node.dataset
    return None


def borrowed_coords(
    reference: xr.DataTree, stations: np.ndarray
) -> dict[str, xr.DataArray]:
    """Site terms from ``reference``, matched to ``stations`` by name.

    These -- vs30, the basin depths, the signed fault distances -- are
    properties of the site and the source geometry, not of the run, so they
    carry over to another simulation of the same event. Matching is on the
    station name, so a station the reference has never seen comes back NaN
    rather than quietly taking a neighbour's site.
    """
    donor = None if reference is None else donor_node(reference)
    if donor is None:
        return {}
    borrowed = {}
    for term in BORROWED_TERMS:
        if term not in donor.coords:
            continue
        da = donor[term].drop_vars(
            [name for name in donor[term].coords if name != "station"]
        )
        borrowed[term] = da.reindex(station=stations).assign_attrs(
            short_units(da.attrs)
        )
    return borrowed


def empirical_groups(
    reference: xr.DataTree, nodes: dict[str, xr.Dataset], stations: np.ndarray
) -> dict[str, xr.Dataset]:
    """Empirical predictions from ``reference``, re-hung on the new stations.

    An empirical prediction is a function of the site and the source, not of
    the simulation, so it transfers to another run of the same event unchanged
    apart from which stations it is stated at. A group whose remaining
    dimensions (the period or frequency axis) disagree with the measure it
    would hang off is skipped: it predicts something the new file cannot line
    up against.
    """
    groups = {}
    for path, node in reference.subtree_with_keys:
        parts = PurePosixPath(path).parts
        if "empirical" not in parts or not node.dataset.data_vars:
            continue
        parent = nodes.get(parts[0])
        if parent is None:
            console_warn(f"dropping {path}: the flat file has no {parts[0]}")
            continue
        ds = node.dataset[list(node.dataset.data_vars)]
        ds = ds.drop_vars(
            [
                name
                for name, coord in ds.coords.items()
                if "station" in coord.dims and name != "station"
            ]
        )
        ds = ds.reindex(station=stations)
        mismatched = [
            dim
            for dim in ds.dims
            if dim != "station"
            and not (dim in parent.dims and parent.sizes[dim] == ds.sizes[dim])
        ]
        if mismatched:
            console_warn(f"dropping {path}: {parts[0]} has different {mismatched}")
            continue
        groups[path] = ds.assign_coords(parent.coords).assign_attrs(node.attrs)
    return groups


def flat_to_tree(flat: xr.Dataset, reference: xr.DataTree | None) -> xr.DataTree:
    """Turn a flat IM dataset into the datatree the plotting commands read.

    The flat form stacks the components of motion into a dimension and keeps
    every measure side by side in one dataset; the datatree gives each measure
    its own group and each component its own variable, so a measure only
    carries the components it has and the groups can differ in shape.

    A ``reference`` tree for the same event supplies what the flat form leaves
    out -- the source geometry and event attributes, the site terms, the
    empirical predictions -- and its measure descriptions and units win where
    the two disagree, so converted files sit alongside native ones.
    """
    coords = {
        str(name): coord.assign_attrs(short_units(coord.attrs))
        for name, coord in flat.coords.items()
        if coord.dims == ("station",)
    }
    stations = flat.station.values
    if reference is not None:
        coords |= borrowed_coords(reference, stations)

    nodes: dict[str, xr.Dataset] = {}
    for name, da in flat.data_vars.items():
        im = str(name)
        if "component" not in da.dims:
            console_warn(f"skipping {im}: no component dimension")
            continue
        group = (
            reference[im]
            if reference is not None and im in reference.children
            else None
        )
        units = short_units(da.attrs)
        attrs = {"name": im, "description": da.attrs.get("description", "")}
        if group is not None:
            attrs = dict(group.attrs)
            first = next(iter(group.dataset.data_vars.values()), None)
            if first is not None:
                units = short_units(first.attrs)
        nodes[im] = xr.Dataset(
            {
                component: da.sel(component=component, drop=True).assign_attrs(units)
                for component in defined_components(da)
            },
            coords=coords,
            attrs=attrs,
        )
        default = DEFAULT_COMPONENT.get(im)
        if default is not None and default not in nodes[im].data_vars:
            console_warn(
                f"{im} has no {default!r} component; plotting it needs --component"
            )

    root = xr.Dataset(attrs=dict(flat.attrs))
    if reference is not None:
        root.attrs.update(reference.attrs)
        nodes |= empirical_groups(reference, nodes, stations)
    return xr.DataTree.from_dict({"/": root, **nodes})


def convert(
    flat_file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Flat intensity measure dataset"
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Datatree file to write")],
    like: Annotated[
        Path | None,
        typer.Option(
            "--like",
            exists=True,
            dir_okay=False,
            help="Datatree for the same event to copy attributes, site terms and "
            "empirical predictions from",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing output file")
    ] = False,
):
    """Convert a flat IM dataset into the datatree these commands read.

    The flat form is one dataset with a ``component`` dimension; the datatree
    is one group per intensity measure holding one variable per component. With
    ``--like`` pointing at a datatree for the same event, everything the flat
    file does not carry is taken from there -- the source geometry, magnitude
    and event name, the site terms (vs30, z1pt0, z2pt5) and signed distances
    (rx, ry), and any empirical predictions -- matched to the new file's
    stations by name::

        eqvis convert flat.h5 intensity_measures.h5 \\
            --like ../validation/2012p001887/intensity_measures.h5
    """
    if output.exists() and not force:
        raise typer.BadParameter(f"{output} exists; pass --force to overwrite")
    flat = xr.open_dataset(flat_file, engine="h5netcdf", mask_and_scale=False)
    reference = open_ims(like) if like is not None else None
    if reference is not None:
        donor = donor_node(reference)
        known = (
            0 if donor is None else np.isin(flat.station.values, donor.station).sum()
        )
        print(f"{like}: {known} of {flat.sizes['station']} stations matched by name")
    tree = flat_to_tree(flat, reference)
    for path, node in tree.subtree_with_keys:
        if node.dataset.data_vars:
            print(f"  {path:<24} {' '.join(str(v) for v in node.dataset.data_vars)}")
    tree.to_netcdf(output, engine="h5netcdf")
    print(f"wrote {output}")
