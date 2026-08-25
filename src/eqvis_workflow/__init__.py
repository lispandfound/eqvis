"""Figures for earthquake simulation workflows.

Everything here is drawn with matplotlib, from one shared vocabulary: the same
colours mean the same things across the commands (green is a recording, blue is
a comparison run or an empirical model, grey is a simulation drawn in bulk and
black the same simulation drawn at a point), the same :class:`~.display.Display`
sizes a figure for the wall it will hang on, and the same coastline and basin
outlines provide the geography.

The commands fall into three groups:

Intensity measures, read from a simulation IM file
    ``map`` draws a measure spatially, ``distance`` plots it against a source
    distance metric, ``bias`` sweeps the log residual against recordings across
    the whole spectrum, ``spectra`` and ``psa-spectrum`` draw single-station
    spectra, ``waveform`` draws time series, ``pick`` opens an interactive
    window for choosing what a crowded map should carry, and ``convert``
    reshapes a flat IM dataset into the datatree the rest of them read.

Sources
    ``rupture-map`` draws a realisation's faults, rupture order and hypocentre
    in plan view over the NZ Community Fault Model.

Wavefields
    ``animate`` renders an EMOD3D XYTS wavefield over topography as a movie.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
