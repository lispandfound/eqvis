"""The vocabulary the figures share.

Units, component names, distance metrics and -- most of all -- colours. A
colour means the same thing in every command that uses it: green is a
recording, blue a comparison run or an empirical model, grey a simulation drawn
as a whole field and black the same simulation drawn at one station. Keeping
them here rather than at each use is what makes that promise checkable.
"""

# Sensible default component for each IM (rotd50 where defined, geometric
# mean for cumulative/duration measures, EAS for Fourier spectra).
DEFAULT_COMPONENT = {
    "PGA": "rotd50",
    "PGV": "rotd50",
    "pSA": "rotd50",
    "CAV": "geom",
    "AI": "geom",
    "Ds575": "geom",
    "Ds595": "geom",
    "FAS": "eas",
}


# IMs spanning orders of magnitude get log-spaced levels; durations stay linear.
LOG_SCALED = {"PGA", "PGV", "pSA", "AI", "CAV", "FAS"}


UNIT_LABEL = {
    "g0": "g",
    "cm/s": "cm/s",
    "m/s": "m/s",
    "s": "s",
    "g0 * s": "g·s",
}


SUBSCRIPTS = {
    "Ds575": r"D$_{s5-75}$",
    "Ds595": r"D$_{s5-95}$",
}


# Source distance metrics: simulation coordinate name -> flatfile column. Which
# ones exist is per-file, so both sides are checked before use.
DISTANCE_COLUMN = {
    "rrup": "r_rup",
    "rjb": "r_jb",
    "rx": "r_x",
    "ry": "r_y",
    "epi": "r_epi",
    "hyp": "r_hyp",
}


DISTANCE_LABEL = {
    "rrup": "R$_{rup}$",
    "rjb": "R$_{JB}$",
    "rx": "R$_x$",
    "ry": "R$_y$",
    "epi": "R$_{epi}$",
    "hyp": "R$_{hyp}$",
}


# Rx and Ry are signed (fault-normal / along-strike), so they cannot go on a
# log axis; the rest are non-negative radial distances and conventionally do.
SIGNED_DISTANCES = {"rx", "ry"}


OBSERVED_GREEN = "#2ca25f"


# Basins are context, not data: neutral so red keeps meaning "positive
# residual", and quiet enough to sit behind the stations.
# What a picker does with an item, cycled by clicking it: left off the map,
# drawn, or drawn with its name beside it.
HIDDEN, SHOWN, NAMED = 0, 1, 2


BASIN_LINE = "#707070"


BASIN_TEXT = "#5a5a5a"


# Green means "a recording" throughout, so a second simulation gets its own
# colour rather than borrowing that meaning.
COMPARISON_BLUE = "#1f6feb"


# Empirical predictions: blue like the comparison run but paler, and always
# drawn as a band with a dashed centre so the two are told apart by shape.
EMPIRICAL_BLUE = "#4c8fd6"


# A simulation is grey where it is drawn as a whole field and black where it is
# drawn at a recording: the same run, quiet in bulk and definite at a point.
SIM_CLOUD_GREY = "#c2c2cc"


SIM_ONE_BLACK = "black"


# ``distance`` is the one view that can show a second run *and* an empirical
# model at once, and blue is spoken for by the model there, so the second run
# goes warm rather than borrowing COMPARISON_BLUE as the other commands do.
SIM_TWO_ORANGE = "#d95f02"


SIM_TWO_CLOUD = "#f0c8a8"


# One run against the other is about the pair, not either of them, so it is
# drawn in neither's colour.
DIFFERENCE_INK = "#3b3b6b"


# A station inside the SW4 supergrid absorbing layer is not another run, so it
# must not take a run's hue: the statement being made is "this number is not a
# ground motion", which is an annotation over the data rather than a series in
# it. This red already carries exactly that meaning in ``spectra``'s per-decade
# rules, and it reads as a warning next to every colour above.
SUPERGRID_RED = "#c1272d"


# Flat IM datasets spell their units out in full (pint's repr); the datatree
# these commands read uses the short forms.
PINT_UNITS = {
    "standard_gravity": "g0",
    "centimeter / second": "cm/s",
    "centimeter": "cm",
    "meter / second": "m/s",
    "second": "s",
    "second * standard_gravity": "g0 * s",
    "standard_gravity * second": "g0 * s",
    "kilometer": "km",
    "degree": "degrees",
}


# Station terms a flat dataset has no room for. They are properties of the
# site, not of the run, so they can be borrowed from another file for the same
# stations -- see ``convert``.
BORROWED_TERMS = ("rx", "ry", "vs30", "z1pt0", "z2pt5")


# Degrees of latitude to kilometres, for the separation the thinning works in.
KM_PER_DEGREE = 111.195


# Simulation waveform components, in file order, and the observed file suffix
# each corresponds to. Verified against the IM file: max|x| == PGA(000) etc.
WAVEFORM_COMPONENTS = {"x": "000", "y": "090", "z": "ver"}


GRAVITY_CM_S2 = 980.665


# Fractions of total Arias intensity bounding the significant duration measures.
ARIAS_LEVELS = (0.05, 0.75, 0.95)


# Spectral columns in the eas ground motion table, one per frequency in Hz.
FAS_PREFIX = "FAS_"


# Decades over which the spectral misfit is summarised.
SPECTRAL_BANDS = ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0))


# Short period -> long period, warm to cool (a conventional seismological
# ordering, and distinct enough at the small size a legend swatch renders at).
DIRECTIVITY_COLOURS = ["#c1272d", "#e8871e", "#2ca25f", "#2166ac"]


# --- Intensity measure vocabulary for the composite database --------------
# Scalar measures, in the order they become columns of the `scalars` table.
# PGD is simulation-only: the observed flatfile does not carry it.
SCALAR_IMS = ("PGA", "PGV", "PGD", "CAV", "AI", "Ds575", "Ds595")

# The spectral groups, and the axis each is resolved over.
SPECTRAL_AXIS = {"pSA": "period", "FAS": "frequency"}

# The one component of each spectral group kept at every station, rather than
# only at stations that record. See `eqvis_workflow.ingest` for the tier rule.
GRID_COMPONENT = {"pSA": "rotd50", "FAS": "eas"}

# The units the IM writer records, carried into the database so that nobody
# has to guess whether a residual compares like with like.
IM_UNITS = {
    "PGA": "g",
    "PGV": "cm/s",
    "PGD": "cm",
    "CAV": "m/s",
    "AI": "m/s",
    "Ds575": "s",
    "Ds595": "s",
    "pSA": "g",
    "FAS": "g s",
}
