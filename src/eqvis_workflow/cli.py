"""The ``eqvis`` command line.

Every command lives in the module that owns its drawing code; this file is only
the register, so that the subcommand names live in one place and importing a
drawing module does not drag the whole CLI in behind it.
"""

import typer

from . import (
    animation,
    attenuation,
    bias,
    compare,
    convert,
    decompose,
    heatmap,
    ingest,
    maps,
    pairwise,
    picker,
    psa,
    residuals,
    rupture,
    store,
    waveforms,
)
from . import spectra as spectra_module

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Figures for earthquake simulation workflows.",
)

app.command("map", no_args_is_help=True)(maps.map_im)
app.command("distance", no_args_is_help=True)(attenuation.distance)
app.command("bias", no_args_is_help=True)(bias.bias)
app.command("spectra", no_args_is_help=True)(spectra_module.spectra)
app.command("psa-spectrum", no_args_is_help=True)(psa.psa_spectrum)
app.command("waveform", no_args_is_help=True)(waveforms.waveform)
app.command("pick", no_args_is_help=True)(picker.pick)
app.command("convert", no_args_is_help=True)(convert.convert)
app.command("rupture-map", no_args_is_help=True)(rupture.rupture_map)
app.command("animate", no_args_is_help=True)(animation.animate)
app.command("ingest", no_args_is_help=True)(ingest.ingest)
app.command("runs", no_args_is_help=True)(store.runs)
app.command("compare", no_args_is_help=True)(compare.compare)
app.command("residual-heat", no_args_is_help=True)(heatmap.residual_heat)
app.command("decompose", no_args_is_help=True)(decompose.decompose)
app.command("residual-map", no_args_is_help=True)(residuals.residual_map)
app.command("pairwise", no_args_is_help=True)(pairwise.pairwise)


if __name__ == "__main__":
    app()
