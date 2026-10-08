"""The ``eqvis`` command line.

Every command lives in the module that owns its drawing code; this file is only
the register, so that the subcommand names live in one place and importing a
drawing module does not drag the whole CLI in behind it.
"""

import typer

from . import (
    adjust,
    animation,
    attenuation,
    bias,
    compare,
    convert,
    decompose,
    domain,
    hazard,
    heatmap,
    ingest,
    maps,
    pairwise,
    picker,
    psa,
    radiation,
    residuals,
    rupture,
    slip,
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
app.command("adjust", no_args_is_help=True)(adjust.adjust)
app.command("rupture-map", no_args_is_help=True)(rupture.rupture_map)
app.command("domain", no_args_is_help=True)(domain.domain_map)
app.command("slip-panels", no_args_is_help=True)(slip.slip_panels)
app.command("animate", no_args_is_help=True)(animation.animate)
app.command("ingest", no_args_is_help=True)(ingest.ingest)
app.command("runs", no_args_is_help=True)(store.runs)
app.command("compare", no_args_is_help=True)(compare.compare)
app.command("residual-heat", no_args_is_help=True)(heatmap.residual_heat)
app.command("decompose", no_args_is_help=True)(decompose.decompose)
app.command("residual-map", no_args_is_help=True)(residuals.residual_map)
app.command("hazard-map", no_args_is_help=True)(hazard.hazard_map)
app.command("hazard-deficit", no_args_is_help=True)(hazard.hazard_deficit)
app.command("pairwise", no_args_is_help=True)(pairwise.pairwise)
app.command("radiation", no_args_is_help=True)(radiation.radiation)


if __name__ == "__main__":
    app()
