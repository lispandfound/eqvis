"""How the commands talk to the terminal.
"""

import typer


def console_warn(message: str) -> None:
    typer.secho(f"warning: {message}", fg=typer.colors.YELLOW, err=True)
