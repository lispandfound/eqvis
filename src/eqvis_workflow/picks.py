"""Pick lists: which stations and basins a figure should carry.

A plain-text file with a section per kind, a name per line, and ``: unnamed``
on anything to be drawn without its label. :mod:`~.picker` writes them
interactively; ``map``, ``distance`` and ``bias`` read them through
``--stations``, so a figure chosen by eye still comes off the command line.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import typer

from .console import console_warn
from .constants import NAMED, SHOWN

# A pick list is grouped under these headings, and a name on its own is drawn
# with its label; the marker below, after a colon, takes the label away and
# leaves the marker. A colon separates them because basin names have spaces in
# them ("Greater Wellington") and splitting on whitespace would eat the name.
PICK_SECTIONS = ("stations", "basins")


TITLE_SECTION = "title"


UNNAMED_MARKER = "unnamed"


def read_pick_list(path: Path) -> dict:
    """A picked selection: what to draw, what to name, and the title to draw it under.

    Returns ``{"stations": {name: named}, "basins": {name: named}, "title": str
    or None}``. ``[stations]``, ``[basins]`` and ``[title]`` open a section, a
    name followed by ``: unnamed`` is drawn without its label, and a line whose
    first character is ``#`` is a comment. Only a whole line comments: a title
    is free text and is entitled to contain a ``#``. A file with no section
    heading at all is read as a bare list of stations, which is what a
    hand-written one usually is.
    """
    picked: dict = {section: {} for section in PICK_SECTIONS}
    picked[TITLE_SECTION] = None
    section = "stations"
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("[") and entry.endswith("]"):
            section = entry[1:-1].strip().casefold()
            if section not in picked:
                raise typer.BadParameter(f"{path} has an unknown section [{section}]")
            continue
        if section == TITLE_SECTION:
            picked[TITLE_SECTION] = entry
            continue
        name, _, marker = entry.partition(":")
        name, marker = name.strip(), marker.strip().casefold()
        if marker and marker != UNNAMED_MARKER:
            raise typer.BadParameter(
                f"{path} marks {name} {marker!r}; the only marker is "
                f"{UNNAMED_MARKER!r}"
            )
        picked[section][name] = marker != UNNAMED_MARKER
    if not any(picked[section] for section in PICK_SECTIONS):
        raise typer.BadParameter(f"{path} names nothing to draw")
    return picked


def write_pick_list(path: Path, picked: dict, provenance: str) -> None:
    """Write a picked selection, with a note on where it came from."""
    lines = [
        f"# {provenance}",
        f"# a name followed by `: {UNNAMED_MARKER}` is drawn without its label",
    ]
    if picked.get(TITLE_SECTION):
        lines += [f"[{TITLE_SECTION}]", picked[TITLE_SECTION]]
    for section in PICK_SECTIONS:
        lines.append(f"[{section}]")
        for name, named in picked[section].items():
            lines.append(name if named else f"{name}: {UNNAMED_MARKER}")
    path.write_text("\n".join([*lines, ""]))


def restrict_to_stations(
    observed: dict[str, np.ndarray], keep: Sequence[str], source: Path
) -> dict[str, np.ndarray]:
    """Keep only the named observed stations, naming any the archive lacks."""
    wanted = list(dict.fromkeys(keep))
    missing = set(wanted) - set(observed["name"].tolist())
    if missing:
        console_warn(f"{source} has no station {', '.join(sorted(missing))}")
    chosen = np.isin(observed["name"], wanted)
    if not chosen.any():
        console_warn("the station list left nothing to plot")
    return {key: value[chosen] for key, value in observed.items()}


def pick_states(picked: dict[str, bool] | None) -> dict[str, int] | None:
    """A pick list section as the states the drawing code reads.

    The file records only what to draw and whether to name it, so a name in it
    is :data:`NAMED` or :data:`SHOWN` and one left out of it is absent -- which
    the drawing code reads as :data:`HIDDEN`.
    """
    if picked is None:
        return None
    return {name: NAMED if named else SHOWN for name, named in picked.items()}


def named_mask(names: np.ndarray, picked: dict[str, bool] | None) -> np.ndarray:
    """Which of ``names`` a pick list asks to label; all of them without one."""
    if picked is None:
        return np.ones(len(names), dtype=bool)
    return np.array([picked.get(name, True) for name in names], dtype=bool)
