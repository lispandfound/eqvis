"""The supergrid flag: what each of its three states means, and what it costs.

The flag says how far into SW4's absorbing layer a station sits, where the
solver deliberately integrates a damped, coordinate-stretched equation instead
of the wave equation -- so a trace from in there is not a ground motion
prediction. Every assertion here is about a decision rather than a picture: is a
missing flag the same as a clean one (no), does a fill value read as a depth
(no), and does a file with no flag at all behave exactly as it did before the
flag existed (yes, and that one is asserted rather than assumed).

The file is written and reopened through :func:`~eqvis_workflow.data.open_ims`
rather than assembled in memory wherever the reader itself is what is on trial:
``open_ims`` passes ``mask_and_scale=False``, and the traps below only exist
because of that.
"""

import numpy as np
import pytest
import xarray as xr

from eqvis_workflow.data import (
    SUPERGRID_DEPTH,
    SUPERGRID_DEPTH_GP,
    Screen,
    Supergrid,
    open_ims,
    supergrid,
    supergrid_note,
)

STATIONS = ["AAAA", "BBBB", "CCCC", "DDDD"]
PERIODS = np.array([0.1, 1.0, 5.0])


def one_group(stations, depth=None, gridpoints=None, encoding=None, attrs=None):
    """A single IM group, optionally carrying the supergrid coordinates."""
    n = len(stations)
    coords = {
        "station": ("station", list(stations)),
        "latitude": ("station", np.linspace(-44.9, -44.1, n)),
        "longitude": ("station", np.linspace(170.1, 171.9, n)),
    }
    if depth is not None:
        coords[SUPERGRID_DEPTH] = ("station", np.asarray(depth, dtype=np.float32))
    if gridpoints is not None:
        coords[SUPERGRID_DEPTH_GP] = (
            "station",
            np.asarray(gridpoints, dtype=np.float32),
        )
    dataset = xr.Dataset(
        {"rotd50": (("station", "period"), np.ones((n, len(PERIODS))))},
        coords={**coords, "period": ("period", PERIODS)},
    )
    if depth is not None:
        dataset[SUPERGRID_DEPTH].attrs.update(attrs or {})
        dataset[SUPERGRID_DEPTH].encoding.update(encoding or {})
    return dataset


def write_tree(path, groups):
    """Write ``groups`` as an IM datatree and reopen it the way a command does."""
    tree = xr.DataTree.from_dict({f"/{name}": ds for name, ds in groups.items()})
    tree.to_netcdf(path, engine="h5netcdf", mode="w")
    return open_ims(path)


class TestThreeStates:
    """0.0 is clean, > 0 is flagged, NaN is unknown -- and they stay distinct."""

    def test_zero_is_clean_and_not_flagged(self):
        node = one_group(STATIONS, depth=[0.0, 0.0, 0.0, 0.0])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert sg.stated
        assert not sg.flagged.any()
        assert sg.clean.all()
        assert not sg.unknown.any()

    def test_a_positive_depth_is_flagged_and_not_clean(self):
        node = one_group(STATIONS, depth=[0.0, 750.0, 5750.0, 0.0])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert list(sg.flagged) == [False, True, True, False]
        assert list(sg.clean) == [True, False, False, True]
        assert not sg.unknown.any()

    def test_nan_is_unknown_and_neither_clean_nor_flagged(self):
        """"We do not know" must never be read as "this station is fine"."""
        node = one_group(STATIONS, depth=[0.0, np.nan, 750.0, np.nan])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert list(sg.unknown) == [False, True, False, True]
        assert list(sg.flagged) == [False, False, True, False]
        assert list(sg.clean) == [True, False, False, False]

    def test_an_unstated_flag_is_unknown_everywhere(self):
        node = one_group(STATIONS)
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert not sg.stated
        assert sg.unknown.all()
        assert not sg.flagged.any()
        assert not sg.clean.any()

    def test_the_flag_covers_every_station_even_when_unstated(self):
        """An absent flag still has to be an array of the right length.

        The whole design is that a file without the flag takes the same code
        path as one with it, so anything that indexes the mask by station must
        keep working rather than broadcasting a scalar.
        """
        node = one_group(STATIONS)
        assert supergrid(xr.DataTree(), node["rotd50"]).depth.shape == (4,)


class TestBackwardCompatibility:
    """A file with no flag must behave exactly as it did before the flag."""

    def test_the_mask_is_identical_to_the_one_computed_without_the_flag(self):
        """The back-compat guarantee, asserted rather than assumed."""
        node = one_group(STATIONS)
        reached = np.array([True, True, False, True])
        nearest = np.array([0, 1, 2, 3, 3])

        today = reached.copy()
        sg = supergrid(xr.DataTree(), node["rotd50"])
        with_flag = reached.copy()
        with_flag &= ~sg.flagged[nearest][: reached.size]

        assert np.array_equal(with_flag, today)

    def test_a_file_with_no_flag_needs_no_special_case_from_the_caller(self, tmp_path):
        """Straight through the real reader, since that is where a file arrives."""
        tree = write_tree(tmp_path / "im.h5", {"pSA": one_group(STATIONS)})
        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert not sg.stated
        assert supergrid_note(sg) is None


class TestFillValues:
    """``open_ims`` passes ``mask_and_scale=False``, so the fill arrives raw.

    Both traps are silent if the fill is not honoured here, and they fail in
    opposite directions -- which is why both are pinned.
    """

    def test_a_negative_fill_reads_as_unknown_rather_than_clean(self, tmp_path):
        """-9999 > 0 is False, so an unhonoured fill would report it clean."""
        node = one_group(
            STATIONS,
            depth=[0.0, 750.0, np.nan, np.nan],
            encoding={"_FillValue": np.float32(-9999.0)},
        )
        tree = write_tree(tmp_path / "im.h5", {"pSA": node})

        raw = tree["pSA"][SUPERGRID_DEPTH].values
        assert raw[2] == pytest.approx(-9999.0), "the trap must really be present"

        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert list(sg.unknown) == [False, False, True, True]
        assert not sg.clean[2], "a fill value is not a station reported as interior"
        assert list(sg.flagged) == [False, True, False, False]

    def test_the_netcdf_default_fill_does_not_flag_every_station(self, tmp_path):
        """9.969e36 > 0 is True, so an unhonoured fill would flag the whole file."""
        node = one_group(
            STATIONS,
            depth=[0.0, 750.0, np.nan, np.nan],
            encoding={"_FillValue": np.float32(9.969209968386869e36)},
        )
        tree = write_tree(tmp_path / "im.h5", {"pSA": node})

        raw = tree["pSA"][SUPERGRID_DEPTH].values
        assert raw[2] > 1e30, "the trap must really be present"

        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert list(sg.flagged) == [False, True, False, False]
        assert list(sg.unknown) == [False, False, True, True]

    def test_missing_value_is_honoured_as_well_as_fill_value(self):
        """The other spelling of the same convention, and just as silent."""
        node = one_group(
            STATIONS,
            depth=[0.0, 750.0, -9999.0, 0.0],
            attrs={"missing_value": -9999.0},
        )
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert list(sg.unknown) == [False, False, True, False]

    def test_a_real_depth_of_zero_survives_a_nonzero_fill(self):
        """A clean station is a claim, and a fill must not be able to erase it."""
        node = one_group(
            STATIONS,
            depth=[0.0, 0.0, -9999.0, 750.0],
            attrs={"_FillValue": -9999.0},
        )
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert list(sg.clean) == [True, True, False, False]
        assert list(sg.unknown) == [False, False, True, False]


class TestGroupResolution:
    """Where the flag is looked for, and why more than one place is needed."""

    def test_a_donor_group_supplies_the_flag_when_the_array_lacks_it(self, tmp_path):
        """``bias`` never reads PGA and ``ingest`` reads only PGA.

        So a flag that landed on one group and not the other would be invisible
        to the command that most needs it. Looking in the other groups is what
        removes that whole class of silent failure.
        """
        tree = write_tree(
            tmp_path / "im.h5",
            {
                "PGA": one_group(STATIONS, depth=[0.0, 750.0, 0.0, 0.0]),
                "pSA": one_group(STATIONS),
            },
        )
        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert sg.stated
        assert list(sg.flagged) == [False, True, False, False]

    def test_the_arrays_own_coords_win_over_a_donor_group(self, tmp_path):
        tree = write_tree(
            tmp_path / "im.h5",
            {
                "PGA": one_group(STATIONS, depth=[750.0, 0.0, 0.0, 0.0]),
                "pSA": one_group(STATIONS, depth=[0.0, 750.0, 0.0, 0.0]),
            },
        )
        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert list(sg.flagged) == [False, True, False, False]

    def test_a_donor_with_a_different_station_set_is_matched_by_name(self, tmp_path):
        """A station the donor never saw comes back unknown, not a neighbour's."""
        tree = write_tree(
            tmp_path / "im.h5",
            {
                "PGA": one_group(STATIONS[:3], depth=[0.0, 750.0, 0.0]),
                "pSA": one_group(STATIONS),
            },
        )
        sg = supergrid(tree, tree["pSA"]["rotd50"])
        assert list(sg.flagged) == [False, True, False, False]
        assert list(sg.unknown) == [False, False, False, True]

    def test_disagreeing_donor_groups_are_reported_once(self, tmp_path, capsys):
        tree = write_tree(
            tmp_path / "im.h5",
            {
                "PGA": one_group(STATIONS, depth=[0.0, 750.0, 0.0, 0.0]),
                "PGV": one_group(STATIONS, depth=[750.0, 0.0, 0.0, 0.0]),
                "PGD": one_group(STATIONS, depth=[0.0, 0.0, 750.0, 0.0]),
                "pSA": one_group(STATIONS),
            },
        )
        sg = supergrid(tree, tree["pSA"]["rotd50"])
        warnings = capsys.readouterr().err
        assert warnings.count("disagree") == 1
        assert list(sg.flagged) == [False, True, False, False], "the first group wins"

    def test_the_flag_is_found_with_no_array_to_align_to(self, tmp_path):
        """``--info`` screens a file without selecting a measure from it."""
        tree = write_tree(
            tmp_path / "im.h5",
            {"pSA": one_group(STATIONS, depth=[0.0, 750.0, 0.0, 0.0])},
        )
        sg = supergrid(tree)
        assert sg.depth.shape == (4,)
        assert list(sg.flagged) == [False, True, False, False]


class TestMetresWin:
    """Metres decide the threshold; grid points are a rounded restatement."""

    def test_metres_win_over_grid_points_rounded_to_zero(self):
        """A station 100 m into a 400 m grid is 0.25 gp, which rounds to clean."""
        node = one_group(
            STATIONS,
            depth=[0.0, 100.0, 750.0, 0.0],
            gridpoints=[0.0, 0.0, 2.0, 0.0],
        )
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert list(sg.flagged) == [False, True, True, False]
        assert list(sg.gridpoints) == [0.0, 0.0, 2.0, 0.0]

    def test_grid_points_alone_are_unknown_and_said_to_be(self, capsys):
        """Both are written under one guard, so one alone is a corrupt file.

        Grid points cannot be converted to metres without the grid spacing, and
        the threshold is stated in metres, so the honest answer is "unknown" --
        loudly, because the file is wrong.
        """
        node = one_group(STATIONS, gridpoints=[0.0, 2.0, 0.0, 0.0])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        assert sg.unknown.all()
        assert not sg.flagged.any()
        assert "grid points alone" in capsys.readouterr().err


class TestBiasComposition:
    """The one line ``bias`` adds, and what it does to the station set."""

    def test_the_mask_drops_only_the_flagged_stations(self):
        node = one_group(STATIONS, depth=[0.0, 750.0, np.nan, 0.0])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        # One recording per simulation station, plus a fifth outside the grid.
        nearest = np.array([0, 1, 2, 3, 3])
        reached = np.array([True, True, True, True, False])

        inside = sg.flagged[nearest] & reached
        reached &= ~inside

        assert list(reached) == [True, False, True, True, False]
        assert int(inside.sum()) == 1, "the unreached recording is not double-counted"

    def test_an_unknown_station_is_kept_rather_than_dropped(self):
        """Excluding on "we do not know" would throw away every old file."""
        node = one_group(STATIONS, depth=[np.nan, np.nan, np.nan, np.nan])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        nearest = np.arange(4)
        reached = np.ones(4, bool)
        reached &= ~sg.flagged[nearest]
        assert reached.all()

    def test_a_wholly_flagged_run_leaves_nothing_to_score(self):
        """The state the refusal exists for: the original investigation's case."""
        node = one_group(STATIONS, depth=[750.0, 800.0, 5750.0, 1200.0])
        sg = supergrid(xr.DataTree(), node["rotd50"])
        reached = np.ones(4, bool)
        inside = sg.flagged[np.arange(4)] & reached
        reached &= ~inside
        assert int(inside.sum()) == 4
        assert not reached.any()


class TestNote:
    """The line ``--info`` prints, which is the whole single-file screen."""

    def test_nothing_is_said_about_a_solver_with_no_absorbing_layer(self):
        sg = supergrid(xr.DataTree(), one_group(STATIONS)["rotd50"])
        assert supergrid_note(sg) is None

    def test_a_clean_run_is_positively_confirmed(self):
        node = one_group(STATIONS, depth=[0.0, 0.0, 0.0, 0.0])
        note = supergrid_note(supergrid(xr.DataTree(), node["rotd50"]))
        assert note is not None
        assert "no stations inside" in note

    def test_the_count_and_the_worst_depth_are_both_named(self):
        node = one_group(STATIONS, depth=[0.0, 750.0, 5750.0, np.nan])
        note = supergrid_note(supergrid(xr.DataTree(), node["rotd50"]))
        assert "2 of 4" in note
        assert "5750 m" in note
        assert "1 not reported" in note

    def test_the_note_is_a_plain_function_of_the_dataclass(self):
        """So a caller can build one without a file, and the wording is one place."""
        sg = Supergrid(
            depth=np.array([0.0, 750.0]),
            gridpoints=np.array([0.0, 2.0]),
            stated=True,
        )
        assert "1 of 2" in supergrid_note(sg)


class TestScreen:
    """The option's three values, spelled the same in every command."""

    def test_the_values_are_the_three_the_commands_share(self):
        assert [s.value for s in Screen] == ["exclude", "mark", "keep"]

    def test_a_screen_is_its_own_string(self):
        """Typer reads a str-Enum for its choice list, as ``View`` already does."""
        assert Screen.exclude == "exclude"
