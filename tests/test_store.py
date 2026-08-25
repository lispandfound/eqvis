"""Tests for the run vocabulary the analysis commands are built on.

The point of these is the last class. Everything else checks that a selection
means what it says; ``TestGenerality`` checks that none of it knows what a
solver is -- which is the property that makes the analysis commands reusable on
a tree ingested along different axes, and the one a careless change would break
without breaking anything else.
"""

import ast
import inspect

import duckdb
import pytest
import typer
from conftest import write_im_file

from eqvis_workflow import database, ingest, store


def code_without_docstrings(module) -> str:
    """A module's source with every docstring removed.

    A docstring may name a real dimension as an example -- explaining what
    ``--label solver=sw4`` means is exactly what a docstring is for. Code may
    not. Stripping by AST rather than by line prefix is what tells the two
    apart; a multi-line docstring has only its first line starting with a quote.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.unparse(tree)


@pytest.fixture(scope="module")
def con(built_path):
    with store.connect(built_path) as connection:
        yield connection


@pytest.fixture(scope="module")
def variant_db(tmp_path_factory, observed):
    """The same runs, ingested under a regex naming one dimension of its own.

    ``solver`` and ``layers`` do not exist in this database; ``variant`` does.
    Nothing in :mod:`~eqvis_workflow.store` should notice the difference.
    """
    root = tmp_path_factory.mktemp("variant")
    for event in ("2020p111111", "2021p222222"):
        for variant in ("coarse", "fine"):
            write_im_file(
                root / event / variant / ingest.IM_FILE, ["AAAA", "BBBB"], 7, False
            )
    work = tmp_path_factory.mktemp("variant_built")
    out = work / "ims.duckdb"
    extract = r"(?P<event>[^/]+)/(?P<variant>[^/]+)/intensity_measures\.h5$"
    sites = database.observation_sites(observed)
    runs, manifest = ingest.stage(root, work / "build", sites, extract=extract)
    database.build(
        out, work / "build", runs, manifest, observed, "rotd50", extract=extract
    )
    with store.connect(out) as connection:
        yield connection


class TestConnect:
    def test_a_database_that_is_not_one_is_refused(self, tmp_path):
        """A wrong path should fail by name, not as a binder error mid-join."""
        other = tmp_path / "other.duckdb"
        with duckdb.connect(str(other)) as con:
            con.execute("CREATE TABLE something (x INTEGER)")
        with pytest.raises(typer.BadParameter, match="not a composite"):
            store.connect(other)


class TestDimensions:
    def test_dimensions_come_from_the_database_not_a_constant(self, con):
        found = store.dimensions(con)
        assert set(found) == {"event", "solver", "layers"}
        assert found["solver"] == ["emod3d", "sw4"]
        assert found["layers"] == ["full", "tomography"]

    def test_the_event_is_a_dimension_even_though_it_is_a_column(self, con):
        """`runs.event` is a column, not a run_labels row, but a caller grouping
        runs has no reason to care which."""
        assert "2020p111111" in store.dimensions(con)["event"]

    @pytest.mark.parametrize(
        "text,expected",
        [("solver=sw4", ("solver", "sw4")), ("a=b=c", ("a", "b=c"))],
    )
    def test_a_label_splits_on_the_first_equals_only(self, text, expected):
        """A value may contain an '='; a dimension name may not."""
        assert store.parse_label(text) == expected

    @pytest.mark.parametrize("text", ["solver", "=sw4", ""])
    def test_a_malformed_label_is_refused(self, text):
        with pytest.raises(typer.BadParameter, match="name=value"):
            store.parse_label(text)


class TestSelection:
    def test_two_dimensions_intersect(self, con):
        found = store.select_runs(con, [("solver", "sw4"), ("layers", "full")])
        assert {r["labels"]["solver"] for r in found} == {"sw4"}
        assert {r["labels"]["layers"] for r in found} == {"full"}

    def test_one_dimension_repeated_is_a_disjunction(self, con):
        """Otherwise the first thing anyone comparing two cells types returns
        nothing at all."""
        found = store.select_runs(con, [("solver", "sw4"), ("solver", "emod3d")])
        assert {r["labels"]["solver"] for r in found} == {"emod3d", "sw4"}

    def test_the_event_selects_like_any_other_dimension(self, con):
        found = store.select_runs(con, [("event", "2020p111111")])
        assert {r["event"] for r in found} == {"2020p111111"}

    def test_pairable_drops_events_with_no_recordings(self, con):
        """2021p222222 was simulated and never recorded, so it joins to nothing."""
        assert "2021p222222" not in {r["event"] for r in store.select_runs(con)}
        assert "2021p222222" in {
            r["event"] for r in store.select_runs(con, pairable=False)
        }

    def test_an_unknown_dimension_lists_the_ones_that_exist(self, con):
        with pytest.raises(typer.BadParameter, match="no dimension for"):
            store.select_runs(con, [("nonsense", "x")])

    def test_a_selection_matching_nothing_prints_the_coverage(self, con):
        """The message a caller needs is which cells exist, not that theirs did
        not."""
        with pytest.raises(typer.BadParameter, match="no run matches"):
            store.select_runs(con, [("solver", "sw4"), ("event", "2022p333333")])

    def test_run_key_is_carried_not_run_id(self, con):
        """run_id is renumbered by a rebuild; run_key is the stable natural key."""
        found = store.select_runs(con)
        assert all(r["run_key"] for r in found)
        assert len({r["run_key"] for r in found}) == len(found)


class TestGrouping:
    def test_varying_names_only_the_dimensions_that_differ(self, con):
        found = store.select_runs(con, [("solver", "sw4")])
        assert "solver" not in store.varying(found)
        assert "layers" in store.varying(found)

    def test_cells_are_sorted_so_a_rerun_reproduces_the_figure(self, con):
        """A figure that renumbered its series between runs could not be compared
        against the one already in the report."""
        found = store.select_runs(con)
        first = [key for key, _ in store.group_runs(con, found, ["solver"])]
        second = [
            key for key, _ in store.group_runs(con, list(reversed(found)), ["solver"])
        ]
        assert first == second == [("emod3d",), ("sw4",)]

    def test_crossing_two_dimensions_gives_their_product(self, con):
        cells = store.group_runs(con, store.select_runs(con), ["solver", "layers"])
        assert [key for key, _ in cells] == [
            ("emod3d", "full"), ("emod3d", "tomography"),
            ("sw4", "full"), ("sw4", "tomography"),
        ]

    @pytest.mark.parametrize(
        "by,key,expected",
        [(["solver"], ("sw4",), "sw4"), (["solver", "layers"], ("sw4", "full"),
                                        "solver=sw4 layers=full")],
    )
    def test_a_cell_is_named_by_value_alone_only_when_that_is_unambiguous(
        self, by, key, expected
    ):
        assert store.cell_label(by, key) == expected


class TestCovariates:
    def test_covariates_are_discovered_from_the_schema(self, con):
        found = store.available_covariates(con)
        assert {"rrup", "vs30", "magnitude"} <= set(found)
        assert found["rrup"] == "run_stations"
        assert found["vs30"] == "stations"
        assert found["magnitude"] == "runs"

    def test_the_enrichment_columns_became_covariates_with_no_code_change(self, con):
        """The argument run_labels makes for the run dimensions, applied to the
        site terms: add a column, get an option."""
        assert {"elevation", "basin"} <= set(store.available_covariates(con))

    def test_keys_and_bookkeeping_are_not_offered(self, con):
        found = store.available_covariates(con)
        assert not {"run_id", "station", "run_key", "ingested_at"} & set(found)


class TestGenerality:
    """The property that makes these commands reusable rather than a report."""

    def test_a_differently_extracted_tree_describes_itself(self, variant_db):
        found = store.dimensions(variant_db)
        assert set(found) == {"event", "variant"}
        assert found["variant"] == ["coarse", "fine"]
        assert "solver" not in found

    def test_selecting_and_grouping_work_on_its_own_dimensions(self, variant_db):
        runs = store.select_runs(variant_db, [("variant", "fine")])
        assert {r["labels"]["variant"] for r in runs} == {"fine"}
        cells = store.group_runs(variant_db, store.select_runs(variant_db), ["variant"])
        assert [key for key, _ in cells] == [("coarse",), ("fine",)]

    def test_a_dimension_from_the_other_tree_is_refused_here(self, variant_db):
        with pytest.raises(typer.BadParameter, match="no dimension for"):
            store.select_runs(variant_db, [("solver", "sw4")])

    @pytest.mark.parametrize("word", ["emod3d", "sw4", "tomography", "layers"])
    def test_no_analysis_code_names_a_dimension_or_a_value(self, word):
        """Crude, and it catches exactly the regression that matters: a shortcut
        that writes this database's own vocabulary into the analysis code.

        ``event`` is the documented exception -- ``runs.event`` is a column, and
        :data:`eqvis_workflow.store.EVENT` is where that is written down.
        """
        code = code_without_docstrings(store)
        assert word not in code.lower(), f"store names {word!r} outside a docstring"
