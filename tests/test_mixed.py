"""Tests for the crossed random-effects REML estimator.

This is statistical code, so it is tested against results that are known
independently of it rather than against its own output. No ground-motion data
appears here at all -- the estimator does not know what an earthquake is, which
is what makes this possible and is the reason it is a module of its own.

Two of these carry most of the weight. ``TestBalancedANOVA`` checks the fit
against a closed form that REML must reproduce exactly, which catches a wrong
log-determinant, a wrong penalised sum of squares, a dropped ``n - p``, or
maximum likelihood in place of REML. ``TestAgainstStatsmodels`` checks it against
an independent implementation of the same model. If both pass, the estimator is
right.
"""

import numpy as np
import pytest

from eqvis_workflow import mixed


def crossed(a: int, b: int, seed: int, tau: float, s2s: float, noise: float,
            per_cell: int = 1):
    """A balanced two-way crossed design with known variance components."""
    rng = np.random.default_rng(seed)
    rows = a * b * per_cell
    first = np.repeat(np.arange(a), b * per_cell)
    second = np.tile(np.repeat(np.arange(b), per_cell), a)
    y = (
        0.5
        + rng.normal(0, tau, a)[first]
        + rng.normal(0, s2s, b)[second]
        + rng.normal(0, noise, rows)
    )
    X = np.ones((rows, 1))
    return y, X, [first, second], [a, b]


def fit(y, X, codes, sizes, groups=("event", "station")):
    return mixed.fit_mixed(y, X, codes, sizes, groups=groups)


class TestBalancedANOVA:
    """On a balanced crossed design REML and ANOVA are the same estimator."""

    @pytest.fixture(scope="class")
    @staticmethod
    def balanced():
        return crossed(12, 15, seed=3, tau=0.35, s2s=0.30, noise=0.5)

    def test_the_variance_components_match_the_closed_form(self, balanced):
        """The sharpest available check: a wrong log-determinant, a wrong r2, an
        `n` where an `n - p` belongs, or ML instead of REML all fail here."""
        y, X, codes, sizes = balanced
        a, b = sizes
        table = y.reshape(a, b)
        grand = table.mean()
        row = table.mean(axis=1)
        col = table.mean(axis=0)
        mean_a = b * ((row - grand) ** 2).sum() / (a - 1)
        mean_b = a * ((col - grand) ** 2).sum() / (b - 1)
        error = ((table - row[:, None] - col[None, :] + grand) ** 2).sum() / (
            (a - 1) * (b - 1)
        )

        got = fit(y, X, codes, sizes)
        assert got.sd["residual"] == pytest.approx(np.sqrt(error), rel=1e-6)
        assert got.sd["event"] == pytest.approx(np.sqrt((mean_a - error) / b), rel=1e-5)
        assert got.sd["station"] == pytest.approx(
            np.sqrt((mean_b - error) / a), rel=1e-5
        )

    def test_the_intercept_is_the_grand_mean(self, balanced):
        """Balance makes the fixed effect free of the variance components."""
        y, X, codes, sizes = balanced
        assert fit(y, X, codes, sizes).a == pytest.approx(y.mean(), rel=1e-10)

    def test_the_components_combine_as_the_papers_state(self, balanced):
        """phi^2 = phi_S2S^2 + phi_ss^2 and sigma^2 = tau^2 + phi^2."""
        got = fit(*balanced)
        assert got.phi == pytest.approx(
            np.hypot(got.sd["station"], got.sd["residual"]), rel=1e-12
        )
        assert got.sigma == pytest.approx(
            np.sqrt(got.sd["event"] ** 2 + got.phi**2), rel=1e-12
        )


class TestAgainstStatsmodels:
    """An independent implementation of the same model, fitted slowly.

    ``MixedLM`` cannot express crossed random effects directly, so the whole
    dataset is made one group and both factors enter as variance components.
    That is unusably slow in a sweep, which is why this estimator exists, and it
    is exactly right, which is why it is the reference.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def reference():
        import pandas as pd
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(11)
        a, b, rows = 7, 40, 140
        first = rng.integers(0, a, rows)
        second = rng.integers(0, b, rows)
        x = rng.normal(size=rows)
        y = (
            0.4 + 0.8 * x
            + rng.normal(0, 0.55, a)[first]
            + rng.normal(0, 0.45, b)[second]
            + rng.normal(0, 0.4, rows)
        )
        frame = pd.DataFrame(
            {"y": y, "x": x, "event": first, "station": second, "const": 1}
        )
        model = smf.mixedlm(
            "y ~ x",
            frame,
            groups="const",
            re_formula="0",
            vc_formula={"event": "0 + C(event)", "station": "0 + C(station)"},
        ).fit(reml=True)
        X = np.column_stack([np.ones(rows), x])
        ours = mixed.fit_mixed(
            y, X, [first, second], [a, b],
            groups=("event", "station"), terms=("intercept", "x"),
        )
        return model, ours

    def test_the_deviance_agrees(self, reference):
        model, ours = reference
        assert ours.deviance == pytest.approx(-2 * model.llf, abs=1e-5)

    def test_the_fixed_effects_agree(self, reference):
        model, ours = reference
        assert ours.beta == pytest.approx(np.asarray(model.fe_params), rel=1e-4)

    def test_the_variance_components_agree(self, reference):
        """``vcomp`` holds variances in the same units as ``scale``, not ratios to
        it -- checked against a dense evaluation of Harville's criterion, which
        matches at ``sqrt(vcomp)`` and is 45 deviance units worse at
        ``sqrt(vcomp * scale)``. Getting that wrong makes this test look like an
        estimator bug when it is a units bug in the reference.
        """
        model, ours = reference
        assert ours.sd["residual"] == pytest.approx(np.sqrt(model.scale), rel=1e-4)
        assert ours.sd["event"] == pytest.approx(np.sqrt(model.vcomp[0]), rel=1e-4)
        assert ours.sd["station"] == pytest.approx(np.sqrt(model.vcomp[1]), rel=1e-4)

    def test_our_fixed_effect_covariance_is_the_conditional_one(self, reference):
        """Ours is (X'V^-1 X)^-1, conditional on the variance parameters, which
        is what lme4's vcov() reports. statsmodels inverts the *joint* Hessian
        over the fixed and variance parameters, so its slope standard error is a
        different -- and also defensible -- quantity, differing by around a
        percent. Do not tighten the loose assertion below into agreement; the
        tight one beside it is the one that pins our implementation down.
        """
        model, ours = reference
        rng = np.random.default_rng(11)
        a, b, rows = 7, 40, 140
        first = rng.integers(0, a, rows)
        second = rng.integers(0, b, rows)
        Z = np.hstack(
            [mixed.indicator(first, a), mixed.indicator(second, b)]
        )
        G = np.diag(
            np.concatenate(
                [np.full(a, ours.sd["event"] ** 2), np.full(b, ours.sd["station"] ** 2)]
            )
        )
        V = Z @ G @ Z.T + ours.sd["residual"] ** 2 * np.eye(rows)
        x = np.column_stack([np.ones(rows), rng.normal(size=rows)])
        # Rebuild the same design the fixture used, deterministically.
        rng2 = np.random.default_rng(11)
        rng2.integers(0, a, rows)
        rng2.integers(0, b, rows)
        x = np.column_stack([np.ones(rows), rng2.normal(size=rows)])
        inverse = np.linalg.inv(V)
        dense = np.linalg.inv(x.T @ inverse @ x)
        assert ours.beta_cov == pytest.approx(dense, rel=1e-8)   # the tight one
        assert np.sqrt(np.diag(ours.beta_cov)) == pytest.approx(
            np.asarray(model.bse_fe), rel=0.05                   # the loose one
        )


class TestShrinkage:
    """With one grouping factor the BLUP has the closed form the papers give."""

    @pytest.fixture(scope="class")
    @staticmethod
    def one_way():
        rng = np.random.default_rng(5)
        a = 9
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        codes = np.repeat(np.arange(a), counts)
        rows = counts.sum()
        y = 0.3 + rng.normal(0, 0.4, a)[codes] + rng.normal(0, 0.6, rows)
        got = mixed.fit_mixed(
            y, np.ones((rows, 1)), [codes], [a], groups=("event",)
        )
        return y, codes, counts, got

    def test_the_blups_match_stafford_equation_eight(self, one_way):
        """The group's mean residual, shrunk by n tau^2 / (n tau^2 + phi^2)."""
        y, codes, counts, got = one_way
        tau2 = got.sd["event"] ** 2
        noise2 = got.sd["residual"] ** 2
        for level in range(len(counts)):
            rows = codes == level
            expected = (
                tau2 * (y[rows] - got.a).sum() / (counts[level] * tau2 + noise2)
            )
            assert got.blup["event"][level] == pytest.approx(expected, abs=1e-10)

    def test_a_singleton_group_is_shrunk_by_exactly_that_factor(self, one_way):
        """The pathology Stafford describes: the two-stage method hands a lone
        record's whole residual to the event term; this hands it a fraction."""
        y, codes, counts, got = one_way
        lone = int(np.flatnonzero(counts == 1)[0])
        tau2, noise2 = got.sd["event"] ** 2, got.sd["residual"] ** 2
        residual = float(y[codes == lone][0] - got.a)
        assert got.blup["event"][lone] == pytest.approx(
            residual * tau2 / (tau2 + noise2), rel=1e-9
        )

    def test_the_conditional_variance_shrinks_as_a_group_gains_records(self, one_way):
        """A level with one observation must announce its own uninformativeness."""
        _, _, counts, got = one_way
        order = np.argsort(counts)
        bars = got.blup_sd["event"][order]
        assert np.all(np.diff(bars) <= 1e-12)


class TestDegenerate:
    """One observation per level: not weakly identified but exactly unidentified."""

    @pytest.fixture(scope="class")
    @staticmethod
    def ridge():
        rng = np.random.default_rng(19)
        a, rows = 6, 60
        first = np.repeat(np.arange(a), rows // a)
        second = np.arange(rows)          # every level has exactly one row
        y = 0.2 + rng.normal(0, 0.4, a)[first] + rng.normal(0, 0.5, rows)
        return y, np.ones((rows, 1)), [first, second], [a, rows]

    def test_the_deviance_surface_is_an_exact_ridge(self, ridge):
        """For every (a, b) there is an equivalent (a/(1+b), 0) with the same
        deviance, so no point on the ridge is preferable to any other."""
        y, X, codes, sizes = ridge
        design = mixed.build_design(y, X, codes, sizes)
        for second in (0.09, 0.49, 2.25, 9.0):
            here = float(mixed.reml_deviance(design, np.array([0.8, second])))
            there = float(mixed.reml_deviance(design, np.array([0.8 / (1 + second), 0.0])))
            assert here == pytest.approx(there, abs=1e-8)

    def test_it_is_detected_and_held_at_zero_rather_than_guessed(self, ridge):
        y, X, codes, sizes = ridge
        got = fit(y, X, codes, sizes)
        assert got.diagnostics["station"].pairs == 0
        assert not got.diagnostics["station"].identified
        assert got.sd["station"] == 0.0
        assert np.all(got.blup["station"] == 0.0)
        assert np.all(got.blup_sd["station"] == 0.0)

    def test_what_survives_the_ridge_is_still_reported(self, ridge):
        """tau, phi and the fixed effects are identified even here; only the
        split of phi into a site term and a remainder is not."""
        y, X, codes, sizes = ridge
        got = fit(y, X, codes, sizes)
        alone = mixed.fit_mixed(y, X, [codes[0]], [sizes[0]], groups=("event",))
        assert got.sd["event"] == pytest.approx(alone.sd["event"], rel=1e-6)
        assert got.a == pytest.approx(alone.a, rel=1e-9)
        assert got.phi == pytest.approx(alone.phi, rel=1e-6)


class TestBoundary:
    def test_the_criterion_is_continuous_into_zero(self):
        """The scaled form has no 1/ratio anywhere, so the boundary is reachable
        rather than approached."""
        y, X, codes, sizes = crossed(8, 10, seed=2, tau=0.4, s2s=0.0, noise=0.5)
        design = mixed.build_design(y, X, codes, sizes)
        limit = float(mixed.reml_deviance(design, np.array([0.5, 0.0])))
        for small in (1e-3, 1e-6, 1e-9, 1e-12):
            approached = float(mixed.reml_deviance(design, np.array([0.5, small])))
            assert approached == pytest.approx(limit, abs=max(1e-8, 40 * small))

    def test_data_with_no_site_effect_lands_on_the_boundary(self):
        """Generated with the component exactly zero; the mixture test must not
        reject, and the estimate should be reported as an exact zero."""
        y, X, codes, sizes = crossed(10, 12, seed=8, tau=0.4, s2s=0.0, noise=0.5)
        got = fit(y, X, codes, sizes)
        assert got.boundary_p["station"] > 0.05

    def test_a_real_effect_is_not_reported_as_a_boundary(self):
        y, X, codes, sizes = crossed(12, 15, seed=4, tau=0.4, s2s=0.5, noise=0.4)
        got = fit(y, X, codes, sizes)
        assert "station" not in got.boundary
        assert got.boundary_p["station"] < 0.05


class TestInvariance:
    @pytest.fixture(scope="class")
    @staticmethod
    def base():
        return crossed(9, 11, seed=6, tau=0.4, s2s=0.3, noise=0.5)

    def test_permuting_the_rows_changes_nothing(self, base):
        y, X, codes, sizes = base
        order = np.random.default_rng(1).permutation(len(y))
        first = fit(y, X, codes, sizes)
        second = fit(y[order], X[order], [c[order] for c in codes], sizes)
        assert second.deviance == pytest.approx(first.deviance, abs=1e-8)
        assert second.a == pytest.approx(first.a, abs=1e-9)
        for name in ("event", "station", "residual"):
            assert second.sd[name] == pytest.approx(first.sd[name], abs=1e-7)

    def test_shifting_the_response_moves_only_the_intercept(self, base):
        y, X, codes, sizes = base
        first = fit(y, X, codes, sizes)
        second = fit(y + 3.7, X, codes, sizes)
        assert second.a - first.a == pytest.approx(3.7, abs=1e-8)
        for name in ("event", "station", "residual"):
            # Loose enough for the optimiser to land on a slightly different
            # point of a flat ridge, tight enough to catch a real shift.
            assert second.sd[name] == pytest.approx(first.sd[name], rel=1e-6)

    def test_scaling_the_response_scales_the_components_but_not_the_ratios(self, base):
        """And the deviance shifts by exactly 2(n - p) log c, which is what
        catches an `n` standing where an `n - p` belongs."""
        y, X, codes, sizes = base
        factor = 2.5
        first = fit(y, X, codes, sizes)
        second = fit(y * factor, X, codes, sizes)
        for name in ("event", "station", "residual"):
            assert second.sd[name] / first.sd[name] == pytest.approx(factor, rel=1e-6)
            assert second.ratio.get(name, 0) == pytest.approx(
                first.ratio.get(name, 0), rel=1e-5
            )
        expected = 2 * (first.n - first.p) * np.log(factor)
        assert second.deviance - first.deviance == pytest.approx(expected, abs=1e-6)


class TestRefusals:
    def test_a_rank_deficient_design_is_refused_by_name(self):
        """z1pt0 beside vs30 is the standing case: deterministic in it."""
        y, X, codes, sizes = crossed(8, 10, seed=7, tau=0.3, s2s=0.3, noise=0.4)
        duplicated = np.column_stack([X, X[:, 0] * 2.0])
        with pytest.raises(ValueError, match="rank deficient"):
            mixed.fit_mixed(
                y, duplicated, codes, sizes,
                groups=("event", "station"), terms=("intercept", "twice"),
            )

    def test_the_denominator_degrees_of_freedom_is_the_group_count(self):
        """Not n - p. The intercept's uncertainty is set by how many groups there
        are, not by how many rows."""
        y, X, codes, sizes = crossed(8, 40, seed=9, tau=0.4, s2s=0.3, noise=0.5)
        got = fit(y, X, codes, sizes)
        assert got.df_denominator == 7
        assert got.n == 320


class TestIdentification:
    @pytest.mark.parametrize(
        "counts,pairs,singletons",
        [([1, 1, 1], 0, 3), ([2, 1], 1, 1), ([3, 2, 1], 4, 1), ([1] * 134 + [2] * 27 + [3] * 2, 33, 134)],
    )
    def test_within_level_pairs_are_counted(self, counts, pairs, singletons):
        """The last case is the reference data's own station structure: 33 pairs
        from 163 stations, which is what makes its site term weak but not void."""
        codes = np.repeat(np.arange(len(counts)), counts)
        found = mixed.identification("station", codes, len(counts))
        assert found.pairs == pairs
        assert found.singletons == singletons
        assert found.identified == (pairs > 0)
