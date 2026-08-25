"""Crossed random-effects REML, for decomposing a residual.

Fits ``y ~ X + sum_f (1|f)`` -- a mean structure plus any number of *crossed*
scalar random intercepts -- by restricted maximum likelihood. Nothing here knows
anything about earthquakes; it takes a long table and returns variance
components, which is what makes it both reusable and testable against textbook
statistics.

The formulation is Lee et al. (2022), *Earthquake Spectra* 38(4) 2548-2579, whose
notation comes from Al Atik et al. (2010), *SRL* 81(5) 794-801, estimated the way
Stafford (2014), *BSSA* 104(2) 702-719 argues for -- one joint fit rather than a
sequence of residual manipulations, because an unbalanced design makes the
sequential estimates of the later components biased. Note that the acronym
"MERA" appears in none of those three papers.

**Sign.** The response this is given is ``ln(simulated) - ln(observed)``, the
convention of :func:`eqvis_workflow.data.residual_label`, so a **positive value
means the simulation over-predicts**. Lee et al. define the residual the other
way up, which inverts every "over-" and "under-prediction" sentence in that
paper against anything computed here. Read the three papers with that in mind.

Two implementation choices are worth stating, because both are places a
hand-rolled variance-component fit is usually silently wrong.

*Dense, not sparse.* At the size this is used for -- a couple of hundred rows,
one or two hundred random-effect levels -- the coefficient matrix is a few
hundred square. One criterion evaluation is a fraction of a millisecond with
LAPACK. A sparse factorisation at that size is slower, gives an LU rather than a
Cholesky (so the log-determinant is no longer free), and would want a
fill-reducing ordering from a library that is not installed. Sparsity is what
``lme4`` needs at a hundred thousand levels; here it is ceremony.

*Parameterised in the variance ratios themselves*, not their square roots and
not their logarithms. See :func:`reml_deviance` for why the square root is a
trap: the gradient vanishes at the boundary, so a gradient method stalls there
and reports a small positive variance where the answer is exactly zero.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2
from scipy.stats import t as student

from .console import console_warn

# The largest variance ratio the optimiser will consider: a random effect a
# hundred times the residual standard deviation. Beyond this the residual is
# being driven to zero, which means the design cannot separate the levels at all
# -- reported as `at_cap` rather than optimised into.
RATIO_MAX = 1.0e4

# Nodes per axis in the coarse scan, including an exact zero. The surface is a
# long shallow ridge -- the components trade off against one another, which is
# the "apparently correlated variance components" Stafford discusses -- so a
# single local search from one start finds a point on the ridge rather than the
# point.
GRID = 14

# A component is reported as exactly zero when profiling it to zero costs less
# deviance than this. Relative, because the deviance scales with the row count.
BOUNDARY_ATOL = 1.0e-6


@dataclass(frozen=True)
class Design:
    """The cross-products a REML evaluation needs, formed once.

    ``T'T``, ``T'y`` and ``y'y`` do not depend on the variance ratios, so the
    rows are touched once and every subsequent evaluation is a few hundred
    square. That is what makes the optimisation and the bootstrap cheap.
    """

    cross: np.ndarray          # T'T, (p + q) square
    projection: np.ndarray     # T'y, (p + q) or (p + q, replicates)
    total: np.ndarray | float  # y'y
    n: int
    p: int
    sizes: tuple[int, ...]     # levels per grouping factor

    @property
    def q(self) -> int:
        return sum(self.sizes)


def indicator(codes: np.ndarray, levels: int) -> np.ndarray:
    """Dense one-hot columns for a grouping factor, in level order."""
    Z = np.zeros((codes.size, levels))
    Z[np.arange(codes.size), codes] = 1.0
    return Z


def build_design(
    y: np.ndarray, X: np.ndarray, codes: Sequence[np.ndarray], sizes: Sequence[int]
) -> Design:
    """Form the cross-products for one response, or for many at once.

    ``y`` may be ``(n,)`` or ``(n, replicates)``. The many-at-once form is what
    lets the parametric bootstrap share one factorisation per grid node across
    every replicate, since the matrix being factorised does not depend on ``y``.
    """
    T = np.hstack([X, *(indicator(code, size) for code, size in zip(codes, sizes))])
    return Design(
        cross=T.T @ T,
        projection=T.T @ y,
        total=(y * y).sum(axis=0),
        n=X.shape[0],
        p=X.shape[1],
        sizes=tuple(sizes),
    )


def scaling(design: Design, ratio: np.ndarray) -> np.ndarray:
    """The diagonal that turns Henderson's matrix into its scaled form."""
    return np.concatenate(
        [np.ones(design.p)]
        + [
            np.full(size, np.sqrt(max(value, 0.0)))
            for size, value in zip(design.sizes, ratio)
        ]
    )


def reml_deviance(design: Design, ratio: np.ndarray) -> np.ndarray | float:
    """The profiled REML deviance at these variance ratios.

    Writing ``Lambda`` for the diagonal of root-ratios and ``T = [X Z]``, the
    *scaled* coefficient matrix is::

        C* = [[ X'X,        X'Z Lambda          ],
              [ Lambda Z'X, Lambda Z'Z Lambda + I ]]

    which is positive definite for **every** ratio at or above zero -- including
    exactly zero, where the lower block is simply the identity. That is the whole
    reason to scale: the unscaled form needs ``G^-1``, which divides by the
    ratio and blows up at precisely the boundary that has to be reachable.

    Because ``D' C D = C*`` with ``D = blkdiag(I, Lambda)``, the two
    log-determinant terms the REML criterion needs collapse into one::

        log det C* = log det C + log det G

    and profiling the residual variance out at ``r2 / (n - p)`` leaves::

        d_R = (n - p) [1 + log(2 pi r2 / (n - p))] + log det C*

    Two ratios, one Cholesky, one solve.

    The ratios themselves are the parameters, not their square roots. ``C*``
    depends on a root-ratio only through its square, so the derivative with
    respect to the root vanishes at zero: any gradient method walks in, stalls,
    and reports a small positive variance where the truth is an exact zero. In
    the ratio the criterion is analytic with a generically non-zero derivative
    at the boundary, and the boundary is a finite, exactly representable point.
    """
    sc = scaling(design, ratio)
    scaled = design.cross * sc[:, None] * sc[None, :]
    scaled[design.p :, design.p :] += np.eye(design.q)
    factor = cho_factor(scaled, lower=True, check_finite=False)
    log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()

    right = design.projection * (sc[:, None] if design.projection.ndim > 1 else sc)
    solved = cho_solve(factor, right, check_finite=False)
    residual = design.total - (right * solved).sum(axis=0)

    free = design.n - design.p
    return free * (1.0 + np.log(2.0 * np.pi * residual / free)) + log_determinant


@dataclass(frozen=True)
class Identification:
    """Why a component is or is not resolvable. Computed every fit, not on request.

    A scalar random intercept is identified only by the covariance between two
    observations sharing a level, so the count of within-level *pairs* is the
    quantity that decides whether its variance can be told apart from the
    residual at all. With one observation per level there are no pairs, the two
    are the same quantity, and the likelihood is flat along their trade-off.
    """

    factor: str
    levels: int
    counts: np.ndarray
    singletons: int
    pairs: int

    @property
    def identified(self) -> bool:
        return self.pairs > 0

    def summary(self) -> str:
        return (
            f"{self.factor}: {self.levels} levels, {self.singletons} with one "
            f"observation, {self.pairs} within-level pairs"
            + ("" if self.identified else "  <- NOT identified")
        )


def identification(factor: str, codes: np.ndarray, levels: int) -> Identification:
    counts = np.bincount(codes, minlength=levels)
    return Identification(
        factor=factor,
        levels=levels,
        counts=counts,
        singletons=int((counts == 1).sum()),
        pairs=int((counts * (counts - 1) // 2).sum()),
    )


@dataclass(frozen=True)
class MixedFit:
    """One REML fit: the mean structure, the variance structure, and the BLUPs."""

    groups: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]
    terms: tuple[str, ...]
    n: int
    p: int
    beta: np.ndarray
    beta_cov: np.ndarray
    sd: dict[str, float]
    ratio: dict[str, float]
    blup: dict[str, np.ndarray]
    blup_sd: dict[str, np.ndarray]
    deviance: float
    boundary: tuple[str, ...]
    boundary_p: dict[str, float]
    diagnostics: dict[str, Identification]
    at_cap: tuple[str, ...]
    interval: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def a(self) -> float:
        """The model prediction bias: the intercept."""
        return float(self.beta[0])

    @property
    def phi(self) -> float:
        """The whole within-event standard deviation.

        The robust number. The split of this into a site term and a remainder
        needs repeated observations at a site to be identified at all, where this
        needs only the rows -- so where station reuse is thin, quote this and
        give the split an interval.
        """
        within = [name for name in self.groups[1:]] + ["residual"]
        return float(np.sqrt(sum(self.sd[name] ** 2 for name in within)))

    @property
    def sigma(self) -> float:
        """The total standard deviation, over every component."""
        return float(np.sqrt(sum(value**2 for value in self.sd.values())))

    @property
    def df_denominator(self) -> int:
        """Degrees of freedom for an interval on a fixed effect.

        The containment rule, ``min(levels) - 1``, and **not** ``n - p``. The
        uncertainty in the intercept is set by how many groups there are, not by
        how many rows: with eight events the standard error of the mean bias is
        the between-event standard deviation over the square root of eight,
        whatever the row count. Taking ``n - p`` would make every interval far
        too narrow. Same argument
        :func:`eqvis_workflow.bias.draw_bias_curve` makes for Student's t over
        the normal, one level up.
        """
        smallest = min(
            (d.levels for d in self.diagnostics.values() if d.levels > 1), default=2
        )
        return max(1, smallest - 1)

    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.beta_cov))

    def confint(self, level: float = 0.95) -> np.ndarray:
        half = student.ppf(0.5 + level / 2, self.df_denominator) * self.se()
        return np.column_stack([self.beta - half, self.beta + half])


def evaluate(design: Design, ratio: np.ndarray) -> dict:
    """Everything the fit reports, at one point in the ratio space."""
    sc = scaling(design, ratio)
    scaled = design.cross * sc[:, None] * sc[None, :]
    scaled[design.p :, design.p :] += np.eye(design.q)
    factor = cho_factor(scaled, lower=True, check_finite=False)
    log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()

    right = design.projection * sc
    solved = cho_solve(factor, right, check_finite=False)
    r2 = float(design.total - right @ solved)
    free = design.n - design.p
    residual_variance = r2 / free

    # Conditional variances come out of the same factorisation: because
    # C^-1 = D C*^-1 D', the random-effect block of phi_ss^2 C^-1 is exactly
    # Var(u_hat - u), which is what a caterpillar bar or a Q-Q envelope needs and
    # what a plain point estimate cannot give.
    inverse = cho_solve(factor, np.eye(design.p + design.q), check_finite=False)
    covariance = residual_variance * inverse * sc[:, None] * sc[None, :]

    estimate = solved * sc
    return {
        "deviance": free * (1.0 + np.log(2.0 * np.pi * r2 / free)) + log_determinant,
        "residual_variance": residual_variance,
        "beta": estimate[: design.p],
        "beta_cov": covariance[: design.p, : design.p],
        "u": estimate[design.p :],
        "u_var": np.maximum(np.diag(covariance)[design.p :], 0.0),
    }


def optimise(design: Design, force_zero: Sequence[bool]) -> np.ndarray:
    """The ratios minimising the REML deviance, boundaries included.

    A coarse scan that contains the exact zeros, then a derivative-free polish
    from the best interior node, then -- unconditionally, not only when the scan
    suggests it -- the profile of every subset of components held at zero. The
    boundary cases are solved exactly rather than approached, which is what makes
    an exact zero reportable instead of merely small.
    """
    k = len(design.sizes)
    free = [n for n in range(k) if not force_zero[n]]
    if not free:
        return np.zeros(k)

    axis = np.concatenate([[0.0], np.geomspace(1e-3, RATIO_MAX, GRID - 1)])
    best, best_value = np.zeros(k), np.inf
    for point in np.ndindex(*([len(axis)] * len(free))):
        trial = np.zeros(k)
        for slot, position in zip(free, point):
            trial[slot] = axis[position]
        value = float(reml_deviance(design, trial))
        if value < best_value:
            best, best_value = trial.copy(), value

    def objective(free_values: np.ndarray) -> float:
        trial = np.zeros(k)
        for slot, value in zip(free, free_values):
            trial[slot] = min(abs(float(value)), RATIO_MAX)
        return float(reml_deviance(design, trial))

    start = np.array([max(best[slot], 1e-3) for slot in free])
    polished = minimize(
        objective, start, method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000},
    )
    if polished.fun < best_value:
        best_value = float(polished.fun)
        best = np.zeros(k)
        for slot, value in zip(free, polished.x):
            best[slot] = min(abs(float(value)), RATIO_MAX)

    # Every component held at zero in turn, and all of them together, profiled
    # exactly over the rest.
    for held in [*([n] for n in free), free]:
        rest = [n for n in free if n not in held]
        if not rest:
            trial = np.zeros(k)
            value = float(reml_deviance(design, trial))
        elif len(rest) == 1:
            def scalar(value, slot=rest[0]):
                trial = np.zeros(k)
                trial[slot] = min(abs(float(value)), RATIO_MAX)
                return float(reml_deviance(design, trial))

            found = minimize_scalar(scalar, bounds=(0.0, RATIO_MAX), method="bounded")
            trial = np.zeros(k)
            trial[rest[0]] = float(found.x)
            value = float(found.fun)
        else:
            def partial(free_values):
                trial = np.zeros(k)
                for slot, value in zip(rest, free_values):
                    trial[slot] = min(abs(float(value)), RATIO_MAX)
                return float(reml_deviance(design, trial))

            found = minimize(
                partial,
                np.array([max(best[slot], 1e-3) for slot in rest]),
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-12},
            )
            trial = np.zeros(k)
            for slot, value in zip(rest, found.x):
                trial[slot] = min(abs(float(value)), RATIO_MAX)
            value = float(found.fun)
        if value < best_value:
            best, best_value = trial, value
    return best


def fit_mixed(
    y: np.ndarray,
    X: np.ndarray,
    codes: Sequence[np.ndarray],
    sizes: Sequence[int],
    groups: Sequence[str],
    terms: Sequence[str] = ("intercept",),
    levels: Sequence[Sequence[str]] | None = None,
    interval: float = 0.95,
) -> MixedFit:
    """Fit ``y ~ X + sum_f (1|f)`` by REML, with crossed random intercepts."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(
            f"the fixed-effect design is rank deficient ({X.shape[1]} columns, "
            f"rank {np.linalg.matrix_rank(X)}): two of {', '.join(terms)} carry "
            "the same information. Drop one -- z1pt0 and z2pt5 beside vs30 are "
            "the standing case, being deterministic functions of it"
        )

    diagnostics = {
        name: identification(name, code, size)
        for name, code, size in zip(groups, codes, sizes)
    }
    # A factor with no within-level pairs is not weakly identified but exactly
    # unidentified: its variance and the residual's enter the likelihood only
    # through their sum, so the surface is a ridge and any point on it fits
    # equally well. Held at zero by fiat, which loses nothing -- the total
    # within-group variance and the fixed effects are unaffected -- and reports
    # honestly instead of picking an arbitrary point.
    force_zero = []
    for name in groups:
        blocked = not diagnostics[name].identified
        force_zero.append(blocked)
        if blocked:
            console_warn(
                f"{name} has no level with two observations, so its variance "
                "cannot be separated from the residual; holding it at zero"
            )

    design = build_design(y, X, codes, sizes)
    ratio = optimise(design, force_zero)
    at = evaluate(design, ratio)

    # Which components are at the boundary, decided by deviance rather than by a
    # small-number test. A grid-and-polish search returns an exact zero only
    # about half the time when the truth is zero, so `sd < eps` systematically
    # under-reports the boundary and over-states confidence in the split.
    boundary, boundary_p = [], {}
    for slot, name in enumerate(groups):
        if force_zero[slot]:
            boundary.append(name)
            boundary_p[name] = 1.0
            continue
        held = ratio.copy()
        held[slot] = 0.0
        rest = [n for n in range(len(groups)) if n != slot and not force_zero[n]]
        if rest:
            def profile(values, held=held, rest=rest):
                trial = held.copy()
                for other, value in zip(rest, np.atleast_1d(values)):
                    trial[other] = min(abs(float(value)), RATIO_MAX)
                return float(reml_deviance(design, trial))

            found = minimize(
                profile,
                np.array([max(ratio[other], 1e-3) for other in rest]),
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-12},
            )
            cost = float(found.fun) - at["deviance"]
        else:
            cost = float(reml_deviance(design, held)) - at["deviance"]
        cost = max(cost, 0.0)
        # The null puts the parameter on the boundary of its space, so the
        # reference is Chernoff's half-and-half mixture, not a plain chi-square.
        # Legitimate as a REML test only because the two models share their
        # fixed effects -- a REML likelihood ratio across mean structures is not.
        boundary_p[name] = float(0.5 * chi2.sf(cost, 1)) if cost > 0 else 1.0
        if cost <= BOUNDARY_ATOL * max(1.0, abs(at["deviance"])):
            boundary.append(name)
            ratio[slot] = 0.0

    if boundary:
        # Refit at the boundary so the BLUPs and their variances are consistent
        # with a component that is exactly zero rather than nearly so.
        at = evaluate(design, ratio)

    residual_sd = float(np.sqrt(at["residual_variance"]))
    sd = {"residual": residual_sd}
    ratios = {}
    blup, blup_sd = {}, {}
    offset = 0
    for slot, (name, size) in enumerate(zip(groups, sizes)):
        sd[name] = float(np.sqrt(max(ratio[slot], 0.0)) * residual_sd)
        ratios[name] = float(ratio[slot])
        blup[name] = at["u"][offset : offset + size]
        blup_sd[name] = np.sqrt(at["u_var"][offset : offset + size])
        offset += size

    return MixedFit(
        groups=tuple(groups),
        levels={
            name: tuple(str(v) for v in (() if values is None else values))
            for name, values in zip(groups, levels or [()] * len(groups))
        },
        terms=tuple(terms),
        n=design.n,
        p=design.p,
        beta=at["beta"],
        beta_cov=at["beta_cov"],
        sd=sd,
        ratio=ratios,
        blup=blup,
        blup_sd=blup_sd,
        deviance=float(at["deviance"]),
        boundary=tuple(boundary),
        boundary_p=boundary_p,
        diagnostics=diagnostics,
        at_cap=tuple(
            name
            for slot, name in enumerate(groups)
            if ratio[slot] >= RATIO_MAX * 0.999
        ),
    )


def profile_interval(
    design: Design,
    groups: Sequence[str],
    ratio: np.ndarray,
    deviance: float,
    level: float = 0.95,
    grid: int = 41,
) -> dict[str, tuple[float, float]]:
    """Profile-likelihood intervals on the standard deviations.

    Read off the same scan the optimisation uses, refined: every node whose
    deviance is within a chi-square threshold of the minimum is in the interval,
    and the interval is the range of the component over those nodes.

    Preferred over a bootstrap because it does not collapse at the boundary. A
    parametric bootstrap started from an estimate of exactly zero generates every
    replicate with a true zero, so its percentile interval is ``[0, small]`` --
    an interval conditional on the boundary estimate, which understates what is
    not known.
    """
    if len(groups) > 2:
        return {}
    threshold = chi2.ppf(level, 1)
    axis = np.concatenate([[0.0], np.geomspace(1e-3, RATIO_MAX, grid - 1)])
    keep: list[np.ndarray] = []
    if len(groups) == 1:
        for value in axis:
            trial = np.array([value])
            if float(reml_deviance(design, trial)) - deviance <= threshold:
                keep.append(trial)
    else:
        for first in axis:
            for second in axis:
                trial = np.array([first, second])
                if float(reml_deviance(design, trial)) - deviance <= threshold:
                    keep.append(trial)
    if not keep:
        return {}

    free = design.n - design.p
    found: dict[str, list[float]] = {name: [] for name in [*groups, "residual"]}
    for trial in keep:
        # The residual variance is profiled out, so it has to be recovered at
        # each node rather than held at its value at the optimum.
        sc = scaling(design, trial)
        scaled = design.cross * sc[:, None] * sc[None, :]
        scaled[design.p :, design.p :] += np.eye(design.q)
        factor = cho_factor(scaled, lower=True, check_finite=False)
        right = design.projection * sc
        r2 = float(design.total - right @ cho_solve(factor, right, check_finite=False))
        residual = np.sqrt(r2 / free)
        found["residual"].append(residual)
        for name, value in zip(groups, trial):
            found[name].append(float(np.sqrt(value) * residual))
    return {name: (min(values), max(values)) for name, values in found.items()}
