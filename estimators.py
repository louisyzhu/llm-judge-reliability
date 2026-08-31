"""Estimators used in "Three Ways Classical Test Theory Misleads for LLM Judges".

Every function takes element-level verdict matrices of shape (n_items, K) with 0/1
entries, or total scores of shape (n_items,). No dependencies beyond numpy/scipy.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import binom


def kr20(P):
    """Kuder-Richardson 20 over K binary elements. P is (n_items, K)."""
    P = np.asarray(P, float)
    n, K = P.shape
    p = P.mean(axis=0)
    tot_var = P.sum(axis=1).var(ddof=1)
    if tot_var <= 0:
        return np.nan
    return (K / (K - 1)) * (1 - (p * (1 - p)).sum() / tot_var)


def kr21(tot, K):
    """KR-21, computable from totals alone. Reported only for comparison."""
    tot = np.asarray(tot, float)
    m, v = tot.mean(), tot.var(ddof=1)
    return (K / (K - 1)) * (1 - (m * (K - m)) / (K * v))


def intra_item_rho(P):
    """Mean pairwise phi between elements within an item."""
    P = np.asarray(P, float)
    K = P.shape[1]
    C = np.corrcoef(P.T)
    iu = np.triu_indices(K, 1)
    return float(np.nanmean(C[iu]))


def _bb_nll(par, tot, K):
    a, b = np.exp(par)
    x = np.asarray(tot, float)
    ll = (betaln(x + a, K - x + b) - betaln(a, b))
    return -ll.sum()


def bb_mle(tot, K):
    """Beta-binomial MLE. Returns (alpha, beta)."""
    tot = np.asarray(tot, float)
    m, v = tot.mean() / K, tot.var(ddof=1) / (K ** 2)
    s = max(m * (1 - m) / v - 1, 0.1) if v > 0 else 1.0
    x0 = np.log([max(m * s, 1e-3), max((1 - m) * s, 1e-3)])
    res = minimize(_bb_nll, x0, args=(tot, K), method="Nelder-Mead")
    return tuple(np.exp(res.x))


def bb_pmf(a, b, K):
    """Beta-binomial pmf over 0..K."""
    x = np.arange(K + 1)
    from scipy.special import gammaln
    logc = gammaln(K + 1) - gammaln(x + 1) - gammaln(K - x + 1)
    return np.exp(logc + betaln(x + a, K - x + b) - betaln(a, b))


def bb_gof(tot, K, min_expected=5):
    """Chi-square goodness of fit of the MLE beta-binomial to observed totals.

    Adjacent cells are pooled until every expected count reaches min_expected.
    Returns (chi2, df, p, (alpha, beta)).
    """
    from scipy.stats import chi2 as chi2_dist
    tot = np.asarray(tot, int)
    n = len(tot)
    a, b = bb_mle(tot, K)
    exp = bb_pmf(a, b, K) * n
    obs = np.bincount(tot, minlength=K + 1).astype(float)
    o, e = [], []
    co = ce = 0.0
    for i in range(K + 1):
        co += obs[i]; ce += exp[i]
        if ce >= min_expected:
            o.append(co); e.append(ce); co = ce = 0.0
    if ce > 0:
        if o:
            o[-1] += co; e[-1] += ce
        else:
            o.append(co); e.append(ce)
    o, e = np.array(o), np.array(e)
    chi2 = ((o - e) ** 2 / e).sum()
    df = len(o) - 1 - 2
    return float(chi2), int(df), float(chi2_dist.sf(chi2, df)), (a, b)


def livingston_lewis(tot, K, cut, n_grid=None):
    """Livingston-Lewis decision consistency and classification accuracy at `cut`.

    IMPORTANT: the accuracy returned is indexed to the examinee's OWN true score on
    this instrument, not to any external criterion. See Section 4 of the paper.
    Returns (decision_consistency, classification_accuracy).
    """
    tot = np.asarray(tot, float)
    a, b = bb_mle(tot, K)
    xs = np.arange(K + 1)
    prior = bb_pmf(a, b, K)
    # true proportion grid; P(observed x | true p) is binomial
    grid = np.linspace(1e-4, 1 - 1e-4, n_grid or 400)
    from scipy.stats import beta as beta_dist
    w = beta_dist.pdf(grid, a, b)
    w = w / w.sum()
    L = binom.pmf(xs[:, None], K, grid[None, :])          # (K+1, n_grid)
    pass_obs = (xs >= cut).astype(float)
    p_pass_given_true = (L * pass_obs[:, None]).sum(axis=0)   # (n_grid,)
    true_pass = (grid * K >= cut).astype(float)
    ca = float((w * (p_pass_given_true * true_pass +
                     (1 - p_pass_given_true) * (1 - true_pass))).sum())
    dc = float((w * (p_pass_given_true ** 2 + (1 - p_pass_given_true) ** 2)).sum())
    return dc, ca


def phi_lambda(P, cut):
    """Generalisability dependability index for absolute decisions at cut lambda.

    This is a RATIO OF VARIANCE COMPONENTS. It is not a classification probability
    and must never be compared numerically to one (Section 4 of the paper).
    """
    P = np.asarray(P, float)
    n, K = P.shape
    lam = cut / K
    X = P.mean(axis=1)
    grand = P.mean()
    v_p = max(X.var(ddof=1) - P.var(ddof=1) / K, 0.0)
    v_e = P.var(ddof=1) / K
    return float((v_p + (grand - lam) ** 2) / (v_p + (grand - lam) ** 2 + v_e))


def classification_accuracy(judge_tot, ref_tot, cut):
    """Agreement of pass/fail decisions between judge totals and a reference."""
    return float(np.mean((np.asarray(judge_tot) >= cut) == (np.asarray(ref_tot) >= cut)))


def cluster_bootstrap(stat_fn, cluster_ids, B=2000, seed=0):
    """Bootstrap `stat_fn(idx)` by resampling clusters with replacement."""
    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    out = []
    for _ in range(B):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(cluster_ids == c)[0] for c in drawn])
        v = stat_fn(idx)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)
