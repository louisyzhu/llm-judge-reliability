"""Simulations for "Three Ways Classical Test Theory Misleads for LLM Judges".

Run `python sweeps.py` to regenerate every simulated quantity reported in the paper.
Seeds are fixed, so output is exact.
"""
import json
import numpy as np
from estimators import kr20, livingston_lewis, classification_accuracy, phi_lambda

K = 10
N_ITEMS = 210
MEASURED_ERROR = 0.0472          # per-element judge error, measured (Section 3)
BANK_SPREADS = [0.0, 0.2, 0.4, 0.6, 0.8]
JUDGE_ERRORS = [0.00, 0.05, 0.10, 0.20, 0.30]


def simulate_bank(spread, err, n=N_ITEMS, K=K, rng=None):
    """One synthetic bank. `spread` controls between-item true-score variance;
    `err` is the per-element probability the judge flips a verdict."""
    rng = rng or np.random.default_rng(0)
    lo, hi = 0.5 - spread / 2, 0.5 + spread / 2
    rates = rng.uniform(lo, hi, n)
    Gm = (rng.random((n, K)) < rates[:, None]).astype(float)
    flip = rng.random((n, K)) < err
    Jm = np.where(flip, 1 - Gm, Gm)
    return Gm, Jm


def two_way_sweep(reps=60, seed=11):
    """Grid of KR-20 over bank spread x judge error. Reproduces Table 2/3."""
    rng = np.random.default_rng(seed)
    mean = np.zeros((len(BANK_SPREADS), len(JUDGE_ERRORS)))
    sd = np.zeros_like(mean)
    for i, s in enumerate(BANK_SPREADS):
        for j, e in enumerate(JUDGE_ERRORS):
            vals = [kr20(simulate_bank(s, e, rng=rng)[1]) for _ in range(reps)]
            vals = np.array([v for v in vals if np.isfinite(v)])
            mean[i, j], sd[i, j] = vals.mean(), vals.std(ddof=1)
    return mean, sd


def phi_control(seed=101, n=4000):
    """Control where the beta-binomial holds exactly, so Livingston-Lewis is
    accurate. Any residual Phi-vs-accuracy gap is a mismatch of quantities."""
    rng = np.random.default_rng(seed)
    p = rng.beta(5.452, 3.663, n)
    P = (rng.random((n, K)) < p[:, None]).astype(float)   # element-level verdicts
    tot = P.sum(axis=1)
    out = {}
    for cut in (4, 5, 6, 7):
        _, ca_ll = livingston_lewis(tot, K, cut)
        ca_true = float(np.mean((tot >= cut) == (p * K >= cut)))
        phi = phi_lambda(P, cut)
        out[cut] = dict(ll=ca_ll, true=ca_true, gap_ll=ca_true - ca_ll,
                        phi=phi, gap_phi=ca_true - phi)
    return out


def ll_estimand_control(seed=303, n=4000):
    """Judge true score is NOT a deterministic function of gold. Compares
    accuracy against the judge's own true score with accuracy against gold."""
    rng = np.random.default_rng(seed)
    gold = rng.binomial(K, 0.55, n).astype(float)
    judge_true = np.clip(0.9 * gold + 0.9 + rng.normal(0, 1.1, n), 0, K)
    obs = np.clip(np.round(judge_true + rng.normal(0, 0.85, n)), 0, K)
    out = {}
    for cut in (4, 5, 6, 7):
        own = classification_accuracy(obs, judge_true, cut)
        ext = classification_accuracy(obs, gold, cut)
        out[cut] = dict(vs_own_true=own, vs_gold=ext, gap=own - ext)
    return out


def phrasing_gstudy(v_phrasing, v_resid, n=N_ITEMS, R=3, sims=200, seed=7):
    """p x ph generalisability study by two-way random-effects ANOVA.

    Phi for absolute decisions is v_p / (v_p + (v_ph + v_e) / R), since a
    deployment commits to one phrasing rather than averaging over R of them.
    The last scenario in the paper sets v_resid = 0: a judge perfectly
    deterministic given a fixed phrasing. The phrasing component stays identified.
    """
    rng = np.random.default_rng(seed)
    comps = []
    for _ in range(sims):
        p = rng.normal(0, 1, (n, 1))
        ph = rng.normal(0, np.sqrt(v_phrasing), (1, R))
        e = rng.normal(0, np.sqrt(v_resid), (n, R))
        X = p + ph + e
        gm = X.mean()
        rm = X.mean(axis=1, keepdims=True)
        cm = X.mean(axis=0, keepdims=True)
        ms_p = R * ((rm - gm) ** 2).sum() / (n - 1)
        ms_ph = n * ((cm - gm) ** 2).sum() / (R - 1)
        ms_e = ((X - rm - cm + gm) ** 2).sum() / ((n - 1) * (R - 1))
        vp = max((ms_p - ms_e) / R, 0.0)
        vph = max((ms_ph - ms_e) / n, 0.0)
        ve = max(ms_e, 0.0)
        phi = vp / (vp + (vph + ve) / R) if vp > 0 else np.nan
        comps.append((vp, vph, ve, phi))
    m = np.nanmean(np.array(comps), axis=0)
    return dict(v_items=m[0], v_phrasing=m[1], v_resid=m[2], Phi=m[3])


if __name__ == "__main__":
    mean, sd = two_way_sweep()
    out = {
        "grid": {"rows_bank_spread": BANK_SPREADS,
                 "cols_judge_error": JUDGE_ERRORS,
                 "kr20_mean": mean.tolist(), "kr20_sd": sd.tolist()},
        "range_at_measured_error": {
            "judge_error": MEASURED_ERROR,
            "kr20_min": float(mean[:, 1].min()), "kr20_max": float(mean[:, 1].max())},
        "judge_error_axis_at_widest_bank": float(mean[-1].max() - mean[-1].min()),
        "bank_axis_at_zero_error": float(mean[:, 0].max() - mean[:, 0].min()),
        "phi_control": phi_control(),
        "ll_estimand_control": ll_estimand_control(),
        "phrasing_gstudy": {
            "irrelevant": phrasing_gstudy(0.000, 0.090),
            "mild": phrasing_gstudy(0.039, 0.090),
            "large": phrasing_gstudy(0.362, 0.090),
            "deterministic_given_phrasing": phrasing_gstudy(0.175, 0.000)},
    }
    with open("sweeps_output.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["range_at_measured_error"], indent=1))
