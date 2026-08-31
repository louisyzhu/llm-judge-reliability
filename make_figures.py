"""Regenerates every figure in "Three Ways Classical Test Theory Misleads for LLM Judges".

Usage:  python make_figures.py     # writes figure1.pdf figure2.pdf figure3.pdf figureA1.pdf
Requires judge_item_bank.csv, sweep_grid.json, estimators.py and sweeps.py alongside.
Output is vector PDF with Type-42 embedded fonts.
"""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from sweeps import ll_estimand_control
from estimators import phi_lambda

K = 10
INK, MUTE = "#1b1e23", "#6c727c"
BLUE, TEAL, RUST, PLUM = "#2c6fa8", "#2a9d8f", "#c1440e", "#7b4b94"
PALEG, PALER = "#eaf4f1", "#fdeee8"

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 7.2, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_bank(path="judge_item_bank.csv"):
    df = pd.read_csv(path)
    g = [f"gold_e{i+1}" for i in range(K)]
    j = [f"judge_e{i+1}" for i in range(K)]
    sc = df.dropna(subset=j)
    return sc[g].to_numpy(float), sc[j].to_numpy(float)


def load_grid(path="sweep_grid.json"):
    d = json.load(open(path))
    return (np.array(d["kr20_mean"]), np.array(d["kr20_sd"]),
            d["rows_bank_spread"], d["cols_judge_error"],
            np.array(d["measured_error_run"]["kr20_mean"]),
            np.array(d["measured_error_run"]["kr20_sd"]))


def figure1(path="figure1.pdf"):
    """Facet schematic: the verdict matrix, what KR-20 resolves, and what it cannot."""
    fig = plt.figure(figsize=(6.2, 1.72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(4, 40); ax.axis("off")
    jx, jy, jw, jh = 2.0, 31.0, 38.0, 6.6
    ax.add_patch(FancyBboxPatch((jx, jy), jw, jh, boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc="white", ec=RUST, lw=1.0))
    ax.text(jx + jw/2, jy + 4.3, "LLM judge applies the rubric", ha="center",
            fontsize=7.0, color=INK)
    ax.text(jx + jw/2, jy + 1.6, "one model, one prompt phrasing", ha="center",
            fontsize=6.3, color=MUTE)
    x0, y0, cw, ch = 8.0, 9.0, 2.85, 2.85
    nr, nc = 6, 7
    ax.annotate("", xy=(x0 + nc*cw - 2.0, y0 + nr*ch + 0.7), xytext=(jx + jw - 6.0, jy - 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=RUST, shrinkA=1, shrinkB=1,
                                connectionstyle="arc3,rad=-0.18"))
    rng = np.random.default_rng(3)
    for r in range(nr):
        for c in range(nc):
            v = rng.random()
            ax.add_patch(Rectangle((x0 + c*cw, y0 + r*ch), cw*0.9, ch*0.9,
                                   fc=(BLUE if v > 0.32 else "#ffffff"),
                                   ec="#c7ccd3", lw=0.4, alpha=0.85 if v > 0.32 else 1.0))
    ax.text(x0, y0 + nr*ch + 2.0, "verdict matrix  $X_{ij}$", ha="left", fontsize=6.9, color=INK)
    ax.annotate("", xy=(x0 - 1.6, y0), xytext=(x0 - 1.6, y0 + nr*ch - 0.3),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTE, shrinkA=0, shrinkB=0))
    ax.text(x0 - 2.8, y0 + nr*ch/2 - 0.2, "responses\n$i=1..n$", ha="right", va="center",
            fontsize=6.2, color=MUTE)
    ax.annotate("", xy=(x0, y0 - 1.7), xytext=(x0 + nc*cw - 0.3, y0 - 1.7),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTE, shrinkA=0, shrinkB=0))
    ax.text(x0 + nc*cw/2 - 1.2, y0 - 2.8, "rubric elements  $j=1..K$", ha="center", va="top",
            fontsize=6.2, color=MUTE)
    bx, bw = 48.0, 50.0
    ax.add_patch(FancyBboxPatch((bx, y0 + 9.0), bw, 9.2,
                                boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc=PALEG, ec=TEAL, lw=0.95))
    ax.text(bx + bw/2, y0 + 15.4, "resolved by KR-20", ha="center", fontsize=6.9, color="#1d6f65")
    ax.text(bx + bw/2, y0 + 11.9,
            r"$\sigma^2(\mathrm{responses})\;+\;\sigma^2(\mathrm{elements})"
            r"\;+\;\sigma^2(\mathrm{residual})$", ha="center", fontsize=6.9, color=INK)
    ax.add_patch(FancyBboxPatch((bx, y0 - 0.5), bw, 6.8,
                                boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc=PALER, ec=RUST, lw=0.95, ls=(0, (3, 2))))
    ax.text(bx + bw/2, y0 + 3.9, r"$\sigma^2(\mathrm{scorer})$", ha="center",
            fontsize=7.2, color=RUST)
    ax.text(bx + bw/2, y0 + 1.2, "absorbed into the residual; no facet identifies it",
            ha="center", fontsize=6.2, color=RUST)
    ax.annotate("", xy=(bx - 0.9, y0 + 13.4), xytext=(x0 + nc*cw + 0.5, y0 + nr*ch - 2.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.95, color=TEAL, shrinkA=2, shrinkB=2))
    ax.annotate("", xy=(bx - 0.9, y0 + 3.0), xytext=(x0 + nc*cw + 0.5, y0 + 2.4),
                arrowprops=dict(arrowstyle="-|>", lw=0.95, color=RUST, shrinkA=2, shrinkB=2,
                                ls=(0, (3, 2))))
    fig.savefig(path, bbox_inches="tight")
    return fig


def figure2(path="figure2.pdf"):
    """Two-way sweep: KR-20 confounds bank design with judge error."""
    M, SD, rows, cols, m472, s472 = load_grid()
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.05),
                            gridspec_kw=dict(width_ratios=[1.40, 1, 1], wspace=0.60))
    a = axs[0]
    # pcolormesh emits vector quads; imshow would embed a bitmap
    im = a.pcolormesh(np.arange(len(cols) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5, M,
                      cmap="YlGnBu", vmin=-0.05, vmax=0.75, shading="flat",
                      rasterized=False, linewidth=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                   color="white" if M[i, j] > 0.42 else INK)
    a.set_xticks(range(5)); a.set_xticklabels([f"{int(v*100)}" for v in cols], fontsize=6)
    a.set_yticks(range(5)); a.set_yticklabels([f"{v:.1f}" for v in rows], fontsize=6)
    a.set_xlim(-0.5, len(cols) - 0.5); a.set_ylim(-0.5, len(rows) - 0.5)
    a.set_xlabel("judge per-element error (%)", fontsize=6.8)
    a.set_ylabel("bank true-score spread", fontsize=6.8)
    a.set_title("(a)  KR-20, both inputs varying", loc="left", fontsize=7.2, color=INK)
    a.add_patch(Rectangle((1.5, 2.5), 1, 1, fill=False, ec=RUST, lw=1.5))
    a.add_patch(Rectangle((2.5, 3.5), 1, 1, fill=False, ec=RUST, lw=1.5))
    a.text(0.0, -0.30, r"outlined: equal KR-20 at $2\times$ the judge error",
           transform=a.transAxes, ha="left", va="top", fontsize=5.9, color=RUST)
    cb = fig.colorbar(im, ax=a, fraction=0.040, pad=0.04)
    cb.solids.set_rasterized(False)  # keep the colorbar vector too
    cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.5)

    b = axs[1]
    b.plot(rows, m472, "-o", color=BLUE, lw=1.4, ms=3.4)
    b.set_xticks(rows)  # match panel (a): one tick per swept bank spread
    b.fill_between(rows, m472 - s472, m472 + s472, color=BLUE, alpha=0.15, lw=0)
    b.axhline(0, color=MUTE, lw=0.5, ls=":")
    b.set_xlabel("bank true-score spread", fontsize=6.8); b.set_ylabel("KR-20", fontsize=6.8)
    b.set_title("(b)  judge fixed at 4.7% err.", loc="left", fontsize=7.2)
    b.text(0.04, 0.88, f"{m472.min():.2f} to {m472.max():.2f}", transform=b.transAxes,
           fontsize=6.4, color=BLUE)
    b.set_ylim(-0.15, 0.92); b.tick_params(labelsize=6)

    c = axs[2]
    xs = [v*100 for v in cols]
    c.plot(xs, M[-1], "-s", color=RUST, lw=1.4, ms=3.2)
    c.fill_between(xs, M[-1] - SD[-1], M[-1] + SD[-1], color=RUST, alpha=0.15, lw=0)
    c.axhline(0, color=MUTE, lw=0.5, ls=":")
    c.set_xlabel("judge per-element error (%)", fontsize=6.8)
    c.set_ylabel("KR-20", fontsize=6.8)
    c.set_title("(c)  bank fixed at widest", loc="left", fontsize=7.2)
    c.text(0.04, 0.20, f"{M[-1].min():.2f} to {M[-1].max():.2f}", transform=c.transAxes,
           fontsize=6.4, color=RUST)
    c.set_ylim(-0.15, 0.92); c.set_xticks([0, 10, 20, 30]); c.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight")
    return fig


def figure3(path="figure3.pdf", ca=None, phi=None):
    """Phi(lambda) against accuracy, and the two Livingston-Lewis estimands."""
    cuts = np.array([4, 5, 6, 7])
    # computed from the released bank so the figure cannot drift from the code
    G, J = load_bank()
    gt, jt = G.sum(1), J.sum(1)
    if ca is None:
        ca = np.array([float(np.mean((jt >= c) == (gt >= c))) for c in cuts])
    if phi is None:
        phi = np.array([phi_lambda(J, c) for c in cuts])
    ctrl = ll_estimand_control()
    own = np.array([ctrl[c]["vs_own_true"] for c in cuts])
    gold = np.array([ctrl[c]["vs_gold"] for c in cuts])
    fig, (p1, p2) = plt.subplots(1, 2, figsize=(5.5, 2.05),
                                 gridspec_kw=dict(wspace=0.34, width_ratios=[1.12, 1.0]))
    w = 0.26
    p1.bar(cuts - w/2, ca, w, color=BLUE, label="classification accuracy",
           edgecolor="white", lw=0.5)
    p1.bar(cuts + w/2, phi, w, color=PLUM, label=r"$\Phi(\lambda)$",
           edgecolor="white", lw=0.5)
    for c_, x_, y_ in zip(cuts, ca, phi):
        # arrow and label centred in the gap between this group and the next
        xa = c_ + 0.5
        p1.annotate("", xy=(xa, y_ + 0.010), xytext=(xa, x_ - 0.010),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK, shrinkA=0, shrinkB=0))
        p1.text(xa, (x_ + y_)/2, f"{x_-y_:.2f}", fontsize=5.6,
                va="center", ha="center", color=INK,
                bbox=dict(fc="white", ec="none", pad=0.5))
    p1.set_xticks(cuts); p1.set_xlabel("gate cut score", fontsize=6.8)
    p1.set_ylabel("value", fontsize=6.8)
    p1.set_ylim(0, 1.20); p1.set_xlim(3.4, 7.9); p1.tick_params(labelsize=6)
    p1.set_title("(a)  a coefficient is not a probability", loc="left", fontsize=7.2)
    p1.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.0, handlelength=1.0,
              borderpad=0.1, columnspacing=0.8, bbox_to_anchor=(0.5, 1.03))
    p1.grid(axis="y", color="#e8eaed", lw=0.5); p1.set_axisbelow(True)

    p2.plot(cuts, own, "-o", color=TEAL, lw=1.4, ms=3.4, label="vs judge's own true score")
    p2.plot(cuts, gold, "-o", color=RUST, lw=1.4, ms=3.4, label="vs external gold")
    p2.fill_between(cuts, gold, own, color=RUST, alpha=0.09, lw=0)
    for c_, o_, g_ in zip(cuts, own, gold):
        p2.text(c_, (o_ + g_)/2, f"{o_-g_:.3f}", fontsize=6.0, ha="center", va="center",
                color=INK, bbox=dict(fc="white", ec="none", pad=0.8))
    p2.set_xticks(cuts); p2.set_xlim(3.7, 7.35); p2.tick_params(labelsize=6)
    p2.set_xlabel("gate cut score", fontsize=6.8)
    p2.set_ylabel("classification accuracy", fontsize=6.8)
    p2.set_title("(b)  two accuracies, two estimands", loc="left", fontsize=7.2)
    p2.legend(frameon=False, loc="lower left", fontsize=6.0, handlelength=1.2, borderpad=0.1)
    p2.set_ylim(0.70, 0.98)
    p2.grid(axis="y", color="#e8eaed", lw=0.5); p2.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight")
    return fig


def figureA1(path="figureA1.pdf", bank="judge_item_bank.csv"):
    """Bank and judge diagnostics: no ceiling, concentrated error, leniency."""
    G, J = load_bank(bank)
    gt, jt = G.sum(1), J.sum(1)
    pe = (J != G).mean(axis=0)
    fig, (q1, q2, q3) = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw=dict(wspace=0.42))
    bins = np.arange(-0.5, K + 1.5, 1)
    q1.hist(gt, bins=bins, color="#c2c8cf", label="gold", zorder=1)
    q1.hist(jt, bins=bins, histtype="step", lw=1.4, color=BLUE, label="judge", zorder=2)
    q1.set_xlabel(f"total score (of $K={K}$)", fontsize=6.8)
    q1.set_ylabel("items", fontsize=6.8)
    q1.set_title("(a)  no ceiling mass", loc="left", fontsize=7.2)
    q1.legend(frameon=False, fontsize=6.0, handlelength=1.0, loc="upper left")
    q1.grid(axis="y", color="#e8eaed", lw=0.5); q1.set_axisbelow(True); q1.tick_params(labelsize=6)

    q2.bar(np.arange(1, K + 1), pe, color="#c98a2e", edgecolor="white", lw=0.4)
    q2.axhline(pe.mean(), color=RUST, ls="--", lw=0.9)
    q2.text(K + 0.4, pe.mean() + 0.003, f"mean {pe.mean():.3f}", fontsize=5.9,
            color=RUST, va="bottom", ha="right")
    q2.set_xticks(range(1, K + 1)); q2.set_xlabel("checklist element", fontsize=6.8)
    q2.set_ylabel("error rate", fontsize=6.8)
    q2.set_title("(b)  error concentrates", loc="left", fontsize=7.2)
    q2.grid(axis="y", color="#e8eaed", lw=0.5); q2.set_axisbelow(True)
    q2.set_ylim(0, max(0.168, pe.max()*1.15)); q2.tick_params(labelsize=6)

    rg = np.random.default_rng(7)
    q3.scatter(gt + rg.uniform(-0.16, 0.16, len(gt)), jt + rg.uniform(-0.16, 0.16, len(jt)),
               s=7, alpha=0.32, color=BLUE, edgecolors="none")
    q3.plot([0, K], [0, K], "--", color=INK, lw=0.8)
    q3.set_xlabel("gold total", fontsize=6.8); q3.set_ylabel("judge total", fontsize=6.8)
    q3.set_title("(c)  systematic leniency", loc="left", fontsize=7.2)
    q3.text(0.05, 0.94, f"$r={np.corrcoef(gt, jt)[0,1]:.3f}$\nbias $+{(jt-gt).mean():.2f}$",
            transform=q3.transAxes, fontsize=6.0, va="top", color=INK)
    q3.set_xlim(-0.6, K + 0.6); q3.set_ylim(-0.6, K + 0.6)
    q3.grid(color="#e8eaed", lw=0.5); q3.set_axisbelow(True); q3.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figureA1()
    print("wrote figure1.pdf figure2.pdf figure3.pdf figureA1.pdf")
