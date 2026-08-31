# Three Ways Classical Test Theory Misleads for LLM Judges

Reproducibility bundle for the paper. Every number and figure in the paper regenerates
from the files here.

Paper: arXiv (link to follow once posted).

## Contents

| File | Contents |
|---|---|
| `judge_item_bank.csv` | 210 items: question id, answer text, element and distractor text, element-level gold and judge verdicts, totals |
| `estimators.py` | KR-20, KR-21, intra-item phi, beta-binomial MLE and goodness of fit, Livingston-Lewis DC/CA, Phi(lambda), cluster bootstrap |
| `sweeps.py` | Two-way sweep, Phi(lambda) control (reports both the Livingston-Lewis and Phi(lambda) gaps against true accuracy), Livingston-Lewis estimand control, phrasing-facet identification check |
| `sweep_grid.json` | The sweep values as published, including the 60-replicate run at the measured 4.72% error rate |
| `make_figures.py` | Regenerates Figures 1-3 and A1 as vector PDF |

## Running

    pip install -r requirements.txt
    python sweeps.py          # writes sweeps_output.json
    python make_figures.py    # writes figure1.pdf figure2.pdf figure3.pdf figureA1.pdf

Both scripts must be run from this directory.

## Reproduction notes

Quantities measured on the real bank reproduce bit-for-bit: KR-20 on judge verdicts
0.5223 and on gold 0.5231, per-element error 4.72%, judge-gold correlation 0.921,
leniency +0.46 elements, beta-binomial goodness of fit chi2 = 5.31 on 6 df (p = 0.504).

Simulated quantities carry fixed seeds, so re-running `sweeps.py` reproduces its own
output exactly. The two-way grid printed in the paper was generated on a different RNG
stream from the one this script uses; re-simulating moves individual cells by about
a third of one per-cell standard deviation. `sweep_grid.json` carries the published values
so the figures match the paper.

## Licence

Code (`*.py`) is released under the MIT licence (see `LICENSE`). The item bank
(`judge_item_bank.csv`) and the published grids (`sweep_grid.json`) are released under
CC BY 4.0 (see `LICENSE-DATA`).

## Citation

    @misc{zhu2026threeways,
      author = {Zhu, Louis Yiven},
      title  = {Three Ways Classical Test Theory Misleads for {LLM} Judges},
      year   = {2026},
      note   = {arXiv preprint}
    }
