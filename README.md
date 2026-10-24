# Data Analytics and Strategic Decision Intelligence

KB-SDI is a knowledge-based artificial intelligence system for DBA 802. Gradient boosting reads the operational record. A 100-firm knowledge base prices the value of changing the business model. The decision is HOLD, SCALE, or REDESIGN.

The short communication is [`paper.md`](paper.md). It follows the structure of a knowledge-guided decision-support short communication: numbered equations, an explicit algorithm, a component ablation, paired Wilcoxon tests, a sensitivity sweep, and a stress test. The firms are synthetic. Every reported number is produced by [`kb_sdi.py`](kb_sdi.py).

## Run

```bash
pip install -r requirements.txt
python3 kb_sdi.py
```

The script runs 40 replications, then the sensitivity and assessment-noise checks. It rewrites `results.json`, `fig1_decision_map.png`, and `fig2_paired_accuracy.png`.

## What the experiment compares

| System | Operational model | Business-model knowledge |
|---|---|---|
| Threshold | No. SCALE if capability is at least 0.55 | No |
| Analytics-only | Gradient boosting | No redesign counterfactual |
| Rules-only | No | Four-parameter mechanism |
| KB-SDI | Gradient boosting | Same mechanism, fused with the forecast |

On the 40-replication benchmark, mean accuracy is 0.437, 0.467, 0.770, and 0.929. Analytics-only selects SCALE in 0.577 of the cases where REDESIGN is the oracle action. KB-SDI does so in 0.014. Paired tests favor KB-SDI in 40 of 40 replications.

## Files

| File | Role |
|---|---|
| `kb_sdi.py` | Simulation, knowledge base, boosting model, tests, figures |
| `paper.md` | Short communication |
| `results.json` | Means, standard deviations, p-values, robustness |
| `fig1_decision_map.png` | Oracle regions and false SCALE decisions |
| `fig2_paired_accuracy.png` | Accuracy and paired gaps |
| `requirements.txt` | Python dependencies |
