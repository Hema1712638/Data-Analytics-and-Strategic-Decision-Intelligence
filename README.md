# Data Analytics and Strategic Decision Intelligence

KB-SDI is a knowledge-based artificial intelligence system for DBA 802. It uses operational data analytics to estimate current performance, and a small knowledge base to decide whether an entrepreneurial firm should hold, scale, or redesign its business model before a digital-transformation and innovation program is expanded.

The short communication is [`paper.md`](paper.md). It is written for *Knowledge-Based Systems* and includes two figures and two tables. Every reported number comes from [`kb_sdi.py`](kb_sdi.py).

The firms are synthetic. The data-generating process is stated in Section 3 of the paper and implemented in `signal`. The decision systems do not see that process.

## Run

```bash
pip install -r requirements.txt
python3 kb_sdi.py
```

The script runs 20 replications, writes `results.json`, and refreshes:

- `fig1_architecture.png`
- `fig2_results.png`

## Files

| File | Role |
|---|---|
| `kb_sdi.py` | Simulation, knowledge base, gradient boosting, decision layer, figures |
| `paper.md` | Short communication |
| `results.json` | Means, standard deviations, and robustness checks |
| `fig1_architecture.png` | System figure |
| `fig2_results.png` | Accuracy and false SCALE rate |
| `requirements.txt` | Python dependencies |

## Decision the system makes

| Action | When it is selected |
|---|---|
| SCALE | Current estimated value, after the business-model adjustment, clears the reservation value and beats a redesign |
| REDESIGN | Lifting business-model change to the knowledge-base target is worth more than scaling the current model |
| HOLD | Neither investment clears the reservation value |

Analytics alone never returns REDESIGN, because it has no business-model counterfactual. On the held-out firms in the paper, that policy selects SCALE in 0.578 of the cases where REDESIGN is the better action. KB-SDI selects SCALE in 0.016 of those cases.
