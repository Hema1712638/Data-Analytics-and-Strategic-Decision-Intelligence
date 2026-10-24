"""
KB-SDI: knowledge-based strategic decision intelligence.

A gradient-boosting model reads the operational record. A four-parameter
knowledge base, fit on firms where business-model change is assessed, prices
the counterfactual value of a redesign. The decision is HOLD, SCALE, or
REDESIGN.

The firms are synthetic. The data-generating process in `signal` is not
available to any decision system. Fixed seeds regenerate every reported number.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

N_FIRMS = 5000
TEST_SIZE = 0.28
KNOWLEDGE_SIZE = 100
BMC_TARGET = 0.82
V_HOLD = 0.42
COST_SCALE = 0.05
COST_REDESIGN = 0.11
CAPABILITY_FLOOR = 0.55
SLICE_BMC = 0.40
SEEDS = list(range(802, 842))
PRIMARY_SEED = 802

ACTIONS = ("HOLD", "SCALE", "REDESIGN")
SYSTEMS = ("Threshold", "Analytics-only", "Rules-only", "KB-SDI")
PREDICTORS = ("Analytics-only", "Rules-only", "KB-SDI")
BASELINES = ("Threshold", "Analytics-only", "Rules-only")
REGION_COLORS = ["#F4F4F4", "#E6A15C", "#7FA6C9"]
ERROR_COLOR = "#8C1D18"
FEATURE_NAMES = ("dac", "aic", "dti", "inn", "eo", "dq", "size", "age")
METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "regret",
    "false_scale_given_redesign",
    "redesign_recall",
    "slice_false_scale",
    "slice_redesign_recall",
    "slice_regret",
)


def simulate(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dac = rng.beta(2.0, 2.0, n)
    aic = rng.beta(2.0, 2.0, n)
    dti = rng.beta(2.0, 2.2, n)
    inn = rng.beta(2.2, 2.0, n)
    eo = rng.beta(1.8, 2.2, n)
    dq = rng.beta(2.0, 2.0, n)
    size = rng.normal(0.0, 1.0, n)
    age = rng.uniform(0.0, 1.0, n)
    capability = np.clip(0.55 * dac + 0.45 * aic, 0.0, 1.0)
    # Independent of the operational features, so they cannot proxy it.
    business_model = np.clip(rng.beta(1.7, 2.0, n), 0.0, 1.0)
    return {
        "dac": dac,
        "aic": aic,
        "dti": dti,
        "inn": inn,
        "eo": eo,
        "dq": dq,
        "size": size,
        "age": age,
        "c": capability,
        "bmc": business_model,
    }


def signal(data: dict[str, np.ndarray], business_model: np.ndarray | None = None) -> np.ndarray:
    bmc = data["bmc"] if business_model is None else business_model
    data_quality_gain = 0.20 / (1.0 + np.exp(-10.0 * (data["dq"] - 0.55)))
    return (
        0.02
        + 0.06 * data["c"]
        + 0.04 * bmc
        + 0.62 * data["c"] * bmc
        + 0.10 * data["inn"]
        + 0.08 * data["eo"]
        + 0.06 * data["dti"]
        + data_quality_gain
        + 0.05 * data["dac"] * data["inn"]
        + 0.01 * np.clip(data["size"], -2.0, 2.0)
    )


def payoffs(current: np.ndarray, counterfactual: np.ndarray, v_hold: float = V_HOLD) -> np.ndarray:
    hold = np.full(current.shape, v_hold)
    return np.column_stack([hold, current - COST_SCALE, counterfactual - COST_REDESIGN])


def estimate_mechanism(capability: np.ndarray, business_model: np.ndarray, performance: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(capability)), capability, business_model, capability * business_model])
    coefficients, *_ = np.linalg.lstsq(design, performance, rcond=None)
    return coefficients


def mechanism_prediction(coefficients: np.ndarray, capability: np.ndarray, business_model: np.ndarray) -> np.ndarray:
    intercept, capability_coef, bmc_coef, interaction = coefficients
    return intercept + capability_coef * capability + bmc_coef * business_model + interaction * capability * business_model


def marginal_bmc_effect(coefficients: np.ndarray, capability: np.ndarray) -> np.ndarray:
    return coefficients[2] + coefficients[3] * capability


def fit_analytics(features: np.ndarray, performance: np.ndarray, seed: int) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_leaf=15,
        random_state=seed,
    )
    model.fit(features, performance)
    return model


def _safe_rate(mask: np.ndarray, condition: np.ndarray) -> float:
    denom = int(condition.sum())
    if denom == 0:
        return float("nan")
    return float(mask.sum() / denom)


def score_actions(
    predicted: np.ndarray,
    oracle: np.ndarray,
    true_value: np.ndarray,
    slice_mask: np.ndarray,
) -> dict[str, float]:
    best = true_value.max(axis=1)
    chosen = true_value[np.arange(len(predicted)), predicted]
    redesign = oracle == 2
    slice_redesign = slice_mask & redesign
    return {
        "accuracy": float(accuracy_score(oracle, predicted)),
        "macro_f1": float(f1_score(oracle, predicted, average="macro", zero_division=0)),
        "regret": float((best - chosen).mean()),
        "false_scale_given_redesign": _safe_rate((predicted == 1) & redesign, redesign),
        "redesign_recall": _safe_rate((predicted == 2) & redesign, redesign),
        "slice_false_scale": _safe_rate((predicted == 1) & slice_redesign, slice_redesign),
        "slice_redesign_recall": _safe_rate((predicted == 2) & slice_redesign, slice_redesign),
        "slice_regret": float((best[slice_mask] - chosen[slice_mask]).mean()) if slice_mask.any() else float("nan"),
        "slice_n": int(slice_mask.sum()),
    }


def oracle_grid(template: dict[str, np.ndarray], size: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = {name: float(template[name].mean()) for name in FEATURE_NAMES}
    capability = np.linspace(0.0, 1.0, size)
    business_model = np.linspace(0.0, 1.0, size)
    grid_c, grid_b = np.meshgrid(capability, business_model)
    flat_c = grid_c.ravel()
    flat_b = grid_b.ravel()
    grid = {name: np.full(flat_c.size, means[name]) for name in FEATURE_NAMES}
    grid["dac"] = flat_c
    grid["aic"] = flat_c
    grid["c"] = flat_c
    grid["bmc"] = flat_b
    current = signal(grid)
    counterfactual = signal(grid, np.maximum(flat_b, BMC_TARGET))
    actions = payoffs(current, counterfactual).argmax(axis=1).reshape(size, size)
    return capability, business_model, actions


def evaluate_once(
    seed: int,
    knowledge_size: int = KNOWLEDGE_SIZE,
    v_hold: float = V_HOLD,
    assessment_noise: float = 0.0,
    return_detail: bool = False,
) -> dict:
    data = simulate(N_FIRMS, seed)
    observed = np.clip(signal(data) + np.random.default_rng(seed + 17).normal(0.0, 0.04, N_FIRMS), 0.0, 1.0)
    expected = signal(data)
    redesigned_true = np.maximum(data["bmc"], BMC_TARGET)
    expected_redesign = signal(data, redesigned_true)
    oracle = payoffs(expected, expected_redesign, v_hold).argmax(axis=1)

    index = np.arange(N_FIRMS)
    train, test = train_test_split(index, test_size=TEST_SIZE, random_state=seed, stratify=oracle)
    knowledge, operational = train[:knowledge_size], train[knowledge_size:]
    features = np.column_stack([data[name] for name in FEATURE_NAMES])

    started = time.perf_counter()
    analytics = fit_analytics(features[operational], observed[operational], seed)
    fit_seconds = time.perf_counter() - started
    predicted_now = analytics.predict(features[test])

    coefficients = estimate_mechanism(data["c"][knowledge], data["bmc"][knowledge], observed[knowledge])
    if assessment_noise > 0:
        shock = np.random.default_rng(seed + 99).normal(0.0, assessment_noise, N_FIRMS)
        assessed = np.clip(data["bmc"] + shock, 0.0, 1.0)
    else:
        assessed = data["bmc"]
    effect = marginal_bmc_effect(coefficients, data["c"][test])
    baseline_bmc = float(data["bmc"][knowledge].mean())
    assessed_test = assessed[test]
    assessed_target = np.maximum(assessed_test, BMC_TARGET)

    knowledge_current = mechanism_prediction(coefficients, data["c"][test], assessed_test)
    knowledge_counterfactual = mechanism_prediction(coefficients, data["c"][test], assessed_target)
    fused_current = np.clip(predicted_now + effect * (assessed_test - baseline_bmc), 0.0, 1.0)
    fused_counterfactual = np.clip(predicted_now + effect * (assessed_target - baseline_bmc), 0.0, 1.0)
    threshold = np.where(data["c"][test] >= CAPABILITY_FLOOR, 1, 0)

    decisions = {
        "Threshold": threshold,
        "Analytics-only": payoffs(predicted_now, predicted_now, v_hold).argmax(axis=1),
        "Rules-only": payoffs(knowledge_current, knowledge_counterfactual, v_hold).argmax(axis=1),
        "KB-SDI": payoffs(fused_current, fused_counterfactual, v_hold).argmax(axis=1),
    }
    truth = payoffs(expected[test], expected_redesign[test], v_hold)
    oracle_test = oracle[test]
    high_capability_low_bmc = (data["c"][test] >= CAPABILITY_FLOOR) & (data["bmc"][test] <= SLICE_BMC)

    report: dict = {
        "seed": seed,
        "interaction": float(coefficients[3]),
        "fit_seconds": fit_seconds,
        "slice_n": int(high_capability_low_bmc.sum()),
    }
    predictions = {
        "Analytics-only": predicted_now,
        "Rules-only": knowledge_current,
        "KB-SDI": fused_current,
    }
    for name, action in decisions.items():
        report[name] = score_actions(action, oracle_test, truth, high_capability_low_bmc)
        if name in predictions:
            report[name]["rmse"] = float(mean_squared_error(observed[test], predictions[name]) ** 0.5)
        else:
            report[name]["rmse"] = float("nan")
    if high_capability_low_bmc.any():
        report["slice_observed"] = float(observed[test][high_capability_low_bmc].mean())
        report["slice_predicted"] = {
            name: float(predictions[name][high_capability_low_bmc].mean()) for name in PREDICTORS
        }
    if return_detail:
        report["detail"] = {
            "c": data["c"][test],
            "bmc": data["bmc"][test],
            "oracle": oracle_test,
            "analytics": decisions["Analytics-only"],
            "kbsdi": decisions["KB-SDI"],
            "grid": oracle_grid(data),
        }
    return report


def _mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "sd": float("nan")}
    spread = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "sd": spread}


def summarize(rows: list[dict]) -> dict:
    summary: dict = {"n_replications": len(rows)}
    for system in SYSTEMS:
        summary[system] = {}
        for key in (*METRIC_KEYS, "rmse"):
            summary[system][key] = _mean_sd([row[system][key] for row in rows])
    for system in PREDICTORS:
        summary[system]["slice_predicted_performance"] = _mean_sd(
            [row["slice_predicted"][system] for row in rows]
        )
    summary["slice_observed_performance"] = _mean_sd([row["slice_observed"] for row in rows])
    summary["slice_n"] = _mean_sd([row["slice_n"] for row in rows])
    summary["interaction"] = _mean_sd([row["interaction"] for row in rows])
    summary["fit_seconds"] = _mean_sd([row["fit_seconds"] for row in rows])
    summary["wins"] = {}
    for baseline in BASELINES:
        accuracy_gap = np.array([row["KB-SDI"]["accuracy"] - row[baseline]["accuracy"] for row in rows])
        regret_gap = np.array([row["KB-SDI"]["regret"] - row[baseline]["regret"] for row in rows])
        summary["wins"][baseline] = {
            "accuracy_wins": int((accuracy_gap > 0).sum()),
            "regret_wins": int((regret_gap < 0).sum()),
            "accuracy_gap": _mean_sd(accuracy_gap.tolist()),
            "regret_gap": _mean_sd(regret_gap.tolist()),
        }
    return summary


def paired_tests(rows: list[dict]) -> list[dict]:
    tests = []
    for baseline in BASELINES:
        for metric in ("accuracy", "regret"):
            proposed = np.array([row["KB-SDI"][metric] for row in rows], dtype=float)
            reference = np.array([row[baseline][metric] for row in rows], dtype=float)
            difference = proposed - reference
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(proposed, reference, alternative="two-sided", zero_method="wilcox").pvalue)
            favors = int((difference > 0).sum()) if metric == "accuracy" else int((difference < 0).sum())
            tests.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "baseline_mean": float(reference.mean()),
                    "kbsdi_mean": float(proposed.mean()),
                    "mean_difference": float(difference.mean()),
                    "p_value": p_value,
                    "replications_favoring_kbsdi": favors,
                    "n": int(len(rows)),
                }
            )
    return tests


def robustness() -> dict:
    seeds = SEEDS[:10]
    report: dict = {"knowledge_size": {}, "reservation_value": {}, "assessment_noise": {}}
    for size in (40, 200):
        rows = [evaluate_once(seed, knowledge_size=size) for seed in seeds]
        report["knowledge_size"][str(size)] = {
            system: summarize(rows)[system]["accuracy"] for system in SYSTEMS
        }
    for hold in (0.38, 0.46):
        rows = [evaluate_once(seed, v_hold=hold) for seed in seeds]
        report["reservation_value"][f"{hold:.2f}"] = {
            system: summarize(rows)[system]["accuracy"] for system in SYSTEMS
        }
    noisy = [evaluate_once(seed, assessment_noise=0.10) for seed in seeds]
    report["assessment_noise"]["0.10"] = {
        system: {
            "accuracy": summarize(noisy)[system]["accuracy"],
            "false_scale_given_redesign": summarize(noisy)[system]["false_scale_given_redesign"],
        }
        for system in SYSTEMS
    }
    return report


def draw_decision_map(detail: dict, path: Path) -> None:
    capability, business_model, grid = detail["grid"]
    extent = [capability[0], capability[-1], business_model[0], business_model[-1]]
    cmap = ListedColormap(REGION_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    panels = (
        (axes[0], detail["analytics"], "(a) Analytics only"),
        (axes[1], detail["kbsdi"], "(b) KB-SDI"),
    )
    for ax, chosen, title in panels:
        ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
        false_scale = (chosen == 1) & (detail["oracle"] == 2)
        ax.scatter(
            detail["c"][false_scale],
            detail["bmc"][false_scale],
            s=9,
            c=ERROR_COLOR,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Capability index", fontsize=8)
        ax.set_title(f"{title}   false SCALE n = {int(false_scale.sum())}", loc="left", fontsize=9, color="#1C1C1C")
        for spine in ax.spines.values():
            spine.set_color("#1C1C1C")
            spine.set_linewidth(0.6)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Business-model change", fontsize=8)
    legend = [
        Patch(facecolor=REGION_COLORS[0], edgecolor="#B0B0B0", label="Oracle HOLD"),
        Patch(facecolor=REGION_COLORS[1], edgecolor="#B0B0B0", label="Oracle SCALE"),
        Patch(facecolor=REGION_COLORS[2], edgecolor="#B0B0B0", label="Oracle REDESIGN"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ERROR_COLOR, markersize=5, label="False SCALE"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cell(stat: dict, digits: int = 3) -> str:
    if stat["mean"] != stat["mean"]:
        return "—"
    return f"{stat['mean']:.{digits}f} ({stat['sd']:.{digits}f})"


def markdown_tables(summary: dict, tests: list[dict]) -> str:
    lines = [
        "| System | Accuracy | Macro-F1 | Regret | RMSE | False SCALE | REDESIGN recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        rmse = summary[system]["rmse"]
        rmse_text = "—" if rmse["mean"] != rmse["mean"] else _cell(rmse)
        lines.append(
            "| {name} | {acc} | {f1} | {reg} | {rmse} | {fs} | {rec} |".format(
                name=system,
                acc=_cell(summary[system]["accuracy"]),
                f1=_cell(summary[system]["macro_f1"]),
                reg=_cell(summary[system]["regret"]),
                rmse=rmse_text,
                fs=_cell(summary[system]["false_scale_given_redesign"]),
                rec=_cell(summary[system]["redesign_recall"]),
            )
        )
    lines.append("")
    lines.append("| Comparison | Metric | Baseline | KB-SDI | Difference | p | Favoring |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for test in tests:
        lines.append(
            "| KB-SDI vs {baseline} | {metric} | {base:.3f} | {kb:.3f} | {diff:.3f} | {p:.3e} | {wins}/{n} |".format(
                baseline=test["baseline"],
                metric=test["metric"],
                base=test["baseline_mean"],
                kb=test["kbsdi_mean"],
                diff=test["mean_difference"],
                p=test["p_value"],
                wins=test["replications_favoring_kbsdi"],
                n=test["n"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    rows = []
    detail = None
    for seed in SEEDS:
        row = evaluate_once(seed, return_detail=(seed == PRIMARY_SEED))
        if seed == PRIMARY_SEED:
            detail = row.pop("detail")
        rows.append(row)
    summary = summarize(rows)
    tests = paired_tests(rows)
    checks = robustness()
    payload = {
        "design": {
            "n_firms": N_FIRMS,
            "test_size": TEST_SIZE,
            "knowledge_size": KNOWLEDGE_SIZE,
            "replications": len(SEEDS),
            "seeds": SEEDS,
            "primary_seed": PRIMARY_SEED,
            "bmc_target": BMC_TARGET,
            "v_hold": V_HOLD,
            "cost_scale": COST_SCALE,
            "cost_redesign": COST_REDESIGN,
            "capability_floor": CAPABILITY_FLOOR,
            "slice_bmc_at_most": SLICE_BMC,
        },
        "summary": summary,
        "paired_tests": tests,
        "robustness": checks,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if detail is None:
        raise RuntimeError("Primary replication did not return the decision map.")
    draw_decision_map(detail, ROOT / "fig1_decision_map.png")
    print(markdown_tables(summary, tests))
    print("SLICE", json.dumps({
        "observed": summary["slice_observed_performance"],
        "n": summary["slice_n"],
        "predicted": {system: summary[system]["slice_predicted_performance"] for system in PREDICTORS},
        "false_scale": {system: summary[system]["slice_false_scale"] for system in SYSTEMS},
        "recall": {system: summary[system]["slice_redesign_recall"] for system in SYSTEMS},
    }))
    print("INTERACTION", json.dumps(summary["interaction"]))
    print("FIT", json.dumps(summary["fit_seconds"]))
    print("ROBUST", json.dumps(checks))


if __name__ == "__main__":
    main()
