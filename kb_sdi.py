"""
KB-SDI: a knowledge-based AI system for strategic decision intelligence.

The operational feature store records data-analytics and innovation signals.
It does not record business-model change. A small knowledge sample estimates
the capability x business-model-change mechanism. At decision time the two
are fused, and the firm is assigned HOLD, SCALE, or REDESIGN.

The firms in this study are synthetic. The data-generating process is
implemented in `signal` and is not available to the decision systems.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
SLICE_C = 0.55
SLICE_BMC = 0.40
SEEDS = list(range(802, 822))

ACTIONS = ("HOLD", "SCALE", "REDESIGN")
SYSTEMS = ("Rules-only", "Analytics-only", "KB-SDI")
COLORS = {"Rules-only": "#8C6A4A", "Analytics-only": "#C47B2B", "KB-SDI": "#1F4E79"}

FEATURE_NAMES = ("dac", "aic", "dti", "inn", "eo", "dq", "size", "age")


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
    # Business-model change is a strategic assessment. It is generated
    # independently of the operational features so the warehouse cannot proxy it.
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
    """Expected performance. `business_model` overrides the observed assessment."""
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
        + 0.02 * np.clip(data["size"], -2.0, 2.0) / 2.0
    )


def payoffs(current: np.ndarray, counterfactual: np.ndarray, v_hold: float = V_HOLD) -> np.ndarray:
    hold = np.full(current.shape, v_hold)
    scale = current - COST_SCALE
    redesign = counterfactual - COST_REDESIGN
    return np.column_stack([hold, scale, redesign])


def estimate_mechanism(capability: np.ndarray, business_model: np.ndarray, performance: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(capability)), capability, business_model, capability * business_model]
    )
    coefficients, *_ = np.linalg.lstsq(design, performance, rcond=None)
    return coefficients


def mechanism_prediction(coefficients: np.ndarray, capability: np.ndarray, business_model: np.ndarray) -> np.ndarray:
    intercept, capability_coef, bmc_coef, interaction = coefficients
    return intercept + capability_coef * capability + bmc_coef * business_model + interaction * capability * business_model


def marginal_bmc_effect(coefficients: np.ndarray, capability: np.ndarray) -> np.ndarray:
    _, _, bmc_coef, interaction = coefficients
    return bmc_coef + interaction * capability


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


def metrics_for(predicted: np.ndarray, oracle: np.ndarray, true_value: np.ndarray, slice_mask: np.ndarray) -> dict[str, float]:
    best = true_value.max(axis=1)
    chosen = true_value[np.arange(len(predicted)), predicted]
    redesign = oracle == 2
    slice_redesign = slice_mask & redesign
    return {
        "accuracy": float(accuracy_score(oracle, predicted)),
        "macro_f1": float(f1_score(oracle, predicted, average="macro", zero_division=0)),
        "regret": float((best - chosen).mean()),
        "false_scale_given_redesign": float(((predicted == 1) & redesign).sum() / max(redesign.sum(), 1)),
        "redesign_recall": float(((predicted == 2) & redesign).sum() / max(redesign.sum(), 1)),
        "slice_redesign_recall": float(((predicted == 2) & slice_redesign).sum() / max(slice_redesign.sum(), 1)),
        "slice_false_scale": float(((predicted == 1) & slice_redesign).sum() / max(slice_redesign.sum(), 1)),
        "slice_regret": float((best[slice_mask] - chosen[slice_mask]).mean()) if slice_mask.any() else float("nan"),
        "slice_n": int(slice_mask.sum()),
    }


def evaluate_once(seed: int, knowledge_size: int = KNOWLEDGE_SIZE, v_hold: float = V_HOLD) -> dict:
    data = simulate(N_FIRMS, seed)
    noise = np.random.default_rng(seed + 17).normal(0.0, 0.04, N_FIRMS)
    expected = signal(data)
    redesigned = np.maximum(data["bmc"], BMC_TARGET)
    expected_redesign = signal(data, redesigned)
    observed = np.clip(expected + noise, 0.0, 1.0)
    oracle = payoffs(expected, expected_redesign, v_hold).argmax(axis=1)

    index = np.arange(N_FIRMS)
    train, test = train_test_split(index, test_size=TEST_SIZE, random_state=seed, stratify=oracle)
    knowledge, operational = train[:knowledge_size], train[knowledge_size:]
    features = np.column_stack([data[name] for name in FEATURE_NAMES])

    analytics = fit_analytics(features[operational], observed[operational], seed)
    predicted_now = analytics.predict(features[test])
    coefficients = estimate_mechanism(data["c"][knowledge], data["bmc"][knowledge], observed[knowledge])
    effect = marginal_bmc_effect(coefficients, data["c"][test])
    baseline_bmc = float(data["bmc"][knowledge].mean())

    knowledge_current = mechanism_prediction(coefficients, data["c"][test], data["bmc"][test])
    knowledge_counterfactual = mechanism_prediction(coefficients, data["c"][test], redesigned[test])
    fused_current = np.clip(predicted_now + effect * (data["bmc"][test] - baseline_bmc), 0.0, 1.0)
    fused_counterfactual = np.clip(predicted_now + effect * (redesigned[test] - baseline_bmc), 0.0, 1.0)

    systems = {
        "Rules-only": payoffs(knowledge_current, knowledge_counterfactual, v_hold),
        "Analytics-only": payoffs(predicted_now, predicted_now, v_hold),
        "KB-SDI": payoffs(fused_current, fused_counterfactual, v_hold),
    }
    truth = payoffs(expected[test], expected_redesign[test], v_hold)
    high_capability_low_bmc = (data["c"][test] >= SLICE_C) & (data["bmc"][test] <= SLICE_BMC)
    oracle_test = oracle[test]

    report = {
        "seed": seed,
        "interaction": float(coefficients[3]),
        "rmse": {
            "Rules-only": float(mean_squared_error(observed[test], knowledge_current) ** 0.5),
            "Analytics-only": float(mean_squared_error(observed[test], predicted_now) ** 0.5),
            "KB-SDI": float(mean_squared_error(observed[test], fused_current) ** 0.5),
        },
        "slice_predicted_performance": {
            "Analytics-only": float(predicted_now[high_capability_low_bmc].mean()),
            "KB-SDI": float(fused_current[high_capability_low_bmc].mean()),
            "observed": float(observed[test][high_capability_low_bmc].mean()),
        },
    }
    for name, value in systems.items():
        report[name] = metrics_for(value.argmax(axis=1), oracle_test, truth, high_capability_low_bmc)
        report[name]["rmse"] = report["rmse"][name]
    return report


def _mean_sd(rows: list[dict], system: str, key: str) -> tuple[float, float]:
    values = np.array([row[system][key] for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def summarize(rows: list[dict]) -> dict:
    summary = {"n_replications": len(rows), "wins": {}}
    for system in SYSTEMS:
        summary[system] = {}
        for key in (
            "accuracy",
            "macro_f1",
            "regret",
            "false_scale_given_redesign",
            "redesign_recall",
            "slice_redesign_recall",
            "slice_false_scale",
            "slice_regret",
            "rmse",
        ):
            mean, sd = _mean_sd(rows, system, key)
            summary[system][key] = {"mean": mean, "sd": sd}
    kb_vs_rules = np.array([row["KB-SDI"]["accuracy"] - row["Rules-only"]["accuracy"] for row in rows])
    kb_vs_analytics = np.array([row["KB-SDI"]["accuracy"] - row["Analytics-only"]["accuracy"] for row in rows])
    summary["wins"] = {
        "accuracy_beats_rules": int((kb_vs_rules > 0).sum()),
        "accuracy_beats_analytics": int((kb_vs_analytics > 0).sum()),
        "accuracy_gap_vs_rules": {"mean": float(kb_vs_rules.mean()), "sd": float(kb_vs_rules.std(ddof=1))},
        "accuracy_gap_vs_analytics": {"mean": float(kb_vs_analytics.mean()), "sd": float(kb_vs_analytics.std(ddof=1))},
    }
    interaction = np.array([row["interaction"] for row in rows])
    summary["interaction"] = {"mean": float(interaction.mean()), "sd": float(interaction.std(ddof=1))}
    for system in ("Analytics-only", "KB-SDI"):
        predicted = np.array([row["slice_predicted_performance"][system] for row in rows])
        summary[system]["slice_predicted_performance"] = {
            "mean": float(predicted.mean()),
            "sd": float(predicted.std(ddof=1)),
        }
    observed = np.array([row["slice_predicted_performance"]["observed"] for row in rows])
    summary["slice_observed_performance"] = {"mean": float(observed.mean()), "sd": float(observed.std(ddof=1))}
    summary["slice_n_mean"] = float(np.mean([row["KB-SDI"]["slice_n"] for row in rows]))
    return summary


def robustness(knowledge_sizes: tuple[int, ...] = (40, 200), holds: tuple[float, ...] = (0.38, 0.46)) -> dict:
    seeds = SEEDS[:10]
    report: dict = {"knowledge_size": {}, "reservation_value": {}}
    for size in knowledge_sizes:
        rows = [evaluate_once(seed, knowledge_size=size) for seed in seeds]
        report["knowledge_size"][str(size)] = {
            system: summarize(rows)[system]["accuracy"] for system in SYSTEMS
        }
    for hold in holds:
        rows = [evaluate_once(seed, v_hold=hold) for seed in seeds]
        report["reservation_value"][f"{hold:.2f}"] = {
            system: summarize(rows)[system]["accuracy"] for system in SYSTEMS
        }
    return report


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, face, edge="#1F2A33"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=face, edgecolor=edge, linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4, color="#1F2A33", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#1F2A33"))

    box(0.25, 2.85, 2.5, 1.25, "Operational data\nanalytics capability, AI capability,\ninnovation, entrepreneurship,\ndata quality", "#E7EEF5")
    box(0.25, 0.45, 2.5, 1.25, "Knowledge sample\ncapability and\nbusiness-model change\n(100 firms)", "#F4EBE1")
    box(3.35, 2.85, 2.7, 1.25, "Analytics model\ngradient boosting\npredicts current\nperformance", "#E7EEF5")
    box(3.35, 0.45, 2.7, 1.25, "Knowledge base\nperformance rises with\ncapability only if the\nbusiness model changes", "#F4EBE1")
    box(6.65, 1.45, 2.15, 1.65, "Decision layer\nHOLD\nSCALE\nREDESIGN", "#E5EFE8", "#1F4E79")
    box(9.05, 1.7, 1.15, 1.15, "Strategic\ndecision", "#1F4E79", "#1F4E79")
    ax.texts[-1].set_color("white")

    arrow(2.75, 3.47, 3.35, 3.47)
    arrow(2.75, 1.07, 3.35, 1.07)
    arrow(6.05, 3.47, 7.7, 2.85)
    arrow(6.05, 1.07, 7.7, 1.70)
    arrow(8.80, 2.27, 9.05, 2.27)
    ax.set_title("Knowledge-based strategic decision intelligence", loc="left", fontsize=11, color="#1F2A33", pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_results(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    fig.patch.set_facecolor("white")
    x = np.arange(len(SYSTEMS))
    panels = (
        (axes[0], "accuracy", "Decision accuracy", "Accuracy"),
        (axes[1], "false_scale_given_redesign", "False SCALE rate when REDESIGN is best", "Rate"),
    )
    for ax, key, title, ylabel in panels:
        means = [summary[system][key]["mean"] for system in SYSTEMS]
        sds = [summary[system][key]["sd"] for system in SYSTEMS]
        bars = ax.bar(
            x, means, yerr=sds, color=[COLORS[system] for system in SYSTEMS],
            width=0.72, capsize=4, error_kw={"elinewidth": 1.0, "ecolor": "#1F2A33"},
            edgecolor="white",
        )
        ax.set_xticks(x, ["Rules only", "Analytics only", "KB-SDI"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, color="#1F2A33")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#B7BFC7")
        ax.set_axisbelow(True)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, min(mean + 0.06, 0.98), f"{mean:.2f}", ha="center", va="bottom", fontsize=8, color="#1F2A33")
    fig.suptitle("Decision quality across 20 replications", fontsize=11, color="#1F2A33", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(summary: dict) -> str:
    def cell(system: str, key: str) -> str:
        stat = summary[system][key]
        return f"{stat['mean']:.3f} ({stat['sd']:.3f})"

    header = "| System | Accuracy | Macro-F1 | Regret | RMSE | False SCALE | REDESIGN recall |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for system in SYSTEMS:
        lines.append(
            "| {name} | {acc} | {f1} | {reg} | {rmse} | {fs} | {rec} |".format(
                name=system,
                acc=cell(system, "accuracy"),
                f1=cell(system, "macro_f1"),
                reg=cell(system, "regret"),
                rmse=cell(system, "rmse"),
                fs=cell(system, "false_scale_given_redesign"),
                rec=cell(system, "redesign_recall"),
            )
        )
    slice_header = "\n| System | REDESIGN recall | False SCALE | Regret | Predicted performance |\n|---|---:|---:|---:|---:|"
    slice_lines = [slice_header]
    for system in SYSTEMS:
        predicted = "—"
        if "slice_predicted_performance" in summary[system]:
            stat = summary[system]["slice_predicted_performance"]
            predicted = f"{stat['mean']:.3f} ({stat['sd']:.3f})"
        slice_lines.append(
            "| {name} | {rec} | {fs} | {reg} | {pred} |".format(
                name=system,
                rec=cell(system, "slice_redesign_recall"),
                fs=cell(system, "slice_false_scale"),
                reg=cell(system, "slice_regret"),
                pred=predicted,
            )
        )
    return "\n".join(lines) + "\n" + "\n".join(slice_lines) + "\n"


def main() -> None:
    rows = [evaluate_once(seed) for seed in SEEDS]
    summary = summarize(rows)
    checks = robustness()
    payload = {
        "design": {
            "n_firms": N_FIRMS,
            "test_size": TEST_SIZE,
            "knowledge_size": KNOWLEDGE_SIZE,
            "replications": len(SEEDS),
            "seeds": SEEDS,
            "bmc_target": BMC_TARGET,
            "v_hold": V_HOLD,
            "cost_scale": COST_SCALE,
            "cost_redesign": COST_REDESIGN,
            "slice": {"capability_at_least": SLICE_C, "business_model_at_most": SLICE_BMC},
        },
        "summary": summary,
        "robustness": checks,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_architecture(ROOT / "fig1_architecture.png")
    draw_results(summary, ROOT / "fig2_results.png")
    print(markdown_table(summary))
    print("wins", json.dumps(summary["wins"]))
    print("interaction", json.dumps(summary["interaction"]))
    print("slice_observed", json.dumps(summary["slice_observed_performance"]))
    print("robustness", json.dumps(checks))


if __name__ == "__main__":
    main()
