from __future__ import annotations

from typing import Any

from .utils import require_core_dependencies


def qq_plot(values: Any, *, title: str) -> None:
    deps = require_core_dependencies()
    plt = deps["plt"]
    sm = deps["sm"]
    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    sm.qqplot(values, line="45", ax=axis, fit=True)
    axis.set_title(title)
    figure.tight_layout()
    plt.show()


def boxplot_by_group(values: Any, groups: Any, *, title: str, y_label: str) -> None:
    deps = require_core_dependencies()
    np = deps["np"]
    plt = deps["plt"]
    groups = np.asarray(groups)
    values = np.asarray(values)
    unique_groups = np.unique(groups)
    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    axis.boxplot([values[groups == group] for group in unique_groups], labels=[str(group) for group in unique_groups])
    axis.set_title(title)
    axis.set_xlabel("Group")
    axis.set_ylabel(y_label)
    figure.tight_layout()
    plt.show()


def residuals_vs_fitted_plot(fitted: Any, residuals: Any, *, title: str) -> None:
    deps = require_core_dependencies()
    plt = deps["plt"]
    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    axis.scatter(fitted, residuals, alpha=0.75)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Fitted values")
    axis.set_ylabel("Residuals")
    figure.tight_layout()
    plt.show()


def residuals_vs_leverage_plot(
    leverage: Any,
    studentized_residuals: Any,
    cooks_distance: Any,
    *,
    title: str,
) -> None:
    deps = require_core_dependencies()
    plt = deps["plt"]
    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    scatter = axis.scatter(leverage, studentized_residuals, c=cooks_distance, cmap="viridis", alpha=0.8)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Leverage")
    axis.set_ylabel("Studentized residuals")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Cook's distance")
    figure.tight_layout()
    plt.show()


def binned_logit_plot(values: Any, outcome: Any, *, title: str, bins: int = 10) -> None:
    deps = require_core_dependencies()
    np = deps["np"]
    plt = deps["plt"]
    values = np.asarray(values, dtype=float)
    outcome = np.asarray(outcome, dtype=int)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 3:
        return
    centers = []
    logits = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (values >= lower) & (values <= upper if upper == edges[-1] else values < upper)
        if mask.sum() == 0:
            continue
        rate = outcome[mask].mean()
        rate = min(max(rate, 1e-4), 1 - 1e-4)
        centers.append(values[mask].mean())
        logits.append(np.log(rate / (1 - rate)))

    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    axis.plot(centers, logits, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Predictor")
    axis.set_ylabel("Observed log-odds")
    figure.tight_layout()
    plt.show()


def roc_curve_plot(fpr: Any, tpr: Any, auc: float | None, *, title: str) -> None:
    deps = require_core_dependencies()
    plt = deps["plt"]
    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111)
    axis.plot(fpr, tpr, label=f"AUC = {auc:.3f}" if auc is not None else "AUC unavailable")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    plt.show()
