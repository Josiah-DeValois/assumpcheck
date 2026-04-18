from __future__ import annotations

from typing import Iterable

from .types import AssumptionCheck, AssumptionReport


def print_report(
    report: AssumptionReport,
    *,
    verbose: bool = False,
    show_all: bool = False,
) -> None:
    title = report.title or f"{report.model_type.replace('_', ' ').upper()} ASSUMPTION CHECKS"
    print(title)
    for check in report.checks:
        print(f"[{check.status}] {check.assumption}")

    counts = report.summary()
    summary_parts = []
    for key in ("pass", "fail", "warn", "info"):
        value = counts[key]
        if value:
            summary_parts.append(f"{value} {key}")
    print()
    print(f"Summary: {', '.join(summary_parts)}")

    detailed_checks = [
        check
        for check in report.checks
        if show_all or verbose or check.status in {"FAIL", "WARN"}
    ]
    if detailed_checks:
        print()
        print("Details:")
        for check in detailed_checks:
            _print_detail(check)

    if report.diagnostics and (show_all or verbose):
        print()
        print("Diagnostics:")
        for diagnostic in report.diagnostics:
            _print_detail(diagnostic)


def _print_detail(check: AssumptionCheck) -> None:
    print(f"- {check.assumption} [{check.status}]")
    if check.metric:
        print(f"  Metric: {check.metric}")
    if check.threshold:
        print(f"  Threshold: {check.threshold}")
    if check.interpretation:
        print(f"  Interpretation: {check.interpretation}")
    if check.visual:
        print(f"  Visual: {check.visual}")
    if check.mitigation:
        print("  Possible mitigation:")
        for item in check.mitigation:
            print(f"    - {item}")


def maybe_show_plots(
    report: AssumptionReport,
    *,
    show_all: bool = False,
    plots_on_fail: bool = True,
) -> None:
    for plotter in report.iter_plots(show_all=show_all, plots_on_fail=plots_on_fail):
        plotter()


def counts_as_text(counts: dict[str, int]) -> str:
    ordered: Iterable[tuple[str, int]] = (
        ("pass", counts.get("pass", 0)),
        ("fail", counts.get("fail", 0)),
        ("warn", counts.get("warn", 0)),
        ("info", counts.get("info", 0)),
    )
    return ", ".join(f"{value} {name}" for name, value in ordered if value)


def finalize_report(
    report: AssumptionReport,
    *,
    verbose: bool = False,
    show_all: bool = False,
    plots_on_fail: bool = True,
    return_dict: bool = False,
):
    print_report(report, verbose=verbose, show_all=show_all)
    maybe_show_plots(report, show_all=show_all, plots_on_fail=plots_on_fail)
    return report.to_dict() if return_dict else report
