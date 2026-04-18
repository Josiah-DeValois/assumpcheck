from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

VALID_STATUSES = ("PASS", "FAIL", "WARN", "INFO")
Plotter = Callable[[], None]


@dataclass
class AssumptionCheck:
    """Structured result for one diagnostic or assumption check."""

    assumption: str
    status: str
    metric: str | None = None
    threshold: str | None = None
    interpretation: str = ""
    mitigation: list[str] = field(default_factory=list)
    visual: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    category: str = "assumption"
    plotter: Plotter | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption": self.assumption,
            "status": self.status,
            "metric": self.metric,
            "threshold": self.threshold,
            "interpretation": self.interpretation,
            "mitigation": list(self.mitigation),
            "visual": self.visual,
            "details": dict(self.details),
            "category": self.category,
        }


@dataclass
class AssumptionReport:
    """Collection of assumption checks and optional diagnostics for one model."""

    model_type: str
    checks: list[AssumptionCheck]
    diagnostics: list[AssumptionCheck] = field(default_factory=list)
    title: str | None = None

    def summary(self) -> dict[str, int]:
        counts = {status.lower(): 0 for status in VALID_STATUSES}
        for check in self.checks:
            counts[check.status.lower()] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "summary": self.summary(),
            "checks": [check.to_dict() for check in self.checks],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }

    def iter_plots(self, *, show_all: bool = False, plots_on_fail: bool = True) -> list[Plotter]:
        plotters: list[Plotter] = []
        for check in [*self.checks, *self.diagnostics]:
            if check.plotter is None:
                continue
            if show_all or (plots_on_fail and check.status in {"FAIL", "WARN"}):
                plotters.append(check.plotter)
        return plotters
