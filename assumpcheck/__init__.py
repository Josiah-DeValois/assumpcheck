"""Simple assumption checks for common statistical models."""

from .anova import check_anova
from .linear_regression import check_linear_regression
from .logistic_regression import check_logistic_regression
from .types import AssumptionCheck, AssumptionReport

__all__ = [
    "AssumptionCheck",
    "AssumptionReport",
    "check_anova",
    "check_linear_regression",
    "check_logistic_regression",
]

