from __future__ import annotations

import base64
import io
import json
import os
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assumpcheck-mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assumpcheck import check_anova, check_linear_regression, check_logistic_regression

EXAMPLES_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXAMPLES_DIR / "assets"
NOTEBOOK_PATH = EXAMPLES_DIR / "assumpcheck_examples.ipynb"


@dataclass
class ExampleCase:
    title: str
    description: str
    code: str
    runner: Callable[[], dict | None]
    image_prefix: str


@dataclass
class CapturedCase:
    case: ExampleCase
    stdout: str
    image_paths: list[Path]
    result: dict | None


def capture_case(case: ExampleCase) -> CapturedCase:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for existing in ASSETS_DIR.glob(f"{case.image_prefix}_*.png"):
        existing.unlink()

    buffer = io.StringIO()
    original_show = plt.show
    saved_paths: list[Path] = []

    def capture_show(*args, **kwargs) -> None:
        figure_numbers = list(plt.get_fignums())
        for figure_number in figure_numbers:
            figure = plt.figure(figure_number)
            path = ASSETS_DIR / f"{case.image_prefix}_{len(saved_paths) + 1}.png"
            figure.savefig(path, bbox_inches="tight")
            saved_paths.append(path)
        plt.close("all")

    plt.close("all")
    plt.show = capture_show
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buffer):
                result = case.runner()
    finally:
        plt.show = original_show
        plt.close("all")

    return CapturedCase(
        case=case,
        stdout=buffer.getvalue().rstrip(),
        image_paths=saved_paths,
        result=result,
    )


def build_cases() -> dict[str, list[ExampleCase]]:
    return {
        "ANOVA": [
            ExampleCase(
                title="Clean Example",
                description="A small balanced one-way ANOVA example where all four current checks pass.",
                image_prefix="anova_clean",
                code="""groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)

anova_clean = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", anova_clean["summary"])""",
                runner=run_anova_clean,
            ),
            ExampleCase(
                title="Problematic Example",
                description="A similar ANOVA setup with an injected outlier and extra spread in one group.",
                image_prefix="anova_fail",
                code="""groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)
y[4] = 14.50
y[24:36] = y[24:36] + np.linspace(-0.45, 0.45, 12)

anova_fail = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", anova_fail["summary"])""",
                runner=run_anova_fail,
            ),
            ExampleCase(
                title="Full Diagnostics",
                description="Even when the model looks fine, you can request every available metric and plot.",
                image_prefix="anova_show_all",
                code="""groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)

anova_full = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", anova_full["summary"])""",
                runner=run_anova_full,
            ),
        ],
        "Linear Regression": [
            ExampleCase(
                title="Clean Example",
                description="A well-behaved synthetic linear model. Under the current MVP, the influence check still gives a mild warning.",
                image_prefix="linear_clean",
                code="""rng = np.random.default_rng(27)
X = pd.DataFrame({
    "x1": rng.normal(size=35),
    "x2": rng.normal(size=35),
})
y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)

linear_clean = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", linear_clean["summary"])""",
                runner=run_linear_clean,
            ),
            ExampleCase(
                title="Problematic Example",
                description="A nonlinear signal with non-constant variance to trigger multiple diagnostics.",
                image_prefix="linear_fail",
                code="""rng = np.random.default_rng(7)
x = np.linspace(-2.5, 2.5, 90)
X = pd.DataFrame({"x1": x})
noise = rng.normal(scale=0.20 + 0.20 * np.abs(x), size=x.size)
y = 1.0 + 1.8 * x + 0.9 * x**2 + noise

linear_fail = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", linear_fail["summary"])""",
                runner=run_linear_fail,
            ),
            ExampleCase(
                title="Full Diagnostics",
                description="Request all detail and all plots for the cleaner linear example.",
                image_prefix="linear_show_all",
                code="""rng = np.random.default_rng(27)
X = pd.DataFrame({
    "x1": rng.normal(size=35),
    "x2": rng.normal(size=35),
})
y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)

linear_full = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", linear_full["summary"])""",
                runner=run_linear_full,
            ),
        ],
        "Logistic Regression": [
            ExampleCase(
                title="Clean Example",
                description="A stable logistic model. The current influence heuristic still tends to emit a mild warning.",
                image_prefix="logistic_clean",
                code="""rng = np.random.default_rng(44)
X = pd.DataFrame({
    "x1": rng.normal(size=45),
    "x2": rng.normal(size=45),
})
logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
p = 1 / (1 + np.exp(-logits))
y = rng.binomial(1, p)

logistic_clean = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", logistic_clean["summary"])
print("Model fit metric:", logistic_clean["diagnostics"][0]["metric"])""",
                runner=run_logistic_clean,
            ),
            ExampleCase(
                title="Problematic Example",
                description="A near-perfect separation setup that should clearly fail the stability check.",
                image_prefix="logistic_fail",
                code="""x = np.linspace(-2.5, 2.5, 80)
X = pd.DataFrame({
    "x1": x,
    "x2": np.sin(x),
})
y = (x > 0.0).astype(int)

logistic_fail = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", logistic_fail["summary"])""",
                runner=run_logistic_fail,
            ),
            ExampleCase(
                title="Full Diagnostics",
                description="Request the full logistic diagnostic view, including the ROC curve.",
                image_prefix="logistic_show_all",
                code="""rng = np.random.default_rng(44)
X = pd.DataFrame({
    "x1": rng.normal(size=45),
    "x2": rng.normal(size=45),
})
logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
p = 1 / (1 + np.exp(-logits))
y = rng.binomial(1, p)

logistic_full = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", logistic_full["summary"])
print("Model fit metric:", logistic_full["diagnostics"][0]["metric"])""",
                runner=run_logistic_full,
            ),
        ],
    }


def run_anova_clean() -> dict:
    groups = np.repeat(["A", "B", "C"], 12)
    base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
    y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)
    result = check_anova(
        y=y,
        groups=groups,
        design_independent=True,
        plots_on_fail=False,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_anova_fail() -> dict:
    groups = np.repeat(["A", "B", "C"], 12)
    base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
    y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)
    y[4] = 14.50
    y[24:36] = y[24:36] + np.linspace(-0.45, 0.45, 12)
    result = check_anova(
        y=y,
        groups=groups,
        design_independent=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_anova_full() -> dict:
    groups = np.repeat(["A", "B", "C"], 12)
    base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
    y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)
    result = check_anova(
        y=y,
        groups=groups,
        design_independent=True,
        show_all=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_linear_clean() -> dict:
    rng = np.random.default_rng(27)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=35),
            "x2": rng.normal(size=35),
        }
    )
    y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)
    result = check_linear_regression(
        X=X,
        y=y,
        design_independent=True,
        plots_on_fail=False,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_linear_fail() -> dict:
    rng = np.random.default_rng(7)
    x = np.linspace(-2.5, 2.5, 90)
    X = pd.DataFrame({"x1": x})
    noise = rng.normal(scale=0.20 + 0.20 * np.abs(x), size=x.size)
    y = 1.0 + 1.8 * x + 0.9 * x**2 + noise
    result = check_linear_regression(
        X=X,
        y=y,
        design_independent=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_linear_full() -> dict:
    rng = np.random.default_rng(27)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=35),
            "x2": rng.normal(size=35),
        }
    )
    y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)
    result = check_linear_regression(
        X=X,
        y=y,
        design_independent=True,
        show_all=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_logistic_clean() -> dict:
    rng = np.random.default_rng(44)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=45),
            "x2": rng.normal(size=45),
        }
    )
    logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    result = check_logistic_regression(
        X=X,
        y=y,
        design_independent=True,
        plots_on_fail=False,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    print("Model fit metric:", result["diagnostics"][0]["metric"])
    return result


def run_logistic_fail() -> dict:
    x = np.linspace(-2.5, 2.5, 80)
    X = pd.DataFrame(
        {
            "x1": x,
            "x2": np.sin(x),
        }
    )
    y = (x > 0.0).astype(int)
    result = check_logistic_regression(
        X=X,
        y=y,
        design_independent=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    return result


def run_logistic_full() -> dict:
    rng = np.random.default_rng(44)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=45),
            "x2": rng.normal(size=45),
        }
    )
    logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    result = check_logistic_regression(
        X=X,
        y=y,
        design_independent=True,
        show_all=True,
        return_dict=True,
    )
    print()
    print("Returned summary:", result["summary"])
    print("Model fit metric:", result["diagnostics"][0]["metric"])
    return result


def split_lines(text: str) -> list[str]:
    return [f"{line}\n" for line in text.splitlines()]


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": split_lines(text),
    }


def make_code_cell(source: str, stdout: str, image_paths: list[Path], execution_count: int) -> dict:
    outputs = []
    if stdout:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": stdout + "\n",
            }
        )
    for image_path in image_paths:
        outputs.append(
            {
                "output_type": "display_data",
                "data": {
                    "image/png": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                    "text/plain": [f"<Image: {image_path.name}>"],
                },
                "metadata": {},
            }
        )
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": split_lines(source),
    }


def generate_notebook(cases_by_section: dict[str, list[CapturedCase]]) -> dict:
    cells = [
        make_markdown_cell(
            "# assumpcheck example notebook\n\n"
            "This notebook shows an initial workflow for `check_anova(...)`, "
            "`check_linear_regression(...)`, and `check_logistic_regression(...)`."
        ),
        make_markdown_cell(
            "> Note\n>\n> The current MVP uses conservative influence heuristics. "
            "In the clean linear and logistic examples, the package still emits a mild "
            "`WARN` for influential points even when the rest of the diagnostics look reasonable."
        ),
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": split_lines(
                "%matplotlib inline\n\n"
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "from assumpcheck import (\n"
                "    check_anova,\n"
                "    check_linear_regression,\n"
                "    check_logistic_regression,\n"
                ")\n"
            ),
        },
    ]

    execution_count = 2
    for section, captured_cases in cases_by_section.items():
        cells.append(make_markdown_cell(f"## {section}"))
        for captured in captured_cases:
            source = f"# {captured.case.title}\n# {captured.case.description}\n\n{captured.case.code}"
            cells.append(make_code_cell(source, captured.stdout, captured.image_paths, execution_count))
            execution_count += 1

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    cases = build_cases()
    captured = {
        section: [capture_case(case) for case in section_cases]
        for section, section_cases in cases.items()
    }
    NOTEBOOK_PATH.write_text(json.dumps(generate_notebook(captured), indent=1) + "\n")
    print(f"Wrote {NOTEBOOK_PATH.relative_to(EXAMPLES_DIR.parent)}")
    print(f"Wrote assets to {ASSETS_DIR.relative_to(EXAMPLES_DIR.parent)}")


if __name__ == "__main__":
    main()
