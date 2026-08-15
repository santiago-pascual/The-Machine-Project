from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

KEYWORDS = (
    "threshold",
    "multiplier",
    "penalty",
    "confidence",
    "weight",
    "clip",
    "lambda",
    "alpha",
    "rate",
    "max_",
    "min_",
    "lookback",
    "window",
    "horizon",
    "generation",
    "population",
    "elite",
    "mutation",
)


def _classification(name: str, value: object, line: str) -> str:
    text = f"{name} {line}".lower()
    if any(token in text for token in ("252", "trading_days", "annual", "sqrt(252)")):
        return "statistically justified"
    if any(token in text for token in ("ledoit", "garch", "egarch", "hurst", "kalman", "ou_", "entropy", "hawkes")):
        return "statistically justified"
    if any(token in text for token in ("threshold", "penalty", "multiplier", "clip", "boost", "fallback", "guard")):
        return "heuristic"
    return "arbitrary"


def _recommendation(classification: str, name: str) -> str:
    if classification == "statistically justified":
        return "Keep, monitor stability."
    if classification == "heuristic":
        return "Calibrate with forecast history / walk-forward validation."
    return "Document rationale or replace with data-driven calibration."


def audit_file(path: Path) -> list[dict[str, object]]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    rows: list[dict[str, object]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return rows

    for node in ast.walk(tree):
        name = ""
        value: object | None = None
        lineno = getattr(node, "lineno", None)

        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not targets:
                continue
            name = targets[0]
            value = node.value.value
        elif isinstance(node, ast.arg) and node.annotation is not None:
            continue
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if lineno is None:
                continue
            line = lines[lineno - 1].strip()
            if not any(keyword in line.lower() for keyword in KEYWORDS):
                continue
            name = "literal"
            value = node.value
        else:
            continue

        if lineno is None or value is None:
            continue
        line = lines[lineno - 1].strip()
        if name != "literal" and not any(keyword in f"{name} {line}".lower() for keyword in KEYWORDS):
            continue
        classification = _classification(name, value, line)
        rows.append(
            {
                "file": path.name,
                "variable_name": name,
                "current_value": value,
                "purpose": line[:160],
                "classification": classification,
                "recommendation": _recommendation(classification, name),
            }
        )
    return rows


def build_heuristic_audit_report(root: str | Path = ".") -> pd.DataFrame:
    root_path = Path(root)
    rows: list[dict[str, object]] = []
    for path in sorted(root_path.glob("*.py")):
        if path.name.startswith("__"):
            continue
        rows.extend(audit_file(path))
    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(
            columns=[
                "file",
                "variable_name",
                "current_value",
                "purpose",
                "classification",
                "recommendation",
            ]
        )
    return report.drop_duplicates().sort_values(["classification", "file", "variable_name"])


def print_heuristic_audit_report(root: str | Path = ".") -> pd.DataFrame:
    report = build_heuristic_audit_report(root)
    print("\n===== HEURISTIC AUDIT =====")
    print(report.to_string(index=False) if not report.empty else "No constants detected.")
    return report
