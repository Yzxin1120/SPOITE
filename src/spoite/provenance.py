"""Result-schema validation required before any row is written."""

from __future__ import annotations

REQUIRED_RESULT_FIELDS = {
    "experiment", "config_id", "git_commit", "dataset", "regime", "seed",
    "split_id", "nuisance", "method", "method_config", "decision_config",
    "metric_config", "runtime_seconds_scenario", "failure_status",
}


def validate_result_rows(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        missing = REQUIRED_RESULT_FIELDS - row.keys()
        if missing:
            raise ValueError(f"result row {i} missing provenance fields: {sorted(missing)}")
        if not row["git_commit"] or row["git_commit"].startswith("unavailable"):
            raise ValueError("final result generation requires a Git commit")
