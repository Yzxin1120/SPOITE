import pytest

from spoite.provenance import REQUIRED_RESULT_FIELDS, validate_result_rows


def test_provenance_schema_rejects_missing_or_uncommitted_rows():
    row = {field: "x" for field in REQUIRED_RESULT_FIELDS}
    row["git_commit"] = "abc123"
    validate_result_rows([row])
    broken = dict(row)
    broken.pop("metric_config")
    with pytest.raises(ValueError, match="metric_config"):
        validate_result_rows([broken])
    broken = dict(row, git_commit="unavailable-no-project-git-repository")
    with pytest.raises(ValueError, match="Git commit"):
        validate_result_rows([broken])
