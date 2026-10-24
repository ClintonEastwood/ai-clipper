import json

import gap_analysis
import validate_results


def test_validation_summary_with_no_segments(tmp_path, capsys):
    # Every analysis file empty: the removal rate divided by zero.
    (tmp_path / "a_analysis.json").write_text(json.dumps({"segments": []}))
    validate_results.validate_directory(str(tmp_path), str(tmp_path / "out"))
    assert "Total original segments: 0" in capsys.readouterr().out


def test_gap_export_records_the_flags_used(tmp_path):
    out = tmp_path / "c.json"
    gap_analysis.export_candidates([], str(out), max_gap=12.0, max_duration=60.0)
    assert json.loads(out.read_text())["analysis_params"] == {"max_gap": 12.0, "max_duration": 60.0}
