import json

from research.ga_ssw.audit_vc_mature_baselines import audit_arm


def _write_case(tmp_path, *, reported=3, landings=None, checks=None, records=None, evals=None):
    landings = [{"index": -1}] if landings is None else landings
    checks = [{"index": i, "certified": True} for i in range(len(landings))] if checks is None else checks
    records = [{"stage": "initial", "certificate": {"certified": True}}] if records is None else records
    (tmp_path / "result.json").write_text(json.dumps({
        "requests": reported, "search_requests": reported - 1,
        "records": records, "landings": landings,
        "fresh": {"requested": len(checks), "checked": len(checks), "checks": checks},
    }))
    evals = [{"stage": "search", "charged": True}] if evals is None else evals
    (tmp_path / "evaluations.jsonl").write_text("\n".join(json.dumps(x) for x in evals) + "\n")
    return audit_arm(tmp_path)


def test_initial_landing_index_minus_one_is_not_new(tmp_path):
    result = _write_case(tmp_path, reported=1)
    assert result["new_landing_count"] == 0
    assert result["new_landing_certified"] == 0


def test_denied_attempt_is_recorded_but_not_paid(tmp_path):
    result = _write_case(
        tmp_path, reported=1,
        records=[{"stage": "initial", "certificate": {"certified": True}},
                 {"index": 0, "status": "budget_denied", "requests": 0}],
        evals=[{"stage": "search", "charged": True},
               {"stage": "search", "charged": False}],
    )
    assert result["record_attempts"] == 1
    assert result["paid_attempts"] == 0


def test_certified_new_landing_is_counted(tmp_path):
    result = _write_case(tmp_path, reported=2,
                         landings=[{"index": -1}, {"index": 0}],
                         checks=[{"index": 0, "certified": True},
                                 {"index": 1, "certified": True}],
                         evals=[{"stage": "search", "charged": True},
                                {"stage": "fresh", "charged": True}])
    assert result["new_landing_count"] == 1
    assert result["new_landing_certified"] == 1


def test_reported_request_mismatch_is_consistency_error(tmp_path):
    result = _write_case(tmp_path, reported=9)
    assert "reported_requests_eq_charged" in result["consistency"]["errors"]
