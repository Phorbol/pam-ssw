import json

from research.ga_ssw.analyze_ls_pool_routing_panel import load_arm, render_report, shared_prefix


def test_partial_arm_report_keeps_unknown_fresh_denominator_without_crashing():
    analysis = {
        "arms": [{
            "case": "C4H6", "mode": "mc", "status": "missing_summary",
            "budget_censor": "unknown", "outer_records": None,
            "landing_events": None, "converged_landing_events": None,
            "nonconverged_landing_events": None,
            "outer_records_without_landing": None,
            "fresh_certified": None, "fresh_failed_or_uncertified": None,
            "fresh_missing": None, "search_requests": None,
            "search_requests_lower_bound": 15, "fresh_requests": None,
            "fresh_requests_lower_bound": 2, "issues": ["summary missing"],
        }],
        "cases": {"C4H6": {"common_search_prefix_requests": None,
                            "excluded_from_exact_prefix": ["mc"],
                            "prefix_readouts": {}}},
    }
    report = render_report(analysis)
    assert "unknown" in report
    assert "15 lower bound" in report
    assert "Common search prefix: None" in report


def test_mismatched_arm_is_excluded_from_exact_common_prefix():
    rows = [
        {"mode": "mc", "scientific_prefix_eligible": True, "search_requests": 100},
        {"mode": "uniform", "scientific_prefix_eligible": False, "search_requests": 99},
        {"mode": "pam", "scientific_prefix_eligible": True, "search_requests": 95},
    ]
    assert shared_prefix(rows, ("mc", "uniform", "pam")) is None


def test_closed_arms_use_minimum_paid_search_total_as_shared_prefix():
    rows = [
        {"mode": "mc", "scientific_prefix_eligible": True, "search_requests": 100},
        {"mode": "uniform", "scientific_prefix_eligible": True, "search_requests": 99},
        {"mode": "pam", "scientific_prefix_eligible": True, "search_requests": 95},
    ]
    assert shared_prefix(rows, ("mc", "uniform", "pam")) == 95


def test_missing_mc_identity_archive_is_optional_diagnostic(tmp_path):
    arm = tmp_path / "mc"
    arm.mkdir()

    def write(name, value):
        (arm / name).write_text(json.dumps(value) + "\n")

    result = {
        "initial": {"evaluation_requests": 1, "converged": True},
        "records": [{"index": 0, "evaluation_requests": 1,
                     "status": "gaussian_limit", "landing": None,
                     "accepted": False}],
        "minima": [{}], "evaluation_requests": 2,
    }
    write("summary.json", {
        "status": "completed", "algorithm_status": "completed",
        "budget_censor": False, "search_requests": 2, "fresh_requests": 1,
        "total_requests": 3, "outer_attempt_records": 1,
        "returned_landings": 0, "fresh_candidate_count": 1,
        "fresh_checks_recorded": 1, "fresh_qualified_count": 1,
    })
    write("search-result.json", {"status": "completed", "search_requests": 2,
                                  "result": result})
    write("fresh-checks.json", {
        "candidate_count": 1, "qualified_count": 1, "fresh_requests": 1,
        "checks": [{"candidate_index": 0, "source": "initial",
                    "record_index": -1, "minimum_index": 0,
                    "converged": True, "accepted": True,
                    "status": "fresh_completed", "certified": True}],
    })
    write("selector-contract.json", {"mode": "mc"})
    for filename, stage, requests in (
        ("search-ledger.jsonl", "search", 2),
        ("fresh-ledger.jsonl", "fresh", 1),
    ):
        with (arm / filename).open("w") as stream:
            for attempt in range(1, requests + 1):
                stream.write(json.dumps({"event": "attempt_started", "stage": stage,
                                         "attempt": attempt}) + "\n")
                stream.write(json.dumps({"event": "attempt_completed", "stage": stage,
                                         "attempt": attempt, "request": attempt,
                                         "charged": True}) + "\n")

    row = load_arm(arm, {"name": "C4H6"}, "mc", {"steps": 1})

    assert row["scientific_prefix_eligible"] is True
    assert row["offline_identity_diagnostic"] == "missing_optional"
    assert row["pool_report_entries"] is None
    assert not any("offline-identity.json" in issue for issue in row["issues"])
