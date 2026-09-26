from research.ga_ssw.analyze_ls_pool_routing_panel import render_report, shared_prefix


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
