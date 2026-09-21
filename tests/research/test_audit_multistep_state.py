from copy import deepcopy

from research.ga_ssw.audit_multistep_state import audit_payload


def _atoms(x):
    return {"numbers": [1], "positions": [[x, 0.0, 0.0]],
            "cell": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            "pbc": [True, True, True]}


def test_audit_catches_rejected_state_used_as_next_chart_reference():
    initial = _atoms(0.0)
    rejected = _atoms(1.0)
    payload = {"result": {
        "initial": {"atoms": initial, "energy": 0.0, "certificate": {"certified": True}, "optimizer": {"status": "converged"}},
        "current": {"atoms": initial, "energy": 0.0},
        "best": {"atoms": initial, "energy": 0.0},
        "minima": [{"atoms": initial, "energy": 0.0}, {"atoms": rejected, "energy": 1.0}, {"atoms": initial, "energy": 0.0}],
        "records": [
            {"stage": "initial", "status": "converged", "requests": 1},
            {"index": 0, "status": "valid_landing", "accepted": False,
             "requests": 3, "chart_reference": initial,
             "landing": {"atoms": rejected, "energy": 1.0,
                         "certificate": {"certified": True}, "optimizer": {"status": "converged"}}},
            {"index": 1, "status": "valid_landing", "accepted": True,
             "requests": 3, "chart_reference": rejected,
             "landing": {"atoms": initial, "energy": 0.0,
                         "certificate": {"certified": True}, "optimizer": {"status": "converged"}}},
        ],
        "requests": 7,
        "status": "completed",
    }}
    report = audit_payload(payload)
    assert report["status"] == "violations"
    assert any("prior selected current" in item["detail"] for item in report["failures"])


def test_audit_rejects_certificate_true_but_optimizer_not_converged():
    initial = _atoms(0.0)
    landing = {"atoms": _atoms(1.0), "energy": 1.0,
               "certificate": {"certified": True},
               "optimizer": {"status": "line_search_failed"}}
    payload = {"result": {
        "initial": {"atoms": initial, "certificate": {"certified": True}, "optimizer": {"status": "converged"}},
        "current": landing, "best": landing,
        "minima": [{"atoms": initial}, {"atoms": landing["atoms"]}],
        "records": [{"stage": "initial", "status": "converged", "requests": 1},
                    {"index": 0, "status": "valid_landing", "accepted": True,
                     "requests": 2, "chart_reference": initial, "landing": landing}],
        "requests": 3,
    }}
    report = audit_payload(payload)
    assert report["status"] == "violations"
    assert any("optimizer" in item["check"] for item in report["failures"])
