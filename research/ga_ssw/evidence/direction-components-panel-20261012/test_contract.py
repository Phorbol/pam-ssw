import importlib.util
from pathlib import Path


RUNNER_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("direction_components_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_frozen_panel_has_36_arms_and_declared_caps():
    contract = RUNNER.panel_contract()
    assert len(contract["cases"]) == 3
    assert len(contract["methods"]) == 3
    assert len(contract["seeds"]) == 4
    assert contract["arm_count"] == 36
    assert contract["max_search_requests"] == 72_000
    assert contract["max_fresh_requests"] == 36


def test_second_gaussian_exposure_includes_zero_release():
    step = {"climb": [
        {"index": 0, "recovered_direction": {"anchor": [1, 0, 0]}},
        {"index": 1, "status": "direction_zero_release",
         "recovered_direction": {"reason": "zero_direction"}},
    ]}
    exposure = RUNNER.second_gaussian_exposures(step)
    assert exposure == [{"index": 1, "status": "direction_zero_release",
                         "recovered_direction": {"reason": "zero_direction"}}]
