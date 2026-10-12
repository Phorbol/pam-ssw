import importlib.util
from dataclasses import asdict
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "vc_phase139_transfer_runner", Path(__file__).with_name("run_phase139.py"))
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_phase139_input_and_seed_pair_are_explicit_but_configs_are_frozen():
    plan = RUN.build_plan()
    assert plan["cases"] == [{
        "id": "phase139_48",
        "path": str(RUN.PHASE139_SOURCE),
        "sha256": RUN.PHASE139_SHA256,
        "natoms": 48,
        "paired_seed": RUN.SEEDS[0],
    }]
    base = RUN.base_module()
    phase87 = base.load_plan()
    assert asdict(base.treatment_configurations(plan)["cell_only"]["block"]) == \
           asdict(base.treatment_configurations(phase87)["cell_only"]["block"])
    assert asdict(base.treatment_configurations(plan)["cell_plus_atomic"]["block"]) == \
           asdict(base.treatment_configurations(phase87)["cell_plus_atomic"]["block"])
    assert RUN.SEEDS == (26101251, 26101252)


def test_phase139_plan_retains_phase87_control_provenance_without_mislabelling_case():
    plan = RUN.build_plan()
    source = plan["transfer_provenance"]
    assert source["control_case_id"] == "phase87_48"
    assert source["control_input"]["path"].endswith("phase-87.extxyz")
    assert plan["cases"][0]["id"] == "phase139_48"
