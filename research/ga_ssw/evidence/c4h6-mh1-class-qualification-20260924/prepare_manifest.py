"""Build the fixed-prefix class-representative manifest using stdlib only."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COVERAGE = REPO / "research/ga_ssw/evidence/c4h6-mh1-coverage-20260924"
LIFECYCLE = REPO / "research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924"
CURVATURE = REPO / "research/ga_ssw/evidence/c4h6-mh1-curvature-20260924"
PREFIX = 200_000
MODEL = "/home/gengjianrui/.cache/mace/mace-mh-1.model"
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_COUNTS = {
    "ssw-seed61": 5,
    "ssw-seed67": 5,
    "paper_ls-seed61": 6,
    "paper_ls-seed67": 8,
    "native_ls-seed61": 10,
    "native_ls-seed67": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    analysis_path = COVERAGE / "analysis.json"
    coverage_plan_path = COVERAGE / "plan.json"
    lifecycle_plan_path = LIFECYCLE / "plan.json"
    curvature_runner_path = CURVATURE / "run.py"
    for path in (analysis_path, coverage_plan_path, lifecycle_plan_path, curvature_runner_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis = json.loads(analysis_path.read_text())
    coverage_plan = json.loads(coverage_plan_path.read_text())
    lifecycle_plan = json.loads(lifecycle_plan_path.read_text())
    if analysis.get("common_prefix_requests") != PREFIX:
        raise ValueError("coverage analysis does not record the fixed 200000-request prefix")
    if coverage_plan.get("common_request_prefix") != PREFIX:
        raise ValueError("coverage plan prefix differs from requested 200000 requests")
    if analysis.get("issues") or analysis.get("errors"):
        raise ValueError("coverage analysis records issues/errors; refuse class selection")
    if lifecycle_plan.get("model_sha256") != MODEL_SHA256:
        raise ValueError("lifecycle model hash differs from class-qualification model")
    if lifecycle_plan.get("model") != MODEL or lifecycle_plan.get("head") != "omol":
        raise ValueError("lifecycle model path/head differs from fixed MH-1/omol protocol")

    arms = analysis.get("arms")
    if not isinstance(arms, list) or len(arms) != len(EXPECTED_COUNTS):
        raise ValueError("expected six arm records in coverage analysis")
    frames = []
    arm_counts = {}
    source_hashes = {}
    for arm_record in arms:
        arm = arm_record["arm"]
        seed = arm_record["seed"]
        label = f"{arm}-seed{seed}"
        if label not in EXPECTED_COUNTS:
            raise ValueError(f"unexpected arm/seed {label}")
        if arm_record.get("errors"):
            raise ValueError(f"coverage arm {label} records errors")
        prefix_summary = arm_record.get("common_prefix_and_endpoint", {})
        if prefix_summary.get("prefix") != PREFIX or prefix_summary.get("prefix_reached") is not True:
            raise ValueError(f"coverage arm {label} did not reach the fixed prefix")

        source_json = COVERAGE / label / "result.json"
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        source_hash = sha256_file(source_json)
        source_hashes[label] = source_hash

        best_by_class = {}
        for minimum in arm_record.get("minima", []):
            if minimum.get("cumulative_requests") is None or minimum["cumulative_requests"] > PREFIX:
                continue
            if minimum.get("graph", {}).get("component_count") != 1:
                continue
            class_id = minimum.get("graph_class_id")
            if not isinstance(class_id, int):
                raise ValueError(f"missing global graph class ID in {label} minimum {minimum.get('index')}")
            candidate = (minimum["energy_eV"], minimum["index"])
            incumbent = best_by_class.get(class_id)
            if incumbent is None or candidate < incumbent[0]:
                best_by_class[class_id] = (candidate, minimum)

        arm_counts[label] = len(best_by_class)
        for class_id in sorted(best_by_class):
            _, minimum = best_by_class[class_id]
            frames.append({
                "arm": arm,
                "seed": seed,
                "arm_seed": label,
                "minimum_index": minimum["index"],
                "global_graph_class_id": class_id,
                "component_count": minimum["graph"]["component_count"],
                "component_formulas": minimum["component_formulas"],
                "energy_eV": minimum["energy_eV"],
                "cost_requests": minimum["cumulative_requests"],
                "source_json": f"../c4h6-mh1-coverage-20260924/{label}/result.json",
                "source_json_sha256": source_hash,
            })

    if arm_counts != EXPECTED_COUNTS:
        raise ValueError(f"selected class counts differ from fixed expected matrix: {arm_counts}")
    if len(frames) != 42:
        raise ValueError(f"expected 42 representative frames, got {len(frames)}")
    frames.sort(key=lambda frame: (frame["arm_seed"], frame["global_graph_class_id"]))

    helper_hash = sha256_file(curvature_runner_path)
    manifest = {
        "status": "prepared_for_review",
        "selection_input": "../c4h6-mh1-coverage-20260924/analysis.json",
        "selection_input_sha256": sha256_file(analysis_path),
        "coverage_plan": "../c4h6-mh1-coverage-20260924/plan.json",
        "coverage_plan_sha256": sha256_file(coverage_plan_path),
        "lifecycle_plan": "../c4h6-mh1-lifecycle-20260924/plan.json",
        "lifecycle_plan_sha256": sha256_file(lifecycle_plan_path),
        "protocol_sha256": sha256_file(HERE / "plan.md"),
        "manifest_builder": "prepare_manifest.py",
        "manifest_builder_sha256": sha256_file(HERE / "prepare_manifest.py"),
        "curvature_helper": "../c4h6-mh1-curvature-20260924/run.py",
        "curvature_helper_sha256": helper_hash,
        "common_request_prefix": PREFIX,
        "selection_rule": "per arm/seed/global connected graph class, cumulative_requests <= 200000; minimum energy, tie by minimum index",
        "class_counts_by_arm_seed": arm_counts,
        "total_frames": len(frames),
        "source_json_sha256_by_arm_seed": source_hashes,
        "model": MODEL,
        "model_sha256": MODEL_SHA256,
        "head": "omol",
        "device": "cpu",
        "dtype": "float64",
        "energy_units_to_eV": 1.0,
        "length_units_to_A": 1.0,
        "frames": frames,
    }
    output = HERE / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "total_frames": len(frames),
        "class_counts_by_arm_seed": arm_counts,
        "analysis_sha256": manifest["selection_input_sha256"],
        "manifest": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
