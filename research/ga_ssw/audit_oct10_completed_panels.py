"""Offline closure audit of the two completed Oct 7 C60 panels; no PES calls.

Run with the random panel's frozen source first on PYTHONPATH. Original
inputs, ledgers, checkpoints and readouts are never modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_center_curvatures(records):
    """FD real-PES directional curvature + analytic old/new Gaussian terms.

    The frozen rotation callback excludes old Gaussians. This is a local FD
    estimate on the saved directions, not a new Hessian or escape certificate.
    This panel has no LS or mutable historical heights.
    """
    values = []
    for record in records:
        previous = []
        for event in record.climb:
            required = ("weight", "width", "center", "direction", "recovered_rotation")
            if not all(key in event for key in required):
                continue
            direction = np.asarray(event["direction"]).ravel()
            center = np.asarray(event["center"]).ravel()
            width, weight = event["width"], event["weight"]
            curvature = event["recovered_rotation"]["real_curvature"]
            assert np.isclose(np.linalg.norm(direction), 1.)
            for old_center, old_direction, sigma, amplitude in previous:
                projection = float(np.dot(center-old_center, old_direction))
                curvature += (amplitude*np.exp(-.5*(projection/sigma)**2)
                    * (projection**2/sigma**4-1/sigma**2)
                    * float(np.dot(direction, old_direction))**2)
            values.append(curvature-weight/width**2)
            previous.append((center, direction, width, weight))
    values = np.asarray(values)
    assert len(values) and np.isfinite(values).all()
    return dict(count=len(values), estimated_negative=int((values < 0).sum()),
        minimum=float(values.min()), maximum=float(values.max()),
        median=float(np.median(values)), unit="eV/Angstrom^2",
        scope="saved one-sided FD real curvature plus analytic Gaussian Hessian; no PES call")


def audit(root):
    import pamssw
    from pamssw.standalone import load_ssw_checkpoint

    evidence = root / "research/ga_ssw/evidence"
    random = evidence / "c60-direction-transfer-20261007"
    prepared = random / "prepared-1664056"
    assert Path(pamssw.__file__).resolve().is_relative_to((prepared / "source").resolve())
    manifest = load(prepared / "source-manifest.json")
    for entry in manifest["core_files"]:
        assert sha(prepared / "source" / entry["path"]) == entry["sha256"]
    results = dict(scope="offline source, ledger, checkpoint and cost audit; no PES",
                   random_source=manifest["git_head"], random=[])
    readout = load(random / "readout-1664073/analysis.json")
    for row in readout["rows"]:
        folder = Path(row["path"])
        budget = load(folder / "budget.json")
        plan = load(folder / "plan.json")
        assert sha(folder / "plan.json") == load(folder / "summary.json")["plan_sha256"]
        cp = load_ssw_checkpoint(folder / "last-result.pkl")
        count = 0
        with (folder / "requests.jsonl").open() as stream:
            for line in stream:
                r = json.loads(line)
                count += 1
                assert r["id"] == count and "error" not in r
                assert np.isfinite(r["energy_eV"]) and np.isfinite(r["fmax_eV_A"])
        assert count == budget["search"] == row["search_requests"] == cp.evaluation_requests == 60000
        assert cp.initial.evaluation_requests + sum(s.evaluation_requests for s in cp.records) == count
        assert cp.records[-1].status == "evaluation_failed"
        assert cp.records[-1].error == "climb: RuntimeError: search_budget_exhausted"
        events = [e for s in cp.records for e in s.climb]
        phases = dict(initial=cp.initial.evaluation_requests,
                      rotation=sum(e.get("rotation_force_requests", 0) for e in events),
                      biased_quench=sum(e.get("quench_requests", 0) for e in events),
                      true_landing=sum(s.landing.evaluation_requests for s in cp.records if s.landing))
        phases["other_and_unfinished"] = count - sum(phases.values())
        assert phases["other_and_unfinished"] >= 0
        observations = load(random / "readout-1664073" / f"{row['case']}-{row['arm']}-observations.json")
        assert not any(x["graph_energy_candidate"] or x["cage_all_cutoffs"] for x in observations)
        for cap, prefix in row["cost_prefixes"].items():
            valid = [x for x in observations if x["search_force_qualified"] and x["cumulative_search_requests"] <= int(cap)]
            assert len(valid) == prefix["qualified_observations"]
            assert min(x["energy_eV"] for x in valid) == prefix["best_energy_eV"]
        assert all(x["numerical_qualified"] for x in row["fresh_checks"].values())
        assert plan["ssw_config"]["fmax"] == .03
        assert plan.get("ls_settings") is None
        results["random"].append(dict(case=row["case"], arm=row["arm"], paid=count,
            actual_calculate=row["actual_search_calculator_calls"], fresh=row["fresh_requests"],
            qualified_observations=row["qualified_observations"], phases=phases,
            gaussians=sum("weight" in e for e in events), target=False,
            new_gaussian_center_curvature=gaussian_center_curvatures(cp.records),
            best_energy=row["fresh_checks"]["best"]["energy_eV"]))

    native = evidence / "c60-source3-native-reference-20261007"
    stage = native / "prepared-20261007-b"
    source = load(stage / "source-manifest.json")
    for name, entry in source["sources"].items():
        assert sha(stage / name) == entry["sha256"]
    for folder, entries in source["seed_runs"].items():
        for name, expected in entries.items():
            assert sha(stage / folder / name) == expected
    spec = importlib.util.spec_from_file_location("native_oct7_analyzer", stage / "analyze.py")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    results.update(native_source=source["git_head"], native=[])
    for row in load(native / "readout-1665018/analysis.json")["rows"]:
        folder = stage / f"seed-{row['seed']}"
        events = analyzer.parse_events((folder / "lasp.out").read_text())
        counts, samples = analyzer.stream_ledger(folder / "request.jsonl", {e["search_cost"] for e in events})
        assert counts == row["counts"]
        for e in events:
            actual = samples[e["search_cost"]]
            assert abs(actual["energy"] - e["printed_energy_eV"]) <= 1e-4
            assert abs(np.abs(actual["forces"]).max() - e["printed_force_component"]) <= 5e-4
        summary = load(folder / "summary.json")
        assert summary["process"]["state"] == "timeout" and not summary["process"]["cleanup_survivors"]
        assert row["comparison_eligible"] and len(events) == row["minimum_events"]
        observations = load(native / "readout-1665018" / f"seed-{row['seed']}-observations.json")
        assert not any(x["joint_target"] for x in observations)
        results["native"].append(dict(seed=row["seed"], counts=counts,
            actual_calculate=summary["actual_calculate_calls"], minimum_events=len(events),
            unfinished_paid_tail=counts["paid"]-events[-1]["search_cost"],
            process_state="supervised_wall_limit", target=False))
    cold = load(native / "qualified-1665019/summary.json")
    assert cold["paid_requests"] == cold["actual_calculate_calls"] == len(cold["records"]) == 4
    assert all(r["numerical_qualified"] and sha(Path(r["source"])) == r["source_sha256"] for r in cold["records"])
    results["native_cold"] = dict(paid=4, actual_calculate=4, qualified=4)
    results["checks_passed"] = True
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[2])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
