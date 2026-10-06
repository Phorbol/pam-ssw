"""Compare saved-stage escape with direct quench; no new PES evaluations."""
import argparse
import json
from pathlib import Path
import sys
from analyze import ledger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))


def compare(direct_paths):
    from ase.io import read
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    plan = json.loads((HERE / "plan.json").read_text())
    rows = []
    for slot, path in enumerate(direct_paths):
        result = json.loads((path / "result.json").read_text())
        released_path = HERE / f"run-1663813-{1+2*slot}"
        released = json.loads((released_path / "result.json").read_text())
        if result["case"] != released["case"]:
            raise ValueError("direct/escape input identity mismatch")
        if (path / "input.extxyz").read_bytes() != (released_path / "input.extxyz").read_bytes():
            raise ValueError("direct/escape coordinates differ")
        for kind, name in (("search", "requests.jsonl"), ("fresh", "fresh-requests.jsonl")):
            actual = ledger(path / name)
            if any(result[f"{kind}_cost"][k] != v for k, v in actual.items()):
                raise ValueError(f"direct {kind} costs do not close")
        q = result.get("true_quench") or {}
        if q.get("reported_requests", 0) != result["search_cost"]["requests"]:
            raise ValueError("direct quench reported/ledger cost mismatch")
        matches = {}
        if q.get("candidate_minimum") and released["true_quench"].get("candidate_minimum"):
            direct_atoms = read(path / "certified-true-quench-minimum.extxyz")
            escape_atoms = read(released_path / "certified-true-quench-minimum.extxyz")
            for name, tol in plan["identity_tolerances"].items():
                matches[name] = bool(pymatgen_identity(**tol)(direct_atoms, escape_atoms))
        cold = result.get("candidate_cold_check") or {}
        rows.append({"case": result["case"], "direct_path": str(path),
                     "escape_path": str(released_path), "endpoint_matches": matches,
                     "direct_quench": q, "direct_cold": cold,
                     "direct_search_cost": result["search_cost"],
                     "direct_fresh_cost": result["fresh_cost"],
                     "escape_search_cost": released["search_cost"],
                     "escape_fresh_cost": released["fresh_cost"],
                     "direct_cold_qualified": cold.get("cold_confirmed_minimum", False),
                     "direct_cold_target": cold.get("cold_confirmed_target_candidate", False),
                     "escape_cold_target": released.get("cold_confirmed_target_candidate", False)})
    return {"audit": "passed", "pairs": rows,
            "scope": "Saved post-cell states; approximate endpoint matching, not global success-rate or optimizer ranking."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("direct", type=Path, nargs=2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.direct)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"audit": result["audit"], "pairs": [
        {k: r[k] for k in ("case", "endpoint_matches", "direct_cold_qualified", "direct_cold_target", "escape_cold_target")}
        for r in result["pairs"]]}))


if __name__ == "__main__":
    main()
