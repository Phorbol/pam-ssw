"""Zero-PES parser checks using archived pilot data and one synthetic failure.

Run on a CPU compute node. Never reads the six ongoing coverage arms and never
changes pilot evidence. Temporary fixtures are parser checks, not new searches.
"""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "c4h6-mh1-lifecycle-20260924"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    output = HERE / "analysis-preflight.json"
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location("coverage_analysis", HERE / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    h = mod.helpers()
    mod.helpers = lambda: h  # Retain the actual repository helpers in temporary fixtures.
    plan = json.loads((PILOT / "plan.json").read_text())
    plan.update(seeds=[59], common_request_prefix=6000)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="c4h6-readout-") as scratch:
        root = Path(scratch)
        mod.HERE = root
        dump(root / "plan.json", plan)
        for arm in plan["arms"]:
            dst, src = root / f"{arm}-seed59", PILOT / arm
            dst.mkdir()
            for name in ("result.json", "requests.jsonl", "fresh-requests.jsonl", "fresh-checks.json"):
                (dst / name).symlink_to(src / name)
            summary = json.loads((src / "summary.json").read_text())
            summary.update(protocol_completed=True, budget_censored=False, execution_ok=True)
            dump(dst / "summary.json", summary)
        mod.main()
        report = json.loads((root / "analysis.json").read_text())
        assert report["errors"] == [], report["errors"]
        counts = {a["arm"]: a["connected_graph_class_count"] + a["fragmented_graph_class_count"]
                  for a in report["arms"]}
        assert counts == {"ssw": 3, "paper_ls": 4, "native_ls": 1}, counts
        assert sum(a["fresh"]["checked_count"] for a in report["arms"]) == 39
        checks["archived_pilot_full_readout"] = {"graph_class_counts": counts, "fresh": 39}

        # Insert a zero-cost, no-landing record before the real saved records.
        # This tests indexing across a failed attempt, not scientific behavior.
        synthetic = root / "synthetic"
        folder = synthetic / "ssw-seed59"
        folder.mkdir(parents=True)
        original = json.loads((PILOT / "ssw" / "result.json").read_text())
        data = copy.deepcopy(original)
        first_accept = next(i for i, r in enumerate(original["records"]) if r["accepted"])
        prefix = original["initial"]["evaluation_requests"] + sum(
            r["evaluation_requests"] for r in original["records"][:first_accept + 1])
        for r in data["records"]:
            r["index"] += 1
        data["records"].insert(0, dict(index=0, status="true_quench_failed", landing=None,
            accepted=False, evaluation_requests=0, error="synthetic parser-only record", climb=[]))
        dump(folder / "result.json", data)
        summary = json.loads((root / "ssw-seed59" / "summary.json").read_text())
        summary["outer_records"] = 13
        dump(folder / "summary.json", summary)
        for name in ("requests.jsonl", "fresh-requests.jsonl", "fresh-checks.json"):
            (folder / name).symlink_to(PILOT / "ssw" / name)
        mod.HERE = synthetic
        synthetic_plan = dict(plan, steps=13, common_request_prefix=prefix)
        row = mod.audit_arm("ssw", 59, synthetic_plan, h)
        assert row["errors"] == [], row["errors"]
        h[0](row["class_entries"])
        for m in row["minima"]:
            m["graph_class_id"] = m["_class_entry"]["class_id"]
        summary = mod.summarize_arm(row, synthetic_plan)
        assert summary["prefix_accepted_landings"] == 1, summary
        assert len(summary["no_converged_landing_costs"]) == 1
        checks["failed_attempt_does_not_shift_minimum_identity"] = True
        # Removing only the temporary link must leave the evidence untouched.
        (folder / "fresh-checks.json").unlink()
        missing = mod.audit_arm("ssw", 59, synthetic_plan, h)
        assert missing["errors"] and not missing["fresh"]["all_qualified"]
        checks["missing_fresh_is_not_success"] = True
    dump(output, {"kind": "zero_PES_parser_verification", "checks": checks})
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
