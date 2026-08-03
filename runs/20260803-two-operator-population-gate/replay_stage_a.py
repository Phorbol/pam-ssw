from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
G_UP0_SUMMARY = (
    REPO_ROOT / "runs/20260731-uphill-relax-counterfactual-gate/evidence.json"
)
G_UP0_SUMMARY_SHA256 = (
    "0a05e4bd7b1a4f1b83e288c47159a7faa1e15775181150cff1fdbb00e6f05a33"
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_source(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if digest(path) != expected_sha256:
        raise RuntimeError(f"source SHA256 drifted: {path}")


def summarize_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise RuntimeError("G-UP0 pair corpus is empty")
    outcomes = Counter(str(row["causal_outcome"]) for row in pairs)
    direct_costs = [int(row["new_force_evaluations"]) for row in pairs]
    for row in pairs:
        counts = {
            str(key): int(value)
            for key, value in row["explicit_purpose_counts"].items()
        }
        if counts.get("unattributed", 0) != 0 or sum(counts.values()) != int(
            row["new_force_evaluations"]
        ):
            raise RuntimeError("G-UP0 direct ledger does not close")
    return {
        "new_force_evaluations": 0,
        "pair_count": len(pairs),
        "relaxed_only_escape_pairs": outcomes["RELAXED_ONLY_ESCAPE"],
        "explicit_only_escape_pairs": outcomes["EXPLICIT_ONLY_ESCAPE"],
        "same_escaped_landing_pairs": outcomes["SAME_ESCAPED_LANDING"],
        "different_escaped_landing_pairs": outcomes["DIFFERENT_ESCAPED_LANDINGS"],
        "direct_quench_observed_force_evaluations": sum(direct_costs),
        "direct_quench_median_force_evaluations": median(direct_costs),
    }


def run(output: Path) -> dict[str, Any]:
    verify_source(G_UP0_SUMMARY, G_UP0_SUMMARY_SHA256)
    summary = json.loads(G_UP0_SUMMARY.read_text(encoding="utf-8"))
    raw = Path(summary["raw_evidence_path"])
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    verify_source(raw, str(summary["raw_evidence_sha256"]))
    payload = json.loads(raw.read_text(encoding="utf-8"))
    replay = summarize_pairs(payload["pairs"])
    replay.update(
        {
            "schema_version": 1,
            "source_path": str(raw.relative_to(REPO_ROOT)),
            "source_sha256": digest(raw),
            "decision": (
                "ADMIT_FRESH_STAGE_B"
                if replay["relaxed_only_escape_pairs"] > 0
                and (
                    replay["same_escaped_landing_pairs"]
                    + replay["different_escaped_landing_pairs"]
                    + replay["explicit_only_escape_pairs"]
                )
                > 0
                else "CLOSE_TWO_OPERATOR_PORTFOLIO"
            ),
            "claim_ceiling": (
                "descriptive zero-new-FE replay of the existing C60 G-UP0 "
                "corpus; full fresh action cost is deferred to Stage B"
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "output/stage-a.json",
    )
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
