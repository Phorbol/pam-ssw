#!/usr/bin/env python3
"""Read-only provenance and checkpoint audit for the C60 continuation."""
from __future__ import annotations

import hashlib
import argparse
import json
import tempfile
import sys
from pathlib import Path


PARENT_PREPARED = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/"
    "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/"
    "prepared-20261007-a"
)
EXPECTED_MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_SOURCE_HEAD = "4fec3e3f7d09009f0808dfc43cc1c2c2e78900d6"
EXPECTED_INPUT_SHA256 = "dcc81d45c4e7193b306a0c1957b49c19e44751231fa95fef75201183280f0758"
ARMS = (
    (26100791, "ssw"),
    (26100791, "paper_ls"),
    (26100792, "ssw"),
    (26100792, "paper_ls"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed_ls(plan):
    from pamssw.standalone import LSPrequenchSettings, LSSettings

    spec = plan.get("paper_ls")
    if spec is None:
        return None
    spec = dict(spec)
    spec.pop("parameter_source", None)
    for key in ("bond_energies", "bond_lengths"):
        spec[key] = {tuple(map(int, k.split(","))): v for k, v in spec[key].items()}
    if spec.get("prequench") is not None:
        spec["prequench"] = LSPrequenchSettings(**spec["prequench"])
    return LSSettings(**spec)


def inspect_parent(parent_prepared: Path = PARENT_PREPARED) -> dict:
    """Validate every parent's plan, source, budget, ledger and resume state.

    This function makes no files and constructs no Calculator.
    """
    parent_prepared = Path(parent_prepared).resolve(strict=True)
    source_root = parent_prepared / "source"
    manifest_path = parent_prepared / "source-manifest.json"
    search_manifest_path = parent_prepared / "search-plan-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("git_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("parent frozen source commit differs from the qualified source-3 run")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for item in manifest["core_files"]:
        path = source_root / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"frozen core hash mismatch: {path}")

    sys.path.insert(0, str(source_root))
    from pamssw.standalone import (
        LSPrequenchSettings,
        NativeMCSettings,
        SSWConfig,
        load_ssw_checkpoint,
    )
    from pamssw.standalone.paper_reference import _policy_signature

    if not Path(sys.modules["pamssw"].__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError("preflight imported pamssw outside the parent frozen source")

    model_path = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
    if sha(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("cached MH-1 model hash differs from the qualified parent")

    search_manifest = json.loads(search_manifest_path.read_text())
    rows = []
    for seed, arm in ARMS:
        folder = parent_prepared / f"seed-{seed}" / arm
        plan_path = folder / "plan.json"
        plan_sha = sha(plan_path)
        plan = json.loads(plan_path.read_text())
        if plan["provenance"]["prepared_git_head"] != EXPECTED_SOURCE_HEAD:
            raise ValueError(f"prepared source head mismatch: {folder}")
        if plan["seed"] != seed or plan["arm"] != arm:
            raise ValueError(f"seed/arm mapping mismatch: {folder}")
        if plan["model_sha256"] != EXPECTED_MODEL_SHA256 or plan["head"] != "omol":
            raise ValueError(f"model/head mismatch: {folder}")
        if plan["input_sha256"] != EXPECTED_INPUT_SHA256:
            raise ValueError(f"source-3 input hash mismatch: {folder}")
        if plan["outer_steps"] != 100 or plan["search_cap"] != 20000 or plan["wall_seconds"] != 1200:
            raise ValueError(f"unexpected historical budget or horizon: {folder}")
        for rel, expected in plan.get("frozen_files", {}).items():
            path = (folder / rel).resolve()
            if sha(path) != expected:
                raise ValueError(f"parent frozen input/source hash mismatch: {path}")
        manifest_rows = [entry for entry in search_manifest["arm_map"]
                         if entry["seed"] == seed and entry["arm"] == arm]
        if len(manifest_rows) != 1 or manifest_rows[0]["input_sha256"] != EXPECTED_INPUT_SHA256:
            raise ValueError(f"parent arm map mismatch: {folder}")
        if plan_sha != sha(folder / "plan.json"):
            raise ValueError(f"plan changed while reading: {folder}")

        budget_path = folder / "budget.json"
        budget_sha = sha(budget_path)
        budget = json.loads(budget_path.read_text())
        if budget["plan_sha256"] != plan_sha:
            raise ValueError(f"parent budget is not bound to its frozen plan: {folder}")
        if budget["status"] != "wall_exhausted":
            raise ValueError(f"parent arm is not in the expected wall-censored state: {folder}")
        if budget["search"] >= plan["search_cap"] or budget["reserved_seconds"] != plan["wall_seconds"]:
            raise ValueError(f"parent budget totals do not match expected continuation boundary: {folder}")
        if budget.get("unconfirmed_search_reservations", 0):
            raise ValueError(f"parent contains unconfirmed request reservations: {folder}")
        request_path = folder / "requests.jsonl"
        request_sha = sha(request_path)
        raw = [json.loads(line) for line in request_path.read_text().splitlines()]
        if len(raw) != budget["search"] or [r["id"] for r in raw] != list(range(1, len(raw) + 1)):
            raise ValueError(f"parent search ledger count/sequence mismatch: {folder}")
        if any("error" in row for row in raw):
            raise ValueError(f"parent contains a failed paid E/F request: {folder}")
        if budget["fresh"] != len(budget["fresh_checks"]):
            raise ValueError(f"parent fresh-call ledger mismatch: {folder}")

        checkpoint_path = folder / "checkpoint.pkl"
        checkpoint_sha = sha(checkpoint_path)
        if checkpoint_sha != sha(folder / "last-result.pkl"):
            raise ValueError(f"parent safe checkpoint differs from last-result snapshot: {folder}")
        checkpoint = load_ssw_checkpoint(checkpoint_path)
        if checkpoint.status != "completed":
            raise ValueError(f"parent checkpoint is not at a resumable outer boundary: {folder}")
        if checkpoint.next_index != len(checkpoint.records) or checkpoint.next_index >= 100:
            raise ValueError(f"parent checkpoint outer index is invalid: {folder}")
        if checkpoint.evaluation_requests != budget["search"]:
            raise ValueError(f"checkpoint lineage cost differs from parent paid requests: {folder}")
        if checkpoint.rng_state.get("bit_generator") != "PCG64":
            raise ValueError(f"unexpected RNG state: {folder}")
        if checkpoint.native_mc_state is None or checkpoint.mc_settings != NativeMCSettings(
                plan["native_mc"]["energy_tol_eV"], plan["native_mc"]["maxtrap"]):
            raise ValueError(f"native Monte Carlo state/settings mismatch: {folder}")
        expected_config = SSWConfig(**plan["ssw_config"])
        if checkpoint.config != expected_config:
            raise ValueError(f"SSW configuration differs from checkpoint: {folder}")
        expected_ls = _typed_ls(plan)
        if _policy_signature(checkpoint.ls) != _policy_signature(expected_ls):
            raise ValueError(f"LS settings differ from checkpoint: {folder}")
        if (arm == "ssw") != (checkpoint.ls is None):
            raise ValueError(f"LS arm state mismatch: {folder}")
        if arm == "paper_ls":
            if checkpoint.frozen is None or checkpoint.response is None:
                raise ValueError(f"LS checkpoint lacks frozen/response state: {folder}")
            if checkpoint.response.steps != checkpoint.next_index:
                raise ValueError(f"LS response step count differs from checkpoint index: {folder}")

        rows.append({
            "seed": seed,
            "arm": arm,
            "parent_run": str(folder),
            "parent_plan_sha256": plan_sha,
            "parent_budget_sha256": budget_sha,
            "parent_request_ledger_sha256": request_sha,
            "parent_checkpoint_sha256": checkpoint_sha,
            "parent_paid_search_requests": budget["search"],
            "parent_fresh_requests": budget["fresh"],
            "parent_reserved_seconds": budget["reserved_seconds"],
            "checkpoint_status": checkpoint.status,
            "next_index": checkpoint.next_index,
            "remaining_attempts_to_100": 100 - checkpoint.next_index,
            "checkpoint_lineage_requests": checkpoint.evaluation_requests,
            "rng_bit_generator": checkpoint.rng_state["bit_generator"],
            "native_mc_state": checkpoint.native_mc_state.nsame,
            "ls_response_steps": None if checkpoint.response is None else checkpoint.response.steps,
            "ls_response_eV_per_atom": None if checkpoint.response is None else checkpoint.response.last_response,
            "parent_fresh_labels": sorted(budget["fresh_checks"]),
        })

    return {
        "status": "ready_no_pes_calls",
        "parent_prepared": str(parent_prepared),
        "parent_source_manifest_sha256": sha(manifest_path),
        "parent_search_plan_manifest_sha256": sha(search_manifest_path),
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "core_import": str(Path(sys.modules["pamssw"].__file__).resolve()),
        "arms": rows,
        "cumulative_outer_horizon": 100,
        "cumulative_search_cap_each": 60000,
        "additional_fresh_cap_each": 3,
        "child_allocation_seconds_each": 1800,
        "checks": [
            "parent files and hashes verified read-only",
            "all paid search ledgers sequential and equal parent budget totals",
            "all checkpoints load under the parent's frozen core and match charged lineage costs",
            "RNG, native MC, SSW configuration, and LS state are resumable and consistent",
            "no Calculator or PES calls made",
        ],
    }


def terminal_contract_check(parent_prepared: Path = PARENT_PREPARED) -> dict:
    """Exercise failed-boundary serialization using a no-PES denial stub."""
    parent_prepared = Path(parent_prepared).resolve(strict=True)
    source_root = parent_prepared / "source"
    sys.path.insert(0, str(source_root))
    import numpy as np
    from pamssw.standalone import (NativeMCSettings, SSWConfig, load_ssw_checkpoint,
                                   run_ssw, save_ssw_checkpoint)
    folder = parent_prepared / "seed-26100791" / "ssw"
    plan = json.loads((folder / "plan.json").read_text())
    checkpoint = load_ssw_checkpoint(folder / "checkpoint.pkl")
    settings = SSWConfig(**plan["ssw_config"])
    mc = NativeMCSettings(plan["native_mc"]["energy_tol_eV"], plan["native_mc"]["maxtrap"])

    class DeniedBeforePES(RuntimeError):
        pass

    class DeniedSurface:
        requests = 0

        def evaluate(self, atoms):
            raise DeniedBeforePES("preflight sentinel: no Calculator was constructed")

    callbacks = []
    result = run_ssw(checkpoint.current.copy(), DeniedSurface(), steps=1,
                     config=settings, rng=np.random.default_rng(plan["seed"]),
                     checkpoint=checkpoint,
                     checkpoint_callback=lambda snapshot: callbacks.append(snapshot) or False,
                     ls=None, mc=mc)
    if result.status != "evaluation_failed" or result.checkpoint is None:
        raise RuntimeError(f"unexpected terminal status: {result.status}")
    if callbacks or result.checkpoint.next_index != checkpoint.next_index + 1:
        raise RuntimeError("terminal boundary/callback contract changed")
    if (result.checkpoint.evaluation_requests != checkpoint.evaluation_requests or
            result.checkpoint.records[-1].evaluation_requests != 0):
        raise RuntimeError("pre-PES denial changed paid request accounting")
    with tempfile.TemporaryDirectory(prefix="pamssw-terminal-contract-") as temp:
        path = Path(temp) / "terminal.pkl"
        save_ssw_checkpoint(path, result.checkpoint)
        restored = load_ssw_checkpoint(path)
        if (restored.next_index != result.checkpoint.next_index or
                restored.evaluation_requests != checkpoint.evaluation_requests or
                restored.records[-1].status != "evaluation_failed"):
            raise RuntimeError("terminal checkpoint failed serialization round-trip")
    return {"status": "terminal_contract_ok_no_pes_calls", "arm": "seed-26100791/ssw",
            "result_status": result.status, "parent_next_index": checkpoint.next_index,
            "terminal_next_index": result.checkpoint.next_index,
            "terminal_record_requests": result.checkpoint.records[-1].evaluation_requests,
            "callback_invocations": len(callbacks), "calculator_constructed": False,
            "pes_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-contract-check", action="store_true")
    args = parser.parse_args()
    result = terminal_contract_check() if args.terminal_contract_check else inspect_parent()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
