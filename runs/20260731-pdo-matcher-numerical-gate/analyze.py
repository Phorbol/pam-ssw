#!/usr/bin/env python3
"""Run the zero-FE PdO matcher/numerical decomposition."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
FIRST_PASSAGE_PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-current-action-first-passage"
    / "protocol.py"
)
DEFAULT_RAW = (
    REPO_ROOT
    / "runs"
    / "20260731-current-action-first-passage"
    / "output-v3"
    / "evidence.json"
)
LOCKED_GATE_RELATIVE = (
    Path("runs")
    / "20260731-direction-candidate-counterfactual-gate"
    / "output"
    / "groups"
)
FROZEN_PDO_INPUT = Path(
    "/mnt/d/download/trae-research-code/ssw/PdO.xyz"
)
FROZEN_PDO_INPUT_SHA256 = (
    "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0"
)
FROZEN_PDO_BOTTOM_FRACTION = 0.35
FROZEN_PDO_FIXED_COUNT = 40


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    PROTOCOL_PATH,
    "_pdo_matcher_numerical_protocol_analysis",
)
first_passage_protocol = _load_module(
    FIRST_PASSAGE_PROTOCOL_PATH,
    "_pdo_matcher_first_passage_protocol_analysis",
)


def _starter_path(
    state_source_root: Path,
    system: str,
    state_id: str,
) -> Path:
    return (
        state_source_root
        / LOCKED_GATE_RELATIVE
        / system
        / state_id
        / "seed-00000042"
        / "candidate-0"
        / "starter.xyz"
    )


def _pdo_fixed_mask() -> np.ndarray:
    from ase.io import read

    if sha256(FROZEN_PDO_INPUT.read_bytes()).hexdigest() != (
        FROZEN_PDO_INPUT_SHA256
    ):
        raise RuntimeError("frozen PdO input checksum drifted")
    positions = np.asarray(read(FROZEN_PDO_INPUT).positions, dtype=float)
    z = positions[:, 2]
    threshold = float(np.quantile(z, FROZEN_PDO_BOTTOM_FRACTION))
    fixed_mask = z <= threshold
    if int(np.count_nonzero(fixed_mask)) != FROZEN_PDO_FIXED_COUNT:
        raise RuntimeError("frozen PdO fixed-mask count drifted")
    return fixed_mask


def _masked_rmsd(displacement: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(displacement, dtype=float)[np.asarray(mask, dtype=bool)]
    if selected.shape[0] == 0:
        raise ValueError("regional RMSD requires at least one selected atom")
    return float(np.sqrt(np.mean(np.sum(selected * selected, axis=1))))


def _pair_metrics(
    *,
    starter_path: Path,
    landing_path: Path,
    checkpoint: Mapping[str, Any],
    case: Mapping[str, Any],
    fixed_mask: np.ndarray,
) -> dict[str, Any]:
    from pamssw.archive import MinimaArchive
    from pamssw.io import read_state
    from pamssw.pbc import mic_displacement

    starter = read_state(starter_path)
    landing = read_state(landing_path)
    indexed_rmsd = float(MinimaArchive._rmsd(starter, landing))
    displacement = mic_displacement(
        landing.positions,
        starter.positions,
        starter.cell,
        starter.pbc,
    )
    atom_displacements = np.linalg.norm(displacement, axis=1)
    energy_tol = float(case["effective_config"]["dedup_energy_tol"])
    rmsd_tol = float(case["effective_config"]["dedup_rmsd_tol"])
    descriptor_tol = float(
        case["effective_config"]["min_escape_descriptor_delta"]
    )
    decomposition = protocol.decompose_pair(
        energy_delta_eV=float(checkpoint["landing_delta_eV"]),
        energy_tol_eV=energy_tol,
        indexed_mic_rmsd_A=indexed_rmsd,
        rmsd_tol_A=rmsd_tol,
        descriptor_delta=float(checkpoint["descriptor_delta"]),
        descriptor_tol=descriptor_tol,
    )
    if fixed_mask.shape != (starter.n_atoms,):
        raise RuntimeError("frozen PdO fixed mask does not match checkpoint")
    movable_rmsd = _masked_rmsd(displacement, ~fixed_mask)
    fixed_rmsd = _masked_rmsd(displacement, fixed_mask)
    if decomposition["mechanism"] == "energy_only_archive_split":
        local_decomposition = protocol.decompose_local_region(
            movable_indexed_mic_rmsd_A=movable_rmsd,
            rmsd_tol_A=rmsd_tol,
        )
    else:
        local_decomposition = {
            "movable_geometry_same": None,
            "local_region_mechanism": "not_applicable_global_geometry_split",
        }
    return {
        "system": str(case["system"]),
        "state_id": str(case["state_id"]),
        "seed": int(case["seed"]),
        "arm": str(case["arm"]),
        "horizon": int(checkpoint["horizon"]),
        "certificate": bool(checkpoint["certificate"]),
        "geometry_valid": bool(checkpoint["geometry_valid"]),
        "fragmented": bool(checkpoint["fragmented"]),
        "landing_delta_eV": float(checkpoint["landing_delta_eV"]),
        "energy_tol_eV": energy_tol,
        "indexed_mic_rmsd_A": indexed_rmsd,
        "rmsd_tol_A": rmsd_tol,
        "descriptor_delta": float(checkpoint["descriptor_delta"]),
        "descriptor_tol": descriptor_tol,
        "max_indexed_mic_displacement_A": float(atom_displacements.max()),
        "median_indexed_mic_displacement_A": float(
            np.median(atom_displacements)
        ),
        "movable_atom_count": int(np.count_nonzero(~fixed_mask)),
        "fixed_atom_count": int(np.count_nonzero(fixed_mask)),
        "movable_indexed_mic_rmsd_A": movable_rmsd,
        "fixed_indexed_mic_rmsd_A": fixed_rmsd,
        "starter_path": str(starter_path.relative_to(REPO_ROOT)),
        "landing_path": str(landing_path.relative_to(REPO_ROOT)),
        **decomposition,
        **local_decomposition,
    }


def _counterfactual_first_passage(
    raw: Mapping[str, Any],
    *,
    relabel: bool,
) -> Mapping[str, Any] | None:
    if not relabel:
        return None
    cases = deepcopy(raw["cases"])
    for case in cases:
        changed = False
        for checkpoint in case["checkpoints"]:
            if checkpoint["label"] == "AMBIGUOUS_MATCH":
                checkpoint["label"] = "ESCAPED_CERTIFIED"
                changed = True
        if changed:
            case["trajectory_summary"] = (
                first_passage_protocol.summarize_trajectory(
                    case["checkpoints"]
                )
            )
    evidence = first_passage_protocol.build_evidence(
        cases,
        max_force_evaluations=int(raw["max_force_evaluations"]),
    )
    return {
        key: evidence[key]
        for key in (
            "label_counts",
            "horizon_gate_contexts",
            "action_support_gap_contexts",
            "numerical_matcher_gate_systems",
        )
    }


def _render(evidence: Mapping[str, Any]) -> str:
    gate = evidence["gate"]
    lines = [
        "# PdO matcher/numerical ambiguity result",
        "",
        "## Decision",
        "",
        f"- Classification: **{gate['classification']}**.",
        (
            "- Offline relabel as escaped allowed: "
            f"**{gate['relabel_as_escaped_allowed']}**."
        ),
        (
            "- Strict re-quench gate required: "
            f"**{gate['strict_requench_gate_required']}**."
        ),
        "- Production matcher change allowed: **False**.",
        "- New force evaluations: **0**.",
        "",
        "## Pair decomposition",
        "",
        "| Starter | Seed | Arm | H | ΔE (eV) | All MIC RMSD (Å) | Movable MIC RMSD (Å) | Descriptor Δ | Mechanism |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence["pairs"]:
        lines.append(
            f"| {row['state_id']} | {row['seed']} | {row['arm']} | "
            f"{row['horizon']} | {row['landing_delta_eV']:.6f} | "
            f"{row['indexed_mic_rmsd_A']:.6f} | "
            f"{row['movable_indexed_mic_rmsd_A']:.6f} | "
            f"{row['descriptor_delta']:.6f} | {row['mechanism']} |"
        )
    energy_tol = evidence["pairs"][0]["energy_tol_eV"]
    rmsd_tol = evidence["pairs"][0]["rmsd_tol_A"]
    descriptor_tol = evidence["pairs"][0]["descriptor_tol"]
    lines.extend(
        [
            "",
            f"The gate uses the effective {energy_tol:g} eV energy tolerance, "
            f"{rmsd_tol:g} Å indexed MIC RMSD tolerance, and "
            f"{descriptor_tol:g} descriptor threshold. The movable-region "
            "subgate reuses exactly the same RMSD tolerance. No threshold was "
            "fitted to these five outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def run(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    state_source_root = Path(raw["state_source_root"]).resolve()
    fixed_mask = _pdo_fixed_mask()
    pairs = []
    for case in raw["cases"]:
        for checkpoint in case["checkpoints"]:
            if checkpoint["label"] != "AMBIGUOUS_MATCH":
                continue
            pairs.append(
                _pair_metrics(
                    starter_path=_starter_path(
                        state_source_root,
                        str(case["system"]),
                        str(case["state_id"]),
                    ),
                    landing_path=Path(checkpoint["landing_path"]),
                    checkpoint=checkpoint,
                    case=case,
                    fixed_mask=fixed_mask,
                )
            )
    initial_gate = protocol.evaluate_gate(pairs)
    gate = protocol.evaluate_local_region_gate(
        pairs,
        initial_gate=initial_gate,
    )
    evidence = {
        "schema_version": 2,
        "new_force_evaluations": 0,
        "raw_evidence_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_evidence_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "fixed_mask_provenance": {
            "input_path": str(FROZEN_PDO_INPUT),
            "input_sha256": FROZEN_PDO_INPUT_SHA256,
            "bottom_fraction": FROZEN_PDO_BOTTOM_FRACTION,
            "fixed_atom_count": int(np.count_nonzero(fixed_mask)),
            "movable_atom_count": int(np.count_nonzero(~fixed_mask)),
        },
        "pair_count": len(pairs),
        "pairs": pairs,
        "initial_gate": initial_gate,
        "gate": gate,
        "counterfactual_first_passage": _counterfactual_first_passage(
            raw,
            relabel=gate["relabel_as_escaped_allowed"],
        ),
    }
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "conclusion.md").write_text(
        _render(evidence),
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-evidence", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    evidence = run(args.raw_evidence.resolve())
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
