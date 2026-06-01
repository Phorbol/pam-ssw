from __future__ import annotations

import csv
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from pamssw import LSSSWConfig, State
from pamssw.calculators import ASECalculator
from pamssw.reference_dimer import ReferenceDimerRotator, sample_mixed_mode
from pamssw.walker import DirectionCandidateKind, ProposalPotential, SurfaceWalker


RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "output"
RESULTS_CSV = RUN_DIR / "results.csv"
RESULTS_JSON = RUN_DIR / "results.json"
SUMMARY_MD = RUN_DIR / "summary.md"
ARTIFACTS_JSON = RUN_DIR / "artifacts.json"
EVENT_LOG = RUN_DIR / "event_log.jsonl"
STATUS_MD = RUN_DIR / "status.md"

SOURCE_ROOT = Path("/mnt/d/Download/trae-research-code/SSW")
DEFAULT_MODEL = Path("/root/.cache/mace/mace-omat-0-small.model")
SYSTEMS: dict[str, dict[str, Any]] = {
    "c60": {
        "input": SOURCE_ROOT / "runs/20260428-c60-mace-production/prerelaxed_c60.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (False, False, False),
        "fix_bottom_fraction": 0.0,
        "oracle_candidates": 12,
        "n_bond_pairs": 2,
    },
    "cuo": {
        "input": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO_opt_input.arc",
        "model": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO-OMAT_finetune.model",
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "oracle_candidates": 8,
        "n_bond_pairs": 2,
    },
    "pdo": {
        "input": SOURCE_ROOT / "PdO.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "oracle_candidates": 8,
        "n_bond_pairs": 2,
    },
}


def safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_event(payload: dict[str, Any]) -> None:
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def bottom_fixed_mask(positions: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros(len(positions), dtype=bool)
    z = np.asarray(positions, dtype=float)[:, 2]
    threshold = float(np.min(z) + fraction * (np.max(z) - np.min(z)))
    return z <= threshold


def state_from_input(system: str, spec: dict[str, Any]) -> State:
    atoms = read(spec["input"])
    atoms.pbc = spec["pbc"]
    fixed_mask = bottom_fixed_mask(atoms.positions, float(spec["fix_bottom_fraction"]))
    cell = atoms.cell.array if atoms.cell.rank > 0 else None
    if not any(spec["pbc"]):
        cell = None
    return State(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=cell,
        pbc=spec["pbc"],
        fixed_mask=fixed_mask,
        metadata={"system": system, "input": str(spec["input"])},
    )


def make_calculator(model: Path, device: str) -> ASECalculator:
    calc = MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype="float32",
        inference_precision="float32",
        enable_cueq=False,
    )
    return ASECalculator(calc)


def normalize(direction: np.ndarray, fixed_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(direction, dtype=float).reshape(-1, 3).copy()
    if fixed_mask is not None:
        values[fixed_mask] = 0.0
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("cannot normalize zero direction")
    return (values / norm).reshape(-1)


def hvp(state: State, proposal: ProposalPotential, direction: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    plus = state.displaced(direction, epsilon)
    minus = state.displaced(direction, -epsilon)
    _, grad_plus = proposal.evaluate(plus.flatten_positions(), plus)
    _, grad_minus = proposal.evaluate(minus.flatten_positions(), minus)
    values = (grad_plus - grad_minus) / (2.0 * epsilon)
    values = np.asarray(values, dtype=float).reshape(state.n_atoms, 3)
    values[state.fixed_mask] = 0.0
    return values.reshape(-1)


def rayleigh(state: State, proposal: ProposalPotential, direction: np.ndarray) -> tuple[float, np.ndarray]:
    direction = normalize(direction, state.fixed_mask)
    h_direction = hvp(state, proposal, direction)
    return float(np.dot(direction, h_direction)), h_direction


def lanczos_lowest(
    state: State,
    proposal: ProposalPotential,
    dim: int,
    rng: np.random.Generator,
    initial_direction: np.ndarray | None = None,
    penalty_alpha: float = 0.0,
) -> dict[str, Any]:
    q_vectors: list[np.ndarray] = []
    alphas: list[float] = []
    betas: list[float] = []
    if initial_direction is None:
        q = normalize(rng.normal(size=state.positions.shape), state.fixed_mask)
        n0 = None
    else:
        q = normalize(initial_direction, state.fixed_mask)
        n0 = q.copy()
    beta_prev = 0.0
    q_prev = np.zeros_like(q)
    hvp_count = 0
    for _ in range(dim):
        hq_true = hvp(state, proposal, q)
        hvp_count += 1
        if n0 is not None and penalty_alpha != 0.0:
            z = hq_true - penalty_alpha * n0 * float(np.dot(n0, q))
        else:
            z = hq_true
        alpha = float(np.dot(q, z))
        z = z - alpha * q - beta_prev * q_prev
        for basis in q_vectors:
            z = z - float(np.dot(z, basis)) * basis
        beta = float(np.linalg.norm(z))
        q_vectors.append(q)
        alphas.append(alpha)
        if len(q_vectors) == dim or beta <= 1e-12:
            break
        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    n = len(q_vectors)
    tri = np.diag(alphas)
    for index, beta in enumerate(betas[: max(0, n - 1)]):
        tri[index, index + 1] = beta
        tri[index + 1, index] = beta
    eigenvalues, eigenvectors = np.linalg.eigh(tri)
    lowest_index = int(np.argmin(eigenvalues))
    q_matrix = np.column_stack(q_vectors)
    direction = q_matrix @ eigenvectors[:, lowest_index]
    direction = normalize(direction.reshape(state.n_atoms, 3), state.fixed_mask)
    curvature, _ = rayleigh(state, proposal, direction)
    align_n0 = float(abs(np.dot(direction, n0))) if n0 is not None else None
    return {
        "direction": direction,
        "ritz_value": float(eigenvalues[lowest_index]),
        "rayleigh": curvature,
        "hvp_count": hvp_count,
        "basis_size": n,
        "q_matrix": q_matrix,
        "projected_eigenvalues": eigenvalues,
        "projected_eigenvectors": eigenvectors,
        "alignment_n0": align_n0,
        "penalty_alpha": float(penalty_alpha),
    }


def short_relax_probe(
    state: State,
    calculator: ASECalculator,
    direction: np.ndarray,
    step: float,
    n_steps: int = 5,
    alpha: float = 0.03,
) -> dict[str, float]:
    start = calculator.evaluate(state)
    displaced = state.displaced(direction, step)
    current = displaced
    initial_delta_norm = float(np.linalg.norm(displaced.positions - state.positions))
    for _ in range(n_steps):
        result = calculator.evaluate(current)
        positions = current.positions - alpha * result.gradient
        positions[current.fixed_mask] = state.positions[current.fixed_mask]
        current = State(
            numbers=current.numbers,
            positions=positions,
            cell=None if current.cell is None else current.cell.copy(),
            pbc=current.pbc,
            fixed_mask=current.fixed_mask.copy(),
            metadata=current.metadata.copy(),
        )
    end = calculator.evaluate(current)
    remaining = float(np.linalg.norm(current.positions - state.positions))
    return {
        "probe_step": float(step),
        "displaced_energy_delta": float(calculator.evaluate(displaced).energy - start.energy),
        "micro_energy_delta": float(end.energy - start.energy),
        "micro_remaining_fraction": remaining / max(initial_delta_norm, 1e-12),
    }


def candidate_rows(
    system: str,
    state: State,
    calculator: ASECalculator,
    proposal: ProposalPotential,
    reference_direction: np.ndarray | None,
    rng: np.random.Generator,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    config = LSSSWConfig(
        rng_seed=42,
        oracle_candidates=spec["oracle_candidates"],
        n_bond_pairs=spec["n_bond_pairs"],
    )
    candidate_walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    choice_walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    candidates = candidate_walker.oracle.generator.generate(
        state,
        previous_direction=None,
        n_bond_pairs=spec["n_bond_pairs"],
    )
    scored = choice_walker.oracle.choose_direction(
        state,
        proposal,
        previous_direction=None,
        score_sigma=1.0,
        n_bond_pairs=spec["n_bond_pairs"],
    )
    rows: list[dict[str, Any]] = []
    selected_direction = scored.direction
    seed_modes: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        curvature, _ = rayleigh(state, proposal, candidate.direction)
        seed_modes.append(
            {
                "label": f"{candidate.kind.value}_{index}",
                "kind": candidate.kind.value,
                "direction": normalize(candidate.direction, state.fixed_mask),
                "rayleigh": curvature,
            }
        )
        rows.append(
            direction_record(
                system=system,
                method="pool_candidate",
                label=f"{candidate.kind.value}_{index}",
                direction=candidate.direction,
                curvature=curvature,
                hvp_count=1,
                reference_direction=reference_direction,
                selected_direction=selected_direction,
                extra={"kind": candidate.kind.value, "score": None},
                state=state,
                calculator=calculator,
            )
        )
    selected_curvature, _ = rayleigh(state, proposal, selected_direction)
    seed_modes.append(
        {
            "label": f"selected_{scored.kind.value}",
            "kind": scored.kind.value,
            "direction": normalize(selected_direction, state.fixed_mask),
            "rayleigh": selected_curvature,
        }
    )
    rows.append(
        direction_record(
            system=system,
            method="scored_pool_selected",
            label=scored.kind.value,
            direction=selected_direction,
            curvature=selected_curvature,
            hvp_count=len(candidates) + 1,
            reference_direction=reference_direction,
            selected_direction=selected_direction,
            extra={"kind": scored.kind.value, "score": scored.score, "candidate_count": scored.candidate_count},
            state=state,
            calculator=calculator,
        )
    )
    return rows, selected_direction, seed_modes


def direction_record(
    *,
    system: str,
    method: str,
    label: str,
    direction: np.ndarray,
    curvature: float,
    hvp_count: int,
    reference_direction: np.ndarray | None,
    selected_direction: np.ndarray | None,
    extra: dict[str, Any],
    state: State,
    calculator: ASECalculator,
) -> dict[str, Any]:
    direction = normalize(direction.reshape(state.n_atoms, 3), state.fixed_mask)
    force = -calculator.evaluate(state).gradient.reshape(-1)
    probes = [short_relax_probe(state, calculator, direction, step) for step in (0.05, 0.10, 0.20)]
    row = {
        "system": system,
        "method": method,
        "label": label,
        "rayleigh": float(curvature),
        "hvp_count": int(hvp_count),
        "force_projection": float(np.dot(force, direction)),
        "overlap_lanczos32": (
            float(abs(np.dot(direction, reference_direction))) if reference_direction is not None else None
        ),
        "overlap_scored_selected": (
            float(abs(np.dot(direction, selected_direction))) if selected_direction is not None else None
        ),
    }
    for probe in probes:
        step = str(probe["probe_step"]).replace(".", "p")
        row[f"energy_delta_step_{step}"] = probe["displaced_energy_delta"]
        row[f"micro_energy_delta_step_{step}"] = probe["micro_energy_delta"]
        row[f"micro_remaining_fraction_step_{step}"] = probe["micro_remaining_fraction"]
    row.update(extra)
    return row


def penalized_lanczos_rows(
    system: str,
    state: State,
    calculator: ASECalculator,
    proposal: ProposalPotential,
    reference_direction: np.ndarray | None,
    selected_direction: np.ndarray | None,
    seed_modes: list[dict[str, Any]],
    dim: int,
    alphas: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seed_modes[:6]):
        n0 = seed["direction"]
        for alpha in alphas:
            result = lanczos_lowest(
                state,
                proposal,
                dim=dim,
                rng=np.random.default_rng(1000 + seed_index),
                initial_direction=n0,
                penalty_alpha=alpha,
            )
            rows.append(
                direction_record(
                    system=system,
                    method="penalized_lanczos",
                    label=f"{seed['label']}_m{dim}_alpha{alpha:g}",
                    direction=result["direction"],
                    curvature=result["rayleigh"],
                    hvp_count=result["hvp_count"],
                    reference_direction=reference_direction,
                    selected_direction=selected_direction,
                    extra={
                        "seed_label": seed["label"],
                        "seed_kind": seed["kind"],
                        "seed_rayleigh": seed["rayleigh"],
                        "basis_size": result["basis_size"],
                        "effective_ritz_value": result["ritz_value"],
                        "alignment_n0": result["alignment_n0"],
                        "penalty_alpha": alpha,
                    },
                    state=state,
                    calculator=calculator,
                )
            )
    return rows


def constrained_ritz_rows(
    system: str,
    state: State,
    calculator: ASECalculator,
    proposal: ProposalPotential,
    reference_direction: np.ndarray | None,
    selected_direction: np.ndarray | None,
    seed_modes: list[dict[str, Any]],
    dim: int,
    etas: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seed_modes[:6]):
        n0 = seed["direction"]
        result = lanczos_lowest(
            state,
            proposal,
            dim=dim,
            rng=np.random.default_rng(2000 + seed_index),
            initial_direction=n0,
        )
        q_matrix = result["q_matrix"]
        eigenvalues = result["projected_eigenvalues"]
        eigenvectors = result["projected_eigenvectors"]
        candidates: list[dict[str, Any]] = []
        for ritz_index in range(len(eigenvalues)):
            direction = normalize(q_matrix @ eigenvectors[:, ritz_index], state.fixed_mask)
            curvature, _ = rayleigh(state, proposal, direction)
            candidates.append(
                {
                    "direction": direction,
                    "rayleigh": curvature,
                    "ritz_index": ritz_index,
                    "ritz_value": float(eigenvalues[ritz_index]),
                    "alignment_n0": float(abs(np.dot(direction, n0))),
                }
            )
        for eta in etas:
            eligible = [candidate for candidate in candidates if candidate["alignment_n0"] >= eta]
            fallback_used = 0
            if not eligible:
                eligible = candidates
                fallback_used = 1
            best = min(eligible, key=lambda item: item["rayleigh"])
            rows.append(
                direction_record(
                    system=system,
                    method="constrained_ritz",
                    label=f"{seed['label']}_m{dim}_eta{eta:g}",
                    direction=best["direction"],
                    curvature=best["rayleigh"],
                    hvp_count=result["hvp_count"] + len(candidates),
                    reference_direction=reference_direction,
                    selected_direction=selected_direction,
                    extra={
                        "seed_label": seed["label"],
                        "seed_kind": seed["kind"],
                        "seed_rayleigh": seed["rayleigh"],
                        "basis_size": result["basis_size"],
                        "selected_ritz_index": best["ritz_index"],
                        "selected_ritz_value": best["ritz_value"],
                        "alignment_n0": best["alignment_n0"],
                        "alignment_eta": eta,
                        "fallback_used": fallback_used,
                    },
                    state=state,
                    calculator=calculator,
                )
            )
    return rows


def dimer_rows(
    system: str,
    state: State,
    calculator: ASECalculator,
    proposal: ProposalPotential,
    reference_direction: np.ndarray | None,
    selected_direction: np.ndarray | None,
    rng: np.random.Generator,
    samples_per_strength: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def evaluate_forces(positions: np.ndarray) -> tuple[float, np.ndarray]:
        trial_positions = np.asarray(positions, dtype=float).copy()
        trial_positions[state.fixed_mask] = state.positions[state.fixed_mask]
        trial_state = State(
            numbers=state.numbers,
            positions=trial_positions,
            cell=None if state.cell is None else state.cell.copy(),
            pbc=state.pbc,
            fixed_mask=state.fixed_mask.copy(),
            metadata=state.metadata.copy(),
        )
        result = calculator.evaluate(trial_state)
        forces = -result.gradient
        forces[state.fixed_mask] = 0.0
        return float(result.energy), forces

    for bias_strength in (500.0, 50.0, 10.0, 5.0, 1.0, 0.0):
        for sample_index in range(samples_per_strength):
            initial, info = sample_mixed_mode(
                state.positions,
                min_distance=3.0,
                cell=state.cell,
                pbc=state.pbc,
                rng=rng,
            )
            initial[state.fixed_mask] = 0.0
            initial = normalize(initial, state.fixed_mask).reshape(state.n_atoms, 3)
            result = ReferenceDimerRotator(
                delta=0.005,
                bias_strength=max(bias_strength, 1e-12),
                max_steps=30,
                rotation_tol=0.01,
                angular_step=0.02,
            ).rotate(
                state.positions,
                initial,
                evaluate_forces,
                lambda_value=float(info["lambda"]),
                local_pair=info["pair"],
            )
            direction = normalize(result.direction, state.fixed_mask)
            curvature, _ = rayleigh(state, proposal, direction)
            rows.append(
                direction_record(
                    system=system,
                    method="reference_dimer",
                    label=f"a{bias_strength:g}_sample{sample_index}",
                    direction=direction,
                    curvature=curvature,
                    hvp_count=31,
                    reference_direction=reference_direction,
                    selected_direction=selected_direction,
                    extra={
                        "bias_strength": bias_strength,
                        "rotations": result.rotations,
                        "converged": int(result.converged),
                        "dot_initial": float(result.dot_initial),
                        "curvature_force_signed": result.curvature_true,
                        "curvature_biased": result.curvature_biased,
                    },
                    state=state,
                    calculator=calculator,
                )
            )
    return rows


def run_system(system: str, device: str, dimer_samples: int) -> list[dict[str, Any]]:
    spec = SYSTEMS[system]
    state = state_from_input(system, spec)
    calculator = make_calculator(spec["model"], device)
    proposal = ProposalPotential(calculator)
    rng = np.random.default_rng(42)
    append_event({"event": "system_start", "system": system, "device": device, "time": time.time()})
    lanczos_rows: list[dict[str, Any]] = []
    lanczos_results = {}
    for dim in (8, 16, 32):
        result = lanczos_lowest(state, proposal, dim=dim, rng=rng)
        lanczos_results[dim] = result
    reference_direction = lanczos_results[32]["direction"]
    selected_rows, selected_direction, seed_modes = candidate_rows(
        system,
        state,
        calculator,
        proposal,
        reference_direction,
        rng,
        spec,
    )
    for dim, result in lanczos_results.items():
        lanczos_rows.append(
            direction_record(
                system=system,
                method="lanczos",
                label=f"m{dim}",
                direction=result["direction"],
                curvature=result["rayleigh"],
                hvp_count=result["hvp_count"],
                reference_direction=reference_direction,
                selected_direction=selected_direction,
                extra={"basis_size": result["basis_size"], "ritz_value": result["ritz_value"]},
                state=state,
                calculator=calculator,
            )
        )
    rows = (
        lanczos_rows
        + selected_rows
        + penalized_lanczos_rows(
            system,
            state,
            calculator,
            proposal,
            reference_direction,
            selected_direction,
            seed_modes,
            dim=12,
            alphas=(1.0, 5.0, 20.0),
        )
        + constrained_ritz_rows(
            system,
            state,
            calculator,
            proposal,
            reference_direction,
            selected_direction,
            seed_modes,
            dim=12,
            etas=(0.3, 0.6),
        )
        + dimer_rows(
            system,
            state,
            calculator,
            proposal,
            reference_direction,
            selected_direction,
            rng,
            dimer_samples,
        )
    )
    append_event({"event": "system_done", "system": system, "rows": len(rows), "time": time.time()})
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    grouped = []
    for system in sorted({row["system"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["system"] == system}):
            subset = [row for row in rows if row["system"] == system and row["method"] == method]
            grouped.append(
                {
                    "system": system,
                    "method": method,
                    "n": len(subset),
                    "rayleigh_min": float(np.min([row["rayleigh"] for row in subset])),
                    "rayleigh_mean": float(np.mean([row["rayleigh"] for row in subset])),
                    "hvp_mean": float(np.mean([row["hvp_count"] for row in subset])),
                    "overlap_lanczos32_mean": float(
                        np.mean([row["overlap_lanczos32"] for row in subset if row["overlap_lanczos32"] is not None])
                    ),
                    "overlap_scored_selected_mean": float(
                        np.mean(
                            [
                                row["overlap_scored_selected"]
                                for row in subset
                                if row["overlap_scored_selected"] is not None
                            ]
                        )
                    ),
                    "micro_remaining_0p2_mean": float(
                        np.mean([row["micro_remaining_fraction_step_0p2"] for row in subset])
                    ),
                }
            )
    RESULTS_JSON.write_text(safe_json({"rows": rows, "summary": grouped}), encoding="utf-8")
    lines = [
        "# Soft-Mode Eigen Experiment Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "## Aggregate",
        "",
    ]
    for item in grouped:
        lines.append(
            "- {system} / {method}: n={n}, rayleigh_min={rmin:.6g}, rayleigh_mean={rmean:.6g}, "
            "hvp_mean={hvp:.1f}, overlap_lanczos32={ol:.3f}, overlap_scored={os:.3f}, "
            "micro_remaining_0p2={mr:.3f}".format(
                system=item["system"],
                method=item["method"],
                n=item["n"],
                rmin=item["rayleigh_min"],
                rmean=item["rayleigh_mean"],
                hvp=item["hvp_mean"],
                ol=item["overlap_lanczos32_mean"],
                os=item["overlap_scored_selected_mean"],
                mr=item["micro_remaining_0p2_mean"],
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ARTIFACTS_JSON.write_text(
        safe_json(
            {
                "artifacts": [
                    {"artifact_id": "results_csv", "type": "csv", "path": str(RESULTS_CSV)},
                    {"artifact_id": "results_json", "type": "json", "path": str(RESULTS_JSON)},
                    {"artifact_id": "summary_md", "type": "markdown", "path": str(SUMMARY_MD)},
                ]
            }
        ),
        encoding="utf-8",
    )
    STATUS_MD.write_text(
        "# Status\n\n- Phase: completed\n- Completed systems: 3/3\n- Current step: summarize\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SSW direction engines by local soft-mode metrics.")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SYSTEMS),
        default=("c60", "cuo", "pdo"),
        help="Systems to evaluate.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="MACE device.")
    parser.add_argument(
        "--dimer-samples",
        type=int,
        default=8,
        help="Number of mixed-mode dimer samples per bias strength.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this approved run")
    rows: list[dict[str, Any]] = []
    for system in args.systems:
        STATUS_MD.write_text(
            f"# Status\n\n- Phase: running\n- Current system: `{system}`\n",
            encoding="utf-8",
        )
        rows.extend(run_system(system, device, args.dimer_samples))
        write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
