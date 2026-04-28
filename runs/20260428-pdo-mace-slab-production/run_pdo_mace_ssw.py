from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from pamssw import SSWConfig, State, run_ssw
from pamssw.calculators import ASECalculator
from pamssw.pbc import mic_distance_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--steps-per-walk", type=int, default=4)
    parser.add_argument("--oracle-candidates", type=int, default=6)
    parser.add_argument("--proposal-relax-steps", type=int, default=80)
    parser.add_argument("--proposal-fmax", type=float, default=0.05)
    parser.add_argument("--proposal-optimizer", default="ase-fire")
    parser.add_argument("--quench-fmax", type=float, default=0.03)
    parser.add_argument("--quench-optimizer", default="scipy-lbfgsb")
    parser.add_argument("--target-uphill-energy", type=float, default=0.25)
    parser.add_argument("--min-step-scale", type=float, default=0.05)
    parser.add_argument("--max-step-scale", type=float, default=0.6)
    parser.add_argument("--proposal-trust-radius", type=_optional_float, default=0.8)
    parser.add_argument("--walk-trust-radius", type=float, default=2.5)
    parser.add_argument("--proposal-pool-size", type=int, default=1)
    parser.add_argument("--dedup-rmsd-tol", type=float, default=0.15)
    parser.add_argument("--fix-bottom-fraction", type=float, default=0.35)
    parser.add_argument("--slab-pbc-z", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "minima_xyz").mkdir(exist_ok=True)

    atoms = read(args.input)
    input_pbc = tuple(bool(x) for x in atoms.pbc.tolist())
    slab_pbc = (True, True, bool(args.slab_pbc_z))
    atoms.pbc = slab_pbc
    fixed_mask = _bottom_fixed_mask(atoms.positions, args.fix_bottom_fraction)

    state = State(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell.array,
        pbc=slab_pbc,
        fixed_mask=fixed_mask,
        metadata={"input_pbc": input_pbc, "slab_pbc": slab_pbc},
    )
    calc = MACECalculator(
        model_paths=str(args.model),
        device=args.device,
        default_dtype=args.dtype,
        inference_precision=args.dtype,
        enable_cueq=False,
    )
    calculator = ASECalculator(calc)
    config = SSWConfig(
        max_trials=args.trials,
        max_steps_per_walk=args.steps_per_walk,
        target_uphill_energy=args.target_uphill_energy,
        quench_fmax=args.quench_fmax,
        dedup_rmsd_tol=args.dedup_rmsd_tol,
        dedup_energy_tol=1e-3,
        rng_seed=args.seed,
        oracle_candidates=args.oracle_candidates,
        proposal_relax_steps=args.proposal_relax_steps,
        proposal_fmax=args.proposal_fmax,
        proposal_optimizer=args.proposal_optimizer,
        quench_optimizer=args.quench_optimizer,
        min_step_scale=args.min_step_scale,
        max_step_scale=args.max_step_scale,
        proposal_trust_radius=args.proposal_trust_radius,
        walk_trust_radius=args.walk_trust_radius,
        fragment_guard_factor=3.0,
        n_bond_pairs=2,
        proposal_pool_size=args.proposal_pool_size,
        max_prototypes=500,
    )

    result = run_ssw(state, calculator, config)
    trace = _energy_trace(result)
    _write_trace_plot(trace, output_dir / "energy_trace.png")
    (output_dir / "energy_trace.json").write_text(json.dumps(trace, indent=2))

    minima_atoms = []
    archive_entries = sorted(result.archive.entries, key=lambda entry: entry.energy)
    for rank, entry in enumerate(archive_entries):
        item = _state_to_atoms(entry.state)
        item.info.update(
            {
                "entry_id": entry.entry_id,
                "rank": rank,
                "energy": entry.energy,
                "parent_id": -1 if entry.parent_id is None else entry.parent_id,
            }
        )
        minima_atoms.append(item)
        write(output_dir / "minima_xyz" / f"minimum_rank{rank:03d}_entry{entry.entry_id:03d}.xyz", item)
    if minima_atoms:
        write(output_dir / "archive_minima.xyz", minima_atoms)
        write(output_dir / "best_minimum.xyz", minima_atoms[0])

    diagnostics = [_geometry_diagnostics(entry.state, entry.energy, entry.entry_id) for entry in archive_entries]
    summary = {
        "input": str(args.input),
        "model": str(args.model),
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": args.dtype,
        "enable_cueq": False,
        "input_pbc": input_pbc,
        "run_pbc": slab_pbc,
        "cell": state.cell.tolist() if state.cell is not None else None,
        "n_atoms": state.n_atoms,
        "n_fixed": int(np.count_nonzero(state.fixed_mask)),
        "fixed_bottom_fraction": args.fix_bottom_fraction,
        "config": asdict(config),
        "best_energy": result.best_energy,
        "n_minima": len(result.archive.entries),
        "stats": result.stats,
        "walk_history": [asdict(record) for record in result.walk_history],
        "archive_energies": [entry.energy for entry in archive_entries],
        "geometry_diagnostics": diagnostics,
        "outputs": {
            "summary": str(output_dir / "ssw_summary.json"),
            "energy_trace_json": str(output_dir / "energy_trace.json"),
            "energy_trace_png": str(output_dir / "energy_trace.png"),
            "archive_minima_xyz": str(output_dir / "archive_minima.xyz"),
            "best_minimum_xyz": str(output_dir / "best_minimum.xyz"),
            "minima_dir": str(output_dir / "minima_xyz"),
        },
    }
    (output_dir / "ssw_summary.json").write_text(json.dumps(summary, indent=2))


def _bottom_fixed_mask(positions: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros(len(positions), dtype=bool)
    threshold = float(np.quantile(positions[:, 2], min(max(fraction, 0.0), 1.0)))
    return positions[:, 2] <= threshold


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


def _state_to_atoms(state: State) -> Atoms:
    return Atoms(
        numbers=state.numbers,
        positions=state.positions,
        cell=state.cell,
        pbc=state.pbc,
    )


def _energy_trace(result) -> list[dict[str, float | int | bool]]:
    best = result.archive.entries[0].energy if result.archive.entries else result.best_energy
    points: list[dict[str, float | int | bool]] = [
        {"step": 0, "energy": float(best), "best_energy": float(best), "accepted_new_basin": True}
    ]
    for index, record in enumerate(result.walk_history, start=1):
        best = min(best, record.energy)
        points.append(
            {
                "step": index,
                "energy": float(record.energy),
                "best_energy": float(best),
                "accepted_new_basin": bool(record.accepted_new_basin),
                "seed_entry_id": int(record.seed_entry_id),
                "discovered_entry_id": int(record.discovered_entry_id),
            }
        )
    return points


def _write_trace_plot(trace: list[dict[str, float | int | bool]], output: Path) -> None:
    steps = [int(item["step"]) for item in trace]
    energies = [float(item["energy"]) for item in trace]
    best = [float(item["best_energy"]) for item in trace]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
    ax.plot(steps, energies, marker="o", label="trial minimum")
    ax.plot(steps, best, marker="s", label="best so far")
    ax.set_xlabel("SSW trial")
    ax.set_ylabel("MACE energy / eV")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _geometry_diagnostics(state: State, energy: float, entry_id: int) -> dict[str, object]:
    distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
    if state.n_atoms:
        np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1) if state.n_atoms else np.asarray([])
    periodic_positions_ok = True
    if state.cell is not None and any(state.pbc):
        lengths = np.linalg.norm(state.cell, axis=1)
        for axis, periodic in enumerate(state.pbc):
            if periodic:
                coords = state.positions[:, axis]
                periodic_positions_ok = periodic_positions_ok and bool(np.all(coords >= -1e-8))
                periodic_positions_ok = periodic_positions_ok and bool(np.all(coords <= lengths[axis] + 1e-8))
    return {
        "entry_id": int(entry_id),
        "energy": float(energy),
        "cell_present": state.cell is not None,
        "pbc": tuple(bool(x) for x in state.pbc),
        "periodic_positions_in_cell": periodic_positions_ok,
        "nearest_min": float(np.min(nearest)) if nearest.size else 0.0,
        "nearest_median": float(np.median(nearest)) if nearest.size else 0.0,
        "nearest_max": float(np.max(nearest)) if nearest.size else 0.0,
        "z_min": float(np.min(state.positions[:, 2])),
        "z_max": float(np.max(state.positions[:, 2])),
    }


if __name__ == "__main__":
    main()
