from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pamssw.relax as relax_module
from pamssw.bias import GaussianBiasTerm
from pamssw.relax import Relaxer
from pamssw.state import State
from pamssw.walker import ProposalPotential


BACKENDS = (
    "ase-fire",
    "ase-fire2",
    "safe-lbfgs-total",
    "bias-separated-lbfgs",
)


@dataclass(frozen=True)
class _Case:
    case_id: str
    state: State
    hessian_diagonal: np.ndarray
    biases: tuple[GaussianBiasTerm, ...]
    fmax: float = 1.0e-6
    maxiter: int = 400


class _CountingQuadraticCalculator:
    def __init__(self, hessian_diagonal: np.ndarray) -> None:
        self.hessian_diagonal = np.asarray(hessian_diagonal, dtype=float).reshape(-1)
        self.calls = 0

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        if flat.shape != self.hessian_diagonal.shape:
            raise ValueError("quadratic dimension mismatch")
        gradient = self.hessian_diagonal * flat
        energy = 0.5 * float(np.dot(flat, gradient))
        return energy, gradient


def _cases() -> tuple[_Case, ...]:
    state_a = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.8, -0.4, 0.3], [-0.6, 0.5, -0.2]]),
    )
    state_b = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.5, -0.7, 0.4], [-0.4, 0.6, -0.5]]),
    )
    return (
        _Case(
            case_id="anisotropic_one_hill",
            state=state_a,
            hessian_diagonal=np.array([1.0, 2.0, 4.0, 3.0, 1.5, 2.5]),
            biases=(
                GaussianBiasTerm(
                    center=np.zeros(6),
                    direction=np.array([1.0, 0.5, 0.0, -0.5, 0.0, 0.0]),
                    sigma=0.7,
                    weight=0.2,
                ),
            ),
        ),
        _Case(
            case_id="anisotropic_two_hills",
            state=state_b,
            hessian_diagonal=np.array([0.8, 1.2, 3.5, 2.8, 1.7, 4.2]),
            biases=(
                GaussianBiasTerm(
                    center=np.zeros(6),
                    direction=np.array([1.0, 0.0, 0.5, 0.0, -0.5, 0.0]),
                    sigma=0.8,
                    weight=0.15,
                ),
                GaussianBiasTerm(
                    center=np.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0]),
                    direction=np.array([0.0, 1.0, 0.0, 0.0, 0.5, -0.5]),
                    sigma=0.6,
                    weight=0.1,
                ),
            ),
        ),
    )


def _unavailable_row(backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "available": False,
        "objective_calls": 0,
        "telemetry_evaluator_calls": 0,
        "backend_evaluations": 0,
        "converged": False,
        "termination_reason": "unavailable",
        "iterations": 0,
        "final_energy": None,
        "final_max_force": None,
        "accepted_steps": None,
        "rejected_steps": None,
        "accepted_secants": None,
        "rejected_secants": None,
        "line_search_evaluations": None,
        "mic_branch_resets": None,
        "bias_secant_curvature_sum": None,
        "final_positions": None,
    }


def _run_case(case: _Case, backend: str) -> dict[str, Any]:
    if backend == "ase-fire2" and relax_module._ASE_FIRE2 is None:
        return _unavailable_row(backend)
    calculator = _CountingQuadraticCalculator(case.hessian_diagonal)
    proposal = ProposalPotential(calculator, biases=list(case.biases))
    custom = backend in {"safe-lbfgs-total", "bias-separated-lbfgs"}
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer=backend,
        component_evaluator=proposal.evaluate_parts if custom else None,
    )
    result = relaxer.relax(case.state, fmax=case.fmax, maxiter=case.maxiter)
    telemetry = result.telemetry
    return {
        "backend": backend,
        "available": True,
        "objective_calls": calculator.calls,
        "telemetry_evaluator_calls": telemetry.evaluator_calls,
        "backend_evaluations": telemetry.backend_evaluations,
        "converged": telemetry.converged,
        "termination_reason": telemetry.termination_reason,
        "iterations": result.n_iter,
        "final_energy": result.energy,
        "final_max_force": result.gradient_norm,
        "accepted_steps": telemetry.accepted_steps if custom else None,
        "rejected_steps": telemetry.rejected_steps if custom else None,
        "accepted_secants": telemetry.accepted_secants if custom else None,
        "rejected_secants": telemetry.rejected_secants if custom else None,
        "line_search_evaluations": telemetry.line_search_evaluations if custom else None,
        "mic_branch_resets": telemetry.mic_branch_resets if custom else None,
        "bias_secant_curvature_sum": telemetry.bias_secant_curvature_sum if custom else None,
        "final_positions": result.state.positions.tolist(),
    }


def run_comparison(output: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "backends": list(BACKENDS),
        "cases": [],
    }
    for case in _cases():
        payload["cases"].append(
            {
                "case_id": case.case_id,
                "initial_numbers": case.state.numbers.tolist(),
                "initial_positions": case.state.positions.tolist(),
                "cell": None if case.state.cell is None else case.state.cell.tolist(),
                "pbc": list(case.state.pbc),
                "fixed_mask": case.state.fixed_mask.tolist(),
                "hessian_diagonal": case.hessian_diagonal.tolist(),
                "biases": [
                    {
                        "center": bias.center.tolist(),
                        "direction": bias.direction.tolist(),
                        "sigma": bias.sigma,
                        "weight": bias.weight,
                    }
                    for bias in case.biases
                ],
                "fmax": case.fmax,
                "maxiter": case.maxiter,
                "runs": [_run_case(case, backend) for backend in BACKENDS],
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw deterministic proposal-relaxation ablation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_comparison(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
