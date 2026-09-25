"""Bounded Cu13/EMT Hookean composition and restart interface check.

This checks existing APIs only. It is not a Hookean parameter study or an
SSW efficacy benchmark. Run only after reviewing README.md and obtaining the
planned CPU allocation.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "research/ga_ssw/evidence/startup-order-20260926/runs/input.extxyz"
OUT = HERE / "runs"
SEED = 25092531
TOTAL_EF_CAP = 6000
TRAJECTORY_EF_CAP = 3000
WALL_SECONDS = 270.0


def _jsonable(value):
    from ase import Atoms

    if isinstance(value, Atoms):
        return {
            "numbers": value.numbers.tolist(),
            "positions": value.positions.tolist(),
            "cell": value.cell.array.tolist(),
            "pbc": value.pbc.tolist(),
            "constraints": [c.todict() for c in value.constraints],
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()}
    return value


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


class EvaluationBudget:
    """Count calculator E/F API requests across every phase of this run."""

    def __init__(self):
        self.started = time.monotonic()
        self.total = 0
        self.by_label = {}
        self.denials = []

    def counted(self, surface, label, per_trajectory=False):
        budget = self

        class CountedSurface:
            @property
            def requests(self):
                return surface.requests

            @property
            def exhausted(self):
                return self.requests >= TOTAL_EF_CAP

            def evaluate(self, atoms):
                local = budget.by_label.get(label, 0)
                reason = None
                if budget.total >= TOTAL_EF_CAP:
                    reason = "total_ef_cap"
                elif per_trajectory and local >= TRAJECTORY_EF_CAP:
                    reason = "trajectory_ef_cap"
                elif time.monotonic() - budget.started >= WALL_SECONDS:
                    reason = "wall_cap"
                if reason:
                    budget.denials.append({"label": label, "reason": reason,
                                           "total_requests": budget.total,
                                           "label_requests": local})
                    raise RuntimeError(f"qualification budget exhausted: {reason}")
                budget.total += 1
                budget.by_label[label] = local + 1
                return surface.evaluate(atoms)

        return CountedSurface()


def _make_bound_surface(label, specs, budget, *, trajectory=False):
    from ase.calculators.emt import EMT
    from pamssw.standalone import ASESurface
    from pamssw.standalone.ase_constraints import bind_hookean_surface

    physical = budget.counted(ASESurface(EMT()), label,
                              per_trajectory=trajectory)
    return bind_hookean_surface(physical, specs)


def _comparison_state(result):
    checkpoint = result.checkpoint
    return _jsonable({
        "initial": result.initial,
        "current": result.current,
        "best": result.best,
        "minima": result.minima,
        "records": result.records,
        "evaluation_requests": result.evaluation_requests,
        "checkpoint_rng": checkpoint.rng_state,
        "recovered_direction_state": checkpoint.recovered_direction_state,
    })


def _main():
    from ase.calculators.emt import EMT
    from ase.constraints import Hookean
    from ase.io import read
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.paper_reference import load_ssw_checkpoint
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    from pamssw.standalone.ase_constraints import bind_hookean_surface, normalize_constraints
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings

    if not INPUT.is_file():
        raise FileNotFoundError(f"required startup-order input missing: {INPUT}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(INPUT, OUT / "input.extxyz")
    input_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    atoms = read(INPUT)
    if len(atoms) != 13 or set(atoms.get_chemical_symbols()) != {"Cu"}:
        raise ValueError("startup-order input must be the saved Cu13 structure")
    if atoms.pbc.any() or atoms.constraints:
        raise ValueError("expected a free, nonperiodic, initially unconstrained Cu13")

    # One explicit pair spring; this is an active interface stress, not a
    # confinement choice or a searched/tuned parameter.
    pair = (0, 1)
    threshold = 0.9 * float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    hookean = Hookean(pair[0], pair[1], k=1.0, rt=threshold)
    constrained_input = atoms.copy()
    constrained_input.set_constraint(hookean)
    constraints = normalize_constraints(constrained_input)
    clean_atoms = constraints.clean_atoms(constrained_input)
    specs = tuple(constraints.hookean_specs)
    if constraints.fixed_indices or len(specs) != 1:
        raise AssertionError("expected exactly one pair Hookean and no fixed atoms")
    if clean_atoms.constraints:
        raise AssertionError("clean_atoms did not remove ASE constraints")
    original_positions = constrained_input.positions.copy()
    initial_pair_distance = float(np.linalg.norm(
        original_positions[pair[1]] - original_positions[pair[0]]))

    config = SSWConfig(
        width=0.6, rotation_bias=1.0, max_gaussians=6,
        temperature_K=300.0, fmax=0.03, bias_fmax=0.1,
        relax_steps=1000, fd_step=0.001, rotation_hvp=39,
        rotation_tol=0.02, forward_force=0.1,
        direction_sampling="global", rotation_solver="broyden-euclidean",
        cluster_frame="direction_only", quench_optimizer="safe-lbfgs-total",
        lbfgs_memory=500, rotation_exit_policy="force_or_budget",
    )
    direction = RecoveredDirectionSettings(
        50, 0.5, 0.5, 5, 15, 0.2, 0.02, "euclidean", 40,
        startup_order="randomized",
    )
    budget = EvaluationBudget()
    protocol = {
        "purpose": "Cu13/EMT interface stress check; not search efficacy or Hookean tuning",
        "source_input": str(INPUT),
        "input_sha256": input_sha256,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": SEED,
        "config": config,
        "recovered_direction": direction,
        "hookean": {"indices": list(pair), "k_eV_A2": 1.0,
                    "rt_A": threshold, "initial_pair_distance_A": initial_pair_distance},
        "budget": {"total_EF_requests": TOTAL_EF_CAP,
                   "per_trajectory_EF_requests": TRAJECTORY_EF_CAP,
                   "internal_wall_seconds": WALL_SECONDS},
        "native_ls": "omitted to keep this an isolated wrapper/direction/checkpoint composition check",
        "pool": "omitted; no selector required for exact two-step restart comparison",
        "outputs": [
            "input.extxyz", "protocol.json", "preflight.json",
            "continuous/result.json", "continuous/checkpoint.pkl",
            "split/paused-result.json", "split/paused-checkpoint.pkl",
            "split/result.json", "split/checkpoint.pkl",
            "continuity.json", "identity-check.json", "endpoint-check.json",
            "summary.json", "failure.json (only if execution fails)",
        ],
    }
    _dump(OUT / "protocol.json", protocol)

    try:
        preflight = _make_bound_surface("preflight", specs, budget)
        augmented_energy, augmented_forces = preflight.evaluate(clean_atoms)
        initial_terms = preflight.last_evaluation
        if initial_terms is None:
            raise AssertionError("Hookean wrapper did not expose its decomposition")
        correction_norm = float(np.linalg.norm(initial_terms["hookean_forces"]))
        if not initial_terms["hookean_energy"] > 0.0 or not correction_norm > 0.0:
            raise AssertionError("chosen pair Hookean was not active at the initial structure")
        if not np.isclose(augmented_energy,
                          initial_terms["physical_energy"] + initial_terms["hookean_energy"],
                          rtol=0.0, atol=1e-12):
            raise AssertionError("augmented energy does not equal physical plus one Hookean term")
        if not np.allclose(augmented_forces,
                           initial_terms["physical_forces"] + initial_terms["hookean_forces"],
                           rtol=0.0, atol=1e-12):
            raise AssertionError("augmented forces do not equal physical plus one Hookean term")
        if preflight.requests != 1:
            raise AssertionError("one wrapper evaluation must cause exactly one physical E/F request")
        _dump(OUT / "preflight.json", {
            "pair": list(pair), "distance_A": initial_pair_distance,
            "hookean_spec": specs, "physical_energy_eV": initial_terms["physical_energy"],
            "hookean_energy_eV": initial_terms["hookean_energy"],
            "augmented_energy_eV": augmented_energy,
            "hookean_force_norm_eV_A": correction_norm,
            "physical_requests": preflight.requests,
            "one_physical_request_per_augmented_evaluation": True,
            "input_geometry_preserved": bool(np.array_equal(
                constrained_input.positions, original_positions)),
        })

        def execute(label, steps, *, checkpoint=None, checkpoint_path, pause=False,
                    rng_seed=SEED):
            surface = _make_bound_surface(label, specs, budget, trajectory=True)
            result = run_ssw(
                clean_atoms, surface, steps=steps, config=config,
                rng=np.random.default_rng(rng_seed),
                recovered_direction=direction, checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                progress_callback=(lambda event: bool(
                    pause and event.next_index == 1)),
            )
            if result.evaluation_requests != budget.by_label.get(label, 0):
                raise AssertionError(f"{label}: checkpoint/result cost differs from physical requests")
            expected = result.initial.evaluation_requests + sum(
                record.evaluation_requests for record in result.records)
            if result.evaluation_requests != expected:
                raise AssertionError(f"{label}: record request ledger does not sum to result cost")
            return result, surface.specs

        continuous_dir = OUT / "continuous"
        split_dir = OUT / "split"
        continuous_dir.mkdir()
        split_dir.mkdir()
        continuous, continuous_specs = execute(
            "continuous", 2, checkpoint_path=continuous_dir / "checkpoint.pkl")
        if continuous.status != "completed" or len(continuous.records) != 2:
            raise AssertionError("continuous arm did not complete exactly two outer steps")
        _dump(continuous_dir / "result.json", continuous)

        paused, paused_specs = execute(
            "split", 2, checkpoint_path=split_dir / "checkpoint.pkl", pause=True)
        if paused.status != "paused" or paused.checkpoint is None or paused.checkpoint.next_index != 1:
            raise AssertionError("split arm did not pause after exactly one outer step")
        _dump(split_dir / "paused-result.json", paused)
        save_ssw_checkpoint(split_dir / "paused-checkpoint.pkl", paused.checkpoint)
        boundary = load_ssw_checkpoint(split_dir / "paused-checkpoint.pkl")
        if boundary.next_index != 1:
            raise AssertionError("on-disk boundary checkpoint did not retain next_index=1")

        resumed, resumed_specs = execute(
            "split", 1, checkpoint=boundary,
            checkpoint_path=split_dir / "checkpoint.pkl", rng_seed=999)
        if resumed.status != "completed" or len(resumed.records) != 2:
            raise AssertionError("resumed arm did not complete the same two outer steps")
        _dump(split_dir / "result.json", resumed)

        continuous_state = _comparison_state(continuous)
        resumed_state = _comparison_state(resumed)
        records_equal = continuous_state["records"] == resumed_state["records"]
        rng_equal = continuous.checkpoint.rng_state == resumed.checkpoint.rng_state
        direction_equal = (
            continuous_state["recovered_direction_state"] ==
            resumed_state["recovered_direction_state"])
        cost_equal = continuous.evaluation_requests == resumed.evaluation_requests
        exact_state_equal = continuous_state == resumed_state
        specs_equal = (continuous_specs == paused_specs == resumed_specs == specs)
        if not (records_equal and rng_equal and direction_equal and cost_equal and
                exact_state_equal and specs_equal):
            raise AssertionError("continuous and paused/resumed two-step states differ")
        if budget.by_label.get("continuous", 0) > TRAJECTORY_EF_CAP:
            raise AssertionError("continuous trajectory exceeded its E/F cap")
        if budget.by_label.get("split", 0) > TRAJECTORY_EF_CAP:
            raise AssertionError("split trajectory exceeded its E/F cap")
        _dump(OUT / "continuity.json", {
            "continuous_status": continuous.status,
            "resumed_status": resumed.status,
            "steps_each": 2,
            "records_exact": records_equal,
            "main_rng_exact": rng_equal,
            "recovered_direction_state_exact": direction_equal,
            "total_request_cost_exact": cost_equal,
            "complete_compared_state_exact": exact_state_equal,
            "continuous_requests": budget.by_label.get("continuous", 0),
            "split_requests_across_pause_and_resume": budget.by_label.get("split", 0),
            "hookean_specs_equal_across_all_walk_segments": specs_equal,
            "input_geometry_preserved": bool(np.array_equal(
                constrained_input.positions, original_positions)),
        })

        # Deliberately test only whether the ordinary SSW checkpoint carries
        # the wrapped oracle identity. The changed-spec result is not a valid
        # continuation under the original scientific objective.
        changed_input = atoms.copy()
        changed_input.set_constraint(Hookean(pair[0], pair[1], k=2.0, rt=threshold))
        changed_constraints = normalize_constraints(changed_input)
        changed_clean = changed_constraints.clean_atoms(changed_input)
        changed_surface = _make_bound_surface(
            "changed_identity_zero_step", changed_constraints.hookean_specs, budget)
        identity = {"saved_hookean_specs": specs,
                    "changed_hookean_specs": changed_constraints.hookean_specs,
                    "scientific_interpretation": (
                        "oracle-identity boundary only; never use this zero-step result "
                        "as a changed-potential scientific continuation")}
        try:
            changed_resume = run_ssw(
                changed_clean, changed_surface, steps=0, config=config,
                rng=np.random.default_rng(777), recovered_direction=direction,
                checkpoint=boundary)
            identity.update({"accepted": True,
                             "status": changed_resume.status,
                             "new_physical_requests": changed_surface.requests,
                             "saved_rng_unchanged": (
                                 changed_resume.checkpoint.rng_state == boundary.rng_state)})
            if changed_surface.requests != 0:
                raise AssertionError("zero-step changed-spec identity check made a PES request")
        except Exception as error:
            identity.update({"accepted": False, "exception": repr(error),
                             "new_physical_requests": changed_surface.requests})
        _dump(OUT / "identity-check.json", identity)
        if changed_surface.requests != 0:
            raise AssertionError('identity-only resume unexpectedly invoked the PES')

        best = resumed.best.copy()
        raw_surface = budget.counted(ASESurface(EMT()), "endpoint_raw")
        raw_energy, raw_forces = raw_surface.evaluate(best)
        augmented_surface = _make_bound_surface(
            "endpoint_augmented", specs, budget)
        augmented_final_energy, augmented_final_forces = augmented_surface.evaluate(best)
        final_terms = augmented_surface.last_evaluation
        if final_terms is None:
            raise AssertionError("final wrapper evaluation has no component ledger")
        energy_recomposition = np.isclose(
            augmented_final_energy,
            final_terms["physical_energy"] + final_terms["hookean_energy"],
            rtol=0.0, atol=1e-12)
        force_recomposition = np.allclose(
            augmented_final_forces,
            final_terms["physical_forces"] + final_terms["hookean_forces"],
            rtol=0.0, atol=1e-12)
        physical_match = (np.isclose(raw_energy, final_terms["physical_energy"],
                                     rtol=0.0, atol=1e-12) and
                          np.allclose(raw_forces, final_terms["physical_forces"],
                                      rtol=0.0, atol=1e-12))
        if not (energy_recomposition and force_recomposition and physical_match):
            raise AssertionError("raw/augmented final checks do not recompose exactly once")
        if not np.isclose(augmented_final_energy, resumed.checkpoint.best.energy, rtol=0., atol=1e-10):
            raise AssertionError('fresh augmented energy differs from stored best')
        # Independent ASE Atoms constraint machinery, without our surface wrapper.
        direct = constraints.attach(best)
        direct.calc = EMT()
        if budget.total >= TOTAL_EF_CAP:
            raise RuntimeError("no budget for independent ASE check")
        direct.calc.calculate(direct)
        budget.total += 1
        budget.by_label['endpoint_direct_ase'] = 1
        direct_energy, direct_forces = direct.get_potential_energy(), direct.get_forces()
        if not (np.isclose(direct_energy, augmented_final_energy, rtol=0., atol=1e-12)
                and np.allclose(direct_forces, augmented_final_forces, rtol=0., atol=1e-12)):
            raise AssertionError('surface composition differs from direct ASE Hookean')
        augmented_fmax = float(np.linalg.norm(augmented_final_forces, axis=1).max())
        if augmented_fmax > config.fmax:
            raise AssertionError('augmented endpoint is not force qualified')
        _dump(OUT / "endpoint-check.json", {
            "endpoint": best,
            "raw_emt_energy_eV": raw_energy,
            "raw_emt_fmax_eV_A": float(np.linalg.norm(raw_forces, axis=1).max()),
            "augmented_energy_eV": augmented_final_energy,
            "augmented_fmax_eV_A": augmented_fmax,
            "direct_ase_energy_and_forces_match": True,
            "hookean_energy_eV": final_terms["hookean_energy"],
            "hookean_force_norm_eV_A": float(np.linalg.norm(
                final_terms["hookean_forces"])),
            "physical_component_matches_independent_raw_emt": bool(physical_match),
            "energy_recomposes_once": bool(energy_recomposition),
            "forces_recompose_once": bool(force_recomposition),
            "raw_requests": raw_surface.requests,
            "augmented_physical_requests": augmented_surface.requests,
        })

        if budget.total > TOTAL_EF_CAP:
            raise AssertionError("global E/F cap exceeded")
        if not np.array_equal(constrained_input.positions, original_positions):
            raise AssertionError("input geometry was mutated")
        summary = {
            "status": "completed",
            "total_physical_EF_requests": budget.total,
            "requests_by_phase": budget.by_label,
            "denials": budget.denials,
            "elapsed_seconds": time.monotonic() - budget.started,
            "wall_cap_seconds": WALL_SECONDS,
            "input_geometry_preserved": True,
            "hookean_active_initially": True,
            "exact_continuation": exact_state_equal,
            "changed_spec_zero_step_accepted": identity.get("accepted", False),
        }
        _dump(OUT / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except Exception as error:
        _dump(OUT / "failure.json", {
            "error": repr(error), "total_physical_EF_requests": budget.total,
            "requests_by_phase": budget.by_label, "denials": budget.denials,
            "elapsed_seconds": time.monotonic() - budget.started,
            "checkpoint_paths_may_contain_last_completed_outer_boundary": True,
        })
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "--execute":
        raise SystemExit("prepared only; pass --execute after review and CPU allocation")
    _main()
