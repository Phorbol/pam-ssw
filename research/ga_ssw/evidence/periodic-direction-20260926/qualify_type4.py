"""Bounded TYPE4 TiO2@Au24O4 direction qualification; never submits a job."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rootqualify", HERE / "qualify.py")
rootqualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rootqualify)
ledger = rootqualify.ledger
SOURCE = (ROOT.parent / "ga-ssw-behavior-parity" / "research" / "ga_ssw"
          / "evidence" / "type4-source-direction-mace-v100")
MODEL = "/home/gengjianrui/.cache/mace/mace-omat-0-small.model"
FIXED = np.arange(297)       # physical FixAtoms, source indices 1..297
EXCLUDED = np.arange(351)    # direction-only exclusion, source indices 1..351
SEED, LIVE_CAP, REPLAY_CAP, FRESH_CAP, WALL_CAP = 26092631, 4000, 4000, 6, 1200


def _settings():
    from pamssw.standalone.constrained_reference import ConstrainedSSWConfig
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    rotation = RecoveredRotationSettings(5, 15, .2, .02, "euclidean", 40)
    common = dict(width=.6, rotation_bias=100., temperature_K=100., forward_force=.1,
        max_gaussians=3, gradient_tol=.1, fmax=.05, max_step=.2, relax_steps=500,
        fd_step=.001, rotation_hvp=100, rotation_tol=.02, lbfgs_memory=500,
        rotation_exit_policy="force_or_budget")
    configs = {"global_rotation": ConstrainedSSWConfig(**common, recovered_rotation=rotation),
               "periodic_local": ConstrainedSSWConfig(**common)}
    direction = RecoveredDirectionSettings(50, .5, .5, 5, 15, .2, .02, "euclidean", 40,
        c1_radius_policy="per_atom", startup_order="randomized", geometry="periodic_local")
    return configs, rotation, direction


class Oracle:
    """Live counted MACE stream, or same-oracle replay from recorded E/F."""
    def __init__(self, calculator, folder, atoms, deadline, tape=None, replay=False):
        from pamssw.standalone.surface import ASESurface
        self.base = ASESurface(calculator) if calculator is not None else None
        self.folder, self.deadline = Path(folder), deadline
        self.tape = [] if tape is None else tape
        self.phase = "replay" if replay else "live"
        self.cursor = self.live = self.replayed = self.denials = 0
        self.positions, self.cell = atoms.positions.copy(), atoms.cell.array.copy()
        self.pbc, self.numbers = atoms.pbc.copy(), atoms.numbers.copy()

    def evaluate(self, atoms):
        if (not np.array_equal(atoms.positions[FIXED], self.positions[FIXED]) or
                not np.array_equal(atoms.cell.array, self.cell) or
                not np.array_equal(atoms.pbc, self.pbc) or
                not np.array_equal(atoms.numbers, self.numbers)):
            raise AssertionError("fixed geometry/composition changed")
        if time.monotonic() >= self.deadline:
            self.denials += 1
            raise RuntimeError("combined 20-minute wall cap reached")
        if self.phase == "replay":
            if self.replayed >= REPLAY_CAP or self.cursor >= len(self.tape):
                self.denials += 1
                raise RuntimeError("same-oracle replay cap/stream exhausted")
            item = self.tape[self.cursor]
            self.cursor += 1; self.replayed += 1
            match = (np.array_equal(atoms.positions, item["positions"]) and
                     np.array_equal(atoms.cell.array, item["cell"]) and
                     np.array_equal(atoms.pbc, item["pbc"]) and
                     np.array_equal(atoms.numbers, item["numbers"]))
            ledger.append(self.folder / "replay.jsonl", dict(request=self.replayed,
                replay_of=self.cursor, match=match, atoms=atoms))
            if not match:
                raise AssertionError("split trajectory diverged from recorded local E/F stream")
            return item["energy"], item["forces"].copy()
        if self.live >= LIVE_CAP:
            self.denials += 1
            raise RuntimeError("per-arm live E/F cap reached")
        self.live += 1
        try:
            energy, forces = self.base.evaluate(atoms)
            item = dict(numbers=atoms.numbers.copy(), positions=atoms.positions.copy(),
                cell=atoms.cell.array.copy(), pbc=atoms.pbc.copy(), energy=float(energy),
                forces=np.asarray(forces, dtype=float).copy())
            ledger.append(self.folder / "requests.jsonl", dict(request=self.live,
                atoms=atoms, energy=energy, forces=forces))
            if self.phase == "record": self.tape.append(item)
            return energy, forces
        except Exception as error:
            ledger.append(self.folder / "failures.jsonl", dict(request=self.live,
                error=repr(error), atoms=atoms))
            raise


def _stage_audit(result, physical_mobile, allowed_mobile):
    routes, coeffs = Counter(), Counter()
    direction_count = multi = 0
    support_ok = True
    attempts = []
    expected_rotation = np.flatnonzero(np.repeat(allowed_mobile[physical_mobile], 3))
    for event in result.records:
        if not isinstance(event, dict): continue
        stages = event.get("climb", [])
        multi += len(stages) > 1
        attempts.append(dict(index=event.get("index"), gaussian_count=len(stages)))
        if "rotation_coordinate_indices" in event:
            support_ok &= np.array_equal(event["rotation_coordinate_indices"], expected_rotation)
        for stage in stages:
            direction = stage.get("initial_direction")
            if direction is not None:
                x = np.asarray(direction, dtype=float).reshape(-1, 3)
                if len(x) != len(physical_mobile): support_ok = False
                else:
                    support_ok &= bool(np.all(x[~allowed_mobile[physical_mobile]] == 0.))
                    direction_count += 1
            diag = stage.get("recovered_direction") or {}
            routes[str(diag.get("route", "unreported"))] += 1
            c = diag.get("coefficients")
            if c is not None and len(c) == 10:
                coeffs["c4"] += bool(c[4] > 1e-6); coeffs["c6"] += bool(c[6] > 1e-6)
    return dict(direction_records_checked=direction_count, direction_mask_support_ok=bool(support_ok),
        multi_gaussian_attempts=int(multi), attempts=attempts, route_counts=dict(routes),
        route_status="reported" if routes and "unreported" not in routes else "unreported",
        c4_coefficient_stage_count=int(coeffs["c4"]), c6_coefficient_stage_count=int(coeffs["c6"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="new evidence directory; must not exist")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists(): raise FileExistsError(out)
    for path in (SOURCE / "input.arc", SOURCE / "lasp.in", Path(MODEL)):
        if not path.is_file(): raise FileNotFoundError(path)

    import torch
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from mace.calculators import MACECalculator
    from pamssw.standalone.constrained_reference import run_constrained_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.surface import ASESurface
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False)
    atoms = read(SOURCE / "input.arc", index=0)
    if len(atoms) != 514 or not atoms.pbc.all():
        raise ValueError("source must remain the intact 514-atom 3D-periodic structure")
    original = atoms.copy(); atoms.set_constraint(FixAtoms(indices=FIXED))
    write(out / "initial.extxyz", atoms)
    configs, rotation, direction = _settings()
    local_mobile = np.ones(len(atoms), dtype=bool); local_mobile[FIXED] = False; local_mobile[EXCLUDED] = False
    plan = dict(source=str(SOURCE), input="input.arc", source_config="lasp.in",
        model=MODEL, model_head="omat_pbe", device="CUDA", dtype="float64",
        natoms=len(atoms), pbc=atoms.pbc.tolist(), cell=atoms.cell.array.tolist(), seed=SEED,
        outer_steps=2, physical_fixed_indices_0based=FIXED.tolist(),
        direction_excluded_indices_0based=EXCLUDED.tolist(),
        physical_mobile_count=int((~np.isin(np.arange(len(atoms)), FIXED)).sum()),
        direction_mobile_count=int(local_mobile.sum()), segmentation_support_not_fixed="1..486",
        common_numerics=dict(width_A=.6, inner_gradient_tol_eV_A=.1, forward_force_eV_A=.1,
            temperature_K=100., fmax_eV_A=.05, fd_step_A=.001, relax_steps=500,
            lbfgs_history=500, max_gaussians=3), rotation=rotation,
        direction=direction, live_cap_per_arm=LIVE_CAP, replay_cap=REPLAY_CAP,
        fresh_cap_total=FRESH_CAP, wall_cap_seconds=WALL_CAP,
        replay_contract="continuous local 2 steps vs split 1+1 from the recorded live E/F stream",
        acceptance="fixed positions/cell/PBC exact; fresh true-force max over physical mobile atoms <=0.05 eV/A",
        scope="one fixed-cell model feasibility case; no DFT, phase, novelty, efficacy, or native parity claim",
        stopping="retain failures/costs; no retry or budget extension")
    ledger.dump(out / "plan.json", plan); ledger.dump(out / "input.json", original)
    deadline = time.monotonic() + WALL_CAP
    physical_mobile = np.setdiff1d(np.arange(len(atoms)), FIXED)
    global_dir = np.ones(len(atoms), bool); global_dir[FIXED] = False; global_dir[EXCLUDED] = False
    fresh_count = 0; rows = []
    for arm in ("global_rotation", "periodic_local"):
        folder = out / arm; folder.mkdir()
        local = arm == "periodic_local"
        config = configs[arm]
        kwargs = dict(recovered_direction=direction) if local else {}
        tape = []; row = dict(arm=arm, qualified=False)
        calc = MACECalculator(model_paths=MODEL, head="omat_pbe", device="cuda",
            default_dtype="float64", enable_cueq=False, enable_oeq=False)
        oracle = Oracle(calc, folder, original, deadline, tape=tape)
        oracle.phase = "record" if local else "live"
        try:
            result = run_constrained_ssw(atoms, oracle, steps=2, config=config,
                rng=np.random.default_rng(SEED), direction_fixed_indices=EXCLUDED,
                checkpoint_path=folder / "continuous.pkl", **kwargs)
            ledger.dump(folder / "search-result.json", result)
            row.update(status=result.status, search_live=oracle.live, denials=oracle.denials,
                minima_available=len(result.minima))
            row.update(_stage_audit(result, physical_mobile, local_mobile if local else global_dir))
            if local:
                state = result.checkpoint.recovered_direction_state
                cell_ok = np.array_equal(np.asarray(state.geometry_cell), original.cell.array)
                pbc_ok = np.array_equal(np.asarray(state.geometry_pbc), original.pbc)
                row.update(direction_state_cell_exact=bool(cell_ok), direction_state_pbc_exact=bool(pbc_ok))
            fresh_calc = MACECalculator(model_paths=MODEL, head="omat_pbe", device="cuda",
                default_dtype="float64", enable_cueq=False, enable_oeq=False)
            fresh = ASESurface(fresh_calc); endpoints = []
            for i, minimum in enumerate(result.minima):
                if fresh_count >= FRESH_CAP: raise RuntimeError("six fresh E/F checks exhausted")
                endpoint = minimum.atoms.copy(); endpoint.set_constraint()
                fixed_ok = np.array_equal(endpoint.positions[FIXED], original.positions[FIXED])
                cell_ok = np.array_equal(endpoint.cell.array, original.cell.array)
                pbc_ok = np.array_equal(endpoint.pbc, original.pbc)
                fresh_count += 1
                energy, forces = fresh.evaluate(endpoint)
                fmax = float(np.linalg.norm(forces[physical_mobile], axis=1).max())
                item = dict(index=i, energy=energy, stored_energy=minimum.energy,
                    energy_delta=energy - minimum.energy, physical_mobile_fmax=fmax,
                    force_ok=fmax <= .05, energy_ok=abs(energy - minimum.energy) <= 1e-6,
                    fixed_positions_exact=bool(fixed_ok),
                    cell_exact=bool(cell_ok), pbc_exact=bool(pbc_ok), atoms=endpoint, forces=forces)
                ledger.append(folder / "fresh.jsonl", item)
                endpoints.append({k: v for k, v in item.items() if k not in ("atoms", "forces")})
            row["fresh_endpoints"] = endpoints
            row["physical_ok"] = bool(endpoints) and all(e["force_ok"] and e["energy_ok"] and
                e["fixed_positions_exact"] and e["cell_exact"] and e["pbc_exact"] for e in endpoints)

            if local and result.status == "completed" and result.checkpoint is not None and tape:
                replay = Oracle(None, folder, original, deadline, tape=tape, replay=True)
                first = run_constrained_ssw(atoms, replay, steps=1, config=config,
                    rng=np.random.default_rng(SEED), direction_fixed_indices=EXCLUDED,
                    checkpoint_path=folder / "split-1.pkl", **kwargs)
                if first.status != "completed":
                    raise RuntimeError(f"split first segment status: {first.status}")
                from pamssw.standalone.constrained_reference import load_constrained_checkpoint
                checkpoint = load_constrained_checkpoint(folder / "split-1.pkl")
                second = run_constrained_ssw(atoms, replay, steps=1, config=config,
                    rng=np.random.default_rng(SEED + 1), checkpoint=checkpoint,
                    direction_fixed_indices=EXCLUDED,
                    checkpoint_path=folder / "split-2.pkl", **kwargs)
                equal = ledger._jsonable(result.checkpoint) == ledger._jsonable(second.checkpoint)
                row.update(replay_requests=replay.replayed, replay_stream_exact=replay.cursor == len(tape),
                    continuous_split_checkpoint_equal=equal)
                if replay.cursor != len(tape) or not equal:
                    raise AssertionError("continuous 2-step and split 1+1 checkpoints differ")
            row["qualified"] = bool(row["status"] == "completed" and row.get("physical_ok") and
                row.get("direction_mask_support_ok") and
                (not local or (row.get("direction_records_checked", 0) > 0 and
                 row.get("route_status") == "reported" and row.get("direction_state_cell_exact") and
                 row.get("direction_state_pbc_exact") and row.get("continuous_split_checkpoint_equal"))))
        except Exception as error:
            row.update(error=repr(error), traceback=traceback.format_exc())
            ledger.append(folder / "failures.jsonl", dict(error=repr(error), traceback=row["traceback"]))
        finally:
            row.update(live_requests=oracle.live, denials=oracle.denials, tape_entries=len(tape))
            ledger.dump(folder / "result.json", row); rows.append(row)
            calc = fresh_calc = None
            import gc; gc.collect(); torch.cuda.empty_cache()
            ledger.dump(out / "summary.json", dict(rows=rows, live_total=sum(r["live_requests"] for r in rows),
                replay_total=sum(r.get("replay_requests", 0) for r in rows), fresh_total=fresh_count,
                qualified=all(r.get("qualified") for r in rows)))


if __name__ == "__main__":
    main()
