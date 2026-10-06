#!/usr/bin/env python3
"""True-PES refinement and two-stencil internal curvature of saved candidates.

This post hoc qualification does not rerun SSW, tune LS, or validate pathways.
All selected representatives, including failures, remain in the output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("c4_transfer_run", HERE / "run.py")
transfer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(transfer)

# A deterministic, predeclared representative for every connected graph class
# in this panel; no selection by refinement outcome or lowest post-hoc energy.
SELECTED = (
    ("cyclobutene", "ssw", 2, 3, "SSW-only connected graph"),
    ("bicyclobutane", "paper_ls", 11, 2, "LS-only connected graph for this start"),
    ("bicyclobutane", "ssw", 2, 1, "shared butadiene graph control"),
    ("bicyclobutane", "ssw", 0, 4, "starting bicyclobutane graph control"),
)
REFINE_CAP = 400
CHECK_CAP = 121  # one fresh certificate + two 60-force Cartesian stencils
FMAX = 0.005  # supplementary stationary-point qualification; search stays .03
WALL = 480


def inputs():
    from research.ga_ssw.analyze_c4h6_ls_reaction_coverage import atoms_from_dict

    analysis = json.loads((HERE / "readout-1662786/analysis.json").read_text())
    rows = {(r["case"], r["arm"], r["minimum_index"]): r
            for r in analysis["candidate_rows"]}
    selected = []
    for case, arm, index, graph_class, role in SELECTED:
        row = rows[(case, arm, index)]
        assert row["global_graph_class"] == graph_class
        assert row["fresh_qualified"] and row["connected_intact_c4h6"]
        source = HERE / "run-1662702" / f"{case}-{arm}" / "result.json"
        minimum = json.loads(source.read_text())["minima"][index]
        atoms = atoms_from_dict(minimum["atoms"])
        assert len(atoms) == 10 and not atoms.pbc.any() and not atoms.constraints
        selected.append((atoms, dict(case=case, arm=arm, minimum_index=index,
                                    graph_class=graph_class, role=role,
                                    source=str(source), source_energy_eV=minimum["energy"])))
    return selected


def execute(output):
    import networkx as nx
    import numpy as np
    import torch
    from ase.calculators.calculator import Calculator, all_changes
    from ase.collections import g2
    from ase.io import write
    from ase.optimize import LBFGSLineSearch
    from mace.calculators import MACECalculator
    from scipy.linalg import null_space
    from research.ga_ssw.analyze_c4h6_ls_reaction_coverage import graph, graph_label

    provenance = transfer.preflight(output, require_cuda=True)
    chosen = inputs()
    output.mkdir()
    for file in (Path(__file__), HERE / "candidate-qualification.md"):
        shutil.copy2(file, output / file.name)
    ledger = transfer.load_ledger(Path(provenance["ledger_path"]))
    ledger.dump(output / "provenance.json", provenance)
    ledger.dump(output / "protocol.json", dict(selected=[r for _, r in chosen],
                refine_cap_per_structure=REFINE_CAP, checks_cap_per_structure=CHECK_CAP,
                total_cap=4 * (REFINE_CAP + CHECK_CAP), wall_seconds=WALL,
                refine_fmax_eV_A=FMAX, refine_steps=200, optimizer="ASE LBFGSLineSearch",
                displacements_A=[0.01, 0.005], rigid_projection="Cartesian translations and rotations"))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    calculator = MACECalculator(model_paths=str(transfer.MODEL), head="omol", device="cuda",
                               default_dtype="float64", enable_cueq=False, enable_oeq=False)
    actual_calls = ledger.instrument_calculate(calculator)
    load_seconds = time.monotonic() - started
    references = {name: graph(g2[name].copy()) for name in
                  ("butadiene", "cyclobutene", "bicyclobutane", "2-butyne", "methylenecyclopropane")}
    matcher = nx.algorithms.isomorphism.categorical_node_match("number", None)
    all_rows = []
    total_requests = 0

    class CountedCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, folder):
            super().__init__()
            self.folder = folder
            self.phase = "refine"
            self.counts = {"refine": 0, "checks": 0}

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            nonlocal total_requests
            super().calculate(atoms, properties, system_changes)
            cap = REFINE_CAP if self.phase == "refine" else CHECK_CAP
            if self.counts[self.phase] >= cap or time.monotonic() - started >= WALL:
                raise RuntimeError("qualification request or wall cap reached")
            self.counts[self.phase] += 1
            total_requests += 1
            calculator.reset()
            work = self.atoms.copy()
            work.calc = calculator
            try:
                energy = float(work.get_potential_energy())
                forces = np.asarray(work.get_forces(), float)
                if not np.isfinite(energy) or not np.isfinite(forces).all():
                    raise ValueError("nonfinite E/F")
                self.results = {"energy": energy, "forces": forces.copy()}
                ledger.append(self.folder / "requests.jsonl", dict(phase=self.phase,
                              request=total_requests, energy=energy, forces=forces, atoms=work))
            except Exception as error:
                ledger.append(self.folder / "requests.jsonl", dict(phase=self.phase,
                              request=total_requests, error=repr(error), atoms=work))
                raise

    for ordinal, (atoms, row) in enumerate(chosen):
        folder = output / f"representative-{ordinal}"
        folder.mkdir()
        write(folder / "source.extxyz", atoms)
        before = graph(atoms)
        row["source_graph"] = graph_label(atoms, references)
        row["status"] = "failed"
        counted = CountedCalculator(folder)
        atoms.calc = counted
        try:
            optimizer = LBFGSLineSearch(atoms, logfile=str(folder / "refine.log"),
                                       trajectory=str(folder / "refine.traj"))
            optimizer.run(fmax=FMAX, steps=200)
            counted.phase = "checks"
            counted.reset()
            energy = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            row.update(refined_energy_eV=energy, refined_fmax_eV_A=fmax,
                       refined_graph=graph_label(atoms, references),
                       same_graph=nx.is_isomorphic(before, graph(atoms), node_match=matcher),
                       optimizer_steps=optimizer.nsteps)
            write(folder / "refined.extxyz", atoms)
            if fmax > FMAX:
                raise RuntimeError("refinement did not reach fixed force certificate")
            x = atoms.positions.copy()
            centered = x - x.mean(axis=0)
            rigid = np.column_stack([np.tile(v, (len(atoms), 1)).ravel() for v in np.eye(3)] +
                                    [np.cross(v, centered).ravel() for v in np.eye(3)])
            basis = null_space(rigid.T)
            assert basis.shape == (30, 24), "expected nonlinear isolated C4H6 geometry"
            matrices = []
            projected = []
            for h in (0.01, 0.005):
                matrix = np.empty((30, 30))
                for j in range(30):
                    dx = np.zeros(30)
                    dx[j] = h
                    atoms.positions[:] = x + dx.reshape(-1, 3)
                    plus = atoms.get_forces().ravel().copy()
                    atoms.positions[:] = x - dx.reshape(-1, 3)
                    minus = atoms.get_forces().ravel().copy()
                    matrix[:, j] = -(plus - minus) / (2 * h)
                matrices.append(matrix)
                projected.append(basis.T @ ((matrix + matrix.T) / 2) @ basis)
            atoms.positions[:] = x
            eigenvalues = [np.linalg.eigvalsh(h) for h in projected]
            difference = float(np.linalg.norm(projected[0] - projected[1], ord=2))
            np.savez(folder / "curvature.npz", positions=x, rigid_basis=rigid,
                     internal_basis=basis, raw_hessians=np.array(matrices),
                     projected_hessians=np.array(projected), eigenvalues=np.array(eigenvalues))
            row.update(status="completed", internal_eigenvalues_eV_A2=[v.tolist() for v in eigenvalues],
                       antisymmetric_norm_eV_A2=[float(np.linalg.norm((h-h.T)/2, ord=2)) for h in matrices],
                       stencil_difference_norm_eV_A2=difference,
                       two_stencil_positive=bool(all(v[0] > 0 for v in eigenvalues)),
                       lowest_exceeds_stencil_difference=bool(min(v[0] for v in eigenvalues) > difference))
            row["physical_candidate_qualified_on_MH1"] = bool(row["same_graph"]
                and row["refined_graph"]["component_count"] == 1
                and row["two_stencil_positive"] and row["lowest_exceeds_stencil_difference"])
        except Exception as error:
            row["error"] = repr(error)
            row["physical_candidate_qualified_on_MH1"] = False
        row["requests"] = dict(counted.counts)
        all_rows.append(row)
        ledger.dump(folder / "result.json", row)
        ledger.dump(output / "summary.json", dict(rows=all_rows, total_requests=total_requests,
                    actual_calculator_calls=actual_calls["calls"], model_load_seconds=load_seconds,
                    elapsed_seconds=time.monotonic()-started, requested_representatives=len(chosen),
                    completed_representatives=sum(r["status"] == "completed" for r in all_rows)))
    assert total_requests == actual_calls["calls"], "MACE calculation ledger mismatch"
    print(json.dumps(dict(total_requests=total_requests, completed=len(all_rows),
                         qualified=sum(r["physical_candidate_qualified_on_MH1"] for r in all_rows))))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.output.resolve())
    else:
        transfer.preflight(args.output.resolve())
        chosen = inputs()
        print(json.dumps(dict(status="preflight_ok_zero_PES", selected=[r for _, r in chosen],
                              total_cap=4*(REFINE_CAP+CHECK_CAP))))


if __name__ == "__main__":
    main()
