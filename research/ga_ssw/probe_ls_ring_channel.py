#!/usr/bin/env python
"""Bounded same-surface NEB probe for the C4H6 ring-closure channel.

The NEB highest image is only a seed for the existing stationary-point and
downhill diagnostic. This script never treats a converged NEB path as a
qualified transition state or a validated connection.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory, read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import FIRE


ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = ROOT / "research/ga_ssw/qualify_ls_torsion_channel.py"
RING_CLOSURE = (2, 3)  # source cyclobutene indices; must be the two CH2 carbons
NEB_CAP = 1500
NEB_WALL_SECONDS = 600
TOTAL_WALL_SECONDS = 19 * 60  # leave one minute for Slurm cleanup under a 20-min limit
N_IMAGES = 7
SPRING_K_EV_A2 = 0.1
PATH_FMAX_EV_A = 0.05
PHASE_STEPS = {"ordinary": 50, "climbing": 150}


class TotalDeadline(RuntimeError):
    pass


def load_qualifier():
    if not QUALIFIER_PATH.is_file():
        raise FileNotFoundError(QUALIFIER_PATH)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("ls_torsion_qualifier", QUALIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qualification helper: {QUALIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(atoms: Atoms, bond_lengths: dict, margin: float = 0.1) -> np.ndarray:
    n = len(atoms)
    adjacent = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((int(atoms.numbers[i]), int(atoms.numbers[j]))))
            cutoff = float(bond_lengths[key]) + margin
            if atoms.get_distance(i, j, mic=False) <= cutoff:
                adjacent[i, j] = adjacent[j, i] = True
    return adjacent


def carbon_path_and_closure(initial: Atoms, product: Atoms,
                            bond_lengths: dict) -> tuple[np.ndarray, np.ndarray]:
    if len(initial) != 10 or len(product) != 10:
        raise ValueError("both endpoints must contain exactly C4H6 (10 atoms)")
    expected = np.array([6] * 4 + [1] * 6)
    if not np.array_equal(initial.numbers, expected) or not np.array_equal(product.numbers, expected):
        raise ValueError("source endpoint ordering must be C0..C3,H0..H5 for both files")
    for label, atoms in (("gauche", initial), ("cyclobutene", product)):
        if atoms.constraints or np.any(atoms.pbc) or not np.isfinite(atoms.positions).all():
            raise ValueError(f"{label} must be finite, isolated, and unconstrained")

    initial_graph = graph(initial, bond_lengths)
    product_graph = graph(product, bond_lengths)
    initial_cc = initial_graph[:4, :4]
    product_cc = product_graph[:4, :4]
    if int(initial_cc.sum() // 2) != 3 or sorted(initial_cc.sum(axis=0).tolist()) != [1, 1, 2, 2]:
        raise ValueError("gauche carbon graph is not a four-carbon path at HC lengths + 0.1 A")
    if int(product_cc.sum() // 2) != 4 or not np.all(product_cc.sum(axis=0) == 2):
        raise ValueError("cyclobutene carbon graph is not a four-carbon cycle at HC lengths + 0.1 A")

    ch2 = [i for i in range(4) if int(product_graph[i, 4:].sum()) == 2]
    if (len(ch2) != 2 or set(ch2) != set(RING_CLOSURE) or
            not product_cc[ch2[0], ch2[1]]):
        raise ValueError("ring-closure edge is not the edge between the two CH2 carbons")
    deleted = product_graph.copy()
    deleted[ch2[0], ch2[1]] = deleted[ch2[1], ch2[0]] = False
    deleted_cc = deleted[:4, :4]
    if int(deleted_cc.sum() // 2) != 3 or sorted(deleted_cc.sum(axis=0).tolist()) != [1, 1, 2, 2]:
        raise ValueError("deleting the CH2-CH2 edge did not leave a four-carbon path")
    return initial_graph, deleted


def proper_kabsch(source: np.ndarray, target: np.ndarray):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    translation = target_center - source_center @ rotation
    aligned = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, rotation, translation, rmsd


def best_graph_mapping(initial: Atoms, product: Atoms, initial_graph: np.ndarray,
                       product_graph: np.ndarray):
    source_indices = [np.flatnonzero(product.numbers == z).tolist() for z in (6, 1)]
    target_indices = [np.flatnonzero(initial.numbers == z).tolist() for z in (6, 1)]
    matches = []
    for carbon_map in itertools.permutations(target_indices[0]):
        for hydrogen_map in itertools.permutations(target_indices[1]):
            mapping = [-1] * len(product)
            for b, a in zip(source_indices[0], carbon_map):
                mapping[b] = a
            for b, a in zip(source_indices[1], hydrogen_map):
                mapping[b] = a
            if any(product_graph[i, j] != initial_graph[mapping[i], mapping[j]]
                   for i in range(len(product)) for j in range(i + 1, len(product))):
                continue
            inverse = np.argsort(mapping)
            coords = product.positions[inverse]
            aligned, rotation, translation, rmsd = proper_kabsch(coords, initial.positions)
            matches.append((rmsd, tuple(mapping), inverse, aligned, rotation, translation))
    if not matches:
        raise ValueError("no element- and C-H-graph-preserving endpoint mapping exists")
    matches.sort(key=lambda row: (row[0], row[1]))
    return matches


class CountedSurfaceCalculator(Calculator):
    """Per-image ASE cache backed by one request-counted physical surface."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, surface):
        super().__init__()
        self.surface = surface

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.surface.evaluate(atoms)
        self.results = {"energy": float(energy), "forces": np.asarray(forces, dtype=float)}


def phase_record(name, optimizer, converged, step_limit, start, surface, calc_counter):
    return {"name": name, "converged": bool(converged), "steps": int(optimizer.nsteps),
            "step_limit": int(step_limit), "elapsed_seconds": time.monotonic() - start,
            "requests_total": int(surface.requests),
            "calculate_calls_total": int(calc_counter["calls"])}


def run(gauche_path: Path, cyclobutene_path: Path, out: Path):
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "gauche_path": str(gauche_path.resolve()),
              "cyclobutene_path": str(cyclobutene_path.resolve()),
              "limits": {"neb_requests": NEB_CAP, "neb_wall_seconds": NEB_WALL_SECONDS,
                         "qualification_requests": 1500, "qualification_wall_seconds": 600,
                         "combined_request_upper_bound": 3000,
                         "total_wall_deadline_seconds": TOTAL_WALL_SECONDS},
              "neb": {"n_images": N_IMAGES, "spring_k_eV_A2": SPRING_K_EV_A2,
                      "method": "improvedtangent", "interpolation": "linear_then_ASE_IDPP_defaults",
                      "fmax_eV_A": PATH_FMAX_EV_A, "phase_step_limits": PHASE_STEPS},
              "connectivity_validated": False,
              "interpretation_limit": "NEB convergence and helper diagnostics do not automatically validate endpoint identity or a chemical connection"}
    utilities = None
    calc_counter = {"calls": 0}
    surface = None
    started_total = time.monotonic()
    timer_installed = False
    qualifier = None
    try:
        if hasattr(signal, "setitimer"):
            def deadline_handler(signum, frame):
                raise TotalDeadline(f"overall script deadline reached ({TOTAL_WALL_SECONDS}s)")
            signal.signal(signal.SIGALRM, deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, TOTAL_WALL_SECONDS)
            timer_installed = True

        qualifier = load_qualifier()
        utilities = qualifier.load_ledger()
        runner_path = Path(__file__).resolve()
        (out / "runner.py").write_bytes(runner_path.read_bytes())
        gauche = read(gauche_path, index=0)
        product = read(cyclobutene_path, index=0)
        result["sources"] = {
            "runner": str(runner_path), "runner_sha256": qualifier.sha256(runner_path),
            "qualification_helper": str(QUALIFIER_PATH),
            "qualification_helper_sha256": qualifier.sha256(QUALIFIER_PATH),
            "gauche_sha256": qualifier.sha256(gauche_path),
            "cyclobutene_sha256": qualifier.sha256(cyclobutene_path),
        }
        initial_graph, product_graph = carbon_path_and_closure(
            gauche, product, qualifier.HC_BOND_LENGTHS)
        mapping_rows = best_graph_mapping(gauche, product, initial_graph, product_graph)
        best = mapping_rows[0]
        rmsd, mapping, inverse, aligned_positions, rotation, translation = best
        final = product[inverse]
        final.positions = aligned_positions
        final.cell = gauche.cell
        final.pbc = gauche.pbc
        if len(final.constraints):
            raise ValueError("mapped product unexpectedly has constraints")
        write(out / "gauche.extxyz", gauche)
        write(out / "cyclobutene-aligned-mapped.extxyz", final)
        utilities.dump(out / "mapping.json", {
            "graph_rule": "HC_BOND_LENGTHS + 0.1 A; preserve element and all adjacency including C-H",
            "deleted_product_ring_edge_source_indices": list(RING_CLOSURE),
            "detected_two_CH2_product_carbons": [i for i in range(4)
                                                   if int(product_graph[i, 4:].sum()) == 2],
            "product_to_gauche_index_map": list(mapping),
            "proper_rotation_product_to_gauche": rotation,
            "translation_A": translation,
            "aligned_all_atom_rmsd_A": rmsd,
            "number_of_graph_isomorphisms": len(mapping_rows),
            "next_mapping_rmsd_A": mapping_rows[1][0] if len(mapping_rows) > 1 else None,
            "mapping_policy": "minimum proper-Kabsch RMSD; tie-break lexicographically; freeze numbering throughout NEB and qualification",
        })

        import torch
        from ase import __version__ as ase_version
        from mace.calculators import MACECalculator

        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        observed_hash = qualifier.sha256(qualifier.MODEL)
        if observed_hash != qualifier.MODEL_SHA256:
            raise ValueError(f"MH-1 model hash mismatch: {observed_hash}")
        calc = MACECalculator(model_paths=str(qualifier.MODEL), head="omol", device="cuda",
                              default_dtype="float64", enable_cueq=False, enable_oeq=False)
        calc_counter = utilities.instrument_calculate(calc)
        surface = utilities.CountedSurface(calc, out / "neb-requests.jsonl", cap=NEB_CAP,
                                           wall=NEB_WALL_SECONDS)
        result["model"] = {"path": str(qualifier.MODEL), "sha256": observed_hash,
                           "head": "omol", "device": "cuda", "dtype": "float64",
                           "torch": torch.__version__, "ase": ase_version,
                           "numpy": np.__version__, "scipy": qualifier.importlib.metadata.version("scipy"),
                           "torch_num_threads": torch.get_num_threads(),
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                           "cuda_device_name": torch.cuda.get_device_name(0),
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        images = [gauche.copy()] + [gauche.copy() for _ in range(N_IMAGES - 2)] + [final.copy()]
        for image in images:
            image.calc = CountedSurfaceCalculator(surface)
        neb = NEB(images, k=SPRING_K_EV_A2, climb=False, method="improvedtangent",
                  allow_shared_calculator=True)
        neb.interpolate(method="linear")
        idpp_interpolate(neb, traj=str(out / "idpp.traj"), log=str(out / "idpp.log"))
        phases = []
        with Trajectory(str(out / "neb.traj"), "w", atoms=neb) as trajectory:
            for name, climb, steps in (("ordinary", False, PHASE_STEPS["ordinary"]),
                                       ("climbing", True, PHASE_STEPS["climbing"])):
                neb.climb = climb
                phase_start = time.monotonic()
                request_start = surface.requests
                call_start = calc_counter["calls"]
                optimizer = FIRE(neb, logfile=str(out / f"{name}.log"))
                optimizer.attach(trajectory.write, interval=1, atoms=neb)
                converged = optimizer.run(fmax=PATH_FMAX_EV_A, steps=steps)
                trajectory.write(neb)
                row = phase_record(name, optimizer, converged, steps, phase_start,
                                   surface, calc_counter)
                row.update(requests=int(surface.requests - request_start),
                           calculate_calls=int(calc_counter["calls"] - call_start))
                phases.append(row)
                result["neb"]["phases"] = phases
                result["neb"]["status"] = f"{name}_completed"
                utilities.dump(out / "result.json", result)

        # Explicitly evaluate each terminal image through the same counted surface.
        image_records = []
        terminal_images = []
        for index, image in enumerate(images):
            energy, forces = surface.evaluate(image)
            force_array = np.asarray(forces, dtype=float)
            row = {"image": index, "energy_eV": float(energy),
                   "physical_fmax_eV_A": float(np.linalg.norm(force_array, axis=1).max()),
                   "positions_A": image.positions.tolist(), "forces_eV_A": force_array.tolist()}
            image_records.append(row)
            saved = image.copy()
            saved.info["energy"] = float(energy)
            saved.arrays["forces"] = force_array
            terminal_images.append(saved)
        write(out / "terminal-images.extxyz", terminal_images, format="extxyz")
        internal_energies = [row["energy_eV"] for row in image_records[1:-1]]
        highest_index = 1 + int(np.argmax(internal_energies))
        highest = images[highest_index]
        write(out / "highest-neb-image.extxyz", highest)
        candidate_input = {"numbers": highest.numbers.tolist(),
                           "positions_A": highest.positions.tolist(),
                           "cell_A": highest.cell.array.tolist(),
                           "pbc": highest.pbc.tolist(),
                           "provenance": {"gauche": str(gauche_path.resolve()),
                                          "cyclobutene": str(cyclobutene_path.resolve()),
                                          "mapping": str((out / "mapping.json").resolve()),
                                          "neb_highest_internal_image_1based": highest_index,
                                          "neb_energy_eV": image_records[highest_index]["energy_eV"],
                                          "neb_fmax_eV_A": image_records[highest_index]["physical_fmax_eV_A"]}}
        candidate_input_path = out / "highest-neb-image.json"
        candidate_input_path.write_text(json.dumps(candidate_input, indent=2, allow_nan=False) + "\n")
        result["neb"].update({"status": "completed" if phases[-1]["converged"] else "climbing_not_converged",
                              "images": image_records, "highest_internal_image": highest_index,
                              "terminal_calculator_requests": len(images),
                              "total_requests": int(surface.requests),
                              "actual_calculate_calls": int(calc_counter["calls"]),
                              "denials": int(surface.denials), "boundary": surface.boundary,
                              "counted_wall_seconds": time.monotonic() - surface.started})
        utilities.dump(out / "result.json", result)
        if not phases[-1]["converged"]:
            result["status"] = "neb_path_diagnostic_only_climbing_not_converged"
            return result

        result["status"] = "neb_completed_highest_image_pending_qualification"
        utilities.dump(out / "result.json", result)
        qualification = qualifier.run(candidate_input_path, out / "qualification")
        result["qualification"] = qualification
        if qualification.get("status") == "completed_stationary_and_downhill_diagnostics":
            result["status"] = "completed_neb_and_stationary_diagnostics_endpoint_review_pending"
        else:
            result["status"] = "qualification_stopped"
        result["connectivity_validated"] = False
        result["combined_actual_calculate_calls"] = int(result["neb"].get("actual_calculate_calls", 0)) + int(
            qualification.get("actual_calculate_calls", 0))
        result["interpretation_limit"] = (
            "even a stationary first-order saddle and two qualified released minima require parent review "
            "to confirm the mapped endpoints are gauche and cyclobutene; no connection is automatically certified")
        return result
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = repr(error)
        if surface is not None:
            result["neb"].update({"total_requests": int(surface.requests),
                                  "actual_calculate_calls": int(calc_counter["calls"]),
                                  "denials": int(surface.denials), "boundary": surface.boundary,
                                  "counted_wall_seconds": time.monotonic() - surface.started})
        return result
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        result["total_wall_seconds"] = time.monotonic() - started_total
        if "neb" in result:
            result["combined_request_total"] = int(result["neb"].get("total_requests", 0)) + int(
                result.get("qualification", {}).get("requests", 0))
        if out.exists():
            dump_result = utilities.dump if utilities is not None else (
                lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))
            dump_result(out / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gauche", type=Path)
    parser.add_argument("cyclobutene", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = run(args.gauche, args.cyclobutene, args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out.resolve()),
                      "combined_request_total": result.get("combined_request_total"),
                      "error": result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
