"""Bounded numerical HVP diagnostic for Cu13 and a Cu(111) EMT slab.

This compares forward and centered force finite differences at fixed random
directions.  It is a numerical step-size diagnostic, not an SSW performance
or scientific efficacy experiment.
"""
from pathlib import Path
import json
import numpy as np
import ase
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.io import read

OUT = Path("research/ga_ssw/evidence/hvp-step-diagnostic-20260912")
STEPS = (1e-4, 1e-3, 5e-3, 1e-2)

def force(atoms, positions):
    probe = atoms.copy()
    probe.positions[:] = positions
    probe.calc = EMT()
    return np.asarray(probe.get_forces(), dtype=float)

def one_case(name, atoms, seed):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=atoms.positions.shape)
    direction /= np.linalg.norm(direction)
    x = atoms.positions.copy()
    f0 = force(atoms, x)
    rows = []
    centered_by_h = {}
    for h in STEPS:
        fp = force(atoms, x + h * direction)
        fm = force(atoms, x - h * direction)
        forward = (f0 - fp) / h
        centered = (fm - fp) / (2.0 * h)
        centered_by_h[h] = centered.copy()
        u = forward.ravel(); v = centered.ravel()
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cosine = float(np.dot(u, v) / (nu * nv)) if nu and nv else None
        rows.append(dict(step=h, forward_requests=1, centered_requests=2,
                         forward_norm=float(nu), centered_norm=float(nv),
                         relative_difference=float(np.linalg.norm(u-v)/nv) if nv else None,
                         angle_rad=float(np.arccos(np.clip(cosine,-1,1))) if cosine is not None else None,
                         centered_vs_small_h_relative=None,
                         base_force_norm=float(np.linalg.norm(f0))))
    reference = centered_by_h[STEPS[0]].ravel()
    for row, h in zip(rows, STEPS):
        value = centered_by_h[h].ravel()
        row['centered_vs_small_h_relative'] = float(np.linalg.norm(value-reference)/np.linalg.norm(reference))
    return dict(name=name, atoms=len(atoms), composition=atoms.get_chemical_formula(),
                cell=atoms.cell.array.tolist(), pbc=atoms.pbc.tolist(), seed=seed,
                positions=atoms.positions.tolist(), direction=direction.tolist(),
                actual_force_requests=1 + 2*len(STEPS), rows=rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cu13 = read("research/ga_ssw/evidence/ase-basin-hopping-baseline-20260912-v2/cu13.extxyz", index=0)
    slab = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=8.0)
    add_adsorbate(slab, "Cu", height=2.0, position="fcc")
    results = [one_case("Cu13_EMT", cu13, 41), one_case("Cu111_2x2x3_plus_adatom_EMT", slab, 41)]
    payload = dict(status="complete", diagnostic="single-sided versus centered force HVP",
                   model="ASE EMT", ase_version=ase.__version__, steps=list(STEPS),
                   total_requests=sum(r["actual_force_requests"] for r in results),
                   limits="fixed geometry and random direction; no optimization, no GPU; numerical diagnostic only",
                   accounting="Each h evaluates fp and fm once; forward reuses fp, so 1+2*4=9 force calls per case; total 18.",
                   results=results)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "script.py").write_text(Path(__file__).read_text())
    print(json.dumps({"status": payload["status"], "total_requests": payload["total_requests"]}))

if __name__ == "__main__":
    main()
