"""Calibrate existing periodic matching and classify qualification endpoints.

No PES calls. Run on a CPU allocation, including when reading GPU results.
Matching is approximate geometric evidence, not a transition-state certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
import numpy as np
from ase.io import read, write
from pamssw.standalone.periodic_ga_reference import pymatgen_identity

def describe(atoms):
    import spglib
    result = dict(natoms=len(atoms), formula=atoms.get_chemical_formula(),
                  volume_A3=float(atoms.get_volume()),
                  cellpar_A_deg=atoms.cell.cellpar().tolist(),
                  pbc=atoms.pbc.tolist(), spacegroup={})
    for tol in (0.01, 0.1):
        result['spacegroup'][str(tol)] = spglib.get_spacegroup(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=tol)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualified', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    plan = json.loads((HERE / 'plan.json').read_text())
    source = Path(plan['cases'][0]['path']).parent
    names = dict(anatase='anatase.extxyz', anatase_fs=Path(plan['cases'][1]['path']).name,
                 rutile='rutile.extxyz', tio2_ii='tio2-ii.extxyz', brookite='brookite.extxyz',
                 phase87='phase-87.extxyz', phase87_is=Path(plan['cases'][0]['path']).name)
    refs = {key: read(source / name) for key, name in names.items()}
    sources = {}
    for key, atoms in refs.items():
        path = source / names[key]
        sources[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            **describe(atoms))
        write(args.out / f'source-{key}.extxyz', atoms)
    rng = np.random.default_rng(plan['seed_for_geometry_checks'])
    original = refs['anatase_fs']
    relabeled = original[rng.permutation(len(original))]
    wrapped = original.copy()
    wrapped.positions += rng.integers(-2, 3, size=(len(original), 3)) @ original.cell.array
    wrapped.wrap()
    basis_changed = original.copy()
    basis_changed.set_cell(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ original.cell.array,
                           scale_atoms=False)
    basis_changed.wrap()
    positives = dict(relabel=relabeled, wrapping=wrapped, equivalent_basis=basis_changed,
                     supercell=original.repeat((2, 1, 1)))
    negatives = ['rutile', 'tio2_ii', 'brookite', 'phase87', 'phase87_is']
    checks, matrices, matchers = {}, {}, {}
    for name, tolerances in plan['identity_tolerances'].items():
        matcher = pymatgen_identity(**tolerances)
        matchers[name] = matcher
        positive_rows = {key: matcher(original, atoms) for key, atoms in positives.items()}
        negative_rows = {key: matcher(original, refs[key]) for key in negatives}
        checks[name] = dict(positive_matches=positive_rows, negative_matches=negative_rows,
                           passed=all(positive_rows.values()) and not any(negative_rows.values()))
        matrices[name] = {a: {b: matcher(x, y) for b, y in refs.items()} for a, x in refs.items()}
    result = dict(scope='geometry-only calibration; no PES requests', plan=plan,
                  source_info=sources, calibration=checks, source_identity=matrices,
                  calibration_passed=all(row['passed'] for row in checks.values()),
                  qualified_run=None)
    if args.qualified:
        qualification = json.loads((args.qualified / 'qualification.json').read_text())
        endpoints = {}
        for case in plan['cases']:
            path = args.qualified / case['id'] / 'endpoint.extxyz'
            if not path.exists():
                endpoints[case['id']] = dict(present=False)
                continue
            atoms = read(path)
            endpoints[case['id']] = dict(present=True, path=str(path), **describe(atoms),
                reference_matches={name: {key: matcher(atoms, ref) for key, ref in refs.items()}
                                   for name, matcher in matchers.items()})
        # Phase identity may drift on a new PES; compare qualified phases to each
        # other as well, rather than assuming source cells remain valid targets.
        available = {case['id']: read(args.qualified / case['id'] / 'endpoint.extxyz')
                     for case in plan['cases'] if endpoints[case['id']]['present']}
        pairwise = {name: {a: {b: matcher(x, y) for b, y in available.items()}
                              for a, x in available.items()} for name, matcher in matchers.items()}
        result['qualified_run'] = dict(path=str(args.qualified), qualification=qualification,
                                      endpoints=endpoints, endpoint_pairwise=pairwise)
    (args.out / 'geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(calibration_passed=result['calibration_passed'], calibration=checks,
                          output=str(args.out)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
