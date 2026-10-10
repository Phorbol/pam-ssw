"""Offline weak-loading diagnostics on a qualified stationary channel."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads((args.run/'result.json').read_text())
    if raw['status'] != 'completed_stationary_and_downhill_diagnostics':
        raise ValueError('stationary diagnostics did not complete')
    ts = read(args.run/'ts-candidate.extxyz')
    rows = []
    for side in ('minus', 'plus'):
        record = raw['phases'][side+'-0.050']
        if not record['force_qualified'] or not record['stable_minimum_qualified']:
            raise ValueError('unqualified endpoint')
        atoms = read(args.run/f'{side}-0.050.extxyz')
        data = np.load(args.run/f'{side}-0.050-hessian.npz')
        h, q = data['H_h0005'], data['Q']
        soft = FrozenBondSoftening.from_atoms(atoms, bond_energies=HC_BOND_ENERGIES,
            bond_lengths={key: value+.1 for key, value in HC_BOND_LENGTHS.items()})
        wm, force = soft.evaluate(atoms)
        ws, _ = soft.evaluate(ts)
        g = -q.T @ force.ravel()
        chi = float(g @ np.linalg.solve(h, g))
        b = ws-wm
        barrier = raw['ts_certificate']['energy_eV']-record['energy_eV']
        rows.append({'side':side, 'baseline_barrier_eV':barrier, 'slope_eV':b,
                     'chi_eV':chi, 'minus_b_over_sqrt_chi_sqrt_eV':-b/np.sqrt(chi),
                     'torsion':record['torsion_deg_and_order'],
                     'caveat':'weak a=0 branch prediction; numerical bidirectional quench evidence, not IRC or finite-a test'})
    args.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
