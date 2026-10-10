"""Read-only common-W comparison; endpoint identity remains explicitly visible."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read
from pamssw.standalone.native_ls import HC_BOND_LENGTHS, HC_BOND_ENERGIES
from pamssw.standalone.softening import FrozenBondSoftening
from probe_ls_ring_channel import graph, proper_kabsch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('torsion', type=Path)
    parser.add_argument('ring', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = json.loads((args.ring/'result.json').read_text())
    if result['status'] != 'completed_neb_and_stationary_diagnostics_endpoint_review_pending':
        raise ValueError('ring stationary diagnostics incomplete')
    qrun = args.ring/'qualification'
    qresult = json.loads((qrun/'result.json').read_text())
    old = json.loads((args.torsion/'result.json').read_text())
    gauche = read(args.torsion/'plus-0.050.extxyz')
    product = read(args.ring/'cyclobutene-aligned-mapped.extxyz')
    refs = {'gauche': gauche, 'cyclobutene': product}
    comparisons = {}
    for side in ('minus', 'plus'):
        endpoint = read(qrun/f'{side}-0.050.extxyz')
        comparisons[side] = {}
        for name, ref in refs.items():
            aligned, _, _, rmsd = proper_kabsch(endpoint.positions, ref.positions)
            comparisons[side][name] = {
                'same_index_graph': bool(np.array_equal(graph(endpoint, HC_BOND_LENGTHS), graph(ref, HC_BOND_LENGTHS))),
                'proper_rotation_same_index_rmsd_A': rmsd}
    soft = FrozenBondSoftening.from_atoms(gauche, bond_energies=HC_BOND_ENERGIES,
        bond_lengths={k:v+.1 for k,v in HC_BOND_LENGTHS.items()})
    wm, _ = soft.evaluate(gauche)
    wt, _ = soft.evaluate(read(args.torsion/'ts-candidate.extxyz'))
    wr, _ = soft.evaluate(read(qrun/'ts-candidate.extxyz'))
    e0 = old['phases']['plus-0.050']['energy_eV']
    report = {'endpoint_comparisons': comparisons,
        'same_side_checks': qresult['same_side_endpoint_comparisons'],
        'common_W_origin': str(args.torsion/'plus-0.050.extxyz'),
        'torsion_barrier_eV': old['ts_certificate']['energy_eV']-e0,
        'ring_candidate_barrier_eV': qresult['ts_certificate']['energy_eV']-e0,
        'b_torsion_eV': wt-wm, 'b_ring_candidate_eV': wr-wm,
        'barrier_gap_slope_eV': wr-wt,
        'caveat':'common-W first-order branch predictions, subject to explicit endpoint review; no finite-a or SSW effectiveness claim'}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
