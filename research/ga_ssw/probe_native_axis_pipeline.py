"""Execute findleast -> getpair -> local-group generator on frozen geometries.

This follows the verified output-address relationship, but does not execute
allopt or reconstruct its saved-reference timing. Each function is isolated.
No PES or native main is run; see the imported probes for library hooks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.build import molecule
from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_local_group import native_local_group, select_native_local_group
from pamssw.standalone.native_pair_selection import refresh_native_pair
from research.ga_ssw.probe_native_axis_group_selection_v2 import run as select, ELF, SHA
from research.ga_ssw.probe_native_get_atompair_v2 import run as refresh
from research.ga_ssw.probe_native_group_geometry import GeometryOracle
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_native_weight_emulated import load_elf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", action="store_true", help="Use in-cell inputs required for relative-geometry comparison")
    args = parser.parse_args()
    _, segments = load_elf(ELF)
    rng = np.random.default_rng(519)
    rows = []
    for name in ('C2H6','CH3OH','C6H6'):
        atoms = molecule(name)
        if args.canonical:
            atoms.positions += 15.
        x = atoms.positions
        reference = x + np.arange(len(atoms))[:,None] * [.1,.03,-.02]
        selected = select(len(atoms), [1]*len(atoms), reference=reference,
                          current=x, rng=.5)
        python_selected = select_native_local_group(reference, atoms, iter([.5]))
        assert tuple(i-1 if i else None for i in selected['pair']) == python_selected.pair
        assert np.array_equal(selected['group'], python_selected.group_mask)
        frame = ClusterFrame(atoms)
        seed = normalized(frame.project(rng.normal(size=x.shape)))
        for prefix in ([0.,.2,0.], [.9,.7,0.], [0.,.2,.9]):
            refreshed = refresh(x,selected['pair'],prefix,atoms.numbers)
            assert refreshed['completed']
            pair = tuple(i-1 for i in refreshed['pair_after'])
            if args.canonical:
                python_refresh = refresh_native_pair(atoms, python_selected.pair, iter(refreshed['draws']))
                assert python_refresh.pair == pair
                assert python_refresh.draw_count == len(refreshed['draws'])
                assert python_refresh.geometry_accepted == bool(refreshed['events']['accepted_exit'])
            assert min(pair)>=0 and pair[0]!=pair[1]
            group = np.asarray(selected['group'], dtype=np.int32)
            oracle = GeometryOracle(segments)
            oracle.axis, oracle.group = pair, group
            got = oracle.run_geometry(x,seed,np.zeros_like(x))
            raw = native_local_group(atoms,pair,group)
            projected = frame.project(raw)
            expected = normalized(.6*seed+.5*normalized(projected))
            error = float(np.max(abs(got-expected)))
            assert error<1e-12
            # getpair has no group argument; retain findleast's actual group.
            rows.append(dict(name=name, reference=reference.tolist(), selection=selected,
                             refresh=refreshed, seed=seed.tolist(), output=got.tolist(),
                             max_error=error, raw_norm=float(np.linalg.norm(raw)),
                             pair_changed=selected['pair']!=refreshed['pair_after']))
    report=dict(scope=__doc__,sha256=SHA,canonical=args.canonical,cases=rows)
    suffix = '-canonical' if args.canonical else ''
    Path(f'research/ga_ssw/evidence/native-axis-pipeline{suffix}-20260917.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),pair_changes=sum(r['pair_changed'] for r in rows),
                          max_error=max(r['max_error'] for r in rows),
                          counter_exhausted=sum(not r['refresh']['events']['accepted_exit'] for r in rows))))


if __name__=='__main__': main()
