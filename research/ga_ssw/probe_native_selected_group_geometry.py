"""Bounded selection -> group -> projection -> mixture differential probe.

Each native function is executed in isolation. This does not claim that the
allopt caller passes this pair unchanged, or establish the snapshot lifecycle.
No energy calculator, executable main, or protection code is executed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.build import molecule

from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_local_group import (
    native_local_group, select_native_local_group,
)
from research.ga_ssw.probe_native_axis_group_selection_v2 import run
from research.ga_ssw.probe_native_group_geometry import GeometryOracle
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    rng = np.random.default_rng(20260917)
    rows = []
    for name in ('C2H6', 'CH3OH', 'C6H6'):
        atoms = molecule(name)
        x = atoms.positions.copy()
        reference = x + np.arange(len(x))[:, None] * [.1, .03, -.02]
        frame = ClusterFrame(atoms)
        seed = normalized(frame.project(rng.normal(size=x.shape)))
        for draw in (0., .5, np.nextafter(1., 0.)):
            selected = select_native_local_group(reference, atoms, iter([draw]))
            native_selection = run(len(x), [1] * len(x), reference=reference,
                                   current=x, rng=draw)
            pair = tuple(i - 1 if i else None for i in native_selection['pair'])
            selection_match = (pair == selected.pair and np.array_equal(
                native_selection['group'], selected.group_mask))
            assert selection_match
            if pair[1] is None:
                rows.append(dict(name=name, draw=draw, pair=pair,
                                 selection_match=True, generator_skipped='no axis'))
                continue
            oracle = GeometryOracle(segments)
            oracle.axis, oracle.group = pair, native_selection['group']
            got = oracle.run_geometry(x, seed, np.zeros_like(x))
            raw = native_local_group(atoms, selected.pair, selected.group_mask)
            projected = frame.project(raw)
            expected = normalized(.6 * seed + .5 * normalized(projected))
            errors = dict(raw=float(np.max(abs(oracle.raw_local - raw))),
                          projected=float(np.max(abs(oracle.projected - projected))),
                          mixture=float(np.max(abs(got - expected))))
            rows.append(dict(name=name, draw=draw, pair=pair,
                             selection_match=selection_match, errors=errors,
                             positions=x.tolist(), reference=reference.tolist(),
                             group=selected.group_mask.tolist(), seed=seed.tolist(),
                             native_output=got.tolist(), python_output=expected.tolist()))
            assert max(errors.values()) < 1e-12, rows[-1]
    out = Path('research/ga_ssw/evidence/native-selected-group-geometry-20260917.json')
    out.write_text(json.dumps(dict(scope=__doc__, sha256=ELF_SHA256, cases=rows),
                              indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), generator_cases=sum('errors' in r for r in rows),
                          max_error=max(max(r['errors'].values()) for r in rows if 'errors' in r))))


if __name__ == '__main__':
    main()
