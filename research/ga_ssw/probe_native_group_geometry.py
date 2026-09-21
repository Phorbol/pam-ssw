"""Execute real local-group generation, rigid cleanup and generator mixing.

Axis atoms and group membership are explicit inputs, not native selections.
Only fixed RNG and allocation/memory operations are hooked; same-N projection
cache initialization follows the existing setconstraints instruction probe.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from pamssw.standalone.native_local_group import native_local_group
from research.ga_ssw.probe_native_group_projection import ProjectionOracle
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_setconstraints_emulated import basis
from research.ga_ssw.probe_addgaussian_emulated import DATA
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class GeometryOracle(ProjectionOracle):
    def hook(self, uc, address, size, user):
        if address == 0x5d5c50:
            super().hook(uc, address, size, user)
            obj = self.readq(DATA)
            uc.mem_write(obj+0x1ad8, struct.pack('<ii', *(i+1 for i in self.axis)))
            self.q(obj+0x1ae0, self.alloc(np.asarray(self.group, dtype='<i4').tobytes()))
        elif address == 0x6e4c00:
            self.producer_calls += 1
        elif 0x6e4c00 < address < 0x6e4e4d:
            pass
        elif address == 0x5d825a:
            self.raw_local = self.readarr(self.readq(self.readq(DATA)+0x1848)).copy()
        else:
            super().hook(uc, address, size, user)


def main():
    from ase.build import molecule
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    rng = np.random.default_rng(20260917)
    rows = []
    for name in ('C2H6', 'CH3OH', 'C6H6'):
        atoms = molecule(name)
        x = atoms.positions.copy()
        heavy = np.flatnonzero(atoms.numbers != 1)
        a, b = map(int, heavy[:2])
        _, projector = basis(x)
        seed = normalized((projector @ rng.normal(size=x.size)).reshape(x.shape))
        for selection in ('one_side', 'all_atoms'):
            group = (np.linalg.norm(x-x[a], axis=1) <= np.linalg.norm(x-x[b], axis=1)) if selection == 'one_side' else np.ones(len(x), bool)
            native = GeometryOracle(segments)
            native.axis, native.group = (a, b), group
            got = native.run_geometry(x, seed, np.zeros_like(x))
            raw = native_local_group(atoms, (a, b), group.astype(np.int32))
            projected = (projector @ raw.ravel()).reshape(x.shape)
            expected = normalized(.6*seed + .5*normalized(projected))
            errors = dict(raw=float(np.max(abs(native.raw_local-raw))),
                          projected=float(np.max(abs(native.projected-projected))),
                          output=float(np.max(abs(got-expected))))
            rows.append(dict(name=name, selection=selection, positions=x.tolist(), axis=[a,b], group=group.tolist(),
                             seed=seed.tolist(), raw=native.raw_local.tolist(), projected=native.projected.tolist(), output=got.tolist(),
                             errors=errors, producer_calls=native.producer_calls, projection_calls=native.projection_calls,
                             passed=max(errors.values()) < 1e-12 and native.producer_calls == native.projection_calls == 1))
    report = dict(sha256=ELF_SHA256, scope=__doc__, cases=rows, passed=all(r['passed'] for r in rows))
    Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/group-geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], cases=len(rows), max_error=max(max(r['errors'].values()) for r in rows))))
    assert report['passed']


if __name__ == '__main__':
    main()
