"""Real native rigid cleanup inside c6 mixing; local producer remains injected."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RAX
from research.ga_ssw.probe_native_group_mixture import MixtureOracle, normalized
from research.ga_ssw.probe_setconstraints_emulated import DESCS, basis
from research.ga_ssw.probe_addgaussian_emulated import DATA
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class ProjectionOracle(MixtureOracle):
    def hook(self, uc, address, size, user):
        if address == 0x5d5c50:
            obj = self.readq(DATA)
            uc.mem_write(self.readq(obj+0x170), self.positions.tobytes())
            # Same-N warm cache: only translation basis initialized externally.
            # Original instructions construct and orthogonalize rotations.
            for j, desc in enumerate(DESCS):
                v = np.zeros_like(self.positions)
                if j < 3:
                    v[:, j] = 1/np.sqrt(self.n)
                self.descriptor(desc, self.arr(v), (3, self.n))
            uc.mem_write(0x54c06c0, struct.pack('<i', 0))
        elif address == 0x5790e0:
            self.projection_calls += 1
        elif 0x5790e0 < address < 0x57e980:
            pass
        elif address == 0x4a102b0:
            dst, src, count = (uc.reg_read(r) for r in (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX))
            uc.mem_write(dst, bytes(uc.mem_read(src, count)))
            uc.reg_write(UC_X86_REG_RAX, dst)
            self.ret()
        elif address == 0x5d827f:
            self.projected = self.readarr(self.readq(self.readq(DATA)+0x1848)).copy()
        else:
            super().hook(uc, address, size, user)

    def run_geometry(self, positions, seed, local, weight=.5):
        self.positions = np.asarray(positions, dtype='<f8')
        return self.mixture(seed, local, weight)


def main():
    from ase.build import molecule
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    cases = []
    rng = np.random.default_rng(20260917)
    for name in ('H2O', 'CH4', 'C6H6'):
        x = molecule(name).positions.copy()
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        x = x @ rotation + [2.1, -1.3, .7]
        _, projector = basis(x)
        seed = normalized((projector @ rng.normal(size=x.size)).reshape(x.shape))
        for mode in ('internal_and_rigid', 'rigid_only'):
            rigid = np.cross([.3, -.5, .7], x-x.mean(0)) + [.8, -.4, .2]
            local = rigid + (rng.normal(size=x.shape) if mode == 'internal_and_rigid' else 0)
            expected_local = (projector @ local.ravel()).reshape(x.shape)
            oracle = ProjectionOracle(segments)
            got = oracle.run_geometry(x, seed, local)
            expected = normalized(.6*seed + .5*normalized(expected_local))
            projection_error = float(np.max(np.abs(oracle.projected-expected_local)))
            error = float(np.max(np.abs(got-expected)))
            cases.append(dict(name=name, mode=mode, positions=x.tolist(), seed=seed.tolist(), local=local.tolist(),
                              projected=oracle.projected.tolist(), output=got.tolist(), projection_error=projection_error,
                              output_error=error, projection_calls=oracle.projection_calls,
                              passed=projection_error < 1e-12 and error < 1e-12 and oracle.projection_calls == 1))
    report = dict(sha256=ELF_SHA256, cases=cases, passed=all(c['passed'] for c in cases),
                  scope='actual generator and setconstraints on same-N warm cache; localgroup injected, fixed RNG; ASE molecular geometries only, zero PES; not native cold-cache initialization')
    Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/group-projection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], cases=len(cases), max_projection_error=max(c['projection_error'] for c in cases))))
    assert report['passed']


if __name__ == '__main__':
    main()
