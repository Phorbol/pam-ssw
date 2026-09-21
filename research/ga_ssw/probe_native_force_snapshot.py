"""Execute soften_mode0 force snapshot before curvature/rotation dispatch."""
import hashlib
import json
from pathlib import Path

import numpy as np
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSP, UC_X86_REG_RIP

from research.ga_ssw.probe_native_direction_update import UpdateOracle
from research.ga_ssw.probe_addgaussian_emulated import DATA, STACK, STOP
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class SnapshotOracle(UpdateOracle):
    def hook(self, uc, address, size, user):
        if address == 0x5c3af7:
            uc.emu_stop()
        elif address == 0x4970360:
            super().hook(uc, address, size, user)
        elif not 0x5c3880 <= address < 0x5c3af7:
            raise RuntimeError(f'unexpected address {address:#x}')

    def snapshot(self, force):
        self.order = 'C'
        self.n = len(force)
        self.cursor = DATA + 0x10000
        obj = DATA + 0x100
        self.q(DATA, obj)
        source = self.arr(force)
        saved = self.arr(np.full_like(force, 91.))
        center = self.arr(np.full_like(force, -73.))
        for offset, pointer in ((0x1d0, source), (0x9c8, saved), (0x1a60, center)):
            self.descriptor(obj + offset, pointer, (3, self.n))
        self.q(STACK + 0x80008, STOP)
        self.uc.reg_write(UC_X86_REG_RSP, STACK + 0x80008)
        self.uc.reg_write(UC_X86_REG_RDI, DATA)
        self.uc.emu_start(0x5c3880, STOP, timeout=1_000_000, count=100_000)
        assert self.uc.reg_read(UC_X86_REG_RIP) == 0x5c3af7
        return self.readarr(source), self.readarr(saved), self.readarr(center)


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    cases = []
    for n in (2, 5, 15):
        for sign in (-1, 1):
            force = sign * np.arange(1., 3*n+1).reshape(n, 3)
            source, snapshot, center = SnapshotOracle(segments).snapshot(force)
            cases.append(dict(n=n, sign=sign, passed=bool(
                np.array_equal(source, force) and np.array_equal(snapshot, force)
                and np.all(center == -73.))))
    result = dict(sha256=ELF_SHA256, cases=cases,
                  passed=all(c['passed'] for c in cases),
                  scope='actual soften_mode0 entry snapshot; same-shape allocation hook; stops before curvature and dispatch; no PES')
    path = Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/force-snapshot.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    assert result['passed']


if __name__ == '__main__':
    main()
