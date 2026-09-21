"""Execute the presweep-to-biased-rotation anchor copy, not the presweep solve."""
import hashlib
import json
from pathlib import Path

import numpy as np
from unicorn.x86_const import UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15, UC_X86_REG_RBP, UC_X86_REG_RSP, UC_X86_REG_RIP

from research.ga_ssw.probe_native_direction_update import UpdateOracle
from research.ga_ssw.probe_addgaussian_emulated import DATA, STACK, STOP
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class AnchorOracle(UpdateOracle):
    def hook(self, uc, address, size, user):
        if address == 0x5c4887:
            uc.emu_stop()
        elif address == 0x4970360:
            super().hook(uc, address, size, user)
        elif not 0x5c45fe <= address < 0x5c4887:
            raise RuntimeError(f'unexpected address {address:#x}')

    def copy(self, source, saved):
        self.n = len(source)
        self.order = 'C'
        self.cursor = DATA + 0x10000
        obj = DATA + 0x100
        self.q(DATA, obj)
        src, dst = self.arr(source), self.arr(saved)
        self.descriptor(obj+0x1788, src, (3, self.n))
        self.descriptor(obj+0x17e8, dst, (3, self.n))
        for reg, value in ((UC_X86_REG_R13, DATA), (UC_X86_REG_R14, obj),
                           (UC_X86_REG_R15, obj), (UC_X86_REG_RBP, STACK+0x80000),
                           (UC_X86_REG_RSP, STACK+0x7f000)):
            self.uc.reg_write(reg, value)
        self.uc.emu_start(0x5c45fe, STOP, timeout=1_000_000, count=100_000)
        assert self.uc.reg_read(UC_X86_REG_RIP) == 0x5c4887
        return self.readarr(src).copy(), self.readarr(dst).copy()


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    cases = []
    for n in (2, 5, 15):
        for sign in (-1, 1):
            source = sign * np.arange(1, 3*n+1, dtype=float).reshape(n, 3)
            source /= np.linalg.norm(source)
            old = np.full((n, 3), 17.)
            after, anchor = AnchorOracle(segments).copy(source, old)
            cases.append(dict(n=n, sign=sign, source=source.tolist(), anchor=anchor.tolist(),
                              passed=bool(np.array_equal(after, source) and np.array_equal(anchor, source))))
    result = dict(sha256=ELF_SHA256, cases=cases, passed=all(c['passed'] for c in cases),
                  scope='actual vector copy only; same-shape allocation hook; supplied presweep output; no solver, generator, PES or main')
    path = Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/anchor-copy.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], cases=len(cases))))
    assert result['passed']


if __name__ == '__main__':
    main()
