"""Native c6 mixture with an injected local-group vector (not its producer).

The generator, normalization, weighted addition and final mask execute original
instructions. RNG is fixed; rigid cleanup is explicitly replaced by identity.
This validates mixing arithmetic only, not geometry selection or native RNG.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_R8, UC_X86_REG_RSP, UC_X86_REG_RIP
from research.ga_ssw.probe_native_direction_update import UpdateOracle
from research.ga_ssw.probe_addgaussian_emulated import DATA, STACK, STOP
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class MixtureOracle(UpdateOracle):
    def hook(self, uc, address, size, user):
        if address in (0x5d873c, 0x5d870c):
            self.terminal = address
            uc.emu_stop()
        elif address == 0x580640:
            self.d(uc.reg_read(UC_X86_REG_RDI), .5)
            self.ret()
        elif address == 0x6e4c00:
            uc.mem_write(uc.reg_read(UC_X86_REG_R8), self.local.tobytes())
            self.producer_calls += 1
            self.ret()
        elif address == 0x5790e0:
            self.projection_calls += 1
            self.ret()
        elif address in (0x4970360, 0x4a10430):
            super().hook(uc, address, size, user)
        elif 0x5d5c50 <= address < 0x5dab00 or 0x578e20 <= address < 0x5790e0:
            pass
        else:
            raise RuntimeError(f'unexpected address {address:#x}')

    def mixture(self, seed, local, weight):
        self.n = len(seed)
        self.order = 'C'
        self.local = np.asarray(local, dtype='<f8')
        self.cursor = DATA+0x10000
        obj = DATA+0x100
        self.q(DATA, obj)
        self.q(DATA+0x38, 0x53ca680)
        self.uc.mem_write(obj, struct.pack('<i', self.n))
        self.uc.mem_write(obj+0x1660, struct.pack('<i', 1))
        self.uc.mem_write(0x53ed7a0+0x100, struct.pack('<i', 5))
        self.uc.mem_write(0x1d1031b0, struct.pack('<i', 1))  # suppress rank-0 logging
        shape = (3, self.n)
        direction = self.arr(seed)
        self.descriptor(obj+0x1788, direction, shape)
        self.descriptor(obj+0x1848, self.arr(np.zeros_like(seed)), shape)
        self.descriptor(obj+0x170, self.arr(np.zeros_like(seed)), shape)
        mask = self.alloc(np.ones_like(seed, dtype='<i4').tobytes())
        self.descriptor(obj+0x8a8, mask, shape, 4)
        coeff = np.zeros(10)
        coeff[6], coeff[9] = weight, 1.2*weight
        self.producer_calls = self.projection_calls = 0
        self.q(STACK+0x80008, STOP)
        self.uc.reg_write(UC_X86_REG_RSP, STACK+0x80008)
        self.uc.reg_write(UC_X86_REG_RDI, DATA)
        self.uc.reg_write(UC_X86_REG_RSI, self.arr(coeff))
        self.uc.emu_start(0x5d5c50, STOP, timeout=getattr(self, "timeout_us", 2_000_000),
                          count=getattr(self, "instruction_limit", 150_000))
        assert self.uc.reg_read(UC_X86_REG_RIP) in (0x5d873c, 0x5d870c)
        return self.readarr(direction).copy()


def normalized(x):
    norm2 = float(np.vdot(x, x))
    return np.zeros_like(x) if norm2 <= 1e-6 else x/np.sqrt(norm2)


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    cases = []
    for n in (2, 5):
        s = np.zeros((n, 3)); s[0, 0] = 1.
        t = np.zeros_like(s); t[0, 1] = 1.
        for label, seed, local in (('forward', s, s*3), ('reverse', s, -s*3),
                                   ('transverse', s, t*3), ('zero_seed', s*0, t*3),
                                   ('zero_local', s, t*0), ('both_zero', s*0, t*0),
                                   ('cone_tangent', s, -5*s/6+np.sqrt(11)*t/6)):
            oracle = MixtureOracle(segments)
            got = oracle.mixture(seed, local, .5)
            expected = normalized(.6*seed + .5*normalized(local))
            error = float(np.max(np.abs(got-expected)))
            cases.append(dict(n=n, case=label, output=got.tolist(), error=error,
                              producer_calls=oracle.producer_calls, projection_calls=oracle.projection_calls,
                              terminal=hex(oracle.terminal), passed=error < 1e-14 and oracle.producer_calls == 1
                              and oracle.terminal == (0x5d873c if np.any(expected) else 0x5d870c)))
    result = dict(sha256=ELF_SHA256, scope=__doc__, cases=cases, passed=all(c['passed'] for c in cases))
    Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/group-mixture.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], cases=len(cases), max_error=max(c['error'] for c in cases))))
    assert result['passed']


if __name__ == '__main__':
    main()
