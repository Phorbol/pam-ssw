"""Bounded update_mode0 instruction probe; capture before random generation.

Same-shape allocation is stubbed, n_normal runs from the ELF. No PES/main.
"""
import json
import hashlib
import struct
from pathlib import Path
import numpy as np
from unicorn import UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RAX, UC_X86_REG_RSP, UC_X86_REG_RIP
from research.ga_ssw.probe_addgaussian_emulated import Oracle, DATA, STACK, STOP
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256


class UpdateOracle(Oracle):
    def hook(self, uc, address, size, user):
        if address == 0x4970360:
            lhs = uc.reg_read(UC_X86_REG_RDI)
            rhs = uc.reg_read(UC_X86_REG_RSI)
            assert self.readq(lhs) != 0
            assert all(self.readq(lhs+o) == self.readq(rhs+o) for o in (8, 0x20, 0x30, 0x48))
            self.ret()
        elif address == 0x4a10430:
            pointer = uc.reg_read(UC_X86_REG_RDI)
            byte = uc.reg_read(UC_X86_REG_RSI) & 255
            size = uc.reg_read(UC_X86_REG_RDX)
            uc.mem_write(pointer, bytes([byte])*size)
            uc.reg_write(UC_X86_REG_RAX, pointer)
            self.ret()
        elif address == 0x5d5c50:
            pointer = uc.reg_read(UC_X86_REG_RSI)
            self.captured = dict(coefficients=np.frombuffer(uc.mem_read(pointer, 80), dtype='<f8').tolist(),
                                 normalized_displacement=self.readarr(self.direction).tolist())
            uc.emu_stop()
        elif not (0x5d55f0 <= address < 0x5d5c50 or 0x578e20 <= address < 0x5790e0):
            raise RuntimeError(f'unexpected executed address {address:#x}')

    def probe(self, current, reference, old_direction, coefficients, *, selected=1):
        self.order = 'C'
        self.n = len(current)
        self.cursor = DATA + 0x10000
        obj = DATA + 0x100
        self.q(DATA, obj)
        self.q(DATA+0x38, 0x53ca680)
        self.uc.mem_write(obj, struct.pack('<i', self.n))
        self.uc.mem_write(obj+0x1660, struct.pack('<i', selected))
        # Empty constraint metadata avoids setconstraints; geometry remains full 3N.
        shape = (3, self.n)
        self.descriptor(obj+0x170, self.arr(current), shape)
        self.direction = self.arr(old_direction)
        self.descriptor(obj+0x1788, self.direction, shape)
        references = np.asarray(reference)
        if references.ndim == 2:
            references = references[None, ...]
        if not 1 <= selected <= len(references):
            raise ValueError('selected record out of range')
        records = self.alloc(bytes(0x690*len(references)))
        self.descriptor(obj+0x1668, records, (len(references),), 0x690)
        for index, item in enumerate(references):
            self.descriptor(records+index*0x690+0x170, self.arr(item), shape)
        control = 0x53ed5c0
        self.uc.mem_write(control+0x140, struct.pack('<i', 0))
        self.uc.mem_write(control+0x68, struct.pack('<i', 0))
        for i, value in enumerate(coefficients):
            self.d(control+0x178+8*i, value)
        self.captured = None
        self.q(STACK+0x80008, STOP)
        self.uc.reg_write(UC_X86_REG_RSP, STACK+0x80008)
        self.uc.reg_write(UC_X86_REG_RDI, DATA)
        self.uc.emu_start(0x5d55f0, STOP, timeout=2_000_000, count=100_000)
        assert self.uc.reg_read(UC_X86_REG_RIP) == 0x5d5c50
        return self.captured


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    cases = []
    for n in (2, 5):
        current = np.arange(3*n, dtype=float).reshape(n, 3)*.1 + .3
        reference = np.zeros((n, 3))
        for old_sign in (-1, 1):
            old = np.full((n, 3), old_sign / np.sqrt(3*n))
            coefficients = [.2, .3, .4]
            result = UpdateOracle(segments).probe(current, reference, old, coefficients)
            expected = current / np.linalg.norm(current)
            expected_coefficients = [0., 0., 0., 0., *coefficients, 0., 0., 1.2*sum(coefficients)]
            error = float(np.max(np.abs(np.array(result['normalized_displacement'])-expected)))
            coeff_error = float(np.max(np.abs(np.array(result['coefficients'])-expected_coefficients)))
            cases.append(dict(n=n, old_sign=old_sign, **result, displacement_error=error,
                              coefficient_error=coeff_error, passed=error < 1e-14 and coeff_error < 1e-14))
    report = dict(sha256=ELF_SHA256, cases=cases, passed=all(c['passed'] for c in cases),
                  scope='modelevel=0; no compress; nonzero displacement; constraint projection excluded; stops BEFORE random generator; zero PES')
    path = Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/direction-update.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
