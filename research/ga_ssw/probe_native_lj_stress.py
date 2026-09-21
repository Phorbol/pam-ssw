"""Execute native LJ pair arithmetic and volume normalization, without LASP init.

The cutoff value/derivative are controlled inputs, not emulated tanh4 outputs.
This checks a backend arithmetic block, not its neighbor counting or cal_pes.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RBP, UC_X86_REG_RSP, UC_X86_REG_RDI, UC_X86_REG_R15, UC_X86_REG_RIP, UC_X86_REG_XMM0
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF = '/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
SHA = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == SHA
    machine = Uc(UC_ARCH_X86, UC_MODE_64)
    for page, size in [(0x504000, 0x1000), (0x509000, 0x1000),
                       (0x4a3f000, 0x1000), (0x5ded000, 0x1000),
                       (0x70000000, 0x10000)]:
        machine.mem_map(page, size)
        for address, _, data in segments:
            lo, hi = max(page, address), min(page + size, address + len(data))
            if hi > lo:
                machine.mem_write(lo, data[lo-address:hi-address])

    def write(address, value):
        machine.mem_write(address, struct.pack('<d', float(value)))

    def read(address):
        return struct.unpack('<d', machine.mem_read(address, 8))[0]

    def execute(start, stop):
        machine.emu_start(start, stop, count=10000, timeout=1000000)
        assert machine.reg_read(UC_X86_REG_RIP) == stop

    amplitude, a12, a6, b6, b12, reciprocal = [read(p) for p in
        (0x4a3f558, 0x4a3f560, 0x4a3f568, 0x4a3f570, 0x4a3f578, 0x4a3f580)]
    assert reciprocal == 1.0
    assert np.isclose(b6, 6*a6) and np.isclose(b12, 12*a12)
    bp, cellptr, stressptr = 0x70008000, 0x70001000, 0x70002000
    force1, force2 = 0x70003000, 0x70004000
    rows = []
    for r in (2.5, 3.5, 5.0):
        for decay in (0.0, 0.17):
            cell = np.array([[9., 1., -0.3], [0.4, 8., 0.6], [0.2, -0.7, 10.]])
            if decay:
                cell[:, 0] *= -1  # Explicitly check abs(det), not signed determinant.
            machine.mem_write(cellptr, cell.astype('<f8').tobytes(order='F'))
            sp, stop = bp + 0x1008, 0x70000000
            machine.mem_write(sp, struct.pack('<Q', stop))
            machine.reg_write(UC_X86_REG_RSP, sp)
            machine.reg_write(UC_X86_REG_RDI, cellptr)
            execute(0x504000, stop)
            bits = machine.reg_read(UC_X86_REG_XMM0) & ((1 << 64)-1)
            volume = struct.unpack('<d', struct.pack('<Q', bits))[0]
            np.testing.assert_allclose(volume, abs(np.linalg.det(cell)), rtol=1e-14)
            vector = np.array([1., -2., 3.]) * r / np.sqrt(14.)
            cutoff = np.exp(-decay*r)
            derivative = -decay*cutoff
            machine.mem_write(force1, bytes(24))
            machine.mem_write(force2, bytes(24))
            machine.mem_write(0x5ded500, bytes(72))
            machine.reg_write(UC_X86_REG_RBP, bp)
            locals_ = {0x50:r**-6, 0xd8:r**-12, 0x48:1/r,
                       0x110:a12, 0x108:a6, 0xf0:amplitude,
                       0xf8:b6, 0x100:b12, 0x120:cutoff, 0x118:derivative,
                       0xc8:vector[0], 0xb0:vector[1], 0xb8:vector[2],
                       0xd0:vector[0]**2, 0xc0:vector[1]**2,
                       0x58:vector[2]**2, 0x68:0., 0x190:volume}
            for offset, value in locals_.items():
                write(bp-offset, value)
            for offset, value in ((0xe0, force1), (0xe8, force2), (0x70, 24)):
                machine.mem_write(bp-offset, struct.pack('<Q', value))
            execute(0x509512, 0x509712)
            energy = read(bp-0x68)
            radial = amplitude*((6*a6/r**7-12*a12/r**13)*cutoff
                                +(a12/r**12-a6/r**6)*derivative)
            expected_energy = amplitude*(a12/r**12-a6/r**6)*cutoff
            expected_stress = np.outer(vector, vector)*radial/r/volume
            native_force = np.frombuffer(machine.mem_read(force1, 24), dtype='<f8')
            other_force = np.frombuffer(machine.mem_read(force2, 24), dtype='<f8')
            machine.mem_write(stressptr, bytes(machine.mem_read(0x5ded500, 72)))
            machine.reg_write(UC_X86_REG_R15, stressptr)
            execute(0x509b2f, 0x509b8a)
            native_stress = np.frombuffer(machine.mem_read(stressptr, 72), dtype='<f8').reshape(3, 3, order='F')
            errors = dict(energy=abs(energy-expected_energy),
                          force=float(np.max(abs(native_force-vector*radial/r))),
                          action_reaction=float(np.max(abs(native_force+other_force))),
                          stress=float(np.max(abs(native_stress-expected_stress))))
            assert max(errors.values()) < 1e-11, errors
            rows.append(dict(r=r, cutoff_decay=decay, cell=cell.tolist(), volume=volume,
                             radial_derivative=radial, energy=energy,
                             stress=native_stress.tolist(), errors=errors))
    output = Path(args.output)
    with output.open('x') as handle:
        json.dump(dict(elf_sha256=SHA, scope='isolated original arithmetic blocks; zero PES calls; cutoff inputs supplied; no neighbor or cal_pes emulation',
                       constants=dict(amplitude=amplitude, a12=a12, a6=a6, b6=b6, b12=b12), rows=rows), handle, indent=2)
        handle.write('\n')
    print(json.dumps({'cases': len(rows), 'maximum_errors': {key:max(row['errors'][key] for row in rows) for key in rows[0]['errors']}}))


if __name__ == '__main__':
    main()
