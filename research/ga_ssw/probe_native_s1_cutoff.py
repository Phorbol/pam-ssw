"""Check two isolated S1 cutoff arithmetic blocks, never LASP main or a PES.

The tanhf call is deliberately outside both instruction slices: its rounded
result is supplied by Python. This verifies constants, argument construction,
and the value/derivative arithmetic, not native libm or full Q-mode parity.
"""
import argparse
import hashlib
import json
import math
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import (
    UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RBP, UC_X86_REG_RSP,
    UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R12, UC_X86_REG_R13,
    UC_X86_REG_XMM0, UC_X86_REG_XMM3, UC_X86_REG_XMM7,
)
from research.ga_ssw.probe_native_weight_emulated import load_elf


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def run(elf):
    blob, segments = load_elf(elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':
        raise ValueError('unexpected reference ELF')
    machine = Uc(UC_ARCH_X86, UC_MODE_64)
    for page, size in ((0xa24000, 0x2000), (0x4a75000, 0x1000)):
        machine.mem_map(page, size)
        for va, _, data in segments:
            begin, end = max(page, va), min(page + size, va + len(data))
            if begin < end:
                machine.mem_write(begin, data[begin-va:end-va])
    data, stack = 0x72000000, 0x71000000
    machine.mem_map(data, 0x4000)
    machine.mem_map(stack, 0x4000)
    def put(address, fmt, value):
        machine.mem_write(address, struct.pack(fmt, value))
    def xmm(register, value):
        machine.reg_write(register, int.from_bytes(struct.pack('<f', value), 'little'))
    def read_xmm(register):
        return struct.unpack('<f', machine.reg_read(register).to_bytes(16, 'little')[:4])[0]
    constants = {
        'argument_constant': struct.unpack('<f', machine.mem_read(0x4a75b1c, 4))[0],
        'derivative_factor': struct.unpack('<d', machine.mem_read(0x4a75af8, 8))[0],
        'cutoff_guard': struct.unpack('<f', machine.mem_read(0x4a75b20, 4))[0],
    }
    assert constants['argument_constant'] == 1.
    assert constants['derivative_factor'] == -3.
    rows = []
    for cutoff in (2., 6.48507):
        cutoff = f32(cutoff)
        for ratio in (.1, .25, .5, .9, .99):
            radius = f32(ratio*cutoff)
            for register, value in (
                (UC_X86_REG_RAX, 0), (UC_X86_REG_RBX, data),
                (UC_X86_REG_R13, data+0x100), (UC_X86_REG_R12, 0),
                (UC_X86_REG_R9, 0), (UC_X86_REG_R10, 0),
                (UC_X86_REG_RBP, data+0x1000), (UC_X86_REG_RSP, stack+0x1000),
            ):
                machine.reg_write(register, value)
            put(data+0x38, '<f', cutoff)
            put(data+0x108, '<f', radius)
            put(data+0x1270, '<Q', data+0x2000)
            put(data+0x12a4, '<i', 0)
            xmm(UC_X86_REG_XMM3, constants['argument_constant'])
            machine.emu_start(0xa24bfe, 0xa24c21, count=40)
            argument = read_xmm(UC_X86_REG_XMM0)
            assert argument == f32(1.-f32(radius/cutoff))
            tangent = f32(math.tanh(argument))
            xmm(UC_X86_REG_XMM0, tangent)
            machine.emu_start(0xa24c26, 0xa24c88, count=50)
            value = struct.unpack('<f', machine.mem_read(data+0x2000, 4))[0]
            derivative = read_xmm(UC_X86_REG_XMM7)
            square = f32(tangent*tangent)
            assert value == f32(tangent*square)
            assert derivative == f32(-3.*square*(1.-square)/cutoff)
            rows.append(dict(radius=radius, cutoff=cutoff, argument=argument,
                             value=value, derivative=derivative))
    return dict(elf_sha256=digest, constants=constants, rows=rows,
                pes_calls=0, boundary=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elf', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.elf)
    with open(args.output, 'w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
