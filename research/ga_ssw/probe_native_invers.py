"""Isolated Unicorn oracle for the uploaded ``broyden_module_mp_invers_``.

Only the inverse routine and its in-ELF LUDCM/LUBKS helpers execute. Runtime
allocation/copy/zero/printing calls are handled at the ABI boundary; this
never enters the LASP main program or a PES evaluation.
"""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import (UC_X86_REG_RAX, UC_X86_REG_RDI,
                                UC_X86_REG_RSI, UC_X86_REG_RDX,
                                UC_X86_REG_RCX, UC_X86_REG_R8,
                                UC_X86_REG_R9, UC_X86_REG_RSP,
                                UC_X86_REG_RIP)
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF_DEFAULT = '/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ELF_SHA256 = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
ENTRY, STOP = 0x701290, 0x700000000000
STACK, DATA, HEAP = 0x710000000000, 0x720000000000, 0x730000000000


class Oracle:
    def __init__(self, segments):
        self.uc = Uc(UC_ARCH_X86, UC_MODE_64)
        for va, size, chunk in segments:
            start = va & ~4095
            self.uc.mem_map(start, ((va + size + 4095) & ~4095) - start)
            self.uc.mem_write(va, chunk)
        for va, size in ((STOP, 4096), (STACK, 0x100000),
                         (DATA, 0x200000), (HEAP, 0x1000000)):
            self.uc.mem_map(va, size)
        self.heap = HEAP
        self.calls = {}
        self.uc.hook_add(UC_HOOK_CODE, self.hook)

    def ret(self):
        rsp = self.uc.reg_read(UC_X86_REG_RSP)
        destination = struct.unpack('<Q', self.uc.mem_read(rsp, 8))[0]
        self.uc.reg_write(UC_X86_REG_RSP, rsp + 8)
        self.uc.reg_write(UC_X86_REG_RIP, destination)

    def hook(self, uc, address, size, user):
        if address == STOP:
            uc.emu_stop()
            return
        if 0x701290 <= address < 0x7026c0:
            return
        self.calls[hex(address)] = self.calls.get(hex(address), 0) + 1
        rdi, rsi, rdx = (uc.reg_read(reg) for reg in
                         (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX))
        if address == 0x498a070:
            uc.mem_write(rsi, struct.pack('<Q', self.heap))
            self.heap += ((rdi + 63) // 64) * 64
            uc.reg_write(UC_X86_REG_RAX, 0)
        elif address == 0x4a102b0:
            uc.mem_write(rdi, bytes(uc.mem_read(rsi, rdx))); uc.reg_write(UC_X86_REG_RAX, rdi)
        elif address == 0x4a10430:
            uc.mem_write(rdi, bytes([rsi & 255]) * rdx); uc.reg_write(UC_X86_REG_RAX, rdi)
        elif address in (0x499e470, 0x49a01a0, 0x4998d70):
            uc.reg_write(UC_X86_REG_RAX, 0)
        else:
            raise RuntimeError(f'unexpected runtime call {address:#x}')
        self.ret()

    def run(self, matrix, ld):
        n = matrix.shape[1]
        a = np.asarray(matrix, dtype='<f8', order='F')
        out = np.zeros((ld, n), dtype='<f8', order='F')
        work = np.zeros(max(ld * n, n * n) + 64, dtype='<f8')
        index = np.zeros(n + 8, dtype='<f8')
        vv = np.zeros(n + 8, dtype='<f8')
        ptr = DATA

        def put(raw, align=64):
            nonlocal ptr
            ptr = (ptr + align - 1) // align * align
            address = ptr; self.uc.mem_write(address, raw); ptr += len(raw)
            return address

        pa = put(a.tobytes(order='F')); pn = put(struct.pack('<i', n)); pld = put(struct.pack('<i', ld))
        po = put(out.tobytes(order='F')); pw = put(work.tobytes()); pi = put(index.tobytes(order='F')); pvv = put(vv.tobytes())
        for reg, address in ((UC_X86_REG_RDI, pa), (UC_X86_REG_RSI, pn),
                             (UC_X86_REG_RDX, pld), (UC_X86_REG_RCX, po),
                             (UC_X86_REG_R8, pw), (UC_X86_REG_R9, pi)):
            self.uc.reg_write(reg, address)
        sp = STACK + 0x80008
        self.uc.mem_write(sp, struct.pack('<Q', STOP) + struct.pack('<Q', pvv))
        self.uc.reg_write(UC_X86_REG_RSP, sp)
        self.uc.emu_start(ENTRY, STOP, count=5_000_000)
        if self.uc.reg_read(UC_X86_REG_RIP) != STOP:
            raise RuntimeError(f'stopped at {self.uc.reg_read(UC_X86_REG_RIP):#x}')
        result = np.frombuffer(self.uc.mem_read(po, out.nbytes), dtype='<f8').reshape((ld, n), order='F').copy()
        index_result = np.frombuffer(self.uc.mem_read(pi, index.nbytes), dtype='<f8').copy()
        return result, index_result, self.calls


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--elf', default=ELF_DEFAULT); ap.add_argument('--output', required=True)
    args = ap.parse_args(); blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256: raise ValueError('wrong ELF')
    rng = np.random.default_rng(20260912); cases = []
    for n in (1, 2, 3, 7):
        ld = 50; matrix = rng.normal(size=(ld, n)); matrix[:n] = rng.normal(size=(n, n)); matrix[:n] += np.eye(n) * 3.
        inverse, index, calls = Oracle(segments).run(matrix, ld)
        error = float(np.max(np.abs(matrix[:n] @ inverse[:n] - np.eye(n))))
        cases.append(dict(n=n, ld=ld, matrix=matrix[:n].tolist(), inverse=inverse[:n].tolist(), max_identity_error=error, index=index.tolist(), runtime_calls=calls, passed=error < 1e-8))
    report = dict(elf=str(Path(args.elf).resolve()), sha256=digest, entry='0x701290', helpers=['0x701ad0', '0x701510'], matrix_layout='column-major Fortran doubles', index_storage='float64 array; LUBKS converts entries with cvttsd2si', pivot_result='observed 1..n for nonsingular test matrices', hook_scope='allocation/memcpy/memset/printing only; inverse arithmetic executes ELF', cases=cases, passed=sum(c['passed'] for c in cases), total=len(cases))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
    if report['passed'] != report['total']: raise SystemExit(1)


if __name__ == '__main__': main()
