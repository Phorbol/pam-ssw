"""Actual rotate_dimer force algebra up to BRIONS, with explicit endpoint E/F."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from unicorn.x86_const import (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX,
    UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_RSP, UC_X86_REG_RIP)
from research.ga_ssw.probe_addgaussian_emulated import Oracle, DATA, STACK, STOP
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class ResponseOracle(Oracle):
    def hook(self, uc, address, size, user):
        if address == 0x6f6440:
            self.endpoint = self.readarr(uc.reg_read(UC_X86_REG_RSI)).copy()
            self.response = self.readarr(uc.reg_read(UC_X86_REG_RDX)).copy()
            uc.emu_stop()
        elif address == 0x4a102b0:
            destination=uc.reg_read(UC_X86_REG_RDI)
            source=uc.reg_read(UC_X86_REG_RSI)
            count=uc.reg_read(UC_X86_REG_RDX)
            uc.mem_write(destination,bytes(uc.mem_read(source,count)))
            self.ret()
        elif not 0x6e57d0 <= address < 0x6e8330:
            raise RuntimeError(f'unexpected address {address:#x}')

    def run_response(self, center, direction, f0, f1, dr, factor):
        self.n = len(center)
        self.order = 'C'
        self.cursor = DATA+0x10000
        scalar_i = lambda v: self.alloc(struct.pack('<i',v))
        scalar_d = lambda v: self.alloc(struct.pack('<d',v))
        self.d(0x791b8c0, factor)
        self.uc.mem_write(0x53ed7a0+0x2dcc8,struct.pack('<i',0))
        endpoint = self.arr(center+dr*direction)
        curvature = scalar_d(91.)
        args = [scalar_i(2),scalar_i(self.n),self.arr(np.eye(3)),self.arr(center),endpoint,
                self.arr(f0), self.arr(f1),self.arr(direction),self.arr(direction),
                self.alloc(np.ones(3*self.n,dtype='<i4').tobytes()),scalar_i(0),
                self.alloc(bytes(128)),curvature,scalar_d(.7),scalar_d(.05),scalar_d(dr),
                scalar_i(10),scalar_d(.02),scalar_i(0),self.alloc(b'CBD_PreRot'),10]
        regs = (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX,
                UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9)
        for reg, pointer in zip(regs,args):
            self.uc.reg_write(reg,pointer)
        sp=STACK+0x80008
        self.q(sp,STOP)
        for index,value in enumerate(args[6:]):
            self.q(sp+8*(index+1),value)
        self.uc.reg_write(UC_X86_REG_RSP,sp)
        self.uc.emu_start(0x6e57d0,STOP,timeout=1_000_000,count=100_000)
        assert self.uc.reg_read(UC_X86_REG_RIP)==0x6f6440
        return self.endpoint,self.response,self.readd(curvature)


def main():
    blob,segments=load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    rng=np.random.default_rng(1709)
    cases=[]
    for n in (1,2,5):
        center=rng.normal(size=(n,3))
        direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction)
        f0=rng.normal(size=(n,3));f1=rng.normal(size=(n,3))
        for dr in (.001,.03):
            for factor in (.05,.032):
                endpoint,response,curvature=ResponseOracle(segments).run_response(center,direction,f0,f1,dr,factor)
                expected_curvature=float(np.vdot(f0-f1,direction)/dr)
                expected_response=factor*(f1-f0+dr*expected_curvature*direction)
                error=float(np.max(np.abs(response-expected_response)))
                endpoint_error=float(np.max(np.abs(endpoint-center-dr*direction)))
                cases.append(dict(n=n,dr=dr,factor1=factor,response_error=error,
                                  endpoint_error=endpoint_error,curvature_error=abs(curvature-expected_curvature),
                                  passed=error<1e-12 and endpoint_error<1e-12 and abs(curvature-expected_curvature)<1e-10))
    report=dict(sha256=ELF_SHA256,cases=cases,passed=all(c['passed'] for c in cases),
                scope='actual rotate_dimer prefix, rotnum=2, fixed cell/unconstrained, stops before BRIONS; explicit forces, zero PES; no arithmetic hooks')
    Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/rotation-response.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed']


if __name__=='__main__':
    main()
