"""Instruction oracle through BRZERO4's first history-matrix block.

This deliberately stops after FINF/SMAT/GMAT construction on noninitial calls.
It does NOT emulate the complete BRZERO4 routine. Runtime memory allocation,
copy, zeroing, and printing are replaced; arithmetic executes uploaded ELF bytes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from pamssw.standalone.native_broyden import initial_step, secant_prefix, native_block_sum_product

ENTRY = 0x6f6c00
PREFIX_END = 0x6f9f78
MATRIX_END = 0x6fa815
STOP = 0x700000000000
STACK = 0x710000000000
DATA = 0x720000000000
HEAP = 0x730000000000
DESCRIPTORS = dict(x=0x55222a0, f=0x5522300, g0=0x5522360,
    df=0x5522420, u=0x55224e0, x_last=0x55225a0,
    f_last=0x5522600, dx=0x55227e0, f_ll=0x5522840, x_ll=0x55228a0)
DESCRIPTORS.update(finf=0x5522720, gmat=0x5522660, smat=0x55226c0)

class PrefixOracle:
    def __init__(self, segments):
        self.uc = uc = Uc(UC_ARCH_X86, UC_MODE_64)
        for va, ms, chunk in segments:
            start = va & ~4095
            uc.mem_map(start, ((va + ms + 4095) & ~4095) - start)
            uc.mem_write(va, chunk)
        for addr, size in ((STOP,4096),(STACK,0x100000),(DATA,0x100000),(HEAP,0x1000000)):
            uc.mem_map(addr,size)
        self.heap = HEAP
        self.prefix_stop = False
        self.matrix_stop = False
        self.runtime_calls = {}
        uc.hook_add(UC_HOOK_CODE, self.hook)
    def i(self, addr): return struct.unpack('<i',self.uc.mem_read(addr,4))[0]
    def q(self, addr): return struct.unpack('<Q',self.uc.mem_read(addr,8))[0]
    def ret(self):
        uc=self.uc; rsp=uc.reg_read(UC_X86_REG_RSP)
        dest=self.q(rsp)
        uc.reg_write(UC_X86_REG_RSP,rsp+8); uc.reg_write(UC_X86_REG_RIP,dest)
    def hook(self, uc, address, size, user):
        if address == PREFIX_END and self.prefix_stop and not self.matrix_stop:
            uc.emu_stop(); return
        if address == MATRIX_END and self.matrix_stop:
            uc.emu_stop(); return
        if 0x6f6c00 <= address < 0x701290 or 0x498a9c0 <= address < 0x498ac00:
            return
        rdi=uc.reg_read(UC_X86_REG_RDI); rsi=uc.reg_read(UC_X86_REG_RSI); rdx=uc.reg_read(UC_X86_REG_RDX)
        self.runtime_calls[hex(address)] = self.runtime_calls.get(hex(address),0)+1
        if address == 0x498a070: # for_alloc_allocatable(bytes, descriptor, flags)
            if self.heap+rdi>HEAP+0x1000000: raise RuntimeError('heap budget')
            uc.mem_write(rsi,struct.pack('<Q',self.heap)); self.heap+=(rdi+63)//64*64
            uc.reg_write(UC_X86_REG_RAX,0)
        elif address == 0x4a102b0:
            uc.mem_write(rdi,bytes(uc.mem_read(rsi,rdx)));uc.reg_write(UC_X86_REG_RAX,rdi)
        elif address == 0x4a10430:
            uc.mem_write(rdi,bytes([rsi&255])*rdx);uc.reg_write(UC_X86_REG_RAX,rdi)
        elif address in (0x499e470,0x49a01a0,0x4998d70):
            uc.reg_write(UC_X86_REG_RAX,0)
        else:
            raise RuntimeError(f'unexpected call {address:#x}, args {rdi:#x} {rsi:#x} {rdx:#x}')
        self.ret()
    def array(self,name):
        d=DESCRIPTORS[name]; rank=self.q(d+0x20)
        shape=[self.q(d+0x30+24*i) for i in range(rank)]
        return np.frombuffer(self.uc.mem_read(self.q(d),int(np.prod(shape))*8),dtype='<f8').reshape(shape,order='F').copy()
    def product(self,a,b):
        a=np.asarray(a,dtype='<f8');b=np.asarray(b,dtype='<f8')
        self.uc.mem_write(DATA,a.tobytes());self.uc.mem_write(DATA+0x10000,b.tobytes())
        self.uc.mem_write(DATA+0x20000,struct.pack('<i',a.size))
        for r,p in ((UC_X86_REG_RDI,DATA),(UC_X86_REG_RSI,DATA+0x10000),(UC_X86_REG_RDX,DATA+0x20000)):
            self.uc.reg_write(r,p)
        sp=STACK+0x80008;self.uc.mem_write(sp,struct.pack('<Q',STOP));self.uc.reg_write(UC_X86_REG_RSP,sp)
        self.uc.emu_start(0x700f20,STOP,count=100000)
        if self.uc.reg_read(UC_X86_REG_RIP)!=STOP:raise RuntimeError('product interrupted')
        raw=self.uc.reg_read(UC_X86_REG_XMM0)&((1<<64)-1)
        return struct.unpack('<d',struct.pack('<Q',raw))[0]
    def call(self,x,f,g0,initial, matrix=False):
        self.prefix_stop=not initial
        self.matrix_stop=matrix
        vals=[struct.pack('<i',len(x)), np.asarray(x,dtype='<f8').tobytes(), np.asarray(f,dtype='<f8').tobytes(),np.asarray(g0,dtype='<f8').tobytes(),struct.pack('<i',int(initial)),struct.pack('<i',-1),struct.pack('<i',-1),struct.pack('<d',0.5),struct.pack('<i',0),struct.pack('<i',0),struct.pack('<i',0)]
        ptrs=[];cur=DATA
        for val in vals:
            ptrs.append(cur);self.uc.mem_write(cur,val);cur+=(len(val)+31)//32*32
        for reg,ptr in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9),ptrs):self.uc.reg_write(reg,ptr)
        sp=STACK+0x80008
        self.uc.mem_write(sp,struct.pack('<Q',STOP)+b''.join(struct.pack('<Q',p) for p in ptrs[6:]))
        self.uc.reg_write(UC_X86_REG_RSP,sp)
        self.uc.emu_start(ENTRY,STOP,timeout=5_000_000,count=2_000_000)
        rip=self.uc.reg_read(UC_X86_REG_RIP)
        expected = STOP if initial else (MATRIX_END if matrix else PREFIX_END)
        if rip != expected:raise RuntimeError(f'incomplete at {rip:#x}')
        return dict(output=np.frombuffer(self.uc.mem_read(ptrs[1],len(x)*8),dtype='<f8').copy(),arrays={k:self.array(k) for k in DESCRIPTORS},iteration=self.i(0x7942fa4))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--elf',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    blob,segments=load_elf(args.elf); digest=hashlib.sha256(blob).hexdigest()
    if digest!='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':raise ValueError('wrong ELF')
    rng=np.random.default_rng(20260912); cases=[]
    for ndim in (3,6,9):
        o=PrefixOracle(segments); x=rng.normal(size=ndim); f=rng.normal(size=ndim); g0=rng.uniform(.1,2.,size=ndim)
        first=o.call(x,f,g0,True); x2=first['output']+rng.normal(size=ndim)*.01; f2=rng.normal(size=ndim)
        second=o.call(x2,f2,g0,False,matrix=True); py=secant_prefix(x2,f2,x,f,g0); df=second['arrays']['df'][:,0]; u=second['arrays']['u'][:,0]
        expected=dict(finf=native_block_sum_product(df,df), smat=float(np.sum(g0*df*df)), gmat=float(np.dot(df,u)))
        observed=dict(finf=float(second['arrays']['finf'][0,0]), smat=float(second['arrays']['smat'][0,0]), gmat=float(second['arrays']['gmat'][0,0]))
        errors=dict(prefix=float(np.max(np.abs(df-py.force_difference))),u=float(np.max(np.abs(u-py.u))),**{k:abs(observed[k]-expected[k]) for k in expected})
        cases.append(dict(ndim=ndim,input=dict(x=x.tolist(),f=f.tolist(),g0=g0.tolist(),x2=x2.tolist(),f2=f2.tolist()),observed=observed,expected=expected,errors=errors,passed=all(v<1e-10 for v in errors.values()),runtime_calls=o.runtime_calls,stop='0x6fa815 before DGEGV'))
    report=dict(elf=str(Path(args.elf).resolve()),sha256=digest,seed=20260912,hook_scope='allocation/memcpy/memset/printing only; arithmetic and inproduct execute ELF; emulation stops at 0x6fa815',cases=cases,passed=sum(c['passed'] for c in cases),total=len(cases))
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(report,indent=2))
    if report['passed']!=report['total']:raise SystemExit(1)
if __name__=='__main__':main()
