"""Isolated VMB2 + native RAN3; host cos/log only, no LASP main or PES."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256

BASE=0x720000010000
STACK=0x720000030000
STOP=0x720000050000

def run(segments, n, temperature, mask, initial, seed):
    uc=Uc(UC_ARCH_X86, UC_MODE_64)
    for va,ms,data in segments:
        lo=va&~4095
        uc.mem_map(lo, ((va+ms+4095)&~4095)-lo)
        uc.mem_write(va,data)
    for address in (BASE,STACK,STOP): uc.mem_map(address,0x10000)
    uc.mem_write(BASE,struct.pack('<i',n))
    uc.mem_write(BASE+8,struct.pack('<d',temperature))
    uc.mem_write(BASE+16,struct.pack('<d',seed))
    initial=np.asarray(initial,dtype='<f8').reshape(n,3)
    mask=np.asarray(mask,dtype='<i4').reshape(n,3)
    uc.mem_write(BASE+0x1000,initial.tobytes())
    uc.mem_write(BASE+0x2000,mask.tobytes())
    rsp=STACK+0x8008
    uc.mem_write(rsp,struct.pack('<Q',STOP))
    for reg,value in ((UC_X86_REG_RDI,BASE),(UC_X86_REG_RSI,BASE+8),
                      (UC_X86_REG_RDX,BASE+0x1000),(UC_X86_REG_RCX,BASE+0x2000),
                      (UC_X86_REG_R8,BASE+16),(UC_X86_REG_RSP,rsp)):
        uc.reg_write(reg,value)
    draws=[]; seeds=[]; math_calls=[]
    def xmm0():return struct.unpack('<d',uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')[:8])[0]
    def hook(m,address,size,user):
        if address==STOP:
            m.emu_stop()
        elif address==0x58cbf0:
            seeds.append(struct.unpack('<i',m.mem_read(m.reg_read(UC_X86_REG_RDI),4))[0])
        elif address in (0x58cb9b,0x58cba8):
            draws.append(xmm0())
        elif address in (0x4920770,0x4920840):
            value=xmm0(); fn=math.cos if address==0x4920770 else math.log
            result=fn(value); math_calls.append(dict(function=fn.__name__,argument=value))
            m.reg_write(UC_X86_REG_XMM0,int.from_bytes(struct.pack('<d',result),'little'))
            sp=m.reg_read(UC_X86_REG_RSP); ret=struct.unpack('<Q',m.mem_read(sp,8))[0]
            m.reg_write(UC_X86_REG_RSP,sp+8);m.reg_write(UC_X86_REG_RIP,ret)
    uc.hook_add(UC_HOOK_CODE,hook)
    uc.emu_start(0x58c930,STOP,timeout=2_000_000,count=200_000)
    if uc.reg_read(UC_X86_REG_RIP)!=STOP: raise RuntimeError('native function did not return')
    actual=np.frombuffer(uc.mem_read(BASE+0x1000,24*n),dtype='<f8').reshape(n,3)
    factor=struct.unpack('<d',uc.mem_read(0x4a43918,8))[0]
    two_pi=struct.unpack('<d',uc.mem_read(0x4a43920,8))[0]
    expected=initial.copy(); cursor=0
    for i in range(n):
        for j in range(3):
            if mask[i,j]:
                u1,u2=draws[cursor:cursor+2];cursor+=2
                expected[i,j]=factor*math.sqrt(temperature/1.)*math.sqrt(-2*math.log(u1))*math.cos(two_pi*u2)
    expected-=expected.mean(axis=0)
    err=float(np.max(np.abs(actual-expected)))
    passed=cursor==len(draws)==2*np.count_nonzero(mask) and err<1e-14
    if seeds: passed=passed and seeds[0]==-int((seed+1)*10)
    return dict(n=n,temperature=temperature,seed_input=seed,initial=initial.tolist(),
        mask=mask.tolist(),native=actual.tolist(),reference=expected.tolist(),
        random_draws=draws,first_ran3_seed=seeds[0] if seeds else None,
        math_hooks=math_calls,max_abs_error=err,passed=bool(passed))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    blob,segments=load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    cases=[]
    for temperature in (1.,300.):
        for masked in (False,True):
            flags=np.ones((3,3),dtype=int)
            if masked:flags[[0,1,2],[0,1,2]]=0
            cases.append(run(segments,3,temperature,flags,np.arange(9).reshape(3,3)*.01,.17))
    cases.append(run(segments,1,300.,np.ones((1,3)),np.zeros((1,3)),.83))
    report=dict(elf=ELF_DEFAULT,sha256=ELF_SHA256,entry='0x58c930',
        executed='VMB2, VELO_LOC and RAN3 original instructions; native arithmetic/mean/seed path',
        hooks='host math.cos/log replace only scalar math library calls; native RAN3 executes unmodified',
        scope='synthetic flat output/mask ABI; no surrounding gen_randommode/main/PES or expiry path',
        pes_requests=0,cases=cases,passed=all(c['passed'] for c in cases))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],errors=[c['max_abs_error'] for c in cases])))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
