"""Execute uploaded ELF weight-update instructions under Unicorn (optional).

Only acos is replaced with host math.acos. This is instruction-level evidence,
not native process execution, full LASP parity or a production dependency.
API reference: https://www.unicorn-engine.org/docs/tutorial.html
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
import unicorn
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX,
    UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_RSP,
    UC_X86_REG_RIP, UC_X86_REG_XMM0)
from pamssw.standalone.gaussian import adjust_native_weight

ENTRY=0x6e1730
ACOS=0x49206e0
STOP=0x700000000000
STACK=0x710000000000
DATA=0x720000000000


def load_elf(path):
    blob=Path(path).read_bytes()
    if blob[:6]!=b'\x7fELF\x02\x01':
        raise ValueError('requires little-endian ELF64')
    phoff=struct.unpack_from('<Q',blob,32)[0]
    entsize,count=struct.unpack_from('<HH',blob,54)
    segments=[]
    for i in range(count):
        kind,flags,off,va,pa,fs,ms,align=struct.unpack_from('<IIQQQQQQ',blob,phoff+i*entsize)
        if kind==1:
            segments.append((va,ms,blob[off:off+fs]))
    return blob,segments


def emulate(segments,args):
    uc=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segments:
        start=va&~4095
        uc.mem_map(start,((va+ms+4095)&~4095)-start)
        uc.mem_write(va,chunk)
    for address in (STOP,STACK,DATA):
        uc.mem_map(address,0x10000)
    pointers=[]
    values=[struct.pack('<i',len(args['n']))]
    for key in ('d1','d2','fa0','fa2','n','e2','w'):
        values.append(np.asarray(args[key],dtype='<f8').tobytes())
    values.extend([struct.pack('<d',0.)]*2)
    values.extend(struct.pack('<d',args[key]) for key in ('maxw','step','scalefact0'))
    cursor=DATA
    for value in values:
        pointers.append(cursor);uc.mem_write(cursor,value)
        cursor+=(len(value)+31)//32*32
    for register,pointer in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,
                                 UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9),pointers):
        uc.reg_write(register,pointer)
    sp=STACK+0x8008
    uc.mem_write(sp,struct.pack('<Q',STOP)+b''.join(struct.pack('<Q',p) for p in pointers[6:]))
    uc.reg_write(UC_X86_REG_RSP,sp)
    calls=[]
    def hook(machine,address,size,user):
        if address==ACOS:
            raw=machine.reg_read(UC_X86_REG_XMM0)&((1<<64)-1)
            x=struct.unpack('<d',struct.pack('<Q',raw))[0]
            calls.append(x)
            result=struct.unpack('<Q',struct.pack('<d',math.acos(x)))[0]
            machine.reg_write(UC_X86_REG_XMM0,result)
            rsp=machine.reg_read(UC_X86_REG_RSP)
            ret=struct.unpack('<Q',machine.mem_read(rsp,8))[0]
            machine.reg_write(UC_X86_REG_RSP,rsp+8)
            machine.reg_write(UC_X86_REG_RIP,ret)
        elif not ENTRY<=address<0x6e3000:
            raise RuntimeError(f'unexpected execution at {address:#x}')
    uc.hook_add(UC_HOOK_CODE,hook)
    uc.emu_start(ENTRY,STOP,timeout=5_000_000,count=1_000_000)
    if uc.reg_read(UC_X86_REG_RIP)!=STOP:
        raise RuntimeError('emulation budget exhausted')
    def scalar(index):
        return struct.unpack('<d',uc.mem_read(pointers[index],8))[0]
    return dict(weight=scalar(7),energy=scalar(8),angle_degrees=scalar(9),
                force=np.frombuffer(uc.mem_read(pointers[3],args['fa0'].nbytes),dtype='<f8').reshape(args['fa0'].shape).tolist(),
                updates=len(calls)-1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--elf',required=True)
    parser.add_argument('--output',required=True)
    opt=parser.parse_args()
    blob,segments=load_elf(opt.elf)
    if hashlib.sha256(blob).hexdigest()!='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':
        raise ValueError('address contract requires the inspected uploaded ELF')
    rng=np.random.default_rng(20260909)
    cases=[]
    for natoms in (1,2,3,5,15,45):
        for trial in range(8):
            n=rng.normal(size=(natoms,3));n/=np.linalg.norm(n)
            args=dict(fa0=rng.normal(size=n.shape)-trial*n,fa2=rng.normal(size=n.shape)*.1,
                      n=n,d1=.8,d2=.3,e2=-20.,w=.1,maxw=10.,step=1.2,scalefact0=1.5)
            cases.append(dict(natoms=natoms,trial=trial,args=args))
    for angle in (86.999,87.,87.001):
        rad=angle*math.pi/180
        cases.append(dict(label=f'angle_{angle}',args=dict(
            fa0=np.array([[math.cos(rad)-1,math.sin(rad),0.]]),
            fa2=np.zeros((1,3)),n=np.array([[1.,0.,0.]]),
            d1=1.,d2=1.,e2=-10.,w=1.,maxw=10.,step=1.,scalefact0=2.)))
    for force,w,maxw in (([1.,1.,0.],2.,1.),([-1.,1.,0.],1.,1.5),([-100.,1.,0.],.1,1.)):
        cases.append(dict(label='weight_limit_order',args=dict(
            fa0=np.array([force]),fa2=np.zeros((1,3)),n=np.array([[1.,0.,0.]]),
            d1=1.,d2=1.,e2=-10.,w=w,maxw=maxw,step=2.,scalefact0=2.)))
    for case in cases:
        args=case.pop('args')
        original=emulate(segments,args)
        python=adjust_native_weight(**args)
        errors={k:float(np.max(np.abs(np.asarray(original[k])-getattr(python,k))))
                for k in ('weight','energy','angle_degrees','force','updates')}
        case.update(input={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in args.items()},
                    emulated=original,errors=errors,passed=all(v<1e-10 for v in errors.values()))
    report=dict(elf=str(Path(opt.elf).resolve()),sha256=hashlib.sha256(blob).hexdigest(),
                unicorn_version=unicorn.__version__,seed=20260909,entry=hex(ENTRY),
                evidence='ELF instructions emulated; only acos replaced by host math.acos; not full native execution',
                cases=cases,passed=sum(c['passed'] for c in cases),total=len(cases))
    Path(opt.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    if report['passed']!=report['total']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
