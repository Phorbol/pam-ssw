"""Run_type=5 get_atompair with actual geometry callees; no main or PES.

Only uniform RNG, Fortran allocation/overflow/deallocation, floor and acos are replaced.
Input fixatom=0 is an explicit free-atom fixture, not inferred native defaults.
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_axis_group_selection_v2 import ELF, SHA

ENTRY=0x57f990
STOP=0x700000000100
STACK=0x710000000000
DATA=0x720000000000
PARA=0x53ed7a0


def run(positions, pair, randoms, numbers=None, fixatom=None, trace_angles=False):
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == SHA
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, size, data in segments:
        low = va & ~4095
        uc.mem_map(low, ((va+size+4095)&~4095)-low)
        if data:
            uc.mem_write(va, data)
    for address, size in ((STACK,0x20000),(DATA,0x40000),(STOP&~4095,0x1000)):
        uc.mem_map(address,size)
    def q(address,value): uc.mem_write(address,struct.pack('<Q',value))
    def i(address,value): uc.mem_write(address,struct.pack('<i',value))
    def readq(address): return struct.unpack('<Q',uc.mem_read(address,8))[0]
    positions=np.asarray(positions,dtype='<f8')
    n=len(positions)
    numbers=np.full(n,6,dtype='<i4') if numbers is None else np.asarray(numbers,dtype='<i4')
    fixatom=np.zeros(n,dtype='<f8') if fixatom is None else np.asarray(fixatom,dtype='<f8')
    nptr,cell,xyz,z,mask,pairptr,fixptr=[DATA+v for v in (0,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000)]
    i(nptr,n)
    uc.mem_write(cell,(30*np.eye(3,dtype='<f8')).tobytes())
    uc.mem_write(xyz,positions.tobytes())
    uc.mem_write(z,numbers.tobytes())
    uc.mem_write(mask,np.ones((n,3),dtype='<i4').tobytes())
    uc.mem_write(pairptr,struct.pack('<ii',*pair))
    uc.mem_write(fixptr,fixatom.tobytes())
    i(PARA+0x100,5)
    q(PARA+0x2df30,fixptr)
    q(PARA+0x2df70,1)
    q(0x53e0f98,DATA+0xf000)  # Unresolved dynamic __svml_floor2 GOT entry.
    sp=STACK+0x10000
    q(sp,STOP)
    for register,value in ((UC_X86_REG_RDI,nptr),(UC_X86_REG_RSI,cell),
          (UC_X86_REG_RDX,xyz),(UC_X86_REG_RCX,z),(UC_X86_REG_R8,mask),
          (UC_X86_REG_R9,pairptr),(UC_X86_REG_RSP,sp)):
        uc.reg_write(register,value)
    calls=[]
    checks=[]
    angles=[]
    events={'distance_or_fixatom_rejections':0,'forbidden_rejections':0,
            'element_rejections':0,'accepted_exit':0}
    draws=[]
    rng=np.random.default_rng(817)
    cursor=iter(randoms)
    heap=[DATA+0x10000]
    def ret():
        rsp=uc.reg_read(UC_X86_REG_RSP)
        target=readq(rsp)
        uc.reg_write(UC_X86_REG_RSP,rsp+8)
        uc.reg_write(UC_X86_REG_RIP,target)
    def hook(_,address,size,user):
        if address==0x580520:
            base=uc.reg_read(UC_X86_REG_RBP)
            checks.append(dict(pair=list(struct.unpack('<ii',uc.mem_read(pairptr,8))),
                               allowed=bool(struct.unpack('<i',uc.mem_read(base-0xbc,4))[0])))
        if address==0x58052b: events['distance_or_fixatom_rejections']+=1
        if address==0x580529: events['accepted_exit']+=1
        if address==0x580527 and uc.reg_read(UC_X86_REG_EFLAGS)&0x40:
            events['forbidden_rejections']+=1
        if address==0x5804ed and not (uc.reg_read(UC_X86_REG_EFLAGS)&0x41):
            events['element_rejections']+=1
        if address==0x580640:
            value=next(cursor,None)
            if value is None: value=float(rng.random())
            draws.append(value)
            uc.mem_write(uc.reg_read(UC_X86_REG_RDI),struct.pack('<d',value))
            ret()
        elif address==0x49206e0:
            packed=uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')
            value=struct.unpack('<d',packed[:8])[0]
            if trace_angles:
                angles.append(dict(pair=list(struct.unpack('<ii',uc.mem_read(pairptr,8))),
                    cosine=value,axis=list(struct.unpack('<ddd',uc.mem_read(0x78e1730,24))),
                    neighbor=list(struct.unpack('<ddd',uc.mem_read(0x78e1770,24))),
                    image=list(struct.unpack('<ddd',uc.mem_read(0x78e1750,24)))))
            result=struct.pack('<d',float(np.arccos(value)))+packed[8:]
            uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(result,'little'))
            ret()
        elif address==DATA+0xf000:
            packed=uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')
            values=np.frombuffer(packed,dtype='<f8')
            uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(np.floor(values).tobytes(),'little'))
            ret()
        elif address==0x498a9c0:
            # Integer size multiplication: result pointer, rank, extents, itemsize.
            rank=uc.reg_read(UC_X86_REG_RSI)
            assert rank==2
            product=uc.reg_read(UC_X86_REG_RDX)*uc.reg_read(UC_X86_REG_RCX)
            q(uc.reg_read(UC_X86_REG_RDI),product)
            uc.reg_write(UC_X86_REG_RAX,0)
            ret()
        elif address==0x498a070:
            size=uc.reg_read(UC_X86_REG_RDI)
            assert size<=0x4000
            q(uc.reg_read(UC_X86_REG_RSI),heap[0])
            heap[0]+=0x4000
            uc.reg_write(UC_X86_REG_RAX,0)
            ret()
        elif address==0x498a650:
            uc.reg_write(UC_X86_REG_RAX,0)
            ret()
        elif (ENTRY<=address<0x580640 or 0x580660<=address<0x580df0
              or 0x581240<=address<0x5816b0 or 0x578480<=address<0x578660
              or 0x58f680<=address<0x58f800):
            if address in (0x580660,0x580c50,0x581240): calls.append(hex(address))
        else:
            raise RuntimeError(f'Unapproved callee {address:#x}')
    uc.hook_add(UC_HOOK_CODE,hook)
    try:
        uc.emu_start(ENTRY,STOP,count=10000000)
    except Exception as exc:
        raise RuntimeError(f'{exc}; pc={uc.reg_read(UC_X86_REG_RIP):#x}; '
                           f'sp={uc.reg_read(UC_X86_REG_RSP):#x}; '
                           f'callees={calls[-10:]}; draws={len(draws)}') from exc
    completed=uc.reg_read(UC_X86_REG_RIP)==STOP
    return dict(positions=positions.tolist(),numbers=numbers.tolist(),fixatom=fixatom.tolist(),
                completed=completed,stop_pc=hex(uc.reg_read(UC_X86_REG_RIP)),
                instruction_limit=10000000,events=events,checks=checks,angles=angles,
                pair_before=list(pair),pair_after=list(struct.unpack('<ii',uc.mem_read(pairptr,8))),
                rng_prefix=list(randoms),rng_tail_seed=817,draws=draws,callees=calls)


def main():
    from ase.build import molecule
    rows=[]
    for name in ('C2H6','CH3OH','C6H6'):
        atoms=molecule(name)
        for label,prefix in [('preserve-first',[0.,.2,0.]),('replace-first',[.9,.7,0.]),
                             ('neighbor-first',[0.,.2,.9])]:
            row=run(atoms.positions,(1,2),prefix,atoms.numbers)
            rows.append(dict(name=name,branch=label,**row))
    report=dict(scope=__doc__,sha256=SHA,cases=rows)
    Path('research/ga_ssw/evidence/native-get-atompair-v2-20260917.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],branch=r['branch'],completed=r['completed'],pair=r['pair_after'],draws=len(r['draws']),callees=sorted(set(r['callees']))) for r in rows]))


if __name__=='__main__': main()
