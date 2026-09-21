"""Run5/modelevel0/Q-off/no-compression coefficient selection, no PES."""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256

BASE, STACK = 0x720000010000, 0x720000030000
PARA, CONTROL = 0x53ed7a0, 0x53ed5c0


def run(segments, group, draws, ratio=50, local_probability=.5, group_threshold=.5):
    uc=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,size,data in segments:
        lo=va&~4095
        uc.mem_map(lo,((va+size+4095)&~4095)-lo)
        uc.mem_write(va,data)
    for address in (BASE,STACK): uc.mem_map(address,0x10000)
    def put(a,fmt,v): uc.mem_write(a,struct.pack(fmt,v))
    obj=BASE+0x1000
    put(BASE,'<Q',obj)
    put(obj+0x1660,'<i',1)
    put(obj+0x1ae0,'<Q',BASE+0x4000)
    put(obj+0x1b10,'<q',len(group))
    put(obj+0x2204,'<i',0)
    uc.mem_write(BASE+0x4000,np.asarray(group,dtype='<i4').tobytes())
    for offset,value in ((0x100,5),(0x2db78,0),(0x2db80,0),(0x2e25c,0),(0x2db54,ratio)):
        put(PARA+offset,'<i',value)
    put(PARA+0x2dba8,'<d',local_probability)
    put(PARA+0x2dbb0,'<d',group_threshold)
    put(CONTROL+0x68,'<i',0)
    uc.reg_write(UC_X86_REG_RDI,BASE)
    uc.reg_write(UC_X86_REG_RSP,STACK+0x8008)
    consumed=[]
    iterator=iter(draws)
    def hook(m,address,size,user):
        if address==0x5c050f: m.emu_stop()
        elif address==0x580640:
            value=next(iterator);consumed.append(value)
            put(m.reg_read(UC_X86_REG_RDI),'<d',value)
            sp=m.reg_read(UC_X86_REG_RSP)
            ret=struct.unpack('<Q',m.mem_read(sp,8))[0]
            m.reg_write(UC_X86_REG_RSP,sp+8);m.reg_write(UC_X86_REG_RIP,ret)
    uc.hook_add(UC_HOOK_CODE,hook)
    uc.emu_start(0x5c0100,0x5c050f,timeout=2_000_000,count=100_000)
    assert uc.reg_read(UC_X86_REG_RIP)==0x5c050f
    coeff=np.frombuffer(uc.mem_read(CONTROL+0x158,80),dtype='<f8')
    marker=struct.unpack('<i',uc.mem_read(obj+0x2204,4))[0]
    return dict(group=group,draws=consumed,ratio=ratio,local_probability=local_probability,
                group_threshold=group_threshold,coefficients=coeff.tolist(),group_marker=marker)


def main():
    blob,segments=load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    rows=[]
    for group in ([0,0,0],[0,1,1]):
        for u in (.1,.5,.9):
            for marker in (.1,.5,.9):
                row=run(segments,group,[.3,.2,.7,u,marker])
                expected=np.zeros(10);expected[1]=1
                slot=4 if .5>u or not sum(group) else 6
                expected[slot]=1.6
                assert np.allclose(row['coefficients'],expected,atol=1e-15,rtol=0)
                assert row['group_marker']==(-1 if slot==4 and marker>.5 else 0)
                assert len(row['draws'])==5
                rows.append(row)
    Path('research/ga_ssw/evidence/native-local-coefficients-20260917.json').write_text(json.dumps(
        dict(scope=__doc__,sha256=ELF_SHA256,cases=rows),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),matched=len(rows))))


if __name__=='__main__': main()
