"""Bounded ABI probe for get_atompair_ (no main/PES; external calls logged)."""
import hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE, UC_HOOK_MEM_INVALID
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'; ENTRY=0x57f990; STOP=0x700000000100
STACK=0x710000000000; DATA=0x720000000000; HEAP=0x730000000000
RD=0x580640; NEIGH=0x580c50; FORB=0x580660; ALLOC=0x498a070; DEALLOC=0x498a650; OV=0x498a9c0

def run(pos, pair=(1,2), randoms=()):
 blob,segs=load_elf(ELF); u=Uc(UC_ARCH_X86,UC_MODE_64)
 for va,sz,ch in segs:
  lo=va&~4095; u.mem_map(lo,((va+sz+4095)&~4095)-lo)
  if ch:u.mem_write(va,ch)
 for p,s in ((STACK,0x200000),(DATA,0x200000),(HEAP,0x400000),(STOP&~4095,0x1000)):u.mem_map(p,s)
 n=len(pos); nptr=DATA; cell=DATA+0x1000; xyz=DATA+0x1100; z=DATA+0x2000; mask=DATA+0x2100; out=DATA+0x2200; pairp=DATA+0x2300; alloc=HEAP
 u.mem_write(nptr,struct.pack('<i',n));u.mem_write(cell,np.eye(3,dtype='<f8').tobytes());u.mem_write(xyz,np.asarray(pos,dtype='<f8').tobytes());u.mem_write(z,np.full(n,6,dtype='<i4').tobytes());u.mem_write(mask,np.ones(3*n,dtype='<i4').tobytes());u.mem_write(pairp,struct.pack('<ii',*pair));u.mem_write(alloc,b'\0'*0x10000)
 sp=STACK+0x1ff00;u.mem_write(sp,struct.pack('<Q',STOP));u.reg_write(UC_X86_REG_RSP,sp)
 for reg,val in ((UC_X86_REG_RDI,nptr),(UC_X86_REG_RSI,cell),(UC_X86_REG_RDX,xyz),(UC_X86_REG_RCX,z),(UC_X86_REG_R8,mask),(UC_X86_REG_R9,pairp)):u.reg_write(reg,val)
 calls=[]; ri=iter(randoms); cursor=[alloc]
 def ret():
  s=u.reg_read(UC_X86_REG_RSP);u.reg_write(UC_X86_REG_RIP,struct.unpack('<Q',u.mem_read(s,8))[0]);u.reg_write(UC_X86_REG_RSP,s+8)
 def hook(uc,a,sz,_):
  if a==STOP:uc.emu_stop();return
  if a==RD:
   uc.mem_write(uc.reg_read(UC_X86_REG_RDI),struct.pack('<d',next(ri,0.0)));calls.append('rng');ret();return
  if a==ALLOC:
   p=uc.reg_read(UC_X86_REG_RSI);q=cursor[0];cursor[0]+=0x4000;uc.mem_write(p,struct.pack('<Q',q));uc.mem_write(q,b'\0'*0x4000);calls.append('alloc');ret();return
  if a in (DEALLOC,OV):calls.append(hex(a));ret();return
  if a==NEIGH:
   calls.append('neighboringlist');
   # zero-length list: its output count/data pointers are ABI-dependent; leave
   # caller-provided zero storage and return to expose bounded control flow.
   ret();return
  if a==FORB:calls.append('check_forbiden');uc.reg_write(UC_X86_REG_RAX,0);ret();return
  if a>=ENTRY and a<0x580640:return
  raise RuntimeError(f'unhandled external {a:#x} at {uc.reg_read(UC_X86_REG_RIP):#x}')
 u.hook_add(UC_HOOK_CODE,hook);u.emu_start(ENTRY,STOP,count=500000)
 return dict(pair_before=list(pair),pair_after=list(struct.unpack('<ii',u.mem_read(pairp,8))),calls=calls,position=np.asarray(pos).tolist(),randoms=list(randoms))

if __name__=='__main__':
 rows=[]
 for p in (np.array([[0.,0.,0.],[4.,0.,0.],[0.,4.,0.],[0.,0.,4.]]),np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])):
  try: rows.append(run(p,randoms=[0.,.2,0.,.8,.9]))
  except Exception as e: rows.append({'status':'blocked','error':str(e)})
 out={'entry':hex(ENTRY),'scope':'bounded get_atompair only; no main/PES; neighboringlist/check_forbiden calls logged/stubbed','elf_sha256':hashlib.sha256(load_elf(ELF)[0]).hexdigest(),'rows':rows}
 Path('research/ga_ssw/evidence/native-get-atompair-bounded-20260917.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
