"""Direct bounded oracle for localatompairgroup_mode_ (no main/PES)."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64,UC_HOOK_CODE
from unicorn.x86_const import *
from ase import Atoms
from pamssw.standalone.native_pair_group import native_local_pair_group
from research.ga_ssw.probe_native_weight_emulated import load_elf
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'; SHA='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'; ENTRY=0x6e4aa0; STOP=0x700000000100; STACK=0x710000000000; DATA=0x720000000000

def run(pair,a,b,initial=None,positions=None):
 blob,segs=load_elf(ELF); sha=hashlib.sha256(blob).hexdigest(); assert sha == SHA; u=Uc(UC_ARCH_X86,UC_MODE_64)
 for va,sz,ch in segs:
  lo=va&~4095;u.mem_map(lo,((va+sz+4095)&~4095)-lo)
  if ch:u.mem_write(va,ch)
 for va,sz in ((STACK,0x20000),(DATA,0x10000),(STOP&~4095,0x1000)):u.mem_map(va,sz)
 n=4; nptr=DATA; pos=DATA+0x1000; out=DATA+0x2000; pp=DATA+0x3000; ap=DATA+0x3100; bp=DATA+0x3200
 xyz=np.array([[0.,0.,0.],[2.,1.,0.],[4.,2.,3.],[8.,-1.,2.]]) if positions is None else np.asarray(positions,dtype='<f8')
 u.mem_write(nptr,struct.pack('<i',n));u.mem_write(pos,xyz.tobytes());u.mem_write(pp,struct.pack('<ii',*pair));u.mem_write(ap,struct.pack('<'+'i'*n,*a));u.mem_write(bp,struct.pack('<'+'i'*n,*b));
 init=np.zeros((n,3)) if initial is None else np.asarray(initial,dtype='<f8');u.mem_write(out,init.tobytes())
 sp=STACK+0x10000;u.mem_write(sp,struct.pack('<Q',STOP))
 for r,v in ((UC_X86_REG_RDI,nptr),(UC_X86_REG_RSI,pos),(UC_X86_REG_RDX,out),(UC_X86_REG_RCX,pp),(UC_X86_REG_R8,ap),(UC_X86_REG_R9,bp),(UC_X86_REG_RSP,sp)):u.reg_write(r,v)
 reached=[False]
 def hook(m,addr,size,user):
  if addr==STOP:reached[0]=True;m.emu_stop()
 u.hook_add(UC_HOOK_CODE,hook);u.emu_start(ENTRY,STOP+1,count=100000)
 if not reached[0]:raise RuntimeError(f'no STOP rip={u.reg_read(UC_X86_REG_RIP):#x}')
 return {'pair':list(pair),'mask_a':list(a),'mask_b':list(b),'positions':xyz.tolist(),'initial':init.tolist(),'output':np.frombuffer(u.mem_read(out,24*n),dtype='<f8').reshape(n,3).tolist(),'sha256':sha}

def main():
    base=np.array([[0.,0.,0.],[2.,1.,0.],[4.,2.,3.],[8.,-1.,2.]])
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rows=[run((1,2),[1,1,0,0],[0,1,1,0]), run((1,2),[0,0,1,0],[0,0,1,0]),
          run((2,3),[1,0,1,0],[0,1,1,0],np.arange(12,dtype=float).reshape(4,3)),
          run((1,2),[1,1,0,0],[0,1,1,0],positions=base+17),
          run((1,2),[1,1,0,0],[0,1,1,0],positions=base@rot.T+np.array([4.,-3.,2.]))]
    for row in rows:
        atoms=Atoms('H4',positions=np.asarray(row['positions']))
        got=native_local_pair_group(atoms,tuple(np.asarray(row['pair'])-1),row['mask_a'],row['mask_b'],np.asarray(row['initial']))
        err=float(np.max(np.abs(got-np.asarray(row['output'])))); row['python_max_error']=err
        assert err < 5e-12

    out={'entry':hex(ENTRY),'scope':'direct localatompairgroup only; no main/PES/GPU','cases':rows};p=Path('research/ga_ssw/evidence/native-local-pair-group-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
