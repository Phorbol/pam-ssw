"""Bounded original compute-tail instructions; no LASP execution or physical PES."""
import json,struct
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64,UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
blob,segments=load_elf(ELF);u=Uc(UC_ARCH_X86,UC_MODE_64)
for va,ms,chunk in segments:
 lo=va&~4095;u.mem_map(lo,((va+ms+4095)&~4095)-lo);u.mem_write(va,chunk)
D=0x700000000000;u.mem_map(D,0x20000);obj=D;bp=D+0x18000
w=lambda p,v:u.mem_write(p,struct.pack('<d',v))
ptr=lambda p,v:u.mem_write(p,struct.pack('<Q',v))
def hook(u,addr,size,user):
 if addr in (0x12da8db,0x12da9d1,0x12d9f1d):u.emu_stop();return
 if not (0x12da756<=addr<0x12da8db or 0x12d9c4b<=addr<0x12d9f1d):raise RuntimeError(hex(addr))
u.hook_add(UC_HOOK_CODE,hook)
rows=[]
for radius in (1.7,3.,9.5,12.):
 for wlj,wc in ((.5,5/6),(0.,0.),(.2,.7)):
  u.mem_write(D,bytes(0x20000))
  for reg,val in ((UC_X86_REG_RBX,obj),(UC_X86_REG_RBP,bp),(UC_X86_REG_R12,0),(UC_X86_REG_R15,3),(UC_X86_REG_R14,1)):u.reg_write(reg,val)
  for offset,base,val in ((0x150,D+0x1000,wlj),(0x158,D+0x1100,wc)):
   ptr(obj+offset,base);w(base+8,val)
  X=D+0x2000;ptr(bp-0x30,X)
  for i in range(4):ptr(X+8*i,D+0x3000+32*i)
  w(D+0x3000,radius)
  Q=D+0x4000;ptr(bp-0xe8,Q);w(Q,.3);w(Q+24,-.4)
  T=D+0x5000;ptr(bp-0xf0,T);u.mem_write(T,struct.pack('<4i',1,1,1,1))
  C=332.06371;w(bp-0xf8,C);u.mem_write(bp-0x78,struct.pack('<i',1))
  eps=.21;R=3.3224
  values=(48*eps*R**12,24*eps*R**6,4*eps*R**12,4*eps*R**6)
  for k,value in enumerate(values):
   arr=D+0x6000+k*0x100;data=D+0x7000+k*0x100
   ptr(obj+0x180+k*8,arr);ptr(arr+8,data);w(data+8,value)
  u.emu_start(0x12da756,0x12da9d2,timeout=1000000,count=1000)
  E_lj=struct.unpack('<d',u.mem_read(bp-0x168,8))[0];E_c=struct.unpack('<d',u.mem_read(bp-0x170,8))[0]
  fpair=struct.unpack('<d',int(u.reg_read(UC_X86_REG_XMM2)).to_bytes(16,'little')[:8])[0] if wlj or wc else 0.
  t=(R/radius)**6;expectedE=wlj*eps*(t*t-2*t)+wc*C*.3*(-.4)/radius
  expectedF=(12*wlj*eps*(t*t-t)+wc*C*.3*(-.4)/radius)/radius**2
  np.testing.assert_allclose([E_lj+E_c,fpair],[expectedE,expectedF],rtol=2e-14,atol=1e-12)
  rows.append(dict(r=radius,wlj=wlj,wc=wc,native_energy=E_lj+E_c,formula_energy=expectedE,native_force_prefactor=fpair,formula_force_prefactor=expectedF))
geometry_rows=[]
rng=np.random.default_rng(17)
for index in range(12):
 u.mem_write(D,bytes(0x20000));xyz=rng.normal(size=(4,3))
 u.reg_write(UC_X86_REG_RBX,obj);u.reg_write(UC_X86_REG_RBP,bp)
 ptr(bp-0xa0,D+0x8000);ptr(D+0x8000,D+0x8100);u.mem_write(D+0x8100,struct.pack('<5i',0,1,2,3,1));ptr(bp-0x38,0)
 ptr(bp-0x30,D+0x2000)
 for i in range(4):ptr(D+0x2000+8*i,D+0x3000+32*i);u.mem_write(D+0x3000+32*i,xyz[i].astype('<f8').tobytes())
 u.emu_start(0x12d9c4b,0x12d9f1e,timeout=1000000,count=1000)
 c=struct.unpack('<d',u.mem_read(bp-0x88,8))[0];sn=struct.unpack('<d',u.mem_read(bp-0x98,8))[0]
 v1=xyz[0]-xyz[1];v2=xyz[2]-xyz[1];v3=xyz[3]-xyz[2]
 aa=np.cross(v1,-v2);bb=np.cross(v3,-v2);den=np.linalg.norm(aa)*np.linalg.norm(bb)
 expected=[aa@bb/den,np.linalg.norm(v2)*(aa@v3)/den]
 np.testing.assert_allclose([c,sn],expected,atol=1e-14,rtol=1e-14)
 geometry_rows.append(dict(positions=xyz.tolist(),native_cos=c,native_sin=sn,stock_formula_cos_sin=expected))
Path('research/ga_ssw/evidence/native-dihedral-amber/tail-oracle.json').write_text(json.dumps(dict(entry='0x12da756',stop=['0x12da8db','0x12da9d1'],scope='actual compute tail; geometry/coeff producer supplied explicitly; no external hooks/functions; no whole-engine claim',cases=rows,geometry_cases=geometry_rows),indent=2)+'\n');print(len(rows),'original compute-tail and',len(geometry_rows),'angle-convention cases matched')
