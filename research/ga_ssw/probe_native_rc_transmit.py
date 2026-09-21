"""Original instruction slice of RC divide_force, not whole-function parity."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
blob,segs=load_elf(ELF);sha=hashlib.sha256(blob).hexdigest()
assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
u=Uc(UC_ARCH_X86,UC_MODE_64)
for page,size in [(0x818000,0x1000),(0x4a4e000,0x1000),(0x1d103000,0x1000),(0x70000000,0x10000)]:
 u.mem_map(page,size)
 for addr,_,data in segs:
  lo=max(addr,page);hi=min(addr+len(data),page+size)
  if hi>lo:u.mem_write(lo,data[lo-addr:hi-addr])
def put(p,x):u.mem_write(p,np.asarray(x,dtype='<f8').tobytes())
def ptr(p,x):u.mem_write(p,struct.pack('<Q',x))
def get(p):return np.frombuffer(u.mem_read(p,24),dtype='<f8').copy()
obj=0x70001000;fp=0x70002000;vp=0x70002100;passed=0x70002200;radial=0x70002300;tangent=0x70002400;nonrot=0x70002500;sp=0x70008000
rows=[]
for seed in (113,271):
 rng=np.random.default_rng(seed);axis=rng.normal(size=3);axis/=np.linalg.norm(axis);v=rng.normal(size=3);F=rng.normal(size=3)
 for k in (0.,.5,.7,1.):
  u.mem_write(0x70000000,bytes(0x10000));put(obj+0xf8,axis);put(fp,F);put(vp,v);put(0x1d103690,[k]);ptr(sp+0xb8,tangent);ptr(sp+0x118,nonrot)
  for reg,val in [(UC_X86_REG_R10,obj),(UC_X86_REG_R12,fp),(UC_X86_REG_R13,vp),(UC_X86_REG_R14,24),(UC_X86_REG_R8,radial),(UC_X86_REG_RDX,passed),(UC_X86_REG_RSP,sp)]:u.reg_write(reg,val)
  u.emu_start(0x81826c,0x8184ef,count=1000,timeout=1000000)
  assert u.reg_read(UC_X86_REG_RIP)==0x8184ef
  vpperp=v-axis*(axis@v);axial=axis*(axis@F);radialF=vpperp*(vpperp@F)/(vpperp@vpperp);tangentialF=F-axial-radialF
  native_pass=get(passed);native_keep=get(sp+0x1e0)
  rows.append(dict(seed=seed,krot=k,axis=axis.tolist(),v=v.tolist(),F=F.tolist(),passed=native_pass.tolist(),retained=native_keep.tolist(),
    pass_axis_residual_error=float(abs(native_pass-(F-k*(F-axial))).max()),pass_tangent_residual_error=float(abs(native_pass-(F-k*tangentialF)).max()),retained_tangent_error=float(abs(native_keep-k*tangentialF).max()),nonrot_error=float(abs(get(nonrot)-(axial+radialF)).max())))
out=Path('research/ga_ssw/evidence/native-rc-transmit-slice');out.mkdir(exist_ok=False)
(out/'probe.py').write_text(Path(__file__).read_text());(out/'result.json').write_text(json.dumps(dict(elf_sha256=sha,entry='0x81826c',stop='0x8184ef before crossproduct call',scope='internal original instruction slice with synthetic register/stack/object inputs; not cold-start/full-function/caller validation',rows=rows),indent=2)+'\n')
print([(r['krot'],r['pass_axis_residual_error'],r['pass_tangent_residual_error'],r['retained_tangent_error']) for r in rows])
