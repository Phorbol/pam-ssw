"""Execute native setconstraints warm-cache branch; no projection arithmetic hooks."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import unicorn
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RSP, UC_X86_REG_RIP, UC_X86_REG_RAX
from research.ga_ssw.probe_addgaussian_emulated import Oracle as Base, DATA, STACK, STOP
from research.ga_ssw.probe_native_weight_emulated import load_elf
ENTRY=0x5790e0
DESCS=(0x54c0480,0x54c04e0,0x54c0540,0x54c05a0,0x54c0600,0x54c0660)
class Oracle(Base):
 def hook(self,uc,a,size,user):
  if a in (0x4970360,0x4a10430):return super().hook(uc,a,size,user)
  if a==0x4a102b0:
   dst=uc.reg_read(UC_X86_REG_RDI);src=uc.reg_read(UC_X86_REG_RSI);n=uc.reg_read(UC_X86_REG_RDX)
   uc.mem_write(dst,bytes(uc.mem_read(src,n)));uc.reg_write(UC_X86_REG_RAX,dst);self.hooks.append({'name':'memcpy','size':n});self.ret()
  elif not ENTRY<=a<0x57e980:raise RuntimeError(f'unexpected PC {a:#x}')
 def run(self,x,v):
  x=np.asarray(x,float);v=np.asarray(v,float);self.n=len(x);self.order='C';self.cursor=DATA+0x10000;self.hooks=[]
  self.uc.mem_write(DATA,bytes(0x100000));self.uc.mem_write(STACK,bytes(0x100000))
  # Construct valid cached NX/NY/NZ from a previous same-N initialization.
  for j,a in enumerate(DESCS):
   z=np.zeros_like(x)
   if j<3:z[:,j]=1/np.sqrt(self.n)
   self.descriptor(a,self.arr(z),(3,self.n))
  self.uc.mem_write(0x54c06c0,struct.pack('<i',0))
  pn=self.alloc(struct.pack('<i',self.n));px=self.arr(x);pv=self.arr(v)
  sp=STACK+0x80008;self.q(sp,STOP)
  for r,p in ((UC_X86_REG_RSP,sp),(UC_X86_REG_RDI,pn),(UC_X86_REG_RSI,px),(UC_X86_REG_RDX,pv)):self.uc.reg_write(r,p)
  self.uc.emu_start(ENTRY,STOP,timeout=5_000_000,count=2_000_000)
  if self.uc.reg_read(UC_X86_REG_RIP)!=STOP:raise RuntimeError('instruction/timeout budget exhausted')
  return self.readarr(pv).copy(),[self.readarr(self.readq(a)).copy() for a in DESCS],list(self.hooks)
def basis(x):
 q=x-x.mean(0);b=np.stack([np.broadcast_to(e,x.shape).ravel() for e in np.eye(3)]+[np.cross(e,q).ravel() for e in np.eye(3)],axis=1)
 u,s,_=np.linalg.svd(b,full_matrices=False);u=u[:,s>1e-12*s[0]]
 return u,np.eye(x.size)-u@u.T

def main():
 p=argparse.ArgumentParser();p.add_argument('--elf',required=True);p.add_argument('--output',required=True);o=p.parse_args()
 blob,segments=load_elf(o.elf);assert hashlib.sha256(blob).hexdigest()=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
 oracle=Oracle(segments);rng=np.random.default_rng(20260910);cases=[]
 for n in (2,4,7,13):
  x=rng.normal(size=(n,3))@np.array([[2.,.6,.3],[0.,1.,.4],[0.,0.,.5]])+np.array([3.,-2.,1.])
  v=rng.normal(size=x.shape);u,pj=basis(x)
  out,modes,hooks=oracle.run(x,v)
  # Matrix from complete function responses, one independent input per column.
  columns=[]
  for e in np.eye(x.size):columns.append(oracle.run(x,e.reshape(x.shape))[0].ravel())
  native=np.stack(columns,axis=1)
  rot,_=np.linalg.qr(rng.normal(size=(3,3)))
  if np.linalg.det(rot)<0:rot[:,0]*=-1
  outrot,_,_=oracle.run(x@rot,v@rot)
  trans,_,_=oracle.run(x+np.array([7.,-3.,4.]),v)
  rigid=v*0+np.cross(np.array([.3,-.7,.5]),x-x.mean(0))
  rout,_,_=oracle.run(x,rigid)
  each=np.stack([m.ravel() for m in modes],axis=1)
  independent=np.eye(x.size)-each@each.T
  sequential=(np.eye(x.size)-each[:,3:]@each[:,3:].T)@(np.eye(x.size)-each[:,:3]@each[:,:3].T)
  c=dict(n=n,rotation_matrix=rot.tolist(),rotated_output=outrot.tolist(),translated_output=trans.tolist(),rigid_input=rigid.tolist(),rigid_output=rout.tolist(),projected_reference=(pj@v.ravel()).reshape(x.shape).tolist(),rigid_rank=u.shape[1],positions=x.tolist(),vector=v.tolist(),output=out.tolist(),modes=[m.tolist() for m in modes],hooks=hooks,native_matrix=native.tolist(),orthogonal_matrix=pj.tolist(),mode_gram=(each.T@each).tolist(),max_native_vs_orthogonal=float(np.max(abs(native-pj))),idempotence_error=float(np.max(abs(native@native-native))),symmetry_error=float(np.max(abs(native-native.T))),rigid_residual_norm=float(np.linalg.norm(rout)),input_rigid_norm=float(np.linalg.norm(rigid)),rotation_covariance_error=float(np.max(abs(outrot-out@rot))),translation_covariance_error=float(np.max(abs(trans-out))),translation_output_norm=float(np.linalg.norm(out.mean(0))),sequential_formula_error=float(np.max(abs(native-sequential))),independent_rankone_formula_error=float(np.max(abs(native-independent))),native_eigenvalues=np.linalg.eigvalsh(native).tolist())
  assert c['sequential_formula_error'] < 1e-12
  if n > 2:
   assert c['max_native_vs_orthogonal'] < 1e-12 and c['rotation_covariance_error'] < 1e-12
  cases.append(c)
 report=dict(elf=o.elf,sha256=hashlib.sha256(blob).hexdigest(),unicorn=unicorn.__version__,entry=hex(ENTRY),seed=20260910,scope='complete setconstraints instructions from entry on valid same-N warm-cache state; NX/NY/NZ cache initialized on host, rotation geometry/normalization/projection not hooked; runtime same-shape realloc, memcpy, memset only',cases=cases)
 Path(o.output).parent.mkdir(parents=True,exist_ok=True);Path(o.output).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps([{k:c[k] for k in ('n','max_native_vs_orthogonal','idempotence_error','rigid_residual_norm','rotation_covariance_error','independent_rankone_formula_error')} for c in cases]))
if __name__=='__main__':main()
