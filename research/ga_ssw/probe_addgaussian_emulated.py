"""Whole addgaussian instruction oracle; numerical/runtime boundaries explicit."""
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np
import unicorn
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
ENTRY=0x5cda70
STOP=0x700000000000
STACK=0x710000000000
DATA=0x720000000000
class Oracle:
 def __init__(self,segments):
  self.uc=Uc(UC_ARCH_X86,UC_MODE_64)
  for va,ms,chunk in segments:
   start=va&~4095;self.uc.mem_map(start,((va+ms+4095)&~4095)-start);self.uc.mem_write(va,chunk)
  for address in (STOP,STACK,DATA):self.uc.mem_map(address,0x100000)
  self.uc.hook_add(UC_HOOK_CODE,self.hook)
 def q(self,a,v):self.uc.mem_write(a,struct.pack('<Q',v))
 def d(self,a,v):self.uc.mem_write(a,struct.pack('<d',v))
 def readq(self,a):return struct.unpack('<Q',self.uc.mem_read(a,8))[0]
 def readd(self,a):return struct.unpack('<d',self.uc.mem_read(a,8))[0]
 def alloc(self,blob):
  p=self.cursor;self.cursor+=(len(blob)+31)//32*32;self.uc.mem_write(p,blob);return p
 def arr(self,x):return self.alloc(np.asarray(x,dtype='<f8').tobytes(order=self.order))
 def descriptor(self,addr,ptr,shape,itemsize=8):
  self.q(addr,ptr);self.q(addr+8,itemsize);self.q(addr+0x10,0);self.q(addr+0x18,1);self.q(addr+0x20,len(shape))
  stride=itemsize
  for i,n in enumerate(shape):
   self.q(addr+0x30+24*i,n);self.q(addr+0x38+24*i,stride);self.q(addr+0x40+24*i,1);stride*=n
 def ret(self):
  sp=self.uc.reg_read(UC_X86_REG_RSP);self.uc.reg_write(UC_X86_REG_RIP,self.readq(sp));self.uc.reg_write(UC_X86_REG_RSP,sp+8)
 def hook(self,uc,a,size,user):
  if a in (0x49207c0,0x49206e0):
   x=struct.unpack('<d',struct.pack('<Q',uc.reg_read(UC_X86_REG_XMM0)&((1<<64)-1)))[0]
   y=(math.exp if a==0x49207c0 else math.acos)(x)
   self.hooks.append({'name':'exp' if a==0x49207c0 else 'acos','input':x,'output':y})
   uc.reg_write(UC_X86_REG_XMM0,struct.unpack('<Q',struct.pack('<d',y))[0]);self.ret()
  elif a==0x4970360:
   lhs=uc.reg_read(UC_X86_REG_RDI);rhs=uc.reg_read(UC_X86_REG_RSI)
   assert self.readq(lhs)!=0 and self.readq(lhs+0x20)==self.readq(rhs+0x20)==2
   assert all(self.readq(lhs+o)==self.readq(rhs+o) for o in (8,0x30,0x48))
   self.hooks.append({'name':'for_realloc_lhs','same_shape_noop':True,'lhs':hex(lhs)});self.ret()
  elif a==0x4a10430:
   p=uc.reg_read(UC_X86_REG_RDI);v=uc.reg_read(UC_X86_REG_RSI)&255;n=uc.reg_read(UC_X86_REG_RDX)
   uc.mem_write(p,bytes([v])*n);uc.reg_write(UC_X86_REG_RAX,p);self.hooks.append({'name':'memset','size':n});self.ret()
  elif a in (0x5ce345,0x5ce660,0x5ceb93):
   self.trace.append({'pc':hex(a),'scratch':self.readarr(self.scratch).tolist()})
  elif not (ENTRY<=a<0x5cf6f0 or 0x6e1730<=a<0x6e3000):raise RuntimeError(f'unexpected PC {a:#x}')
 def readarr(self,p):return np.frombuffer(self.uc.mem_read(p,self.n*3*8),dtype='<f8').reshape(self.n,3,order=self.order)
 def run(self,positions,centers,directions,widths,weights,new=False,base_force=None,layout="atom_major"):
  self.order="C" if layout=="atom_major" else "F";shape=(3,len(positions)) if self.order=="C" else (len(positions),3)
  self.uc.mem_write(DATA,bytes(0x100000));self.uc.mem_write(STACK,bytes(0x100000));self.cursor=DATA+0x10000;self.hooks=[];self.trace=[]
  self.n=len(positions);ng=len(widths);obj=DATA+0x100;wrapper=DATA;flag=DATA+8
  self.q(wrapper,obj);self.q(flag,int(new));self.uc.mem_write(obj,struct.pack('<i',self.n))
  self.d(obj+0x230,-3.);self.uc.mem_write(obj+0x1660,struct.pack('<i',ng))
  p=self.arr(positions);f=self.arr(np.ones_like(positions)*.03 if base_force is None else base_force);self.scratch=self.arr(np.zeros_like(positions))
  for off,ptr in ((0x170,p),(0x1d0,f),(0x1a60,self.scratch)):self.descriptor(obj+off,ptr,shape)
  tr=self.alloc(bytes(ng*0x690));dr=self.alloc(bytes(ng*0x138))
  self.descriptor(obj+0x1668,tr,(ng,),0x690);self.descriptor(obj+0x16b0,dr,(ng,),0x138)
  for j in range(ng):
   self.descriptor(tr+j*0x690+0x170,self.arr(centers[j]),shape);self.descriptor(dr+j*0x138,self.arr(directions[j]),shape)
  wptr=self.arr(weights)
  self.descriptor(obj+0x16f8,self.arr(widths),(ng,));self.descriptor(obj+0x1740,wptr,(ng,))
  for off,v in ((0x2dce0,10.),(0x2dcf0,1.2),(0x2dcf8,1.5)):self.d(0x53ed7a0+off,v)
  self.d(0x54c0430,math.pi)
  sp=STACK+0x80008;self.q(sp,STOP);self.uc.reg_write(UC_X86_REG_RSP,sp);self.uc.reg_write(UC_X86_REG_RDI,wrapper);self.uc.reg_write(UC_X86_REG_RSI,flag)
  self.uc.emu_start(ENTRY,STOP,timeout=5_000_000,count=1_000_000)
  assert self.uc.reg_read(UC_X86_REG_RIP)==STOP
  return dict(energy=self.readd(obj+0x230),force=self.readarr(f).tolist(),scratch=self.readarr(self.scratch).tolist(),weights=np.frombuffer(self.uc.mem_read(wptr,ng*8),dtype='<f8').tolist(),angle=self.readd(0x53ed5c0+0x48),gauss=self.readd(0x53ed5c0+0x50),hooks=self.hooks,trace=self.trace)
def main():
 p=argparse.ArgumentParser();p.add_argument('--elf',required=True);p.add_argument('--output',required=True);o=p.parse_args()
 blob,segments=load_elf(o.elf);assert hashlib.sha256(blob).hexdigest()=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
 oracle=Oracle(segments);rng=np.random.default_rng(20260910);cases=[]
 for natoms in (1,2,5,15):
  for ng in (1,2,3):
   x=rng.normal(size=(natoms,3));centers=rng.normal(size=(ng,natoms,3))*.2;directions=rng.normal(size=centers.shape);directions/=np.linalg.norm(directions,axis=(1,2))[:,None,None];widths=np.linspace(.7,1.3,ng);weights=np.linspace(.2,.6,ng)
   if natoms==1 and ng==3:centers[0]=x  # exact s_old=0 control
   for layout,new in (("atom_major",False),("atom_major",True),("component_major",False),("component_major",True)):
    args=dict(positions=x.tolist(),centers=centers.tolist(),directions=directions.tolist(),widths=widths.tolist(),weights=weights.tolist(),new=new,layout=layout,base_force=np.full_like(x,.03).tolist())
    out=oracle.run(**args);w=np.array(out['weights']);s=np.sum((x-centers)*directions,axis=(1,2));b=w*np.exp(-s*s/(2*widths**2));f=.03+np.sum((b*s/widths**2)[:,None,None]*directions,axis=0)
    grad=np.zeros_like(x);delta=1e-5;fd_evaluations=[]
    # Freeze updated weights and branch for finite differences.
    for i in range(x.size):
     xp=x.copy();xm=x.copy();xp.flat[i]+=delta;xm.flat[i]-=delta
     e=[]
     for sign,pos in ((1,xp),(-1,xm)):
      fd_output=oracle.run(**dict(args,positions=pos.tolist(),weights=w.tolist(),new=False));e.append(fd_output['energy'])
      fd_evaluations.append(dict(coordinate=i,sign=sign,delta=delta,energy=fd_output['energy'],force=fd_output['force'],weights=fd_output['weights']))
     grad.flat[i]=.03-(e[0]-e[1])/(2*delta)
    old=np.sum((b*s/widths**2)[:-1,None,None]*directions[:-1],axis=0)
    doubled_error=float(np.max(np.abs(np.array(out['force'])-(f+old))))
    cases.append(dict(fd_evaluations=fd_evaluations,double_old_force_error=doubled_error,input=args,output=out,analytic_energy=-3+float(sum(b)),analytic_force=f.tolist(),finite_difference_force=grad.tolist(),energy_error=abs(out['energy']+3-sum(b)),force_error=float(np.max(np.abs(np.array(out['force'])-f))),fd_error=float(np.max(np.abs(np.array(out['force'])-grad)))))
 report=dict(elf=o.elf,sha256=hashlib.sha256(blob).hexdigest(),entry=hex(ENTRY),unicorn=unicorn.__version__,seed=20260910,scope='whole addgaussian and nested set_thisgaussw ELF instructions; host exp/acos, same-shape realloc no-op, memset hooks; no force or Gaussian arithmetic hooked',cases=cases)
 assert all(c['energy_error']<1e-12 and c['double_old_force_error']<1e-12 for c in cases)
 Path(o.output).parent.mkdir(parents=True,exist_ok=True);Path(o.output).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(cases=len(cases),max_energy_error=max(c['energy_error'] for c in cases),max_double_old_force_error=max(c['double_old_force_error'] for c in cases),max_fd_error=max(c['fd_error'] for c in cases))))
if __name__=='__main__':main()
