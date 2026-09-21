"""Isolated ELF LBFGS/MCSRCH/MCSTEP reverse communication on frozen Cu13 stages.
Numerical instructions execute in Unicorn; host only supplies memcpy and EMT E/G.
Not BFGSDRIVER, native Gaussian, or full LASP execution.
"""
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from .probe_native_weight_emulated import load_elf, STOP, STACK, DATA
from pamssw.standalone import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ENTRY=0x6e87b0
SETTINGS=dict(history=400,gtol=900.,stpmin=.0001,maxstep=.5,ftol=.0001,
              eps=struct.unpack('<d',struct.pack('<Q',0x3ee4f8b588e368f1))[0],
              xtol=struct.unpack('<d',struct.pack('<Q',0x3c9cd2b297d889bc))[0],
              gradient_scale=1.,request_cap=201,step_cap=200,fmax=.01)

class Oracle:
    def __init__(self,segments,x):
        self.uc=u=Uc(UC_ARCH_X86,UC_MODE_64)
        for va,ms,chunk in segments:
            start=va&~4095;u.mem_map(start,((va+ms+4095)&~4095)-start);u.mem_write(va,chunk)
        for p in (STOP,STACK,DATA):u.mem_map(p,0x100000)
        self.cursor=DATA;self.n=np.size(x);self.accepted=[];self.calls={};self.current=None;self.stop_reason=None
        self.ptr=[self.alloc(struct.pack('<i',self.n)),self.alloc(struct.pack('<i',400)),self.arr(x),self.arr([0.]),self.arr(np.zeros(self.n)),self.alloc(struct.pack('<i',0)),self.arr(np.ones(self.n)),self.alloc(struct.pack('<ii',-1,0)),self.arr([SETTINGS['eps']]),self.arr([SETTINGS['xtol']]),self.arr(np.zeros(self.n*801+800)),self.alloc(struct.pack('<i',0)),self.arr([.5]),self.arr([.0001])]
        u.hook_add(UC_HOOK_CODE,self.hook)
    def alloc(self,data):
        p=self.cursor;self.cursor+=(len(data)+31)//32*32;self.uc.mem_write(p,data);return p
    def arr(self,a):return self.alloc(np.asarray(a,dtype='<f8').tobytes())
    def integer(self,p):return struct.unpack('<i',self.uc.mem_read(p,4))[0]
    def x(self):return np.frombuffer(self.uc.mem_read(self.ptr[2],self.n*8),dtype='<f8').copy().reshape(-1,3)
    def hook(self,u,pc,size,data):
        if pc in (ENTRY,0x6ea9d0,0x6eb690):self.calls[hex(pc)]=self.calls.get(hex(pc),0)+1
        if pc==0x4a102b0:
            dst,src,n=[u.reg_read(r) for r in (UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX)]
            u.mem_write(dst,bytes(u.mem_read(src,n)));sp=u.reg_read(UC_X86_REG_RSP)
            ret=struct.unpack('<Q',u.mem_read(sp,8))[0];u.reg_write(UC_X86_REG_RSP,sp+8);u.reg_write(UC_X86_REG_RIP,ret);return
        if not ENTRY<=pc<0x6ebce0:raise RuntimeError(f'unexpected native PC {pc:#x}')
        if pc==0x6e8dad and self.integer(0x791b938)==1:
            self.accepted.append(dict(self.current))
            if self.current['max_force']<=.01:self.stop_reason='converged'
            elif len(self.accepted)>=200:self.stop_reason='step_limit'
            if self.stop_reason:u.emu_stop()
    def advance(self,e,f,current):
        self.current=current;u=self.uc
        u.mem_write(self.ptr[3],struct.pack('<d',e));u.mem_write(self.ptr[4],np.asarray(-f,dtype='<f8').tobytes())
        for r,p in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9),self.ptr):u.reg_write(r,p)
        sp=STACK+0x80008;u.mem_write(sp,struct.pack('<Q',STOP)+b''.join(struct.pack('<Q',p) for p in self.ptr[6:]));u.reg_write(UC_X86_REG_RSP,sp)
        u.emu_start(ENTRY,STOP,timeout=10_000_000,count=10_000_000)
        if not self.stop_reason and u.reg_read(UC_X86_REG_RIP)!=STOP:raise RuntimeError('instruction/time cap')
        return self.integer(self.ptr[11])

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    blob,segments=load_elf(ELF);sha=hashlib.sha256(blob).hexdigest();assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    (out/'script.py').write_text(Path(__file__).read_text())
    (out/'plan.json').write_text(json.dumps(dict(settings=SETTINGS,elf=ELF,sha256=sha,selection='all original biased_quench_failed, sorted source then record; frozen consistent E+B gradient; no driver force scaling',limit=args.limit),indent=2))
    rows=[]
    for source in sorted(Path('research/ga_ssw/evidence/cu13-direction-only').glob('[0-9]*-*.json')):
        for record in json.loads(source.read_text())['result']['records']:
            if record['status']!='biased_quench_failed':continue
            if args.limit and len(rows)>=args.limit:break
            begin=time.monotonic();last=record['climb'][-1];a=Atoms(**record['last_atoms']);a.positions=np.array(last['center'])+last['width']*np.array(last['direction'])
            terms=[ProjectedGaussian(np.array(t['center']),np.array(t['direction']),t['width'],t['weight']) for t in record['climb']]
            surface=ASESurface(EMT());oracle=Oracle(segments,a.positions);evals=[];status='request_limit';flag=None;error=None
            try:
                for request in range(201):
                    a.positions=oracle.x();e,f=surface.evaluate(a)
                    for term in terms:be,bf=term.evaluate(a);e+=be;f+=bf
                    current=dict(request=surface.requests,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),positions=a.positions.tolist(),forces=f.tolist())
                    evals.append(current)
                    if request==0 and current['max_force']<=.01:status='converged';break
                    flag=oracle.advance(e,f,current)
                    if oracle.stop_reason:status=oracle.stop_reason;break
                    if flag!=1:status='native_converged_unqualified' if flag==0 else 'native_failure';break
            except Exception as exc:status='oracle_error';error=repr(exc)
            row=dict(source=source.name,step=record['index'],status=status,error=error,requests=surface.requests,accepted_iterates=len(oracle.accepted),last_accepted_force=oracle.accepted[-1]['max_force'] if oracle.accepted else evals[0]['max_force'],native_flag=flag,native_info=oracle.integer(0x791b938),calls=oracle.calls,seconds=time.monotonic()-begin,evaluations=evals,accepted=oracle.accepted)
            (out/f'{source.stem}-step{record["index"]}.json').write_text(json.dumps(row,indent=2)+'\n');rows.append({k:v for k,v in row.items() if k not in ('evaluations','accepted')});print(json.dumps(rows[-1]),flush=True)
            del oracle
        if args.limit and len(rows)>=args.limit:break
    summary=dict(runs=rows,attempts=len(rows),converged=sum(r['status']=='converged' for r in rows),requests=sum(r['requests'] for r in rows),statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
