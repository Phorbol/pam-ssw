"""Original set_initial_gaussw instructions, isolated normal-return scalar oracle."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64,UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_RSP,UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf
ENTRY=0x6e8700;END=0x6e879a;PARA=0x53ed7a0;CONTROL=0x53ed5c0
STOP=0x700000000000;STACK=0x710000000000;DATA=0x720000000000

def main():
 p=argparse.ArgumentParser();p.add_argument('--elf',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 blob,segments=load_elf(args.elf);sha=hashlib.sha256(blob).hexdigest()
 if sha!='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':raise ValueError('inspected ELF required')
 uc=Uc(UC_ARCH_X86,UC_MODE_64)
 for va,ms,chunk in segments:
  start=va&~4095;uc.mem_map(start,((va+ms+4095)&~4095)-start);uc.mem_write(va,chunk)
 for address in (STOP,STACK,DATA):uc.mem_map(address,0x10000)
 trace=[]
 def hook(machine,address,size,user):
  if not ENTRY<=address<END:raise RuntimeError(f'unexpected executed address {address:#x}')
  trace.append(hex(address))
 uc.hook_add(UC_HOOK_CODE,hook)
 def write(addr,value):uc.mem_write(addr,struct.pack('<d',value))
 settings=dict(maxw=10.,w_initial=.6,w_neg=.07,step=1.2,scalefact=1.5)
 for offset,key in ((0x2dce0,'maxw'),(0x2dce8,'w_initial'),(0x2dcf0,'step'),(0x2dcf8,'scalefact'),(0x2dd00,'w_neg')):write(PARA+offset,settings[key])
 cases=[]
 for ng in (1,2,4):
  for level in (0,1,2):
   for curv in (-.2,0.,.2):
    before=np.arange(8,dtype=float)/10+7.1;uc.mem_write(DATA,struct.pack('<i',ng));uc.mem_write(DATA+0x100,before.astype('<f8').tobytes())
    uc.mem_write(PARA+0x2dcd8,struct.pack('<i',level));write(CONTROL+0x40,curv)
    ptrs=(DATA,DATA+0x100,DATA+0x200,DATA+0x208,DATA+0x210)
    for reg,ptr in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8),ptrs):uc.reg_write(reg,ptr)
    sp=STACK+0x8008;uc.mem_write(sp,struct.pack('<Q',STOP));uc.reg_write(UC_X86_REG_RSP,sp);trace.clear()
    uc.emu_start(ENTRY,STOP,timeout=1_000_000,count=1000)
    if uc.reg_read(UC_X86_REG_RIP)!=STOP:raise RuntimeError('instruction budget/nonreturn')
    after=np.frombuffer(uc.mem_read(DATA+0x100,64),dtype='<f8').copy();expected=before.copy();expected[ng-1]=settings['w_neg'] if curv<0 else settings['w_initial']
    if level==1:expected[0]=5.6
    elif level==2:
     expected[0]=.5
     if ng>1:expected[1]=.5
    outputs=[struct.unpack('<d',uc.mem_read(p,8))[0] for p in ptrs[2:]]
    assert np.array_equal(after,expected) and outputs==[settings['scalefact'],settings['maxw'],settings['step']]
    cases.append(dict(ng=ng,level=level,curv_real=curv,before=before.tolist(),after=after.tolist(),expected=expected.tolist(),scale_maxw_step=outputs,instructions=list(trace)))
 report=dict(elf=args.elf,sha256=sha,entry=hex(ENTRY),settings=settings,note='controlled explicit inputs, not parser defaults; no external hooks or main-program execution; finite valid indices only',cases=cases,max_absolute_error=0.)
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'original-instruction cases exact, no external hooks')
if __name__=='__main__':main()
