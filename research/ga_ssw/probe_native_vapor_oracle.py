"""Instruction-level check_vapor_new_ oracle, no LASP/PES execution."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE, UC_HOOK_MEM_WRITE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF_DEFAULT='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ENTRY,END=0x5950f0,0x5967a0
STOP,STACK,DATA,HEAP=0x700000000000,0x710000000000,0x720000000000,0x730000000000
ALLOC,DEALLOC,MEMSET,OVERFLOW=0x498a070,0x498a650,0x4a10430,0x498a9c0

def q(uc,p): return struct.unpack('<Q',uc.mem_read(p,8))[0]
def ret(uc):
    sp=uc.reg_read(UC_X86_REG_RSP); uc.reg_write(UC_X86_REG_RIP,q(uc,sp)); uc.reg_write(UC_X86_REG_RSP,sp+8)

def run(segments, pos, mode, cri):
    u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segments:
        lo=va&~4095; size=((va+ms+4095)&~4095)-lo
        u.mem_map(lo,size)
        if chunk:u.mem_write(va,chunk)
    for p,s in ((STOP,0x10000),(STACK,0x200000),(DATA,0x200000),(HEAP,0x400000)):u.mem_map(p,s)
    heap=HEAP; calls=[]; allocs=[]; writes=[]
    def put(p,b):u.mem_write(p,b)
    n=len(pos); np_pos=np.asarray(pos,dtype='<f8'); np_pos0=np_pos.copy()
    n_p,xyz,cri_p,mode_p,out=DATA,DATA+0x100,DATA+0x1000,DATA+0x2000,DATA+0x3000
    put(n_p,struct.pack('<i',n)); put(xyz,np_pos.tobytes()); put(cri_p,struct.pack('<d',cri)); put(mode_p,struct.pack('<i',-1 if mode else 0)); put(out,struct.pack('<d',-999.))
    sp=STACK+0x1ff00; put(sp,struct.pack('<Q',STOP)); u.reg_write(UC_X86_REG_RSP,sp)
    for reg,val in ((UC_X86_REG_RDI,n_p),(UC_X86_REG_RSI,xyz),(UC_X86_REG_RDX,cri_p),(UC_X86_REG_RCX,mode_p),(UC_X86_REG_R8,out)):u.reg_write(reg,val)
    def hook(uc,address,size,user):
        nonlocal heap
        if ENTRY<=address<END:return
        calls.append(address)
        if address==ALLOC:
            # Intel Fortran allocatable descriptor: size in rdi, descriptor in rsi.
            size_ptr=uc.reg_read(UC_X86_REG_RDI); amount=size_ptr; desc=uc.reg_read(UC_X86_REG_RSI); flags=uc.reg_read(UC_X86_REG_RDX)
            before=uc.mem_read(desc,0x40).hex() if desc else ''
            if amount<0 or heap+amount+64>HEAP+0x400000: raise RuntimeError(f'alloc {amount} {desc:#x}')
            uc.mem_write(desc,struct.pack('<Q',heap)); uc.mem_write(heap,bytes(min(amount,0x400000))); allocs.append({'size':amount,'size_ptr':hex(size_ptr),'descriptor':hex(desc),'flags':flags,'ptr':hex(heap),'before':before,'after':uc.mem_read(desc,0x40).hex()}); heap += ((amount+63)//64)*64
            uc.reg_write(UC_X86_REG_RAX,0); ret(uc); return
        if address==DEALLOC:
            uc.reg_write(UC_X86_REG_RAX,0); ret(uc); return
        if address==OVERFLOW:
            # Helper receives an output size pointer in rdi; the caller fills
            # it before for_alloc_allocatable. Preserve that contract.
            outp=uc.reg_read(UC_X86_REG_RDI); a1=uc.reg_read(UC_X86_REG_RSI); a2=uc.reg_read(UC_X86_REG_RDX); a3=uc.reg_read(UC_X86_REG_RCX)
            size=max(1, a2*a3 if a1==2 else a2*a3)
            uc.mem_write(outp,struct.pack('<Q',size)); uc.reg_write(UC_X86_REG_RAX,0); ret(uc); return
        if address==MEMSET:
            a=uc.reg_read(UC_X86_REG_RDI); val=uc.reg_read(UC_X86_REG_RSI)&255; count=uc.reg_read(UC_X86_REG_RDX)
            uc.mem_write(a,bytes([val])*count); uc.reg_write(UC_X86_REG_RAX,a); ret(uc); return
        raise RuntimeError(f'unexpected external {address:#x} rip={uc.reg_read(UC_X86_REG_RIP):#x}')
    u.hook_add(UC_HOOK_CODE,hook)
    def memhook(uc, access, address, size, value, user):
        if xyz <= address < xyz + n*24:
            writes.append({'address':hex(address),'offset':address-xyz,'size':size,'value_bits':value,'value':struct.unpack('<d',struct.pack('<Q',value & ((1<<64)-1)))[0] if size==8 else None})
    u.hook_add(UC_HOOK_MEM_WRITE, memhook)
    try:u.emu_start(ENTRY,STOP,count=2000000,timeout=5000000)
    except Exception as e:
        return {'mode':mode,'n':n,'cri':cri,'status':'blocked','error':f'{e} rip={u.reg_read(UC_X86_REG_RIP):#x} rdi={u.reg_read(UC_X86_REG_RDI):#x} rsi={u.reg_read(UC_X86_REG_RSI):#x} rdx={u.reg_read(UC_X86_REG_RDX):#x}','calls':[hex(x) for x in calls],'allocs':allocs,'coordinate_writes':writes}
    got=np.frombuffer(u.mem_read(xyz,n*3*8),dtype='<f8').reshape(n,3).copy()
    return {'mode':mode,'n':n,'cri':cri,'status':'ok' if u.reg_read(UC_X86_REG_RIP)==STOP else 'not_stopped','return':struct.unpack('<d',u.mem_read(out,8))[0],'max_coord_delta':float(np.max(np.abs(got-np_pos0))), 'positions':got.tolist(),'calls':[hex(x) for x in calls],'allocs':allocs,'coordinate_writes':writes}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--elf',default=ELF_DEFAULT);ap.add_argument('--output',required=True);ap.add_argument('--c60-result');a=ap.parse_args()
    blob,segs=load_elf(a.elf); digest=hashlib.sha256(blob).hexdigest(); expected='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    if digest!=expected:raise ValueError('wrong ELF digest')
    cases=[[[0,0,0],[0.8,0,0]], [[0,0,0],[0.8,0,0],[0.8,0.8,0]], [[0,0,0],[0.8,0,0],[0.8,0.8,0],[8,0,0]]]
    if a.c60_result:
        d=json.loads(Path(a.c60_result).read_text()); landing=d['records'][0]['landing']['atoms']; cases.append(landing['positions'])
    rows=[]
    for ci,pos in enumerate(cases):
      for mode in (0,1):
        row=run(segs,pos,mode,1.7); row['case_index']=ci; row['source']='synthetic' if ci<3 else a.c60_result; rows.append(row)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'elf':a.elf,'elf_sha256':digest,'entry':hex(ENTRY),'scope':'Unicorn instruction execution with synthetic allocator/memset stubs; no LASP/PES','rows':rows},indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'statuses':[x['status'] for x in rows]},separators=(',',':')))
if __name__=='__main__':main()
