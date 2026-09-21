"""Bounded native probe for find_leastmoveatoms_ (no main/PES).

The caller passes reference/current Cartesian coordinates and a 3-int/atom
mask; this probe records the pair and raw-group outputs directly.
"""
import hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE, UC_HOOK_MEM_INVALID, UC_HOOK_MEM_WRITE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
SHA='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
ENTRY=0x57e940; STOP=0x700000000100; STACK=0x710000000000; DATA=0x720000000000
RD=0x580640; MEMSET=0x4a10430; ALLOC=0x498a070; DEALLOC=0x498a650; OV=0x498a9c0

def run(n, values, *, rng=0.25, offsets=None, reference=None, current=None, pair_init=(1,1)):
    blob,segs=load_elf(ELF); assert hashlib.sha256(blob).hexdigest()==SHA
    u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,sz,ch in segs:
        lo=va&~4095; u.mem_map(lo,((va+sz+4095)&~4095)-lo)
        if ch:u.mem_write(va,ch)
    u.mem_map(STACK,0x20000); u.mem_map(DATA,0x40000); u.mem_map(STOP & ~0xfff,0x1000)
    nptr=DATA; records=DATA+0x1000; pos=DATA+0x4000; mask=DATA+0x8000
    pair=DATA+0x9000; group=DATA+0x9010; alloc=DATA+0x10000
    u.mem_write(nptr,struct.pack('<i',n)); x=np.array([[3*i,0.,0.] for i in range(n)],dtype='<f8')
    if reference is None: reference=x
    else: reference=np.asarray(reference,dtype='<f8').reshape(n,3)
    if current is None:
        if offsets is None: offsets=[0.2*(i+1) for i in range(n)]
        current=reference+np.array([[offsets[i],0.,0.] for i in range(n)])
    else: current=np.asarray(current,dtype='<f8').reshape(n,3)
    u.mem_write(records,reference.astype('<f8').tobytes())
    u.mem_write(pos,current.astype('<f8').tobytes())
    u.mem_write(mask,b'\0'*(0x18*n))
    for i,v in enumerate(values):
        u.mem_write(mask+0x18*i,struct.pack('<i',int(v)))
        u.mem_write(mask+0x18*i+0xc,struct.pack('<i',int(v)))
    u.mem_write(pair,struct.pack('<ii',*pair_init)); u.mem_write(group,struct.pack('<'+'i'*n,*range(7,7+n)))
    u.mem_write(alloc,b'\0'*0x10000); alloc_cursor=[alloc]; rsp=STACK+0x10000; u.mem_write(rsp,struct.pack('<Q',STOP))
    for reg,val in ((UC_X86_REG_RDI,nptr),(UC_X86_REG_RSI,records),(UC_X86_REG_RDX,pos),(UC_X86_REG_RCX,mask),(UC_X86_REG_R8,pair),(UC_X86_REG_R9,group),(UC_X86_REG_RSP,STACK)):
        u.reg_write(reg,val)
    u.reg_write(UC_X86_REG_RSP,rsp)
    calls=[]; writes=[]; reached=[False]; traces={}
    def wh(m,access,a,sz,val,_):
        if group <= a < group+8*n: writes.append([hex(a-group),sz,val])
    def ret(m):
        sp=m.reg_read(UC_X86_REG_RSP); target=struct.unpack('<Q',m.mem_read(sp,8))[0]
        m.reg_write(UC_X86_REG_RSP,sp+8); m.reg_write(UC_X86_REG_RIP,target)
    def hook(m,a,sz,_):
        if a==STOP: reached[0]=True; m.emu_stop();return
        if a==0x57f53c:
            calls.append('cmp:'+str(m.reg_read(UC_X86_REG_XMM1))+':'+str(m.reg_read(UC_X86_REG_XMM0)))
        if a==0x57ec8f:
            p=m.reg_read(UC_X86_REG_R15); traces['d_first']=list(struct.unpack('<'+'d'*40,m.mem_read(p,8*40)))
        if a==0x57f2f1:
            p=m.reg_read(UC_X86_REG_R15); traces['dist_ptr']=hex(p); traces['dist']=list(struct.unpack('<'+'d'*40,m.mem_read(p,8*40)))
        if a==0x57f557:
            p=m.reg_read(UC_X86_REG_RDI); idx=m.reg_read(UC_X86_REG_RSI); calls.append('candidate:'+str(idx)+':'+str(struct.unpack('<d',m.mem_read(p+idx*16,8))[0]))
        if a==0x57f528:
            p=m.reg_read(UC_X86_REG_RDI); idx=m.reg_read(UC_X86_REG_RSI); calls.append('threshold:'+str(idx)+':'+str(struct.unpack('<d',m.mem_read(p+idx*16,8))[0])+':floorbits='+str(m.reg_read(UC_X86_REG_XMM0)))
        if a==RD:
            p=m.reg_read(UC_X86_REG_RDI);m.mem_write(p,struct.pack('<d',rng));calls.append('random');ret(m)
        elif a==MEMSET:
            p=m.reg_read(UC_X86_REG_RDI); size=m.reg_read(UC_X86_REG_RDX); m.mem_write(p,b'\0'*min(size,0x100000)); ret(m)
        elif a==ALLOC:
            # alloc)
            p=m.reg_read(UC_X86_REG_RSI); size=m.reg_read(UC_X86_REG_RDI)
            ap=alloc_cursor[0]; alloc_cursor[0]+=0x4000
            m.mem_write(p,struct.pack('<Q',ap)); m.mem_write(ap,b'\0'*0x4000)
            m.reg_write(UC_X86_REG_RAX,0);calls.append('alloc');ret(m)
        elif a in (DEALLOC,OV):
            if a==OV:
                p=m.reg_read(UC_X86_REG_RDI); m.mem_write(p,struct.pack('<Q',alloc)); m.reg_write(UC_X86_REG_RAX,0)
            calls.append(hex(a));ret(m)
        elif a<0x57e940 or a>=0x580640: raise RuntimeError(f'external {a:#x}')
    def bad(m,typ,a,size,val,_):
        print('BAD',typ,hex(a),size,hex(val),'pc',hex(m.reg_read(UC_X86_REG_RIP)))
        return False
    u.hook_add(UC_HOOK_MEM_INVALID,bad); u.hook_add(UC_HOOK_MEM_WRITE,wh); u.hook_add(UC_HOOK_CODE,hook); u.emu_start(ENTRY,STOP+1,count=500000)
    if not reached[0]: raise RuntimeError(f'did not reach STOP rip={u.reg_read(UC_X86_REG_RIP):#x} rsp={u.reg_read(UC_X86_REG_RSP):#x}')
    return dict(values=list(values),pair=list(struct.unpack('<ii',u.mem_read(pair,8))),group=list(struct.unpack('<'+'i'*n,u.mem_read(group,4*n))),calls=calls,group_writes=writes,traces=traces)

def main():
    rows=[run(3,[1,1,1],reference=[[100,0,0],[10,0,0],[100,0,0]],current=[[0,0,0],[10,0,0],[20,0,0]],pair_init=(1,1)), run(3,[1,1,1],reference=[[100,0,0],[100,0,0],[20,0,0]],current=[[0,0,0],[10,0,0],[20,0,0]],pair_init=(1,1)), run(3,[1,1,1],reference=[[0,0,0],[10,0,0],[100,0,0]],current=[[0,0,0],[10,0,0],[20,0,0]],pair_init=(1,1))]
    out=dict(entry=hex(ENTRY),sha256=SHA,scope='find_leastmoveatoms only; explicit structured records and Cartesian coordinates; alloc/rd_numb/dealloc/overflow hooked; no main/PES',cases=rows)
    p=Path('research/ga_ssw/evidence/native-axis-group-selection-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
