"""Bounded Unicorn probe for native selected displacement and width."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF_DEFAULT='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ELF_SHA256='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
NORMAL=0x578e20; LOOP=0x5c8044; LOOP_END=0x5c8060; STORE=0x5c80c4
DATA=0x720000000000; STACK=0x710000000000; STOP=0x700000000000

def run(segments,name,selected,center):
    u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,size,chunk in segments:
        lo=va&~4095;u.mem_map(lo,((va+size+4095)&~4095)-lo)
        if chunk:u.mem_write(va,chunk)
    u.mem_map(DATA,0x100000);u.mem_map(STACK,0x20000);u.mem_map(STOP,0x1000)
    selected=np.asarray(selected,dtype='<f8').ravel();center=np.asarray(center,dtype='<f8').ravel()
    if selected.shape!=center.shape or selected.size%3:raise ValueError('equal 3N arrays required')
    n=selected.size;delta=selected-center;put=lambda p,b:u.mem_write(p,b)
    nptr,vec,out=DATA,DATA+0x1000,DATA+0x2000;put(nptr,struct.pack('<i',n//3));put(vec,delta.tobytes());put(out,b'\0'*8)
    sp=STACK+0x10000;put(sp,struct.pack('<Q',STOP));u.reg_write(UC_X86_REG_RSP,sp)
    u.reg_write(UC_X86_REG_RDI,nptr);u.reg_write(UC_X86_REG_RSI,vec);u.reg_write(UC_X86_REG_RDX,out);trace=[]
    def guard(m,a,s,_):
        if not (NORMAL<=a<0x5790a0 or LOOP<=a<=LOOP_END or a in (STORE,0x5c80c9)):raise RuntimeError(f'PC escaped: {a:#x}')
        trace.append(hex(a))
    u.hook_add(UC_HOOK_CODE,guard);u.reg_write(UC_X86_REG_RIP,NORMAL);u.emu_start(NORMAL,STOP,count=100000)
    if u.reg_read(UC_X86_REG_RIP)!=STOP:raise AssertionError('n_normal did not stop at return')
    normalized=np.frombuffer(u.mem_read(vec,n*8),dtype='<f8').copy()
    selected_ptr,center_ptr,norm_ptr,width_ptr=DATA+0x4000,DATA+0x5000,DATA+0x6000,DATA+0x7000;put(selected_ptr,selected.tobytes());put(center_ptr,center.tobytes());put(norm_ptr,normalized.tobytes());put(width_ptr,b'\0'*8)
    rbp=STACK+0x18000;u.reg_write(UC_X86_REG_RBP,rbp);put(rbp-0x330,struct.pack('<Q',1));
    for reg,val in ((UC_X86_REG_RCX,selected_ptr),(UC_X86_REG_RSI,center_ptr),(UC_X86_REG_R8,norm_ptr),(UC_X86_REG_RDI,0),(UC_X86_REG_RDX,n),(UC_X86_REG_RAX,0)):u.reg_write(reg,val)
    u.reg_write(UC_X86_REG_XMM1,0);u.reg_write(UC_X86_REG_RIP,LOOP)
    def stop_loop(m,a,s,_):
        if not (LOOP<=a<=LOOP_END or a in (STORE,0x5c80c9)):raise RuntimeError(f'width PC escaped: {a:#x}')
        if a==LOOP_END:m.reg_write(UC_X86_REG_RIP,STOP)
    u.hook_add(UC_HOOK_CODE,stop_loop);u.emu_start(LOOP,STOP,count=100000)
    if u.reg_read(UC_X86_REG_RIP)!=STOP:raise AssertionError('width loop did not stop at boundary')
    bits=u.reg_read(UC_X86_REG_XMM1)&((1<<64)-1);acc=struct.unpack('<d',struct.pack('<Q',bits))[0]
    u.reg_write(UC_X86_REG_R8,width_ptr);u.reg_write(UC_X86_REG_XMM1,bits);u.reg_write(UC_X86_REG_RIP,STORE)
    def stop_store(m,a,s,_):
        if a==0x5c80c9:m.reg_write(UC_X86_REG_RIP,STOP)
        elif a!=STORE:raise RuntimeError(f'store PC escaped: {a:#x}')
    u.hook_add(UC_HOOK_CODE,stop_store);u.emu_start(STORE,STOP,count=10)
    if u.reg_read(UC_X86_REG_RIP)!=STOP:raise AssertionError('width store did not stop at return')
    got=struct.unpack('<d',u.mem_read(width_ptr,8))[0];expected=direction=delta/np.linalg.norm(delta);width=float(np.dot(delta,direction))
    return {'name':name,'selected':selected.reshape(-1,3).tolist(),'center':center.reshape(-1,3).tolist(),'delta_input_to_n_normal':delta.tolist(),'normalized':normalized.reshape(-1,3).tolist(),'normal_norm':float(np.linalg.norm(normalized)),'expected_normalized':expected.reshape(-1,3).tolist(),'width_accumulator':acc,'width_written':got,'expected_width':width,'trace':trace,'producer_pointer_stub':{'rcx':'selected/current displacement coordinates','rsi':'center coordinates','r8':'n_normal output direction'},'synthetic_dependencies':['all pointers/arrays','n and loop bound','XMM1'],'native_executed':['n_normal 0x578e20','width loop 0x5c8044-0x5c805b','width store 0x5c80c4']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--elf',default=ELF_DEFAULT);ap.add_argument('--output',required=True);a=ap.parse_args();blob,segs=load_elf(a.elf);digest=hashlib.sha256(blob).hexdigest()
    if digest!=ELF_SHA256:raise ValueError('unsupported ELF digest')
    base=np.array([0,0,0,1,2,0],float);zero=np.zeros(6);shift=np.tile(np.array([7,-3,4],float),2)
    rows=[run(segs,'two_atom_noncollinear',base,zero),run(segs,'three_atom_noncollinear',[0,0,0,1,2,0,2,0,3],[0,0,0,.2,.1,0,0,0,.5]),run(segs,'common_translation_invariance',base+shift,zero+shift)]
    for r in rows:
        if abs(r['normal_norm']-1)>1e-12 or abs(r['width_written']-r['expected_width'])>1e-12:raise AssertionError(r)
        if not np.allclose(r['normalized'],r['expected_normalized'],atol=1e-12,rtol=0):raise AssertionError('normalized direction mismatch')
    if abs(rows[0]['width_written']-rows[2]['width_written'])>1e-12:raise AssertionError('translation changed width')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'elf':a.elf,'elf_sha256':digest,'scope':'selected-minus-center normalization and width projection only','supersedes':'native-moveds-success/result.json generated by reversed-pointer stub','rows':rows},indent=2)+'\n');print(json.dumps({'rows':len(rows),'widths':[r['width_written'] for r in rows]}))
if __name__=='__main__':main()
