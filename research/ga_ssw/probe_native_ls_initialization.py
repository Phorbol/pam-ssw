"""Bounded native default LS matrix-construction oracle, before later scheduling.

Executes original bond_info_init_ through the first 0x6c7530 boundary and its
real geometric bond counter/lookup routines. File inquiry is false; allocation,
copy and logging are emulated. No PES/main program/custom-file parser runs.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from ase.collections import g2
from ase.io import read
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

PARA=0x53ed7a0
STOP=0x700000000000
STACK=0x710000000000
DATA=0x720000000000
HEAP=0x730000000000
BOUNDARY=0x6c7530

class Oracle:
    def __init__(self, segments):
        self.uc=u=Uc(UC_ARCH_X86,UC_MODE_64)
        for va,ms,chunk in segments:
            start=va&~4095;u.mem_map(start,((va+ms+4095)&~4095)-start);u.mem_write(va,chunk)
        for addr,size in [(STOP,4096),(STACK,0x100000),(DATA,0x100000),(HEAP,0x100000)]:u.mem_map(addr,size)
        self.heap=HEAP;self.calls={};self.reached=False
        u.hook_add(UC_HOOK_CODE,self.hook)
    def integer(self,addr):return struct.unpack('<i',self.uc.mem_read(addr,4))[0]
    def qword(self,addr):return struct.unpack('<Q',self.uc.mem_read(addr,8))[0]
    def double(self,addr):return struct.unpack('<d',self.uc.mem_read(addr,8))[0]
    def puti(self,addr,value):self.uc.mem_write(addr,struct.pack('<i',value))
    def putq(self,addr,value):self.uc.mem_write(addr,struct.pack('<Q',value))
    def putd(self,addr,value):self.uc.mem_write(addr,struct.pack('<d',value))
    def ret(self):
        s=self.uc.reg_read(UC_X86_REG_RSP);dest=self.qword(s)
        self.uc.reg_write(UC_X86_REG_RSP,s+8);self.uc.reg_write(UC_X86_REG_RIP,dest)
    def hook(self,u,address,size,user):
        if address==BOUNDARY:self.reached=True;u.emu_stop();return
        if 0x6c5000<=address<0x6ce000 or 0x503e10<=address<0x504000 or 0x498a9c0<=address<0x498ac00:return
        a=u.reg_read(UC_X86_REG_RDI);b=u.reg_read(UC_X86_REG_RSI);c=u.reg_read(UC_X86_REG_RDX)
        self.calls[hex(address)]=self.calls.get(hex(address),0)+1
        if address==0x495aa30:
            args=u.reg_read(UC_X86_REG_R8);self.puti(self.qword(args+16),0)
        elif address==0x498a070:
            if self.heap+a>HEAP+0x100000:raise RuntimeError('allocation budget')
            self.putq(b,self.heap);self.heap+=(a+63)//64*64
        elif address==0x498a650:pass
        elif address==0x4a10430:u.mem_write(a,bytes([b&255])*c)
        elif address==0x4a102b0:u.mem_write(a,bytes(u.mem_read(b,c)))
        elif address in (0x499e470,0x49a01a0,0x4998d70,0x499abc0):pass
        else:raise RuntimeError(f'unexpected runtime {address:#x}, {a:#x} {b:#x} {c:#x}')
        u.reg_write(UC_X86_REG_RAX,0);self.ret()
    def matrix(self,desc):
        ptr=self.qword(desc);shape=[self.qword(desc+0x30+24*i) for i in range(2)]
        return np.frombuffer(self.uc.mem_read(ptr,int(np.prod(shape))*8),dtype='<f8').reshape(shape,order='F').copy()
    def run(self,atoms,scale=None):
        u=self.uc
        self.puti(PARA+0x178,1)  # Explicitly enable LS, rather than assume startup flag.
        if scale is not None:self.putd(PARA+0x148,scale)
        n=len(atoms);self.puti(DATA,n);u.mem_write(DATA+0x100,np.diag([50.,50.,50.]).astype('<f8').tobytes())
        u.mem_write(DATA+0x200,atoms.positions.astype('<f8').tobytes());u.mem_write(DATA+0x10000,atoms.numbers.astype('<i4').tobytes());self.puti(DATA+0x20000,0)
        # Physical caller has initialized this allocatable movable-atom mask.
        self.putq(PARA+0x2df30,DATA+0x30000);self.putq(PARA+0x2df70,1)
        u.mem_write(DATA+0x30000,np.ones(n,dtype='<f8').tobytes())
        for reg,ptr in zip([UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8],[DATA,DATA+0x100,DATA+0x200,DATA+0x10000,DATA+0x20000]):u.reg_write(reg,ptr)
        sp=STACK+0x80008;self.putq(sp,STOP);u.reg_write(UC_X86_REG_RSP,sp)
        u.emu_start(0x6c6a30,STOP,count=20000000,timeout=20000000)
        if not self.reached:raise RuntimeError(f'boundary not reached: {u.reg_read(UC_X86_REG_RIP):#x}')
        nt=self.integer(0x1d1032e0)
        elements=np.frombuffer(u.mem_read(0x7915f80,nt*4),dtype='<i4').tolist()
        return dict(elements=elements,bond_count=self.integer(0x7916158),bond_ener_scale=self.double(PARA+0x148),
            amp_c_static_at_boundary=self.double(0x5520450),
            energy_filter=[[self.double(PARA+0x188+8*((a-1)+108*(b-1))) for b in elements] for a in elements],
            length_filter=[[self.double(PARA+0x16e50+8*((a-1)+108*(b-1))) for b in elements] for a in elements],
            energy_matrix=self.matrix(0x55206a0).tolist(),length_matrix=self.matrix(0x55205e0).tolist(),
            len_toller=self.double(0x5520568),runtime_stubs=self.calls)


def main():
    p=argparse.ArgumentParser();p.add_argument('--elf',required=True);p.add_argument('--c60',required=True);args=p.parse_args()
    blob,segments=load_elf(args.elf);digest=hashlib.sha256(blob).hexdigest()
    if digest!='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':raise ValueError('wrong ELF')
    raw=json.loads((Path(__file__).resolve().parent/'evidence/native-ls-pair-table/result.json').read_text())
    if raw['elf_sha256']!=digest:raise ValueError('lookup provenance mismatch')
    energies={tuple(r['pair']):r['raw_return'] for r in raw['rows'] if r['function']=='bondeneval_'}
    lengths={tuple(r['pair']):r['raw_return'] for r in raw['rows'] if r['function']=='bondlenval_'}
    rows=[]
    for name,atoms in [('C60_existing',read(args.c60)),('C4H6_ASE_g2_trans_butadiene',g2['butadiene'])]:
        for scale in (None,2.5):
            result=Oracle(segments).run(atoms,scale)
            factor=float(np.float32(len(atoms))*np.float32(.02))/result['bond_count']
            expected=[[energies[(a,b)]*result['energy_filter'][i][j]*result['bond_ener_scale']*factor/energies[(6,6)]
                       for j,b in enumerate(result['elements'])] for i,a in enumerate(result['elements'])]
            expected_count=sum(np.linalg.norm(atoms.positions[j]-atoms.positions[i])<lengths[(int(atoms.numbers[i]),int(atoms.numbers[j]))]+.1
                               for i in range(len(atoms)) for j in range(i+1,len(atoms)))
            difference=float(np.max(np.abs(np.array(expected)-result['energy_matrix'])))
            if difference>1e-15 or result['bond_count']!=expected_count:raise AssertionError('formula/count mismatch')
            rows.append(dict(system=name,n_atoms=len(atoms),numbers=atoms.numbers.tolist(),positions=atoms.positions.tolist(),
                explicit_scale_override=scale,formula_expected=expected,max_formula_error=difference,
                independent_distance_count=int(expected_count),**result))
    print(json.dumps(dict(elf_sha256=digest,stop_boundary=hex(BOUNDARY),rows=rows,
        limit='matrix-construction prefix; no post-boundary scheduling, adaptive update or real PES'),indent=2))

if __name__=='__main__':main()
