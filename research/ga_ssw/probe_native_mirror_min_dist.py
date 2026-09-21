"""Isolated Unicorn execution of native mirror_min_dist_ only.

No LASP main, protection path, or PES is run.  The probe supplies the six
pointer arguments recovered from pot_bond_add_'s call site and compares the
returned distance/vector with explicit image candidates and ASE find_mic.
"""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9,UC_X86_REG_RSP,UC_X86_REG_RIP

ENTRY=0x6c5930; STOP=0x730000000000; STACK=0x740000000000; DATA=0x750000000000
ELF_SHA256='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'

def load_elf(path):
    blob=Path(path).read_bytes(); assert blob[:6]==b'\x7fELF\x02\x01'
    phoff=struct.unpack_from('<Q',blob,32)[0]; entsize,count=struct.unpack_from('<HH',blob,54)
    seg=[]
    for i in range(count):
        typ,flags,off,va,pa,fs,ms,align=struct.unpack_from('<IIQQQQQQ',blob,phoff+i*entsize)
        if typ==1:seg.append((va,ms,blob[off:off+fs]))
    return blob,seg

def run(segments, cart1, cart2, xfrac1, xfrac2, cell, *, use_reclat):
    uc=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,data in segments:
        lo=va&~4095; size=(va+ms+4095&~4095)-lo
        uc.mem_map(lo,size); uc.mem_write(va,data)
    for base in (STOP,STACK,DATA): uc.mem_map(base,0x20000)
    blocks=[np.asarray(x,dtype='<f8').reshape(-1).tobytes() for x in (cart1,cart2,xfrac1,xfrac2)]
    # A: helper receives the real cell as three C-order vectors. B: caller
    # supplies a reciprocal representation to reclat_loc_, tested separately.
    if use_reclat:
        reciprocal = 2.0*np.pi*np.linalg.inv(np.asarray(cell,dtype=float)).T
        blocks.append(np.asarray(reciprocal,dtype='<f8').reshape(3,3,order='C').tobytes(order='C'))
    else:
        blocks.append(np.asarray(cell,dtype='<f8').reshape(3,3,order='C').tobytes(order='C'))
    ptr=[];cur=DATA
    for b in blocks:
        ptr.append(cur); uc.mem_write(cur,b); cur += (len(b)+63)//64*64
    latout=cur;uc.mem_write(latout,b'\0'*72);cur+=128
    out=cur;uc.mem_write(out,b'\0'*24);cur+=64
    scalar=cur;uc.mem_write(scalar,b'\0'*8);cur+=64
    pbc=cur;uc.mem_write(pbc,struct.pack('<i',1))
    # Produce the same local lattice representation as pot_bond_add_ before the helper.
    sp=STACK+0x1000
    if use_reclat:
        uc.mem_write(sp,struct.pack('<Q',STOP)); uc.reg_write(UC_X86_REG_RSP,sp)
        uc.reg_write(UC_X86_REG_RDI,ptr[4]); uc.reg_write(UC_X86_REG_RSI,latout); uc.reg_write(UC_X86_REG_RDX,pbc)
        uc.emu_start(0x503e10,STOP,timeout=2_000_000,count=100_000)
        lattice=np.frombuffer(uc.mem_read(latout,72),dtype='<f8').copy()
    else:
        lattice=np.asarray(cell,dtype='<f8').reshape(-1).copy()
        uc.mem_write(latout,lattice.tobytes())
    uc.mem_write(sp,struct.pack('<QQ',STOP,out)); uc.reg_write(UC_X86_REG_RSP,sp)
    for reg,p in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9),ptr[:4]+[latout,scalar]): uc.reg_write(reg,p)
    uc.emu_start(ENTRY,STOP,timeout=2_000_000,count=100_000)
    if uc.reg_read(UC_X86_REG_RIP)!=STOP: raise RuntimeError(f'helper did not return: {uc.reg_read(UC_X86_REG_RIP):#x}')
    vec=np.frombuffer(uc.mem_read(out,24),dtype='<f8').copy(); dist=struct.unpack('<d',uc.mem_read(scalar,8))[0]
    return vec,dist,lattice

def expected(cart1,cart2,cell):
    c=np.asarray(cell,float); d=np.asarray(cart1,dtype=float).reshape(3)-np.asarray(cart2,dtype=float).reshape(3)
    opts=[]
    for i in (-1,0,1):
      for j in (-1,0,1):
       for k in (-1,0,1):
        v=d+np.array([i,j,k])@c; opts.append((float(v@v),v,(i,j,k)))
    return min(opts,key=lambda x:x[0])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--elf',default='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp');ap.add_argument('--output',required=True);o=ap.parse_args()
    blob,segments=load_elf(o.elf)
    if hashlib.sha256(blob).hexdigest()!=ELF_SHA256: raise ValueError('unexpected ELF')
    cases=[]
    def add(label,c1,c2,cell, use_reclat=False):
      cell=np.asarray(cell,float); inv=np.linalg.inv(cell.T)
      x1=np.asarray(c1,dtype=float).reshape(3)@np.linalg.inv(cell); x2=np.asarray(c2,dtype=float).reshape(3)@np.linalg.inv(cell)
      vec,dist,lattice=run(segments,c1,c2,x1,x2,cell,use_reclat=use_reclat); ex=expected(c1,c2,cell)
      if not use_reclat:
          assert np.array_equal(lattice, cell.reshape(-1, order='C'))
      micv,micd=find_mic(np.asarray(c1,dtype=float).reshape(3)-np.asarray(c2,dtype=float).reshape(3),cell,pbc=[True,True,True])
      cases.append(dict(label=label,native_matches_27=bool(np.allclose(vec,ex[1],atol=1e-12,rtol=0) and abs(dist-np.sqrt(ex[0]))<1e-12),cart1=np.asarray(c1).tolist(),cart2=np.asarray(c2).tolist(),cell=cell.tolist(),native_vector=vec.tolist(),native_distance=dist,reclat_output=lattice.tolist(),three_by_three_vector=ex[1].tolist(),three_by_three_distance=float(np.sqrt(ex[0])),three_by_three_shift=ex[2],ase_mic_vector=np.asarray(micv).tolist(),ase_mic_distance=float(micd)))
    orth=np.diag([4.,4.,4.])
    add('A_orthogonal_below_half',[[0,0,0]],[[1.9,0,0]],orth)
    add('B_reclat_orthogonal_below_half',[[0,0,0]],[[1.9,0,0]],orth,use_reclat=True)
    add('A_orthogonal_above_half',[[0,0,0]],[[2.1,0,0]],orth)
    add('A_orthogonal_tie',[[0,0,0]],[[2.,0,0]],orth)
    skew=np.array([[5.091168824543143,0,0],[2.5455844122715714,4.409081537009721,0],[0,0,20.156921938165304]])
    add('A_skew_half_boundary',[[3.8262917,2.20911054,12.1196]],[[0,0,12.1196]],skew)
    add('A_skew_over_one_cell',[[9.0,7.0,12.]],[[0,0,12.]],skew)
    # B must reconstruct the same C-order real cell as A before any combined claim.
    a_lat=np.asarray(cases[0]['reclat_output']); b_lat=np.asarray(cases[1]['reclat_output'])
    reclat_matches_direct=bool(np.allclose(b_lat,a_lat,atol=1e-12,rtol=0))
    a_cases=[c for c in cases if c['label'].startswith('A_')]
    assert all(c['native_matches_27'] for c in a_cases)
    assert reclat_matches_direct
    report=dict(reclat_matches_direct=reclat_matches_direct,all_A_match_27=all(c['native_matches_27'] for c in a_cases),elf=str(Path(o.elf).resolve()),sha256=hashlib.sha256(blob).hexdigest(),entry=hex(ENTRY),cases=cases,evidence='isolated mirror_min_dist_ instructions only; no main/PES/protection')
    Path(o.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
