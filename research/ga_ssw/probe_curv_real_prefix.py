"""Bounded original-instruction force-difference block; not whole SSW parity."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_R15, UC_X86_REG_R14, UC_X86_REG_RBX, UC_X86_REG_RBP, UC_X86_REG_RDX, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf

def main():
    p=argparse.ArgumentParser(); p.add_argument('--elf',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    blob, segments=load_elf(a.elf)
    sha=hashlib.sha256(blob).hexdigest()
    assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segments:
        start=va&~4095; u.mem_map(start,((va+ms+4095)&~4095)-start); u.mem_write(va,chunk)
    base=0x720000000000; u.mem_map(base,0x10000)
    def q(addr,x):u.mem_write(addr,struct.pack('<Q',x))
    def d(addr,x):u.mem_write(addr,struct.pack('<d',x))
    cases=[]
    for rows in (3,12):
        for step in (0,1,4):
            for scale in (-1.,1.):
                obj=base; stack=base+0xf000; desc=base+0xe000
                n=np.arange(1,2*rows+1,dtype=float);n/=np.linalg.norm(n)
                tf=np.linspace(-.7,.5,2*rows); fa=tf-scale*.005*n
                for off,addr,array in ((0x1788,base+0x4000,n),(0x1a60,base+0x5000,tf),(0x1d0,base+0x6000,fa)):
                    q(obj+off,addr);u.mem_write(addr,array.astype('<f8').tobytes())
                for off,value in ((0x17d8,rows*8),(0x17e0,1),(0x1a90,rows),(0x1aa8,2),(0x1ab0,rows*8),(0x1ab8,1),(0x220,rows*8),(0x228,1)):
                    q(obj+off,value)
                u.mem_write(0x53ed5c0+4,struct.pack('<i',step));d(0x53ed7a0+0x2db20,.005)
                for reg,value in ((UC_X86_REG_R15,obj),(UC_X86_REG_R14,obj),(UC_X86_REG_RBX,desc),(UC_X86_REG_RBP,stack),(UC_X86_REG_RDX,1)):
                    u.reg_write(reg,value)
                u.emu_start(0x5c3af7,0x5c3e29,count=10000)
                assert u.reg_read(UC_X86_REG_RIP)==0x5c3e29
                got=struct.unpack('<d',u.mem_read(obj+0x18b0,8))[0]
                control=struct.unpack('<d',u.mem_read(0x53ed5c0+0x40,8))[0]
                expected=0. if step==0 else float(np.dot(tf-fa,n)/.005)
                assert abs(got-expected)<1e-12 and control==got
                cases.append(dict(rows=rows,columns=2,rotstep=step,tf0=tf.tolist(),fa=fa.tolist(),n=n.tolist(),dr=.005,expected=expected,object_curv_real=got,control_curv_real=control))
    report=dict(elf=a.elf,sha256=sha,entry='0x5c3af7',stop='0x5c3e29',scope='Arithmetic block only; synthetic arrays, no physical PES, no force-producer or rotation dispatch emulation',cases=cases)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(len(cases),'original-instruction cases passed')
if __name__=='__main__':main()
