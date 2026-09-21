"""Original get_ffix instructions: separate force and mode masks, no PES."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSP, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    blob,segments=load_elf('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp')
    sha=hashlib.sha256(blob).hexdigest()
    assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    machine=Uc(UC_ARCH_X86,UC_MODE_64)
    for page,size in [(0x5a9000,0x1000),(0x53ed000,0x30000),(0x70000000,0x20000)]:
        machine.mem_map(page,size)
        for address,_,data in segments:
            lo=max(page,address);hi=min(page+size,address+len(data))
            if hi>lo:machine.mem_write(lo,data[lo-address:hi-address])
    def q(address,value):machine.mem_write(address,struct.pack('<Q',value))
    def descriptor(address,pointer,n,stride):
        q(address,pointer);q(address+0x30,3);q(address+0x40,1)
        q(address+0x48,n);q(address+0x50,stride);q(address+0x58,1)
    obj,arg,sp,stop=0x70001000,0x70000100,0x7001f008,0x70000000
    para=0x53ed7a0;physical,mode,ffix,modefix=0x70003000,0x70005000,0x70009000,0x7000b000
    rows=[]
    for n in (4,9,514):
        force_mask=np.ones(n)
        mode_mask=np.ones((n,3))
        if n==514:
            force_mask[:297]=0;mode_mask[:351]=0
        else:
            force_mask[0]=0;mode_mask[1,0]=0;mode_mask[2,1:]=0
        machine.mem_write(obj,bytes(0x1000));machine.mem_write(physical,force_mask.astype('<f8').tobytes());machine.mem_write(mode,mode_mask.astype('<f8').tobytes())
        machine.mem_write(ffix,bytes(n*12));machine.mem_write(modefix,bytes(n*12))
        machine.mem_write(obj,struct.pack('<i',n));q(arg,obj)
        q(para+0x2df30,physical);q(para+0x2df70,1)
        descriptor(para+0x2df78,mode,n,24)
        descriptor(obj+0x8a8,ffix,n,12);descriptor(obj+0x908,modefix,n,12)
        q(sp,stop);machine.reg_write(UC_X86_REG_RDI,arg);machine.reg_write(UC_X86_REG_RSP,sp)
        machine.emu_start(0x5a9200,stop,count=200000,timeout=1000000)
        assert machine.reg_read(UC_X86_REG_RIP)==stop
        got=np.frombuffer(machine.mem_read(ffix,n*12),dtype='<i4').reshape(n,3)
        copied=np.frombuffer(machine.mem_read(modefix,n*12),dtype='<i4').reshape(n,3)
        expected=np.minimum(force_mask[:,None],mode_mask).astype(int)
        assert np.array_equal(got,expected) and np.array_equal(copied,mode_mask)
        rows.append(dict(natoms=n,physical_mobile_atoms=int(np.count_nonzero(force_mask)),
                         mode_mobile_atoms=int(np.any(got>0,axis=1).sum()),
                         mode_mobile_components=int((got>0).sum()),
                         force_mask=force_mask.tolist(),input_mode_mask=mode_mask.tolist(),
                         ffix=got.tolist(),modefix=copied.tolist(),exact=True))
    with Path(args.output).open('x') as handle:
        json.dump(dict(elf_sha256=sha,entry='0x5a9200',scope='isolated get_ffix instructions, no initialization or PES; manually formed array descriptors',rows=rows),handle,indent=2);handle.write('\n')
    print([(row['natoms'],row['physical_mobile_atoms'],row['mode_mobile_atoms'],row['exact']) for row in rows])


if __name__=='__main__':main()
