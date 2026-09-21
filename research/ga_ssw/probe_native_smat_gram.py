"""Isolated set_smat prefix; stop before matsqrt/LAPACK, no native main/PES."""
import json,struct,hashlib
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64,UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RSP,UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf,DATA,STACK
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ENTRY=0x5a3b90;STOP=0x5a3d3f

if __name__=='__main__':
    blob,segments=load_elf(ELF);rows=[]
    base=np.array([[4.,.4,.2],[.3,5.,-.1],[.2,.5,6.]])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for name,cell in [('skew',base),('rotated',rotation@base),('anisotropic',np.diag([2.,5.,9.]))]:
        u=Uc(UC_ARCH_X86,UC_MODE_64)
        for va,ms,data in segments:
            start=va&~4095;u.mem_map(start,((va+ms+4095)&~4095)-start);u.mem_write(va,data)
        u.mem_map(DATA,0x100000);u.mem_map(STACK,0x100000)
        obj=DATA+0x100;u.mem_write(DATA,struct.pack('<Q',obj))
        u.mem_write(obj+0xe0,cell.astype('<f8').tobytes(order='F'))
        u.reg_write(UC_X86_REG_RDI,DATA);u.reg_write(UC_X86_REG_RSP,STACK+0x80008)
        def hook(engine,pc,size,data):
            if not ENTRY<=pc<STOP:raise RuntimeError(f'outside isolated prefix {pc:x}')
        u.hook_add(UC_HOOK_CODE,hook)
        u.emu_start(ENTRY,STOP,timeout=1000000,count=10000)
        assert u.reg_read(UC_X86_REG_RIP)==STOP
        dim=struct.unpack('<i',u.mem_read(u.reg_read(UC_X86_REG_RDI),4))[0]
        gram=np.frombuffer(u.mem_read(u.reg_read(UC_X86_REG_RSI),72),dtype='<f8').reshape(3,3,order='F').copy()
        error=float(np.max(np.abs(gram-cell.T@cell)))
        assert dim==3 and error<1e-12
        assert u.reg_read(UC_X86_REG_RDX)==obj+0x788
        rows.append(dict(name=name,cell=cell.tolist(),gram=gram.tolist(),error=error,dimension=dim))
    out=dict(elf_sha256=hashlib.sha256(blob).hexdigest(),scope='original set_smat prefix only; no DPOTRF or reci_latt execution',new_PES=0,rows=rows)
    p=Path('research/ga_ssw/evidence/native-stress-producer-review/smat-gram-prefix.json');p.write_text(json.dumps(out,indent=2)+'\n')
    print([(r['name'],r['error']) for r in rows])
