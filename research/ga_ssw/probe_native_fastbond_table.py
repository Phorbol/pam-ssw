"""Recover full initialized native fastbond table and verify ordered lookups."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from unicorn.x86_const import *
from research.ga_ssw.probe_addgaussian_emulated import Oracle,DATA,STACK,STOP
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT,ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class BondOracle(Oracle):
    def hook(self,uc,address,size,user):
        if address==STOP:uc.emu_stop()
        elif address==0x4a10430:
            pointer=uc.reg_read(UC_X86_REG_RDI);size=uc.reg_read(UC_X86_REG_RDX)
            uc.mem_write(pointer,bytes([uc.reg_read(UC_X86_REG_RSI)&255])*size)
            self.ret()
        elif not 0x58f2a0<=address<=0x58f671:
            raise RuntimeError(hex(address))

    def bond(self,a,b):
        self.uc.mem_write(DATA,struct.pack('<ii',a,b))
        self.q(STACK+0x80008,STOP)
        for reg,value in ((UC_X86_REG_RDI,DATA),(UC_X86_REG_RSI,DATA+4),(UC_X86_REG_RDX,DATA+8),(UC_X86_REG_RSP,STACK+0x80008)):
            self.uc.reg_write(reg,value)
        self.uc.emu_start(0x58f2a0,STOP,count=10000)
        assert self.uc.reg_read(UC_X86_REG_RIP)==STOP
        return self.readd(DATA+8)


def main():
    blob,segments=load_elf(ELF_DEFAULT);assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    oracle=BondOracle(segments);oracle.bond(1,1)
    table=np.frombuffer(oracle.uc.mem_read(0x78e2240,53*53*8),dtype='<f8').reshape(53,53)
    entries={}
    for a in range(1,54):
        for b in range(a,54):
            value=float(table[b-1,a-1])
            assert oracle.bond(a,b)==oracle.bond(b,a)==value
            if value:entries[f'{a},{b}']=value
    for a,b in ((1,54),(53,79),(29,118),(118,118)):
        assert oracle.bond(a,b)==oracle.bond(b,a)==0
    result=dict(scope=__doc__,sha256=ELF_SHA256,entries=entries,lookup_checks=2862+8,
                outside_table='Z>53 returns zero; group caller uses species-radius fallback')
    Path('research/ga_ssw/evidence/native-fastbond-table-20260917.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(nonzero_entries=len(entries),lookup_checks=result['lookup_checks'])))


if __name__=='__main__':main()
