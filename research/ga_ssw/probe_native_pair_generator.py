"""Actual c4 gate/pair producer and mixing; canonical free clusters, no PES.

Supersedes an unsuccessful draft that stubbed forbidden/neighbor functions;
that draft produced no validated output and is not parity evidence.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from ase.cluster import Icosahedron
from research.ga_ssw.probe_native_random_group_generator import RandomGroupOracle
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT,ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_addgaussian_emulated import DATA
from unicorn.x86_const import *
from pamssw.standalone.native_local_pair import native_local_pair
from pamssw.standalone.native_random import native_vmb2
from pamssw.standalone.native_pair_selection import native_pair_allowed
from pamssw.standalone.cluster_frame import ClusterFrame


class C4Oracle(RandomGroupOracle):
    def hook(self,uc,address,size,user):
        if address==0x5d5c50:
            super().hook(uc,address,size,user)
            obj=self.readq(DATA)
            uc.mem_write(obj+0x2204,struct.pack('<i',getattr(self,'group_marker',0)))
            self.q(obj+8,self.alloc(np.asarray(self.atoms.numbers,dtype='<i4').tobytes()))
            uc.mem_write(0x53ed5c0+0x144,struct.pack('<i',1))
        elif address==0x498a070:
            size=uc.reg_read(UC_X86_REG_RDI)
            assert size<=0x10000
            self.q(uc.reg_read(UC_X86_REG_RSI),self.alloc(bytes(size)))
            uc.reg_write(UC_X86_REG_RAX,0);self.ret()
        elif address==0x498a9c0:
            rank=uc.reg_read(UC_X86_REG_RSI)
            assert rank in (2,3)
            self.q(uc.reg_read(UC_X86_REG_RDI),uc.reg_read(UC_X86_REG_RDX)*uc.reg_read(UC_X86_REG_RCX))
            uc.reg_write(UC_X86_REG_RAX,0);self.ret()
        elif address==0x498a650:
            uc.reg_write(UC_X86_REG_RAX,0);self.ret()
        elif address==0x49206e0:
            value=struct.unpack('<d',uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')[:8])[0]
            with np.errstate(invalid='ignore'): result=float(np.arccos(value))
            uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(struct.pack('<d',result),'little'));self.ret()
        elif address==0x6e4490:
            self.pair_calls+=1
        elif (0x6e4490<address<0x6e4aa0 or 0x580660<=address<0x580df0
              or 0x58f680<=address<0x58f800):
            pass
        else:
            super().hook(uc,address,size,user)


def main():
    blob,segments=load_elf(ELF_DEFAULT);assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    rows=[]
    for shells in (2,3):
        atoms=Icosahedron('Cu',shells);atoms.positions+=15
        pair=(0,1);frame=ClusterFrame(atoms)
        assert native_pair_allowed(atoms,pair)
        for initial in (True,False):
            seed=np.zeros_like(atoms.positions) if initial else normalized(frame.project(np.arange(atoms.positions.size).reshape(-1,3)*.013))
            oracle=C4Oracle(segments);oracle.pair_calls=0;oracle.atoms=atoms
            oracle.instruction_limit=5_000_000;oracle.timeout_us=20_000_000
            oracle.axis=pair;oracle.group=np.zeros(len(atoms),np.int32);oracle.uniform=.17
            coeff=np.zeros(10);coeff[4]=.6;coeff[1]=float(initial);coeff[9]=0 if initial else .72
            oracle.coefficients=coeff
            got=oracle.run_geometry(atoms.positions,seed,np.zeros_like(seed))
            local=normalized(frame.project(native_local_pair(atoms,pair,lambda:.17).raw_direction))
            expected=seed*coeff[9]+.6*local
            if initial:
                expected+=normalized(frame.project(native_vmb2(np.zeros_like(seed),np.ones_like(seed,bool),.17)))
            expected=normalized(expected)
            error=float(np.max(abs(got-expected)))
            assert error<1e-12 and oracle.pair_calls==1,(len(atoms),initial,error,oracle.pair_calls)
            rows.append(dict(n=len(atoms),initial=initial,positions=atoms.positions.tolist(),pair=pair,coefficients=coeff.tolist(),input_seed=seed.tolist(),uniform=.17,output=got.tolist(),error=error,pair_calls=oracle.pair_calls))
    Path('research/ga_ssw/evidence/native-pair-generator-20260917.json').write_text(json.dumps(dict(scope=__doc__,sha256=ELF_SHA256,cases=rows),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_error=max(r['error'] for r in rows))))


if __name__=='__main__':main()
