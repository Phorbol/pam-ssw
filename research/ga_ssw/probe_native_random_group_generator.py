"""Actual c1+c6 generator on compact free molecules; Q/c4/compression absent."""
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
from ase.build import molecule
from unicorn.x86_const import *
from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_local_group import native_local_group
from pamssw.standalone.native_random import native_vmb2
from research.ga_ssw.probe_native_group_geometry import GeometryOracle
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_addgaussian_emulated import DATA
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class RandomGroupOracle(GeometryOracle):
    def hook(self,uc,address,size,user):
        if address==0x5d5c50:
            super().hook(uc,address,size,user)
            obj=self.readq(DATA)
            coeff=uc.reg_read(UC_X86_REG_RSI)
            uc.mem_write(coeff,np.asarray(self.coefficients,dtype='<f8').tobytes())
            self.descriptor(obj+0x968,self.alloc(np.ones_like(self.positions,dtype='<i4').tobytes()),(3,self.n),4)
            uc.mem_write(obj+0xe0,(40*np.eye(3,dtype='<f8')).tobytes())
            self.q(0x53e0f98,DATA+0xf000)
        elif address==0x580640:
            self.d(uc.reg_read(UC_X86_REG_RDI),self.uniform)
            self.ret()
        elif address==0x58c930:
            self.vmb_seed=struct.unpack('<d',uc.mem_read(uc.reg_read(UC_X86_REG_R8),8))[0]
        elif address in (0x4920770,0x4920840):
            value=struct.unpack('<d',uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')[:8])[0]
            result=(math.cos if address==0x4920770 else math.log)(value)
            uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(struct.pack('<d',result),'little'))
            self.ret()
        elif address==DATA+0xf000:
            values=np.frombuffer(uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little'),dtype='<f8')
            uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(np.floor(values).astype('<f8').tobytes(),'little'))
            self.ret()
        elif (0x58c930<address<0x58d000 or 0x6e3590<=address<0x6e37c0
              or 0x581240<=address<0x5816b0 or 0x578480<=address<0x578660):
            pass
        else:
            super().hook(uc,address,size,user)


def main():
    blob,segments=load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    rows=[]
    for name in ('C2H6','CH3OH','C6H6'):
        atoms=molecule(name);atoms.positions+=15
        frame=ClusterFrame(atoms)
        pair=tuple(np.flatnonzero(atoms.numbers!=1)[:2])
        group=np.arange(len(atoms))%2
        for weight in (.1,1.6):
            oracle=RandomGroupOracle(segments)
            oracle.axis,oracle.group=pair,group
            oracle.uniform=.17
            oracle.coefficients=np.zeros(10)
            oracle.coefficients[1]=1;oracle.coefficients[6]=weight
            got=oracle.run_geometry(atoms.positions,np.zeros_like(atoms.positions),np.zeros_like(atoms.positions))
            random=normalized(frame.project(native_vmb2(np.zeros_like(atoms.positions),np.ones_like(atoms.positions,bool),oracle.vmb_seed)))
            local=normalized(frame.project(native_local_group(atoms,pair,group)))
            expected=normalized(random+weight*local)
            error=float(np.max(abs(got-expected)))
            assert error<1e-12,(name,error)
            rows.append(dict(name=name,weight=weight,positions=atoms.positions.tolist(),pair=list(map(int,pair)),group=group.tolist(),seed=oracle.vmb_seed,output=got.tolist(),error=error))
    Path('research/ga_ssw/evidence/native-random-group-generator-20260917.json').write_text(json.dumps(dict(scope=__doc__,sha256=ELF_SHA256,cases=rows),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_error=max(r['error'] for r in rows))))


if __name__=='__main__':main()
