"""Bounded original-instruction stress2dedlatt oracle; no native main/init/PES."""
import json,struct,hashlib
from pathlib import Path
import numpy as np
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI,UC_X86_REG_RSP,UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
blob,segs=load_elf(ELF);sha=hashlib.sha256(blob).hexdigest();assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
u=Uc(UC_ARCH_X86,UC_MODE_64)
for page,size in [(0x59c000,0x1000),(0x4a44000,0x1000),(0x53ed000,0x30000),(0x78e8000,0x1000),(0x70000000,0x10000)]:
 u.mem_map(page,size)
 for addr,_,data in segs:
  lo=max(addr,page);hi=min(addr+len(data),page+size)
  if hi>lo:u.mem_write(lo,data[lo-addr:hi-addr])
obj=0x70001000;arg=0x70000100;sp=0x7000f008;stop=0x70000000;para=0x53ed7a0
rows=[]
for k in range(4):
 rng=np.random.default_rng(231+k);C=rng.normal(size=(3,3));S=rng.normal(size=(3,3));S=(S+S.T)/2;V=23.7+k;p=[0.,.013,-.009,.04][k]
 u.mem_write(obj,bytes(0x1000));u.mem_write(arg,struct.pack('<Q',obj));u.mem_write(obj+0x128,S.astype('<f8').tobytes(order='F'));u.mem_write(obj+0x388,C.astype('<f8').tobytes(order='F'));u.mem_write(obj+0x608,struct.pack('<d',V));u.mem_write(para+0x2db38,struct.pack('<d',p));u.mem_write(para+0x2dde0,struct.pack('<iii',1,1,1));u.mem_write(sp,struct.pack('<Q',stop));u.reg_write(UC_X86_REG_RDI,arg);u.reg_write(UC_X86_REG_RSP,sp)
 u.emu_start(0x59cb10,stop,count=10000,timeout=1000000);assert u.reg_read(UC_X86_REG_RIP)==stop
 D=np.frombuffer(u.mem_read(obj+0x3d0,72),dtype='<f8').reshape((3,3),order='F');A=S+p*np.eye(3)
 candidates={'-V*C@A':-V*C@A,'-V*C.T@A':-V*C.T@A,'-V*A@C':-V*A@C,'-V*A@C.T':-V*A@C.T}
 rows.append(dict(stress=S.tolist(),celli=C.tolist(),volume=V,externaltp=p,dedlatt=D.tolist(),candidate_max_errors={s:float(np.max(abs(D-v))) for s,v in candidates.items()}))
out=Path('research/ga_ssw/evidence/native-cell-force');out.mkdir(exist_ok=False);(out/'probe.py').write_text(Path(__file__).read_text());(out/'result.json').write_text(json.dumps(dict(elf_sha256=sha,entry='0x59cb10',no_calls_no_init_no_pes=True,free_cell_mask=[1,1,1],rows=rows),indent=2)+'\n');print([r['candidate_max_errors'] for r in rows])
