"""Native torsion-input slice + explicit two-body fiber rank; no PES."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from ase.build import molecule
from scipy.spatial.transform import Rotation
from unicorn import Uc,UC_ARCH_X86,UC_MODE_64
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from pamssw.standalone.rc_geometry import RigidChainChart
ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
blob,segs=load_elf(ELF);sha=hashlib.sha256(blob).hexdigest();assert sha=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
u=Uc(UC_ARCH_X86,UC_MODE_64)
for page,size in [(0x81d000,0x1000),(0x81f000,0x1000),(0x4a4e000,0x1000),(0x9036000,0x1000),(0x70000000,0x10000)]:
 u.mem_map(page,size)
 for addr,_,data in segs:
  lo=max(page,addr);hi=min(page+size,addr+len(data))
  if hi>lo:u.mem_write(lo,data[lo-addr:hi-addr])
a=molecule('trans-butane');a.positions=a.positions@Rotation.from_rotvec([.37,-.19,.43]).as_matrix().T;bodies=[(0,1,4,6,7),(0,1,2,3,5,8,9,10,11,12,13)]
c=RigidChainChart(a,bodies,parents=(-1,0),joints=(None,(0,1)))
axis=a.positions[1]-a.positions[0];axis/=np.linalg.norm(axis)
obj=0x70001000;coords=0x70002000;bp=0x70008000
native=[]
for theta in (-2.,-.1,0.,.7,2.8):
 u.mem_write(0x70000000,bytes(0x10000));u.mem_write(obj+0x728,struct.pack('<ii',1,2));u.mem_write(coords,a.positions[:2].astype('<f8').tobytes());u.mem_write(bp-0x248,struct.pack('<Q',coords))
 u.reg_write(UC_X86_REG_R10,obj);u.reg_write(UC_X86_REG_RBP,bp);u.reg_write(UC_X86_REG_XMM0,int.from_bytes(struct.pack('<d',theta),'little'))
 u.emu_start(0x81f1b7,0x81d7ae,count=500,timeout=1000000);assert u.reg_read(UC_X86_REG_RIP)==0x81d7ae
 v=np.frombuffer(u.mem_read(obj+0x30,24),dtype='<f8').copy();angle=struct.unpack('<d',u.mem_read(obj+0x48,8))[0]
 native.append(dict(theta=theta,rotation_vector=v.tolist(),angle=angle,error=float(abs(v-theta*axis).max())))
# A representative smooth member of the native-supported axial-inheritance family.
# This is a reconstructed gauge example, NOT the complete native forward map.
def psi(p):
 quat=Rotation.from_rotvec(p).as_quat();return 2*np.arctan2(quat[:3]@axis,quat[3])
def mapq(q,k):
 z=q.copy();z[6]=q[6]-k*psi(q[3:6]);return z
rows=[]
for k in (0.,.7,1.):
 q=np.array([.2,-.1,.3,.31,-.27,.21,.8]);z=mapq(q,k);r,J=c.evaluate(z);J=J.reshape(-1,7)
 A=np.eye(7);eps=1e-6
 for i in range(3,6):
  d=np.eye(7)[i]*eps;A[6,i]=-k*(psi((q+d)[3:6])-psi((q-d)[3:6]))/(2*eps)
 fd=np.column_stack([(c.evaluate(mapq(q+np.eye(7)[i]*eps,k))[0].positions-c.evaluate(mapq(q-np.eye(7)[i]*eps,k))[0].positions).ravel()/(2*eps) for i in range(7)])
 lhs=np.linalg.svd(fd,full_matrices=False)[0];rhs=np.linalg.svd(J,full_matrices=False)[0]
 F=np.random.default_rng(77).normal(size=r.positions.shape);pullback=(J@A).T@F.ravel()
 forcefd=np.array([np.sum(F*(c.evaluate(mapq(q+np.eye(7)[i]*eps,k))[0].positions-c.evaluate(mapq(q-np.eye(7)[i]*eps,k))[0].positions))/(2*eps) for i in range(7)])
 rows.append(dict(krot=k,rank=int(np.linalg.matrix_rank(fd)),singular_values=np.linalg.svd(fd,compute_uv=False).tolist(),map_determinant=float(np.linalg.det(A)),jacobian_error=float(abs(fd-J@A).max()),tangent_projector_error=float(abs(lhs@lhs.T-rhs@rhs.T).max()),work_force_error=float(abs(pullback-forcefd).max()),scope='constructed axial twist gauge using native-supported mechanism, not complete release forward/force'))
out=Path('research/ga_ssw/evidence/rc-reachable-fiber-oblique');out.mkdir(exist_ok=False);(out/'probe.py').write_text(Path(__file__).read_text());(out/'result.json').write_text(json.dumps(dict(elf_sha256=sha,native_slice='0x81f1b7 -> 0x81d7ae; synthetic registers, no entry/global initialization',native=native,reconstructed_gauge=rows),indent=2)+'\n');print(native);print(rows)
