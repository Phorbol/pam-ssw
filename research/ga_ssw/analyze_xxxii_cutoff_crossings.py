"""Zero-EFS finite-cutoff energy jumps in the frozen XXXII table0 diagnostic."""
import json
from pathlib import Path
import numpy as np
from scipy.special import erfc
from ase.io import read
from ase.neighborlist import neighbor_list
from ase import units
from research.ga_ssw.convert_xxxii_amber import sections
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
out=Path('research/ga_ssw/evidence/xxxii-lammps-qualification-table0')
p=json.loads((out/'plan.json').read_text());r=json.loads((out/'result.json').read_text())
a=read(out/'type2_xxxii.extxyz');top=read_rigid_topology(out/'rigidbody',out/'blist',natoms=len(a));a=unwrap_rigid_molecules(a,top.bonds).atoms
rows=sorted(sections((out/'lmp.data').read_text())['Atoms'],key=lambda x:int(x[0]));charges=np.array([float(x[3]) for x in rows])
g=np.sqrt(-np.log(1e-6*np.sqrt(len(a)*10*a.get_volume())/(2*sum(charges**2))))/10
vc=SymmetricLogStrainChart(a,strain_length=5.);q=vc.pack(a);rc=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
def geom(name,t):
 if name=='atomic':
  b=a.copy();b.positions+=t*np.array(p['atomic_direction']);return b
 if name=='RC':return rc.unpack(t*np.array(p['RC_direction']))
 d=np.zeros_like(q);d[-6+int(name[4:])]=t;return vc.unpack(q+d)
def inside(b):
 i,j,S=neighbor_list('ijS',b,10.);return {(int(ii),int(jj),*map(int,ss)) for ii,jj,ss in zip(i,j,S) if ii<jj}
records=[]
for chk in r['derivatives']:
 name=chk['direction'];h=chk['h'];plus=inside(geom(name,h));minus=inside(geom(name,-h));cross=[];jump=0.
 for sign,pairs in ((1,plus-minus),(-1,minus-plus)):
  for pair in sorted(pairs):
   i,j,*_=pair;e=332.06371*(units.kcal/units.mol)*charges[i]*charges[j]*erfc(g*10)/10; jump+=sign*e
   shift=np.array(pair[2:]); distances={}
   for label,b in [('base',a),('plus',geom(name,h)),('minus',geom(name,-h))]:
    distances[label]=float(np.linalg.norm(b.positions[j]-b.positions[i]+shift@b.cell.array))
   cross.append(dict(pair=pair,distances=distances,base_abs_distance_to_cutoff=abs(distances['base']-10),plus_minus_occupancy_change=sign,energy_at_cutoff_eV=e))
 records.append(dict(**chk,crossings=cross,predicted_cutoff_fd=jump/(2*h),fd_error_minus_predicted=chk['finite_difference']-chk['analytic']-jump/(2*h)))
result=dict(scope='0 EFS; analytical jump at r=10 using source-derived G. No correction applied to physical oracle.',g_ewald_source_formula=g,records=records)
(out/'cutoff-crossings.json').write_text(json.dumps(result,indent=2)+'\n')
for x in records:print(x['direction'],x['h'],len(x['crossings']),x['finite_difference']-x['analytic'],x['predicted_cutoff_fd'],x['fd_error_minus_predicted'])
