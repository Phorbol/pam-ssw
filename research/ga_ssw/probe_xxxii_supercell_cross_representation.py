"""Four EFS: equivalent1x1x2 periodic representation across special-image jump.

Repeat identical primitive coordinates/topology, divide E and aggregate forces;
no force-field coefficient, physical crystal or search parameter is changed.
"""
import ctypes,json,time,shutil
from pathlib import Path
import numpy as np
from ase import Atoms,units
from ase.calculators.lammps.coordinatetransform import Prism
from lammps import lammps
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion';OUT=SRC/'supercell-cross-representation';OUT.mkdir(exist_ok=False)
points=[json.loads(x) for x in (SRC/'line-search-gradient/calls.jsonl').open()];points=[x for x in points if x['label'] in ('center','lbfgs+1e-06')];shutil.copy2(__file__,OUT/'runner-executed.py')
(OUT/'plan.json').write_text(json.dumps(dict(EFS=6,representations=[[1,1,2],[1,1,3],[2,2,2]],input=str(SRC),scope='equivalentperiodicreplication numericaldiagnostic; notlargerscientificsystem; returnE/replicas andmeanforces; recordenergycontinuity withoutclaimwholetrajectoryfixed'),indent=2)+'\n')
rows=[]
for rep in ((1,1,2),(1,1,3),(2,2,2)):
 nz=int(np.prod(rep));label="x".join(map(str,rep))
 engine=lammps(cmdargs=['-log',str(OUT/f'engine-{label}.log'),'-screen','none'])
 try:
  for line in (SRC/'in.simple').read_text().splitlines():
   command=line.split('#',1)[0].strip()
   if not command:continue
   if command.startswith('read_data '):command=f'read_data "{SRC / "lmp.data"}"'
   engine.command(command)
  if nz>1:engine.command('replicate '+' '.join(map(str,rep)))
  for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069','compute pam_virial all pressure NULL virial','thermo_style custom step pe ebond eangle edihed eimp evdwl ecoul elong etail'):engine.command(command)
  n=172*nz;assert int(engine.get_natoms())==n
  for p in points:
   primitive=Atoms(**p['atoms']);a=primitive.repeat(rep);prism=Prism(a.cell.array,pbc=True,reduce_cell=False);xx,yy,zz,xy,xz,yz=prism.get_lammps_prism()
   engine.command(f'change_box all x final 0 {xx:.17g} y final 0 {yy:.17g} z final 0 {zz:.17g} xy final {xy:.17g} xz final {xz:.17g} yz final {yz:.17g} units box');engine.command('set atom * image 0 0 0')
   pos=np.ascontiguousarray(prism.vector_to_lammps(a.positions,wrap=False),dtype=np.float64);engine.scatter_atoms('x',1,3,pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)));engine.command('run 0 post no')
   force=np.ctypeslib.as_array(engine.gather_atoms('f',1,3),shape=(n*3,)).copy().reshape(n,3);force=prism.vector_to_ase(force)*units.kcal/units.mol;blocks=force.reshape(nz,172,3)
   pv=engine.extract_compute('pam_virial',0,1);xx,yy,zz,xy,xz,yz=[float(pv[i]) for i in range(6)];pressure=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]);stress=prism.tensor2_to_ase(-pressure*(units.kcal/units.mol)/68568.415)
   parts={k:float(engine.get_thermo(k))*units.kcal/units.mol/nz for k in ('ebond','eangle','edihed','eimp','evdwl','ecoul','elong','etail')}
   charges=np.ctypeslib.as_array(engine.gather_atoms('q',1,1),shape=(n,)).copy().reshape(nz,172);types=np.ctypeslib.as_array(engine.gather_atoms('type',0,1),shape=(n,)).copy().reshape(nz,172);assert np.array_equal(charges,np.tile(charges[:1],(nz,1))) and np.array_equal(types,np.tile(types[:1],(nz,1)))
   rows.append(dict(replication=rep,replicas=nz,label=p['label'],energy=float(engine.get_thermo('pe'))*units.kcal/units.mol/nz,parts=parts,forces=blocks.mean(axis=0).tolist(),stress=stress.tolist(),force_image_max_difference=float(abs(blocks-blocks[:1]).max()),types=types.tolist(),charges=charges.tolist()))
 finally:engine.close()
summary=[]
for rep in ((1,1,2),(1,1,3),(2,2,2)):
 nz=int(np.prod(rep));label="x".join(map(str,rep))
 pair=[x for x in rows if tuple(x['replication'])==rep];summary.append(dict(replication=rep,replicas=nz,energy_delta=pair[1]['energy']-pair[0]['energy'],parts_delta={k:pair[1]['parts'][k]-pair[0]['parts'][k] for k in pair[0]['parts']}))
(OUT/'result.json').write_text(json.dumps(dict(EFS=6,rows=rows,summary=summary),indent=2)+'\n');print(json.dumps(summary,indent=2))
