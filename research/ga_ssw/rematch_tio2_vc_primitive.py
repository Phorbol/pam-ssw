"""Zero-oracle, species-preserving periodic identity audit with tolerance sweep."""
import json,inspect,hashlib,itertools,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher,SpeciesComparator
OUT=Path('research/ga_ssw/evidence/tio2-vc-structure-identity-primitive');OUT.mkdir(exist_ok=False)
(OUT/'script.py').write_text(Path(__file__).read_text());(OUT/'matcher-api.txt').write_text(str(inspect.signature(StructureMatcher))+'\n'+inspect.getdoc(StructureMatcher.__init__)+'\n'+inspect.getsource(StructureMatcher.get_rms_dist))
sources={'rutile_initial':('joint-vc-rutile12-l5','initial'),'rutile_landing':('joint-vc-rutile12-l5','landing'),'anatase_reference':('vc-reference-anatase','initial'),'tio2_b_reference':('vc-reference-tio2-b','initial')}
ss={};report=dict(requests=0,versions={n:importlib.metadata.version(n) for n in ['pymatgen','numpy','spglib']},configuration=dict(scale=False,primitive_cell=True,attempt_supercell=True,allow_subset=False,comparator='SpeciesComparator'),sources={},controls=[],pairs=[],shells={})
for k,(folder,role) in sources.items():
 p=Path('research/ga_ssw/evidence')/folder/'result.json';x=json.loads(p.read_text())['result'];r=x['minima'][-1] if role=='landing' else x['initial']
 ss[k]=Structure(r['cell'],r['symbols'],r['positions'],coords_are_cartesian=True)
 report['sources'][k]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role=role,energy=r['energy'],fmax=r['fmax'],volume=ss[k].volume,volume_per_atom=ss[k].volume/len(ss[k]),cell=r['cell'],positions=r['positions'],symbols=r['symbols'])
# Convert a nominal physical site tolerance using mean volume. The matcher uses its candidate average lattice internally; these Angstrom conversions are nominal.
def compare(a,b,dist,ltol,angle):
 ell=((a.volume+b.volume)/(2*len(a)))**(1/3)
 m=StructureMatcher(stol=dist/ell,ltol=ltol,angle_tol=angle,primitive_cell=True,scale=False,attempt_supercell=True,allow_subset=False,comparator=SpeciesComparator())
 r=m.get_rms_dist(a,b)
 return dict(matched=r is not None,stol=dist/ell,average_free_length_A=ell,nominal_rms_A=None if r is None else r[0]*ell,nominal_max_A=None if r is None else r[1]*ell)
rng=np.random.default_rng(901)
for name,a in ss.items():
 rot=Rotation.from_rotvec([.3,-.4,.2]).as_matrix();order=rng.permutation(len(a));U=np.array([[1,1,0],[0,1,0],[0,0,1]])
 b=Structure(U@a.lattice.matrix@rot,np.array([str(s.specie) for s in a])[order],(a.cart_coords@rot+np.array([.37,-.21,.49]))[order],coords_are_cartesian=True)
 noise=rng.normal(size=a.cart_coords.shape);noise*=1e-4/max(np.linalg.norm(noise,axis=1));c=Structure(a.lattice.matrix,[str(s.specie) for s in a],a.cart_coords+noise,coords_are_cartesian=True)
 for label,obj in [('duplicate',a.copy()),('basis_rotation_translation_permutation',b),('bounded_0.0001A_coordinate_noise',c)]:
  result=compare(a,obj,.002,.001,.1);report['controls'].append(dict(structure=name,control=label,distance_tolerance_A=.002,ltol=.001,angle_tol_deg=.1,**result));assert result['matched'],(name,label)
for ka,kb in itertools.combinations(ss,2):
 for dist,ltol,angle in [(.002,.001,.1),(.02,.005,.5),(.05,.02,2.),(.1,.05,5.)]:
  r=compare(ss[ka],ss[kb],dist,ltol,angle)
  report['pairs'].append(dict(a=ka,b=kb,distance_tolerance_A=dist,ltol=ltol,angle_tol_deg=angle,volume_ratio=ss[kb].volume/ss[ka].volume,**r))
for name,a in ss.items():
 rows=[]
 for i,site in enumerate(a):
  if str(site.specie)!='Ti':continue
  nn=sorted(float(n.nn_distance) for n in a.get_neighbors(site,4.) if str(n.specie)=='O')
  rows.append(dict(site=i,nearest_8_TiO_A=nn[:8]))
 report['shells'][name]=rows
report['status']='completed';(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
for r in report['pairs']:
 if r['matched']:print(r)
print('all controls passed; pairs',len(report['pairs']),'EFS=0')
