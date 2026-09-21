"""Offline geometry audit for phase87 memory400 continuation; no calculator calls."""
import json, numpy as np
from pathlib import Path
from ase import Atoms
from ase.neighborlist import neighbor_list
ROOT=Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity'); BASE=ROOT/'research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step'; SRC=BASE/'joint-memory400-whole-step/result.json'; RUN=BASE/'joint-memory400-continuation-root/result.json'; OUT=BASE/'joint-memory400-continuation-root'
def A(d): return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
def metrics(a):
 # Enumerate periodic images: one MIC distance per atom label misses repeated
 # oxygen images in the short lattice direction. A6A enumeration radius is a
 # reporting bound, not a coordination cutoff; verify six neighbors exist.
 ii,jj,shift,dist=neighbor_list('ijSd',a,6.,self_interaction=False)
 ti=[]
 for i in np.where(a.numbers==22)[0]:
  ids=np.flatnonzero((ii==i)&(a.numbers[jj]==8)); ids=ids[np.argsort(dist[ids])][:6]
  assert len(ids)==6 and dist[ids[-1]]<6.
  ti.append({'atom':int(i),'first6':[{'oxygen_atom':int(jj[k]),'image_shift':shift[k].tolist(),'distance_A':float(dist[k])} for k in ids],'sixth':float(dist[ids[-1]])})
 density=float(a.get_masses().sum()/a.get_volume())
 return {'volume_A3':a.get_volume(),'mass_density_amu_A3':density,'mass_density_g_cm3':density*1.66053906660,'periodic_pair_min_A':float(dist.min()),'ti_o_first6':ti,'ti_o_sixth_min_A':float(min(x['sixth'] for x in ti)),'ti_o_sixth_max_A':float(max(x['sixth'] for x in ti)),'numbers':a.numbers.tolist(),'cell':a.cell.array.tolist(),'method':'image-resolved ASE neighbor_list within6A; sixth neighbor list is not an assertion of sixfold coordination'}
def main():
 src=json.loads(SRC.read_text());run=json.loads(RUN.read_text()); initial=A(src['landings'][0]['atoms']); final=A(run['landing']['atoms']); fresh=run['fresh'];
 out={'source_files':{'original':str(SRC),'continuation':str(RUN)},'classification':{'initial_certificate':src['landings'][0]['certificate'],'final_certificate':run['landing']['certified'],'mc_accepted':run['accepted'],'physical_identity':'geometry metrics only; no phase/landing basin identity inferred','revision':'image-resolved neighbors replace earlier unique-atom MIC lists; old six-neighbor distances were not coordination measures'},'initial_certified':metrics(initial),'final_landing':metrics(final),'energy':{'initial_objective':src['landings'][0]['objective'],'final_objective':run['landing']['objective'],'delta_eV':run['landing']['delta']},'fresh_consistency':{'requests':fresh['requests'],'energy_abs_diff':abs(fresh['energy']-run['landing']['objective']),'fmax_abs_diff':abs(fresh['fmax']-run['landing']['fmax']),'stress_max_abs_diff':abs(fresh['stress_max']-run['landing']['stress_max'])}}
 (OUT/'geometry-review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 (OUT/'geometry-review.md').write_text('# Phase87 memory400 continuation geometry audit\n\nThis is offline ASE analysis of the saved initial certificate and final certified landing. No calculator or PES call was made.\n\n- Final numerical certificate: `fmax=0.0085175750222 eV/A`, `stress_max=2.39552723546e-5 eV/A^3`, volume `763.3494519604 A^3`.\n- Objective change from the saved initial certificate: `+2.6136175021 eV`; MC was rejected.\n- Initial and final image-resolved periodic pair minima, volume/density, and every Ti six-nearest-O-image list are in `geometry-review.json`.\n- Fresh consistency is reported separately; it does not establish phase identity or a valid physical basin.\n')
 print(json.dumps(out['fresh_consistency']))
if __name__=='__main__':main()
