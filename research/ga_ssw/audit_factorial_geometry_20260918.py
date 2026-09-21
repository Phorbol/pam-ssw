"""Read-only geometry audit for factorial saved-minimum controls."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ase.io import read

ROOT=Path('research/ga_ssw/evidence/ssw-allocation-factorial-20260918/controls')
def atoms(d):
    from ase import Atoms
    return Atoms(numbers=d['numbers'],positions=d['positions'],cell=d.get('cell'),pbc=d.get('pbc',False))
def rmsd_kabsch(a,b):
    x=a.positions-a.positions.mean(0); y=b.positions-b.positions.mean(0); h=x.T@y; u,s,v=np.linalg.svd(h); q=u@v
    if np.linalg.det(q)<0: v[-1]*=-1; q=u@v
    return float(np.sqrt(np.mean(np.sum((x@q-y)**2,axis=1))))
def comps(a,cut):
    d=np.linalg.norm(a.positions[:,None]-a.positions[None,:],axis=2); adj=(d<cut)&(d>0); seen=set(); out=[]
    for i in range(len(a)):
        if i in seen: continue
        q=[i]; seen.add(i); n=0
        while q:
            u=q.pop();n+=1
            for v in np.where(adj[u])[0]:
                v=int(v)
                if v not in seen: seen.add(v);q.append(v)
        out.append(n)
    return sorted(out,reverse=True)
def water_diag(a):
    o=np.where(a.numbers==8)[0]; h=np.where(a.numbers==1)[0]; d=np.linalg.norm(a.positions[:,None]-a.positions[None,:],axis=2)
    oh=[float(d[i,j]) for i in o for j in h if d[i,j]<1.25]; oo=[float(d[i,j]) for ii,i in enumerate(o) for j in o[ii+1:] if d[i,j]<2.0]
    adj=(d<1.25)&(d>0); od=[int(adj[i,h].sum()) for i in o]; hd=[int(adj[hidx,o].sum()) for hidx in h]
    return {'atoms':len(a),'composition':dict(Counter(a.get_chemical_symbols())),'oh_pairs_lt_1p25':len(oh),'oh_min_A':min(oh) if oh else None,'oh_o_degree_counts':dict(Counter(od)),'oh_h_degree_counts':dict(Counter(hd)),'oo_pairs_lt_2p0':len(oo),'oo_components_lt_3p5':comps(a[o],3.5),'thresholds_A':{'O-H':1.25,'O-O_overlap':2.0,'O-O_cluster':3.5}}
def main():
    out={'scope':'geometric diagnostics only; atom labels retained for RMSD; no permutation-invariant basin count','cu55':[],'water15':[]}
    initial=read(str(ROOT/'inputs/cu55.traj'))
    for p in sorted(ROOT.glob('cu55-*/result.json')):
        x=json.loads(p.read_text()); q=json.loads((p.parent/'qualification.json').read_text()); rows=[]
        for i,m in enumerate(x.get('minima',[])):
            a=atoms(m['atoms']); rows.append({'index':i,'energy_eV':m['energy'],'fmax_eV_A':m['max_force'],'rmsd_to_initial_A':rmsd_kabsch(initial,a),'qualified':bool(q[i].get('qualified')) if i<len(q) else None,'components_lt_2p8':comps(a,2.8)})
        out['cu55'].append({'run':p.parent.name,'initial_energy_eV':x['initial']['energy'],'initial_fmax_eV_A':x['initial']['max_force'],'status':x['status'],'minima':rows})
    winit=read(str(ROOT/'inputs/water15.traj')) if (ROOT/'inputs/water15.traj').exists() else None
    for p in sorted(ROOT.glob('water15-*/result.json')):
        x=json.loads(p.read_text()); q=json.loads((p.parent/'qualification.json').read_text()); rows=[]
        for i,m in enumerate(x.get('minima',[])):
            z=water_diag(atoms(m['atoms'])); z.update(index=i,energy_eV=m['energy'],fmax_eV_A=m['max_force'],qualified=bool(q[i].get('qualified')) if i<len(q) else None); rows.append(z)
        out['water15'].append({'run':p.parent.name,'initial_energy_eV':x['initial']['energy'],'initial_fmax_eV_A':x['initial']['max_force'],'status':x['status'],'minima':rows})
    (Path('research/ga_ssw/evidence/ssw-allocation-factorial-20260918')/'geometry-audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
