"""Extract published coordinate tables without relaxation or PES evaluation."""
from pathlib import Path
import re,json,hashlib
from io import StringIO
import numpy as np
from ase import Atoms
from ase.io import write,read
from ase.io.dmol import read_dmol_arc

BASE=Path('literature/benchmark-sources')

def main():
    out=BASE/'coordinates';out.mkdir(exist_ok=True)
    text=(BASE/'nn-si.txt').read_text().replace('\f','\n')
    headers=list(re.finditer(r'^\s*(.+?)\s+Energy from VASP is\s+(-?\d+\.\d+)\s*$',text,re.M))
    rows=[]
    for i,m in enumerate(headers):
        name=m.group(1).strip();body=text[m.end():headers[i+1].start() if i+1<len(headers) else len(text)]
        lines=[l.strip() for l in body.splitlines() if l.strip()]
        pbc=next(l for l in lines if l.startswith('PBC'))
        atomlines=[l for l in lines if re.match(r'^(Ti|O)\s+[-\d.]',l)]
        arc='!BIOSYM archive 2\nPBC=ON\n'+name+'\n!DATE\n'+pbc+'\n'+'\n'.join(atomlines)+'\nend\nend\n'
        a=read_dmol_arc(StringIO(arc),index=-1)
        assert len(a)==len(atomlines) and np.isfinite(a.positions).all() and a.get_volume()>0
        assert sum(a.numbers==8)==2*sum(a.numbers==22)
        slug=re.sub('[^a-z0-9]+','-',name.lower()).strip('-')
        arcp=out/(slug+'.arc');xyzp=out/(slug+'.extxyz');arcp.write_text(arc)
        a.info.update(source_doi='10.1039/c7sc01459g',source_section='SI section 7',source_name=name,
                      reported_vasp_energy_rounded=float(m.group(2)),role='transition_state' if name.startswith('TS ') else ('endpoint' if name.startswith(('IS ','FS ')) else 'reported_phase'))
        write(xyzp,a)
        reread=read(xyzp)
        assert np.allclose(reread.positions,a.positions,atol=5.1e-9,rtol=0)
        rows.append(dict(name=name,role=a.info['role'],n_atoms=len(a),formula=a.get_chemical_formula(),cellpar=a.cell.cellpar().tolist(),
            arc=str(arcp),extxyz=str(xyzp),rounded_source_energy=float(m.group(2)),source_si_md5=hashlib.md5((BASE/'SC-008-C7SC01459G-s001.pdf').read_bytes()).hexdigest()))
    for n,energy in [(38,-173.928427),(75,-397.492331)]:
        coords=np.loadtxt(BASE/f'lj{n}.points');assert coords.shape==(n,3) and np.isfinite(coords).all()
        a=Atoms(numbers=np.zeros(n,dtype=int),positions=coords,pbc=False)
        a.info.update(source_url=f'https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/{n}',
                      units='reduced LJ: positions in sigma; published energy in epsilon',published_energy=energy)
        p=out/f'lj{n}-gm.extxyz';write(p,a)
        rows.append(dict(name=f'LJ{n} GM',n_atoms=n,extxyz=str(p),raw_points=str(BASE/f'lj{n}.points'),
                         units='sigma positions / epsilon energy',published_energy=energy,
                         note='X is a dummy ASE element. Supply explicitly untruncated LJ convention; no energy evaluated here.'))
    (BASE/'coordinate-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],n_atoms=r['n_atoms'],role=r.get('role')) for r in rows],indent=2))

if __name__=='__main__':main()
