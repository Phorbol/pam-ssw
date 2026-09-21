"""Compare saved first dimer modes to the full independent physical Hessian.

No PES; uses previously charged TYPE4 two-step Hessian qualification. Negative
rank-one-biased curvature is distinct from a physical negative eigenvalue.
"""
import json
from pathlib import Path
import numpy as np


def main():
    b=Path(__file__).resolve().parent/'evidence';q=b/'type4-certified-start-control/qualification/initial'
    h=np.load(q/'hessian-5e-05.npy');h=(h+h.T)/2
    paths=list((b/'type4-certified-start-control/search').glob('*/result.json'))+list((b/'type4-multistep-heldout').glob('seed*/*/result.json'))
    out=[]
    for p in sorted(paths):
        plan=json.loads((p.parent.parent/'plan.json').read_text());data=json.loads(p.read_text())['result'];stage=data['records'][1]['climb'][0]
        if 'mode' not in stage:continue
        mask=np.arange(651) if p.parent.name=='physical_mask_only' else np.arange(3*(351-297),651)
        anchor=np.random.default_rng(plan['seed']).normal(size=651);anchor=anchor[mask];anchor/=np.linalg.norm(anchor)
        n=np.asarray(stage['mode']['direction'])[mask];beta=plan['config']['rotation_bias'];physical=h[np.ix_(mask,mask)];effective=physical-beta*np.outer(anchor,anchor)
        values,vectors=np.linalg.eigh(effective);exact=vectors[:,0]
        out.append(dict(source=str(p.relative_to(b)),seed=plan['seed'],rotation_bias=beta,dimension=len(mask),anchor_overlap=float(abs(n@anchor)),saved_biased_curvature=stage['mode']['curvature'],physical_rayleigh=float(n@physical@n),matrix_biased_rayleigh=float(n@effective@n),matrix_residual=float(np.linalg.norm(effective@n-(n@effective@n)*n)),exact_effective_min=float(values[0]),exact_mode_overlap=float(abs(n@exact))))
    result=dict(rows=out,scope='only first stage at the common qualified initial; masks and seeded random draws reconstructed from frozen plans; full Hessian was paid in earlier qualification; no physical saddle claim from biased eigenvalues; no claim beta100 is optimal',formula='H_eff = H - beta*a*a^T; q_coordinates are active Cartesian Angstrom; beta in eV/Angstrom^2',reference='Shang and Liu, Stochastic Surface Walking Method for Structure Prediction and Pathway Searching, JCTC2013, DOI10.1021/ct301010b, biased rotation Eqs5-6; beta convention follows generalized_dimer code, not an assertion of identical paper coefficient')
    (b/'type4-certified-start-control/biased-mode-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
