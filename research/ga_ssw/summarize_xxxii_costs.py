"""Saved-artifact accounting only; never replaces API calls by successful calls."""
import json
from pathlib import Path
base=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence'
paths=[('first whole memory10','xxxii-rc-vc-one-step',385),('frozen first memory400','xxxii-rc-vc-one-step/frozen-memory400',291),('first whole memory400','xxxii-rc-vc-memory400-one-step',741),('globalbudget memory10','xxxii-rc-vc-globalbudget-memory10',1499),('globalbudget memory400','xxxii-rc-vc-globalbudget-memory400',1337),('rotation FD','xxxii-rc-vc-globalbudget-memory400/rotation-fd-check',13),('rotation Ritz4arms','xxxii-rc-vc-globalbudget-memory400/rotation-ritz-comparison',164),('central Ritz1500','xxxii-rc-vc-globalbudget-central-ritz',1499),('central Ritzcompletion','xxxii-rc-vc-central-ritz-completion',1526),('line searchFD','xxxii-rc-vc-central-ritz-completion/line-search-gradient',13),('energydecomposition','xxxii-rc-vc-central-ritz-completion/energy-jump-components',2),('supercellrepresentations','xxxii-rc-vc-central-ritz-completion/supercell-representation',4),('supercellcrosscheck','xxxii-rc-vc-central-ritz-completion/supercell-cross-representation',6)]
paths += [('replicated qualification','xxxii-replicated-qualification',33),('replicated full RCVC','xxxii-rc-vc-replicated-completion',3559),('replicated extxyz validation','xxxii-rc-vc-replicated-completion/cross-representation-validation',4),('replicated fullprecision validation','xxxii-rc-vc-replicated-completion/cross-representation-fullprecision',4)]
paths += [('replicated forward control','xxxii-rc-vc-replicated-forward-control',1319)]
paths += [('replicated frozen rotation3arms','xxxii-rc-vc-replicated-forward-control/frozen-rotation-comparison',224)]
paths += [('rotation common certificate','xxxii-rc-vc-replicated-forward-control/rotation-common-certificate',12),('strict endpoints','xxxii-replicated-endpoint-quench',542)]
paths += [('Hessian firstwallcap','xxxii-replicated-hessian-qualification',1850),('Hessian remainingcolumns','xxxii-replicated-hessian-completion',2310)]
rows=[]
for label,p,count in paths:
 d=json.loads((base/p/'result.json').read_text())
 found=next(d[k] for k in ('total_EFS','api_requests','total_API','API','EFS','api_attempts') if k in d)
 assert found==count,(label,found,count)
 rows.append(dict(label=label,path=p,API=count,engine_calls=d.get('engine_calls'),atoms_evaluated=d.get('atoms_evaluated',d.get('internal_atoms_evaluated')),replicas=d.get('replicas','see per-representation rows' if ('supercell' in p or 'replicated' in p) else 1)))
out=dict(qualification_API=196,development_rows=rows,total_API=196+sum(x['API'] for x in rows),failed_zero_PES_preflight_attempts=1,scope='listed convertedGAFF qualification and RCVC diagnostic series only; excludes earlierGFN2 andstatichelpers. Replicated calls evaluate344/516/1376atoms; APIcounts not equal physical cost.',claim='Primitive whole RCVC attempts have no landing and periodic exclusiondiscontinuity invalidates general geometry qualification. Fixed1x1x2 centralRitz whole RCVC completed12stages+landing, higherenergy MC rejected; not search efficiency or phase stability evidence.')
(base/'xxxii-rc-vc-one-step/current-cost-ledger.json').write_text(json.dumps(out,indent=2)+'\n');print(out['total_API'])
