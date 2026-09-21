"""Reconstruct saved Fe7C3 soft-prequench gradients without new PES calls."""
import json
from pathlib import Path
import numpy as np
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from research.ga_ssw.compare_fe7c3_ls_frozen_quenches import _atoms, _softening

ROOT=Path('research/ga_ssw/evidence/fe7c3-80-ls-filter-comparison/comparison')

def main():
    rows=[]
    for path in sorted(ROOT.glob('*/result.json')):
        source=json.loads(path.read_text())
        calls=[json.loads(line) for line in (path.parent/'evaluations.jsonl').open()]
        for record in source['records'][1:]:
            if not record.get('frozen_gaussians'):continue
            chart=SymmetricLogStrainChart(_atoms(record['chart_reference']),strain_length=source['joint_config']['strain_length'])
            q=np.asarray(record['frozen_gaussians'][0]['center']);atoms=chart.unpack(q)
            soft=_softening(record['frozen_softening']);de,df,ds=soft.evaluate_stress(atoms)
            matches=[]
            for call in calls:
                if not call.get('charged') or 'forces' not in call:continue
                err=max(float(np.max(np.abs(np.asarray(call['atoms']['positions'])-atoms.positions))),
                        float(np.max(np.abs(np.asarray(call['atoms']['cell'])-atoms.cell.array))))
                if err<1e-12:matches.append((call,err))
            assert matches,'No raw physical geometry matching prequench endpoint'
            call,err=matches[0];e=call['energy'];f=np.asarray(call['forces']);s=np.asarray(call['stress'])
            gradient=chart.evaluate(q,lambda a:(e+de,f+df,s+ds),pressure=source['joint_config']['pressure']).gradient
            rows.append(dict(arm=source['arm'],seed=source['seed'],outer_index=record['index'],
                source_request=call['request'],matching_requests=[c['request'] for c,_ in matches],geometry_error=err,
                softened_atomic_fmax=float(np.linalg.norm(f+df,axis=1).max()),
                softened_cell_gradient_l2=float(np.linalg.norm(gradient[-6:])),
                softened_stress_max=float(np.abs(s+ds).max()),physical_stress_max=float(np.abs(s).max()),
                LS_stress_max=float(np.abs(ds).max()),joint_gtol=source['joint_config']['gradient_tol']))
    output=Path('research/ga_ssw/evidence/ls-energy-filter/prequench-cell-gradient.json')
    output.write_text(json.dumps(dict(scope='zero-new-PES frozen reconstruction',rows=rows),indent=2)+'\n')
    print(len(rows),'saved preparations audited; cell gradient range',min(r['softened_cell_gradient_l2'] for r in rows),max(r['softened_cell_gradient_l2'] for r in rows))
if __name__=='__main__':main()
