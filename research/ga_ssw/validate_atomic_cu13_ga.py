"""Strict validation of every returned observation, independent of GA descriptor."""
import json
from pathlib import Path
from .validate_cu13_eckart import main as validate


def main():
    base=Path('research/ga_ssw/evidence/atomic-cu13-ga')
    view=base/'strict-input';view.mkdir(exist_ok=False)
    (view/'script.py').write_text(Path(__file__).read_text())
    for seed in (3,17):
        result=json.loads((base/f'{seed}.json').read_text())
        (view/f'{seed}-ga.json').write_text(json.dumps(dict(result=dict(minima=[o['result'] for o in result['observations']])),indent=2)+'\n')
    validate(view)
    summary=json.loads((view/'strict-validation/summary.json').read_text())
    rows=[]
    for seed in (3,17):
        observations=json.loads((base/f'{seed}.json').read_text())['observations']
        strict=json.loads((view/f'strict-validation/{seed}-ga.json').read_text())
        before=set();offspring=set();all_groups=set();by_operator={}
        for o,q in zip(observations,strict):
            g=q['fingerprint_group']
            if not q['qualified']:continue
            all_groups.add(g)
            if o['phase'] in ('initial_quench','quick'):before.add(g)
            if o['phase']=='offspring_quench':
                offspring.add(g);by_operator.setdefault(o['operator'],set()).add(g)
        rows.append(dict(seed=seed,groups=len(all_groups),groups_before_ga=sorted(before),offspring_groups=sorted(offspring),new_offspring_groups=sorted(offspring-before),by_operator={k:sorted(v) for k,v in by_operator.items()}))
    final=dict(runs=rows,combined_groups=summary['fingerprint_groups'],validation_requests=summary['total_validation_requests'],
        positive_internal_spectra=all(s['eigenvalues'][0]>0 for r in summary['representatives'] for s in r['spectra']),
        scope='Cu13 EMT short workflow from three known minima; no matched-cost SSW control; structural fingerprint not a complete basin identity')
    (base/'strict-summary.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))
if __name__=='__main__':main()
