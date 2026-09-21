"""Summarize recorded Gaussian policy choices, without extra PES calls."""
import argparse
import json
from pathlib import Path
import statistics


def stats(values):
    return dict(min=min(values),median=statistics.median(values),max=max(values)) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();rows=[]
    for p in sorted(args.root.glob('*/comparison/*/search-result.json')):
        data=json.loads(p.read_text())
        stages=[e for r in data['records'] if r.get('atomic')
                for e in r['atomic']['climb'] if 'weight' in e and 'width' in e]
        policies=[e['gaussian_policy'] for e in stages if 'gaussian_policy' in e]
        rows.append(dict(arm=p.parts[-4],seed=data['seed'],source=str(p),
            recorded_stages=len(stages),completed_stages=sum('true_energy' in e for e in stages),
            widths=stats([e['width'] for e in stages]),weights=stats([e['weight'] for e in stages]),
            policy_records=len(policies),width_clamped=sum(e['width_clamped'] for e in policies),
            weight_clamped=sum(e['weight_clamped'] for e in policies),
            zero_weight=sum(e['weight']==0 for e in stages),
            k_true=stats([e['k_true'] for e in policies]),k_inner=stats([e['k_inner'] for e in policies]),
            estimated_center_postcurvature=stats([e['k_inner']-e['weight']/e['width']**2 for e in policies])))
    output=dict(scope='Returned stage records only; unfinished pending stages excluded. Clamp counts use policy_records denominator. Not search efficacy.',rows=rows)
    (args.root/'gaussian-policy-summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
