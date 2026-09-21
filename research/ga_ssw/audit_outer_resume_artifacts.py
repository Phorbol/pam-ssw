"""Offline audit of outer SSW/LS checkpoint artifacts; never calls a PES."""
import argparse, json, pickle
from pathlib import Path
import numpy as np

CASES = {
    'cu13-ssw': 'outer-resume-multicase-20260912/cu13-ssw',
    'cu31_fixed-ssw': 'outer-resume-multicase-20260912/cu31_fixed-ssw',
    'bicyclobutane-ssw': 'outer-resume-molecular-20260912/bicyclobutane-ssw',
    'butadiene-paper_ls': 'outer-resume-molecular-20260912-v2/butadiene-paper_ls',
    'butadiene-native_ls': 'outer-resume-molecular-20260912-v2/butadiene-native_ls',
}

def audit(root, rel):
    p = root / rel
    a = json.loads((p/'continuous-ledger.json').read_text())
    b = json.loads((p/'split-ledger.json').read_text())
    continuous = json.loads((p/'continuous-result.json').read_text())
    resumed = json.loads((p/'resumed-result.json').read_text())
    # Boundary pickle is trusted local evidence and supplies the exact paid
    # cost at the split, including cases whose runner stopped before summary.
    cp = pickle.load((p/'boundary.pkl').open('rb'))
    first_n = cp.evaluation_requests
    row = dict(case=p.name.rsplit('-', 1)[0], arm=p.name.rsplit('-', 1)[1])
    same_prefix = len(a) >= first_n and a[:first_n] == b[:first_n]
    first_resume = dict(index=first_n, same_atoms=False, position_max=None,
                        energy_abs=None, force_max=None)
    if first_n < len(a) and first_n < len(b):
        x, y = a[first_n], b[first_n]
        xp, yp = np.asarray(x['atoms']['positions']), np.asarray(y['atoms']['positions'])
        first_resume.update(same_atoms=bool(np.array_equal(xp, yp)),
            position_max=float(np.max(np.abs(xp-yp))),
            energy_abs=abs(x.get('energy', 0.)-y.get('energy', 0.)),
            force_max=float(np.max(np.abs(np.asarray(x.get('forces', []))-np.asarray(y.get('forces', [])))))
            if 'forces' in x and 'forces' in y else None)
    fresh = json.loads((p/'fresh.json').read_text())
    return dict(path=rel, case=row['case'], arm=row['arm'],
        continuous_requests=len(a), first_requests=first_n, resumed_requests=len(b)-first_n,
        ledger_lengths=(len(a), len(b)),
        prefix_exact=same_prefix, first_resume=first_resume,
        final_position_max_difference=float(np.max(np.abs(np.asarray(continuous['current']['positions'])-
                                                         np.asarray(resumed['current']['positions'])))),
        accepted_equal=([r['accepted'] for r in continuous['records']] ==
                        [r['accepted'] for r in resumed['records']]),
        cost_consistent=(continuous['evaluation_requests'] ==
                         continuous['initial']['evaluation_requests'] +
                         sum(r['evaluation_requests'] for r in continuous['records']) == len(a) and
                         resumed['evaluation_requests'] == len(b)),
        fresh_qualified=(len(fresh) == len(continuous['minima']) + len(resumed['minima']) and len(fresh) > 0 and
                         all(c.get('fixed_cell') and c.get('fmax', 1e99) <= .01 and
                             abs(c.get('energy_error', 1e99)) <= 1e-8 for c in fresh)),
        fresh_checks=fresh)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=Path('research/ga_ssw/evidence'))
    ap.add_argument('--output', type=Path, required=True); args=ap.parse_args()
    rows=[audit(args.root, rel) for rel in CASES.values()]
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print(json.dumps(rows, indent=2))
if __name__ == '__main__': main()
