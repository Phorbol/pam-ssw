"""Rebuild the listed TYPE4 research cost ledger; no PES or scheduler calls.

Precheck35 is the retained earlier preparation total, not a new request count.
Only present terminal per-arm result files are charged here; running progress
is reported separately and must not be mistaken for final totals.
"""
import json
from pathlib import Path


def main():
    base=Path(__file__).resolve().parent/'evidence'
    rows=[dict(name='earlier_precheck_initial_fresh',requests=35,source='docs/research/MAINLINE.md historical TYPE4 precheck record',status='completed')]
    paths=[('raw_input_two_mask_search','type4-source-direction-mace-v100/*/result.json','total_requests'),
           ('strict_quench_and_hessians','type4-direction-strict-qualification/*/result.json','requests'),
           ('unbiased_negative_mode_controls','type4-initial-negative-mode/result.json','total_requests'),
           ('certified_start_qualification','type4-certified-start-control/qualification/*/result.json','requests'),
           ('certified_start_two_mask_search','type4-certified-start-control/search/*/result.json','total_requests'),
           ('repeat_landing_qualification','type4-multistep-heldout/landing_qualification/*/result.json','requests'),
           ('heldout_multistep_search','type4-multistep-heldout/seed*/*/result.json','total_requests'),
           ('heldout_best_qualification','type4-heldout-best-qualification/*/result.json','requests')]
    for name,pattern,key in paths:
        for path in sorted(base.glob(pattern)):
            data=json.loads(path.read_text());rows.append(dict(name=name+'/'+path.parent.name,requests=data[key],source=str(path.relative_to(base)),status='recorded_error' if data.get('error') else 'terminal_result'))
    output=dict(rows=rows,total_recorded_requests=sum(r['requests'] for r in rows),scope='listed TYPE4 MACE E/F requests including failed work and independent validation; not equivalent to SSW search-only cost; running unfinalized work excluded')
    target=base/'type4-source-direction-mace-v100/listed-cost-ledger.json';target.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
