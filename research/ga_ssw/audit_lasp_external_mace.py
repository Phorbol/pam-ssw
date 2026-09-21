"""Verify real LASP's exported E/F against the persistent MACE callback ledger."""
import json,argparse
from pathlib import Path
import numpy as np


def audit(root):
    entries=[json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()]
    summaries=json.loads((root/'summary.json').read_text());issues=[];rows=[]
    for s in summaries:
        records=[r for r in entries if r.get('case')==s['case'] and r['response']['ok']]
        lines=(root/s['case']/'allfor.arc').read_text().splitlines();frames=[]
        for i,line in enumerate(lines):
            if not line.lstrip().startswith('For '):continue
            e=float(line.split()[3]);f=np.array([[float(v) for v in item.split()] for item in lines[i+2:i+62]])
            if f.shape!=(60,3):raise ValueError('bad LASP force frame')
            match=min(records,key=lambda r:abs(r['energy']-e))
            ee=abs(match['energy']-e);fe=float(np.max(np.abs(np.array(match['forces'])-f)))
            frames.append(dict(request=match['request'],energy_error=ee,max_force_component_error=fe))
            if ee>5.1e-8 or fe>5.1e-11:issues.append(s['case']+': E/F not equal within printed precision')
        inp=json.loads((root/s['case']/'input.json').read_text())
        xerr=float(np.max(np.abs(np.array(inp['positions'])-np.array(records[0]['positions'])))) if records else None
        if xerr is None or xerr>5.1e-11:issues.append(s['case']+': initial coordinates not equal within external.coord precision')
        if not frames or not s['ssw_done'] or s['process']['returncode']!=0 or s['process']['cleanup_survivors']:issues.append(s['case']+': incomplete native run')
        rows.append(dict(case=s['case'],callback_requests=len(records),initial_coordinate_error=xerr,exported_frames=frames))
    return dict(rows=rows,issues=issues,total_callback_requests=len(entries),scope='Actual fixed-cell nonperiodic LASP consumes MACE E/F; no force convergence or global-search success inferred.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
