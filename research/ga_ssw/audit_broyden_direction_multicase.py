"""Offline source, request ledger and certificate audit; no calculator calls."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    out=Path('research/ga_ssw/evidence/broyden-direction-multicase-20260912')
    rows=json.loads((out/'summary.json').read_text());assert len(rows)==24
    manifest=json.loads((out/'source-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()==manifest['input_sha256']
    inputs=json.loads((out/'inputs.json').read_text())
    for case,source in json.loads((out/'input-sources.json').read_text()).items():
        p=Path(source['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
        assert json.loads(p.read_text())['initial']['atoms']==inputs[case]
    groups=defaultdict(lambda:dict(arms=0,requests=0,certificate_requests=0,qualified=0,failed=0))
    for row in rows:
        folder=out/f"{row['case']}-{row['method']}-seed{row['seed']}"
        ledger=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
        paid=[r for r in ledger if r['kind']!='search_denial'];assert len(paid)==row['search_requests']<=101
        assert [r['request'] for r in paid]==list(range(1,len(paid)+1))
        cert=[json.loads(s) for s in (folder/'certificate.jsonl').read_text().splitlines()]
        assert len(cert)==row['certificate_requests']==2
        assert max(cert[0]['fmax'],paid[0]['fmax'])<=.01
        assert all(r['kind']=='search' for r in paid+cert)
        result=row['result'];assert result['force_calls']==len(paid)
        n=np.asarray(result['direction']);np.testing.assert_allclose(np.linalg.norm(n),1,atol=1e-12)
        centre=np.asarray(inputs[row['case']]['positions'])
        np.testing.assert_array_equal(cert[0]['atoms']['positions'],centre)
        np.testing.assert_allclose(cert[1]['atoms']['positions'],centre+1e-4*n,atol=1e-14)
        hv=(np.asarray(cert[0]['projected_forces'])-np.asarray(cert[1]['projected_forces']))/1e-4
        curvature=float(np.vdot(n,hv));residual=float(np.linalg.norm(hv-curvature*n))
        np.testing.assert_allclose(residual,row['certificate']['residual_norm'],atol=1e-10)
        assert row['certificate']['qualified']==(residual<=.02)
        g=groups[row['method']];g['arms']+=1;g['requests']+=len(paid);g['certificate_requests']+=2;g['qualified']+=int(residual<=.02);g['failed']+=int(row['status']!='completed')
    report=dict(groups=dict(groups),total_requests=sum(g['requests']+g['certificate_requests'] for g in groups.values()),
                checked='24 source-matched arms, all ledger sequences, all initial forces, fresh direct certificates; no backend failures',
                boundary='Fixed-geometry root-solving qualification only, not unique basins or global search efficacy')
    (out/'root-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
