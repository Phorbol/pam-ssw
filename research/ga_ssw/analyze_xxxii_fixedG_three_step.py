"""Compare predeclared erfc residual prediction with three saved FD steps; no PES."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'research/ga_ssw/evidence/xxxii-lammps-qualification-fixedG-three-step-v2'
PRED = ROOT / 'research/ga_ssw/evidence/xxxii-erfc-cell-rc-audit/result.json'
result = json.loads((SRC / 'result.json').read_text())
assert result['status'] == 'completed'
pred = json.loads(PRED.read_text())['predictions']
rows = []
for name, expected in pred.items():
    ds = sorted((d for d in result['derivatives'] if d['direction'] == name), key=lambda d: -d['h'])
    assert len(ds) == 3
    errors = [d['finite_difference'] - d['analytic'] for d in ds]
    remainders = [e - expected for e in errors]
    extrapolated = [(4 * errors[i + 1] - errors[i]) / 3 for i in range(2)]
    rows.append(dict(direction=name, steps=[d['h'] for d in ds], prediction=expected,
                     FD_minus_analytic=errors, remaining=remainders,
                     Richardson_errors=extrapolated,
                     Richardson_remaining=[e - expected for e in extrapolated]))
logs = '\n'.join(p.read_text() for p in sorted(SRC.glob('engine-*.log')))
g = sorted(set(float(v) for v in re.findall(r'G vector \(1/distance\) = ([0-9.eE+-]+)', logs)))
k = sorted(set(re.findall(r'KSpace vectors: actual max1d max3d =\s*(\d+)', logs)))
calls = json.loads((SRC / 'calls.json').read_text())
report = dict(PES_calls=0, measured_PES_calls=sum(c['status'] == 'completed' for c in calls),
              runtime_G=g, runtime_kspace_counts=k, comparisons=rows,
              invariance=result['invariance'], inputs=dict(measurement=str(SRC), prediction=str(PRED)),
              limits='Fixed G does not freeze reciprocal shell membership. No correction was applied to engine forces or stress; no claim of exact conservative oracle or global performance.')
(SRC / 'derivative-attribution.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
