"""Probe record selection and small-displacement normalization before generator."""
import json
from pathlib import Path
import numpy as np
from research.ga_ssw.probe_native_direction_update import UpdateOracle
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
import hashlib


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    threshold = UpdateOracle(segments).readd(0x4a43838)
    rows = []
    for n in (2, 5):
        current = np.arange(3*n, dtype=float).reshape(n, 3)*.1
        refs = np.array([current+1., current-2., current.copy()])
        for selected in (1, 2, 3):
            actual = UpdateOracle(segments).probe(current, refs, np.ones_like(current), [.2,.3,.4], selected=selected)
            delta = current - refs[selected-1]
            expected = delta/np.linalg.norm(delta) if np.linalg.norm(delta)>0 else delta
            error = float(np.max(np.abs(actual['normalized_displacement']-expected)))
            rows.append(dict(n=n, selected=selected, displacement=delta.tolist(),
                             actual=actual, error=error, passed=error<1e-14))
        for amplitude in (0., .000999, .001, .001001):
            current = np.zeros((n,3)); current[0,0] = amplitude
            actual = UpdateOracle(segments).probe(current, np.zeros_like(current), np.ones_like(current), [.2,.3,.4])
            expected = np.zeros_like(current)
            if amplitude*amplitude > threshold:
                expected[0,0] = 1.
            error = float(np.max(np.abs(actual['normalized_displacement']-expected)))
            rows.append(dict(n=n, amplitude=amplitude, actual=actual, error=error, passed=error<1e-14))
    result = dict(sha256=ELF_SHA256, threshold_squared_norm=threshold, cases=rows,
                  passed=all(r['passed'] for r in rows),
                  scope='three-record selection and small displacement before generator; original normalizer; same-shape allocation and memset stubs; no constraints, PES or main')
    path=Path('research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/record-boundary-v2.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} passed; threshold={threshold}")
    assert result['passed']


if __name__ == '__main__':
    main()
