"""Compare independent Python RNG with archived native-instruction outputs."""
import json
from pathlib import Path
import numpy as np
from pamssw.standalone.native_random import native_vmb2


def main():
    root = Path('research/ga_ssw/evidence')
    rows = []
    sources = ('native-vmb2-20260912.json',
               'native-vmb2-seed-quantization-20260917.json',
               'native-vmb2-top-boundary-20260917.json')
    for source in sources:
        artifact = json.loads((root/source).read_text())
        for index, case in enumerate(artifact.get('cases', [artifact])):
            actual = native_vmb2(case['initial'], np.asarray(case['mask'], bool),
                                 case['seed_input'], temperature=case['temperature'])
            error = float(np.max(abs(actual-np.asarray(case['native']))))
            rows.append(dict(source=source, index=index, max_abs_error=error,
                             seed_input=case['seed_input'], passed=error < 1e-14))
    report = dict(scope=__doc__, cases=rows, passed=all(r['passed'] for r in rows))
    (root/'native-random-python-comparison-20260917.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), passed=report['passed'],
                         max_abs_error=max(r['max_abs_error'] for r in rows))))
    assert report['passed']


if __name__ == '__main__':
    main()
