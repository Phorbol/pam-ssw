"""Reproduce the finite-coordinate NumPy/native comparison, no PES calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from research.ga_ssw.native_vapor_reference import vapor_reference
from research.ga_ssw.probe_native_vapor_oracle import load_elf, run, ELF_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(ELF_DEFAULT)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':
        raise ValueError('unexpected native ELF')
    rows = []
    for source in json.loads(Path(args.inputs).read_text()):
        mode, x = source['mode'], source['positions']
        native = run(segments, x, mode, 1.7)
        reference = vapor_reference(x, 1.7, repair=bool(mode))
        if native['status'] != 'ok':
            raise RuntimeError(str(native))
        rows.append(dict(index=source['index'], n=len(x), mode=mode,
            scalar_error=abs(native['return']-reference['scalar']),
            coordinate_error=float(np.max(np.abs(np.asarray(native['positions'])-reference['positions'])))))
    report = dict(elf_sha256=digest, cases=len(rows), rows=rows,
        scope='original routine with allocator stubs versus independent NumPy geometry; no PES or native caller parity')
    report['passed'] = all(r['scalar_error'] <= 1e-10 and r['coordinate_error'] <= 1e-10 for r in rows)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(report['cases'], report['passed'])


if __name__ == '__main__':
    main()
