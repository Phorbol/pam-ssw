"""Post-hoc labels of cold-checked TiO2 structures against published templates.

Does not change the frozen anatase success gate or evaluate a potential.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from ase.io import read
from pamssw.standalone.periodic_ga_reference import pymatgen_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    panel = json.loads(args.panel.read_text())
    plan = json.loads((HERE / 'plan.json').read_text())
    qualification = json.loads((HERE.parent / 'tio2-phase-qualification-20261007/plan.json').read_text())
    source = Path(qualification['cases'][0]['path']).parent
    filenames = ['tio2-b.extxyz', 'anatase.extxyz', 'tio2-ii.extxyz', 'rutile.extxyz',
                 'brookite.extxyz', 'phase-87.extxyz', 'phase-139.extxyz']
    refs = {name: read(source / name) for name in filenames}
    matchers = {label: pymatgen_identity(**values) for label, values in plan['identity_tolerances'].items()}
    rows = []
    for arm in panel['arms']:
        if not arm['result_present']:
            continue
        run = Path(arm['path'])
        for path in sorted((run / 'fresh').glob('*.extxyz')):
            atoms = read(path)
            rows.append(dict(slot=arm['slot'], path=str(path), role=atoms.info.get('fresh_role'),
                natoms=len(atoms), matches={label: {name: bool(matcher(atoms, ref))
                for name, ref in refs.items()} for label, matcher in matchers.items()}))
    payload = dict(scope='post-hoc approximate template identities of cold-checked images; no new success gate or DFT/minimum claim',
        input_analysis=str(args.panel), input_analysis_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        sources={name: dict(path=str(source / name), sha256=hashlib.sha256((source / name).read_bytes()).hexdigest())
                 for name in filenames}, identity_tolerances=plan['identity_tolerances'], rows=rows,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), analysis_pes_requests=0)
    (args.out / 'classification.json').write_text(json.dumps(payload, indent=2)+'\n')
    for row in rows:
        print(row['slot'], row['role'], {key: [name for name, match in values.items() if match]
                                        for key, values in row['matches'].items()})


if __name__ == '__main__':
    main()
