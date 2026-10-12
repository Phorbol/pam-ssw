#!/usr/bin/env python3
"""Private native control-flow probe; preparation is default, execution explicit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012'
BINARY = Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp')
EXPECTED_SHA = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
GEOMETRY = ROOT / 'research/ga_ssw/evidence/lj-bulk-stacking-readiness-20261007/geometry.py'
ARC_WRITER = ROOT / 'research/ga_ssw/evidence/lasp-explicit-seed-qualification-20261007/run-1664773/runner.py'
MPI_LIB = '/opt/devtools/intel/oneapi/mpi/2021.13/lib'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(out):
    from ase.io import write
    if sha(BINARY) != EXPECTED_SHA:
        raise ValueError('ELF differs from the inspected address contract')
    out.mkdir(parents=True, exist_ok=False)
    executable = out / 'lasp'
    shutil.copyfile(BINARY, executable)
    if sha(executable) != EXPECTED_SHA:
        raise ValueError('diagnostic executable copy differs from uploaded ELF')
    executable.chmod(0o700)
    geometry = load(GEOMETRY, 'native_vc_geometry')
    writer = load(ARC_WRITER, 'native_vc_arc_writer')
    atoms, _ = geometry.build_pair((2, 2, 6))
    # Native ljperipot literals, documented in native-lj-stress-producer.md.
    # Scale the existing sigma=1 geometry; no new potential is introduced.
    a12, a6 = 150830.0777733409, 388.36848195153647
    sigma = (a12 / a6) ** (1.0 / 6.0)
    atoms.set_cell(atoms.cell.array * sigma, scale_atoms=True)
    atoms.wrap()
    write(out / 'input.extxyz', atoms)
    arc = writer.arc_text(atoms).splitlines()
    # Existing writer fixes the C60 storage cell at 50 A. Retain its atom-row
    # contract, replacing only the cell line with this periodic geometry.
    arc[4] = 'PBC ' + ' '.join(f'{x:.14f}' for x in atoms.cell.cellpar())
    (out / 'input.arc').write_text('\n'.join(arc) + '\n')
    shutil.copy2(EVIDENCE / 'lasp.in', out / 'lasp.in')
    for name in ('trace.py', 'protocol.md'):
        shutil.copy2(EVIDENCE / name, out / name)
    shutil.copy2(ROOT / 'research/ga_ssw/bounded_process.py', out / 'bounded_process.py')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    (out / 'trace.gdb').write_text(
        'set pagination off\nset confirm off\nset language c\n'
        'set disable-randomization off\nset breakpoint pending off\n'
        'source trace.py\nrun\n')
    provenance = {
        'status': 'prepared_not_executed', 'scope': 'native control-flow diagnostic only',
        'limits': {'native_wall_seconds': 120, 'cpu_job_seconds': 300,
                   'input_outer_steps': 2, 'distinct_observed_outer_steps': 2,
                   'trace_events': 500, 'arms': 1, 'input_run_type': 15},
        'binary': str(BINARY), 'binary_sha256': sha(BINARY),
        'executed_binary': str(executable), 'executed_binary_sha256': sha(executable),
        'geometry_source': str(GEOMETRY), 'geometry_sha256': sha(GEOMETRY),
        'arc_writer_source': str(ARC_WRITER), 'arc_writer_sha256': sha(ARC_WRITER),
        'native_pair_scale': {'a12': a12, 'a6': a6, 'sigma_A': sigma,
                             'source': 'docs/research/native-lj-stress-producer.md'},
        'input': {'formula': atoms.get_chemical_formula(), 'pbc': atoms.pbc.tolist(),
                  'cell_A': atoms.cell.array.tolist(), 'input_arc_sha256': sha(out / 'input.arc')},
        'artifacts': {name: sha(out / name) for name in
                      ('lasp.in', 'trace.py', 'trace.gdb', 'bounded_process.py', Path(__file__).name)},
        'runtime': {'python': sys.executable, 'mpi_lib': MPI_LIB,
                    'I_MPI_FABRICS': 'shm', 'I_MPI_PIN': '0'},
        'qualification': 'initial native LJ convergence and cell-event coverage unverified',
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def execute(out, provenance):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = MPI_LIB + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.update(I_MPI_FABRICS='shm', I_MPI_PIN='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(out / 'bounded_process.py'), '--cwd', str(out),
               '--timeout', '120', '--log', str(out / 'child.log'),
               '--status', str(out / 'process.json'), '--',
               'gdb', '-q', '-nx', '-batch', '-x', 'trace.gdb', str(out / 'lasp')]
    provenance['command'] = command
    provenance['status'] = 'execution_requested'
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    completed = subprocess.run(command, env=env, check=False)
    provenance['supervisor_returncode'] = completed.returncode
    process = json.loads((out / 'process.json').read_text())
    provenance['child_process'] = process
    provenance['status'] = 'execution_returned_unqualified'
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    # Raw completion is never a lifecycle conclusion. Root reviews trace,
    # parser echoes, native failures, process status and cleanup survivors.
    return completed.returncode or (0 if process['state'] == 'completed' and
        process['returncode'] == 0 and not process['cleanup_survivors'] else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, help='new directory, never overwritten')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    provenance = prepare(out)
    if args.execute:
        return execute(out, provenance)
    print(json.dumps({'status': provenance['status'], 'out': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
