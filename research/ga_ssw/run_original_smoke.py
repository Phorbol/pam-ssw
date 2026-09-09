"""Prepare/run one bounded original GA-SSW water workflow, not a performance test.

Run from a shell with `module load intel/mpi/2021.13`. No scheduler submission.
The destination must not exist; original inputs/binaries are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(); root=args.reference_root.resolve(); run=args.run_dir.resolve()
    example=root/'GA-SSW_examples_run/global_exploration'
    template=example/'input-templates/TYPE3-(H2O)15'
    run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(template,run/'input');
    shutil.copytree(example/'GA-SSW/input/ssw_gaussian',run/'input/ssw_gaussian'); (run/'soft').mkdir();(run/'bin').mkdir();(run/'input/mc').mkdir()
    sgn=root/'GA-SSW_program/sgn.jar'; nna=example/'GA-SSW/input/nna1.jar'; native=root/'GA-SSW_program/lasp'
    shutil.copyfile(sgn,run/'soft/sgn.jar');shutil.copyfile(nna,run/'input/nna1.jar')
    # No binary modification: loader executes the original ELF, preserving its mode.
    wrapper=run/'bin/lasp'
    wrapper.write_text('#!/bin/bash\nexec /lib64/ld-linux-x86-64.so.2 '+shlex.quote(str(native))+' "$@"\n');wrapper.chmod(0o755)
    changes=dict(CPU='1',Memory='1',TaskNum='1',SSWTaskNum='1',CombineMultiUnitNum='1',GANum='6',
                 OPTSSWStep='1',QuickSSWStep='2',SSWStep='2',QuickSSWIterations='1',FineSSWIterations='1',
                 PopClassifyNum='1',LaspPath=str(wrapper))
    config=(run/'input/configure.non').read_text()
    for key,value in changes.items():
        config,count=re.subn(r'^'+re.escape(key)+r'=.*$',lambda m:key+'='+value,config,flags=re.M)
        if count!=1: raise ValueError(f'expected exactly one {key}, got {count}')
    (run/'input/configure.non').write_text(config)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                   JAVA_TOOL_OPTIONS='-Xmx512m -XX:ActiveProcessorCount=1')
    env['PATH']=str(root/'tools/jdk-17.0.2/bin')+os.pathsep+env['PATH']
    cmd=[str(root/'tools/jdk-17.0.2/bin/java'),'-jar','sgn.jar']
    inputs=[sgn,nna,native,*sorted(x for x in template.rglob('*') if x.is_file()),
            *sorted(x for x in (example/'GA-SSW/input/ssw_gaussian').rglob('*') if x.is_file())]
    provenance=dict(command=cmd,cwd=str(run/'soft'),config_overrides=changes,timeout_seconds=120,
                    source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},
                    random_seed='Original unseeded Java RNG; actual references and generated structures retained',
                    purpose='Bounded interface smoke run. Original initial OPT steps are multiplied by 3; no performance claim.')
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if not args.execute: print('Prepared',run);return
    supervisor=Path(__file__).resolve().with_name('bounded_process.py')
    os.execve(os.sys.executable,[os.sys.executable,str(supervisor),'--cwd',str(run/'soft'),
                    '--timeout','120','--log',str(run/'stdout.txt'),'--status',str(run/'status.json'),
                    '--',*cmd],env)



if __name__=='__main__': main()
