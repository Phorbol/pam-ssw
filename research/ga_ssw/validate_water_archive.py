"""Fresh original-LASP single-point checks of supplied H30O15 archives.

Same supplied NN oracle, independent process evaluations; not DFT validation.
No search or optimization is performed by this script.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import numpy as np


def read_arc(path):
    lines=path.read_text().splitlines();indices=[i for i,s in enumerate(lines) if ' CORE ' in s]
    symbols=[lines[i].split()[0] for i in indices]
    if len(indices)!=45 or symbols.count('H')!=30 or symbols.count('O')!=15:raise ValueError('Requires one H30O15 ARC frame')
    xyz=np.array([[float(x) for x in lines[i].split()[1:4]] for i in indices])
    energy=float(next(s for s in lines if s.strip().startswith('Energy')).split()[3])
    return lines,indices,symbols,xyz,energy


def displaced_arc(lines,indices,xyz):
    result=list(lines)
    for i,p in zip(indices,xyz):
        parts=result[i].split();result[i]=' '.join([parts[0],*[f'{x:.12f}' for x in p],*parts[4:]])
    return '\n'.join(result)+'\n'


def single_point_config(text):
    for key,value in [('ssw.sswsteps','0'),('ssw.output','T'),('ssw.printevery','T')]:
        text,count=re.subn(r'(?im)^([ \t]*'+re.escape(key)+r'[ \t]+)\S+',lambda m:m[1]+value,text)
        if count!=1:raise ValueError(f'Expected one {key}, got {count}')
    run_type=re.findall(r'(?im)^[ \t]*run_type[ \t]+(\d+)',text)
    if run_type!=['5']:raise ValueError('Single-point water adapter requires Run_type 5')
    return text


class Oracle:
    def __init__(self,root,output):
        self.root=root.resolve();self.output=output.resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.template=self.root/'GA-SSW_examples_run/global_exploration/input-templates/TYPE3-(H2O)15'
        self.native=self.root/'GA-SSW_program/lasp';self.calls=0
        self.config=single_point_config((self.template/'lasp.in').read_text())
    def evaluate(self,text,label):
        directory=self.output/label;directory.mkdir(exist_ok=False)
        (directory/'input.arc').write_text(text);(directory/'lasp.in').write_text(self.config)
        (directory/'H2O_pf.pot').symlink_to(self.template/'H2O_pf.pot')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,str(Path(__file__).resolve().with_name('bounded_process.py')),'--cwd',str(directory),'--timeout','10',
             '--log',str(directory/'stdout.txt'),'--status',str(directory/'status.json'),'--','/lib64/ld-linux-x86-64.so.2',str(self.native)]
        # exec supervisor in a child; forward interruption and allow cleanup instead of killing it.
        child=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL)
        try: child.wait()
        except KeyboardInterrupt:
            child.terminate();child.wait();raise
        status=json.loads((directory/'status.json').read_text())
        log=(directory/'lasp.out').read_text()
        if status['state']!='completed' or status['returncode']!=0 or status['cleanup_survivors'] or 'SSW all done' not in log:
            raise RuntimeError(f'Unsuccessful single point: {directory}')
        output=(directory/'allfor.arc').read_text().splitlines()
        if len([s for s in output if s.strip().startswith('For ')])!=1 or any(s.strip() for s in output[47:]):
            raise ValueError('Expected exactly one 45-atom force frame')
        energy=float(output[0].split()[3]);forces=np.array([[float(x) for x in s.split()] for s in output[2:47]])
        if forces.shape!=(45,3) or not np.isfinite(forces).all() or not np.isfinite(energy):raise ValueError('Invalid force output')
        if (directory/'input.arc').read_text()!=text:raise ValueError('Input was modified')
        self.calls+=1
        return energy,forces,status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',required=True,type=Path)
    p.add_argument('--archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--directional-check',action='store_true');args=p.parse_args()
    oracle=Oracle(args.reference_root,args.output);rows=[];first=None
    for arc in sorted(args.archive.glob('*.arc'),key=lambda p:int(p.stem)):
        lines,indices,symbols,xyz,stored=read_arc(arc)
        energy,forces,status=oracle.evaluate(arc.read_text(),arc.stem)
        z=np.array(symbols);oxygen=xyz[z=='O'];hydrogen=xyz[z=='H']
        oh=np.linalg.norm(hydrogen[:,None,:]-oxygen[None,:,:],axis=2);nearest=oh.argmin(axis=1)
        row=dict(id=arc.stem,source=str(arc.resolve()),source_sha256=hashlib.sha256(arc.read_bytes()).hexdigest(),
                 stored_energy=stored,energy=energy,energy_difference=energy-stored,
                 max_force_component=float(np.abs(forces).max()),max_atom_force=float(np.linalg.norm(forces,axis=1).max()),
                 rms_force_component=float(np.sqrt(np.mean(forces**2))),net_force=forces.sum(axis=0).tolist(),
                 forces=forces.tolist(),nearest_oxygen_hydrogen_counts=np.bincount(nearest,minlength=15).tolist(),
                 nearest_OH_distance_range=[float(oh.min(axis=1).min()),float(oh.min(axis=1).max())],wall_seconds=status['wall_seconds'])
        rows.append(row)
        if first is None:first=(lines,indices,xyz,forces)
    directional=[]
    if args.directional_check:
        lines,indices,xyz,forces=first;v=forces/np.linalg.norm(forces)
        for h in [.01,.005,.002]:
            ep,_,_=oracle.evaluate(displaced_arc(lines,indices,xyz+h*v),f'fd-{h}-plus')
            em,_,_=oracle.evaluate(displaced_arc(lines,indices,xyz-h*v),f'fd-{h}-minus')
            directional.append(dict(h_angstrom=h,central_energy_derivative=(ep-em)/(2*h),negative_force_projection=-float(np.sum(forces*v))))
    sources=[oracle.native,oracle.template/'H2O_pf.pot',oracle.template/'lasp.in']
    report=dict(records=rows,directional_checks=directional,fresh_single_point_processes=oracle.calls,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                scope='Fresh single-point evaluations with same NN/ELF; not independent PES or DFT; counts are process requests, not internal E/F calls')
    (oracle.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),force_component_range=[min(r['max_force_component'] for r in rows),max(r['max_force_component'] for r in rows)],
                         max_energy_difference=max(abs(r['energy_difference']) for r in rows),directional_checks=directional),indent=2))


if __name__=='__main__':main()
