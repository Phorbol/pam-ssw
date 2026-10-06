"""Exercise actual staged server/client/finalization with dummy E/F; no LASP/MACE."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def fixture(prepared):
    original_run=subprocess.run
    original_modules={k:sys.modules.get(k) for k in ['torch','mace','mace.calculators','vacuum_geometry']}
    torch=types.ModuleType('torch')
    torch.set_num_threads=torch.manual_seed=torch.use_deterministic_algorithms=lambda *a,**k:None
    torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True))
    class Dummy(Calculator):
        implemented_properties=['energy','forces']
        def __init__(self,**kwargs):super().__init__();self.r_max=6.0
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results={'energy':0.0,'forces':np.zeros((len(atoms),3))}
    calc=types.ModuleType('mace.calculators');calc.__file__=str(Path(__file__).resolve());calc.MACECalculator=Dummy
    mace=types.ModuleType('mace');mace.calculators=calc
    sys.modules.update({'torch':torch,'mace':mace,'mace.calculators':calc})
    try:
        with tempfile.TemporaryDirectory(dir=Path.home(),prefix='.c60-fixture-') as tmp:
            out=Path(tmp)/'run';shutil.copytree(prepared/'seed-26100791',out)
            plan=json.loads((out/'plan.json').read_text());plan['request_cap']=2
            (out/'plan.json').write_text(json.dumps(plan))
            vacuum=load(out/'vacuum_geometry.py','vacuum_geometry')
            runner=load(out/'runner.py','fixture_native_runner')
            atoms=read(out/'input.extxyz')
            coord='\n'.join(' '.join(map(str,row)) for row in np.eye(3)*50)+'\n'
            coord+='\n'.join('C '+' '.join(map(str,row)) for row in atoms.positions)+'\n'
            replies=[]
            def fake_supervisor(command,env=None,**kwargs):
                assert command[1]==str(out/'bounded_process.py')
                assert env['LASP_MACE_SOCKET'].startswith(str(Path.home()))
                for _ in range(3):
                    (out/'external.coord').write_text(coord)
                    proc=original_run(['bash',str(out/'lasp.external.sh')],cwd=out,env=env,capture_output=True,text=True)
                    replies.append(proc.returncode)
                (out/'process.json').write_text(json.dumps({'state':'completed','returncode':29,'cleanup_survivors':[]}))
                (out/'lasp.out').write_text('fixture, not native output\n')
                return types.SimpleNamespace(returncode=0)
            runner.subprocess.run=fake_supervisor
            runner.main(out)
            summary=json.loads((out/'summary.json').read_text())
            ledger=[json.loads(s) for s in (out/'request.jsonl').open()]
            assert replies[:2]==[0,0] and replies[2]!=0
            assert summary['paid_ef']==2 and summary['requests']==3 and summary['actual_calculate_calls']==1
            assert ledger[-1]['error_kind']=='request_cap' and not ledger[-1]['paid_ef']
            assert sum(r['actual_calculate_calls'] for r in ledger)==1
            provenance=json.loads((out/'runtime-provenance.json').read_text())
            assert provenance['client']==str(out/'client.py') and provenance['runner']==str(out/'runner.py')
            assert not (out/'external.ene').exists() # denied client removes stale output
            print(json.dumps(dict(status='full_path_fixture_passed',dummy_paid=2,dummy_actual=1,
                cap_denials=1,client_stale_response_removed=True,mace_calls=0,physical_pes_calls=0)))
    finally:
        subprocess.run=original_run
        for key,previous in original_modules.items():
            if previous is None:sys.modules.pop(key,None)
            else:sys.modules[key]=previous


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    fixture(p.parse_args().prepared.resolve())
