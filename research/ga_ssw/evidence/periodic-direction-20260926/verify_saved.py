"""Finish failed reporting without repeating the preserved physical searches."""
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ledger',HERE.parent/'periodic-rotation-priority-20260923/ledger.py')
ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)

def main():
    from ase import Atoms
    from mace.calculators import MACECalculator
    import torch
    from pamssw.standalone.paper_reference import run_ssw,load_ssw_checkpoint
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=HERE/'verification';out.mkdir(exist_ok=False)
    calc=MACECalculator(model_paths='/home/gengjianrui/.cache/mace/mace-omat-0-small.model',
        head='omat_pbe',device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    rows=[];fresh_count=0
    for folder in sorted((HERE/'runs').glob('*-*')):
        cp=load_ssw_checkpoint(folder/'continuous.pkl')
        raw=json.loads((folder/'result.json').read_text())
        row=dict(case=folder.name,search=cp.evaluation_requests,status=cp.status,
            landings=len(cp.minima)-1,stages=sum(len(r.climb) for r in cp.records),qualified=False)
        try:
            assert cp.status=='completed' and len(cp.records)==3
            for k,minimum in enumerate(cp.minima):
                a=minimum.atoms.copy();a.calc=calc
                e=a.get_potential_energy();f=a.get_forces();fresh_count+=1
                assert fresh_count<=24
                ledger.append(out/(folder.name+'-fresh.jsonl'),dict(index=k,energy=e,forces=f))
                assert abs(e-minimum.energy)<1e-6
                assert np.max(np.linalg.norm(f,axis=1))<=cp.config.fmax
                np.testing.assert_array_equal(a.cell.array,cp.initial.atoms.cell.array)
            row['best_energy']=min(m.energy for m in cp.minima)
            if folder.name.endswith('local_memory'):
                tape=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
                case_name=folder.name.removesuffix('-local_memory')
                source=next(c for c in json.loads((HERE/'cases.json').read_text())['cases'] if c['case']==case_name)
                if source['source'].endswith('.json'):
                    initial=Atoms(**json.loads(Path(source['source']).read_text())['initial']['atoms'])
                else:
                    from ase.io import read
                    initial=read(source['source'])
                class Replay:
                    requests=0
                    def evaluate(self,a):
                        if self.requests>=len(tape):raise RuntimeError('extra oracle request')
                        event=tape[self.requests]
                        np.testing.assert_array_equal(a.positions,event['atoms']['positions'])
                        np.testing.assert_array_equal(a.cell.array,event['atoms']['cell'])
                        np.testing.assert_array_equal(a.pbc,event['atoms']['pbc'])
                        self.requests+=1
                        return event['energy'],np.asarray(event['forces'])
                replay=Replay();path=out/(folder.name+'-split.pkl')
                run_ssw(initial,replay,steps=1,config=cp.config,rng=np.random.default_rng(26092631),
                    recovered_direction=cp.recovered_direction_state.settings,checkpoint_path=path)
                boundary=load_ssw_checkpoint(path)
                run_ssw(initial,replay,steps=2,config=cp.config,rng=np.random.default_rng(999),
                    checkpoint=boundary,checkpoint_path=path)
                resumed=load_ssw_checkpoint(path)
                assert replay.requests==len(tape)
                assert ledger._jsonable(resumed)==ledger._jsonable(cp),'checkpoint state mismatch'
                row['replayed']=replay.requests
                row['state_equal']=True
                row['routes']=dict(Counter(s.get('recovered_direction',{}).get('route','unreported') for r in cp.records for s in r.climb))
                row['history_used']=any(len(r.climb)>1 for r in cp.records)
            row['qualified']=True
        except Exception as error:row['error']=repr(error)
        rows.append(row);ledger.dump(out/'summary.json',dict(rows=rows,fresh=fresh_count))
        print(json.dumps(row),flush=True)
    assert all(r['qualified'] for r in rows),'preserved qualification failure'

if __name__=='__main__':main()
