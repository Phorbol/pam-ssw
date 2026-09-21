"""Check native accepted-point ABI by recorded-E/F replay, then new EMT certificates."""
import json, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian
from .probe_native_lbfgs_emt import Oracle,ELF,load_elf

class CheckedOracle(Oracle):
    def hook(self,u,pc,size,data):
        if pc==0x6e8dad and self.integer(0x791b938)==1:
            assert np.array_equal(self.x(),np.asarray(self.current['positions'])), 'accepted native X differs from last E/F'
        super().hook(u,pc,size,data)

def main():
    out=Path('research/ga_ssw/evidence/cu13-failed-quench-native-lbfgs');summary=json.loads((out/'summary.json').read_text());_,segments=load_elf(ELF);rows=[];start=time.monotonic()
    for row in summary['runs']:
        data=json.loads((out/f'{Path(row["source"]).stem}-step{row["step"]}.json').read_text());o=CheckedOracle(segments,np.array(data['evaluations'][0]['positions']))
        for v in data['evaluations']:
            assert np.array_equal(o.x(),np.array(v['positions'])), 'requested trajectory mismatch'
            o.advance(v['energy'],np.array(v['forces']),v)
        assert len(o.accepted)==row['accepted_iterates'];assert o.calls==row['calls']
        original=json.loads((Path('research/ga_ssw/evidence/cu13-direction-only')/row['source']).read_text());record=next(r for r in original['result']['records'] if r['index']==row['step'])
        pam=json.loads((Path('research/ga_ssw/evidence/cu13-failed-quench-pam')/f'{Path(row["source"]).stem}-step{row["step"]}-safe-lbfgs-total.json').read_text())
        assert np.array_equal(data['evaluations'][0]['positions'],pam['evaluations'][0]['positions'])
        assert abs(data['evaluations'][0]['energy']-pam['evaluations'][0]['energy'])<1e-12
        certificate=None
        if row['status']=='converged':
            point=data['accepted'][-1];a=Atoms(**record['last_atoms']);a.positions=np.array(point['positions']);surface=ASESurface(EMT());e,f=surface.evaluate(a)
            for t in record['climb']:
                term=ProjectedGaussian(np.array(t['center']),np.array(t['direction']),t['width'],t['weight']);be,bf=term.evaluate(a);e+=be;f+=bf
            certificate=dict(requests=surface.requests,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),energy_error=abs(e-point['energy']),force_error=float(np.max(abs(f-np.array(point['forces'])))))
            assert certificate['max_force']<=.01 and certificate['energy_error']<1e-12 and certificate['force_error']<1e-12
        rows.append(dict(source=row['source'],step=row['step'],accepted_position_checks=len(o.accepted),requested_position_checks=len(data['evaluations']),same_start_as_safe_total=True,certificate=certificate));del o
        print(row['source'],row['step'],'verified',flush=True)
    report=dict(rows=rows,extra_emt_requests=sum(r['certificate']['requests'] for r in rows if r['certificate']),instruction_replay_ef_requests=0,seconds=time.monotonic()-start,accepted_position_checks=sum(r['accepted_position_checks'] for r in rows),mcstep_calls=sum(r['calls'].get('0x6eb690',0) for r in summary['runs']))
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');(out/'verification-script.py').write_text(Path(__file__).read_text());print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
