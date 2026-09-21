"""Original isolated LASP numerical kernel on the existing five frozen starts.

Only callback allocation, memcpy and accepted-step stopping are adapted.
BFGSDRIVER, LASP main/protection, force scaling and PES are not executed.
"""
import json,sys,time,signal,hashlib
from pathlib import Path
import numpy as np


def main(out):
    plan=json.loads((out/'plan.json').read_text());sys.path[:0]=[str(out/'support'),plan['frozen_source']]
    from ledger_helpers import CountedSurface,atoms_from,dump
    from research.ga_ssw.probe_native_lbfgs_emt import Oracle,ELF,load_elf,DATA
    from pamssw.standalone.surface import ASESurface
    from mace.calculators import MACECalculator
    blob,segments=load_elf(ELF)
    if hashlib.sha256(blob).hexdigest()!=plan['elf_sha256']:raise ValueError('ELF differs')
    if hashlib.sha256(Path(plan['model']).read_bytes()).hexdigest()!=plan['model_sha256']:raise ValueError('model differs')
    class NativeOracle(Oracle):
        def alloc(self,data):
            if self.cursor+len(data)>DATA+0x100000 and not getattr(self,'extended',False):
                self.uc.mem_map(DATA+0x100000,0x100000);self.extended=True
            return super().alloc(data)
        def hook(self,u,pc,size,data):
            if pc==0x6e8dad and self.integer(0x791b938)==1:
                err=float(np.max(np.abs(self.x()-np.array(self.current['positions']))))
                if err>1e-12:raise ValueError('accepted-coordinate mismatch')
                self.accepted.append(dict(self.current,accepted_coordinate_error=err))
                if self.current['max_force']<=plan['fmax']:self.stop_reason='force_qualified'
                elif len(self.accepted)>=plan['steps']:self.stop_reason='step_limit'
                if self.stop_reason:u.emu_stop()
                return
            return super().hook(u,pc,size,data)
    kwargs=dict(model_paths=plan['model'],device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    begin=time.monotonic();calc=MACECalculator(**kwargs);fresh=ASESurface(MACECalculator(**kwargs))
    dump(out/'initialization.json',{'seconds':time.monotonic()-begin});rows=[]
    def timeout(*args):raise TimeoutError('arm wall limit')
    signal.signal(signal.SIGALRM,timeout)
    for start in plan['starts']:
        folder=out/start['name'];folder.mkdir();a=atoms_from(start['atoms']);oracle=NativeOracle(segments,a.positions)
        surface=CountedSurface(calc,folder/'requests.jsonl',cap=plan['request_cap'],wall=plan['arm_wall_seconds'])
        status='request_cap';error=None;flag=None;current=None;first=None;check=None;fresh_before=fresh.requests
        signal.setitimer(signal.ITIMER_REAL,plan['arm_wall_seconds'])
        try:
            for k in range(plan['request_cap']):
                a.positions=oracle.x();e,f=surface.evaluate(a)
                current=dict(request=surface.requests,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),positions=a.positions.tolist())
                if first is None:first=current
                if k==0 and current['max_force']<=plan['fmax']:status='force_qualified';break
                flag=oracle.advance(e,f,current)
                if oracle.stop_reason:status=oracle.stop_reason;break
                if flag!=1:status='native_stopped_unqualified' if flag==0 else 'native_failure';break
        except Exception as exc:status='wall_cap' if isinstance(exc,TimeoutError) else 'error';error=repr(exc)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        endpoint=oracle.accepted[-1] if oracle.accepted else first
        if endpoint:
            a.positions=np.array(endpoint['positions'])
            try:
                e,f=fresh.evaluate(a);fm=float(np.linalg.norm(f,axis=1).max())
                check=dict(energy=e,max_force=fm,qualified=fm<=plan['fmax'],energy_error=e-endpoint['energy'])
                if status=='force_qualified' and not check['qualified']:status='fresh_qualification_failed'
            except Exception as exc:check=dict(error=repr(exc));status='fresh_qualification_failed'
        row=dict(name=start['name'],status=status,error=error,requests=surface.requests,
            fresh_requests=fresh.requests-fresh_before,accepted_steps=len(oracle.accepted),native_flag=flag,
            native_info=oracle.integer(0x791b938),calls=oracle.calls,fresh=check,endpoint=endpoint,
            seconds=time.monotonic()-surface.started)
        dump(folder/'accepted.json',oracle.accepted);dump(folder/'summary.json',row);rows.append(row);dump(out/'summary.json',rows)
        print(start['name'],status,surface.requests,check,flush=True)
        del oracle


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
