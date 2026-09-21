"""Replay archived failed rotations on real GFN2 surfaces, original versus verified stop."""
import argparse,json,shutil,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ase import Atoms


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    parent=Path('research/ga_ssw/evidence/c4h6-ls-reaction-coverage-20260912').resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    shutil.copytree(parent/'source',out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py');shutil.copy2('research/ga_ssw/verified_ritz_research.py',out/'verified_ritz_research.py')
    sys.path.insert(0,str(out/'source'))
    from pamssw.standalone.surface import ASESurface
    from pamssw.standalone.cluster_frame import ClusterFrame
    from pamssw.standalone.native_rotation import RotationQuadraticBias
    from pamssw.standalone.direction import reference_soft_mode as original
    from pamssw.standalone.softening import FrozenBondSoftening,LSResponseState
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES,HC_BOND_LENGTHS
    from pamssw.standalone.ls_native_reference import NativeLSSettings,NativeLSRuntime
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_verified_ritz',out/'verified_ritz_research.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from tblite.ase import TBLite
    manifest=dict(status='prepared',parent=str(parent),selection='all 39 rotation_failed outer records in six archived C4H6 arms',
        method='same failed center, stored evaluated presweep anchor/coefficient, reconstructed frozen LS, unchanged .02 tolerance and 1e-4 A finite displacement; main budget100 minus archived pre force calls',
        scope='local real-surface replay, not end-to-end search efficiency or native CBD parity',
        ls_reconstruction='replay saved selected-current and true energy_response updates; restore delta as response*N, retaining roundoff boundary')
    def dump(path,v):path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    dump(out/'manifest.json',manifest)
    if not args.execute:return
    results=[];start=time.monotonic()
    for folder in sorted(parent.glob('butadiene-*-seed*')):
        trajectory=json.loads((folder/'result.json').read_text());current=Atoms(**trajectory['initial']['atoms'])
        frozen=response=None
        if '-paper_ls-' in folder.name:
            lengths={k:v+.1 for k,v in HC_BOND_LENGTHS.items()}
            frozen=FrozenBondSoftening.from_atoms(current,bond_energies=HC_BOND_ENERGIES,bond_lengths=lengths,initial_fraction=.03,xi=.2)
            response=LSResponseState(.7,learning_rate=1.8)
        elif '-native_ls-' in folder.name:
            lengths=HC_BOND_LENGTHS;response=NativeLSRuntime(current,NativeLSSettings(HC_BOND_ENERGIES,lengths,target_mev_per_atom=700.));frozen=response.frozen
        for record in trajectory['records']:
            if record['status']=='rotation_failed':
                event=record['climb'][-1];atoms=Atoms(**record['last_atoms']);frame=ClusterFrame(atoms)
                anchor=np.array(event['actual_anchor']);a=event['actual_rotation_bias'];budget=100-event['pre_rotation']['force_calls']
                bias=RotationQuadraticBias(atoms.positions,anchor,a)
                row=dict(arm=folder.name,step=record['index'],gaussian=event['index'],archived=event['main_rotation'],main_hvp_budget=budget)
                for name,solver in [('original',original),('verified',module.reference_soft_mode)]:
                    surface=ASESurface(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));ledger=[]
                    def evaluate(candidate):
                        candidate=candidate.copy();candidate.positions=frame.positions(candidate.positions)
                        e,f=surface.evaluate(candidate)
                        if frozen is not None:
                            de,df=frozen.evaluate(candidate);e+=de;f+=df
                        f=frame.project(f)
                        # The rank-one bias acts after the physical/frozen-LS frame wrapper.
                        de,df=bias.evaluate(candidate);e+=de;f+=df
                        ledger.append(dict(request=surface.requests,energy=e,forces=f.tolist(),positions=candidate.positions.tolist()))
                        return e,f
                    try:
                        result=solver(atoms,anchor,fd_step=1e-4,max_hvp=budget,residual_tol=.02,evaluate=evaluate,finite_difference='forward')
                        data=asdict(result);data['direction']=result.direction.tolist();data['surface_requests']=surface.requests
                        assert result.force_calls==surface.requests and result.hvp_calls<=budget
                        row[name]=data
                    except Exception as error:row[name]=dict(error=repr(error),surface_requests=surface.requests)
                    dump(out/f'{folder.name}-step{record["index"]}-{name}-ledger.json',ledger)
                results.append(row);dump(out/'results.json',results)
            landing=record.get('landing')
            if record['accepted']:current=Atoms(**landing['atoms'])
            if response is not None and record['energy_response'] is not None:
                frozen=response.update(frozen,current,energy_before=0.,energy_after=record['energy_response']*len(current),bond_energies=HC_BOND_ENERGIES,bond_lengths=lengths)
    manifest.update(status='completed',replays=len(results),requests=sum(x[k]['surface_requests'] for x in results for k in ('original','verified')),seconds=time.monotonic()-start)
    dump(out/'manifest.json',manifest);print(json.dumps(manifest))
if __name__=='__main__':main()
