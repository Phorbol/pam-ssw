import json
from pathlib import Path
import runner
from ase.io import read
p=json.loads((runner.HERE/'plan.json').read_text())
runner.check_frozen(p)
rows=[]
for case in p['input_sources']:
    a=read(runner.HERE/'inputs'/f'{case}.extxyz')
    cfg=runner.effective_config(p,case)
    recovered_cfg=runner.replace(cfg, **p['recovered_config_overrides'])
    assert recovered_cfg.pre_rotation_hvp is None
    settings=runner.RecoveredRotationSettings(**p['recovered_rotation'])
    class NoPES:
        requests=0
        reached=False
        def evaluate(self, atoms):
            self.reached=True
            raise RuntimeError('PREFLIGHT_REACHED_ORACLE')
    for method, conf in [('ritz',cfg),('recovered',recovered_cfg)]:
        surface=NoPES()
        try:
            runner.run_ssw(a,surface,steps=0,config=conf,rng=runner.np.random.default_rng(41),recovered_rotation=settings if method=='recovered' else None)
        except Exception as error:
            assert type(error) is RuntimeError and str(error)=='PREFLIGHT_REACHED_ORACLE', repr(error)
            assert surface.reached, 'did not reach oracle sentinel'
        else:
            raise AssertionError('Expected oracle sentinel')
    assert a.pbc.all() and not a.constraints
    assert cfg.cluster_frame=='translation_only'
    rows.append(dict(case=case,natoms=len(a),fmax=cfg.fmax,bias_fmax=cfg.bias_fmax,lbfgs_memory=cfg.lbfgs_memory,rotation_max_calls=settings.max_force_calls))
(runner.HERE/'preflight-v4.json').write_text(json.dumps(dict(status='passed',pes_requests=0,cases=rows),indent=2)+'\n')
print('preflight passed; zero PES',flush=True)
