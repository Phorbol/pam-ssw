"""Bounded real Cu13 integration qualification; not a search ranking."""
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase.calculators.emt import EMT
from ase.io import read, write
from pamssw.standalone import ASESurface, SSWConfig, PaperGAConfig, run_ga_ssw
from pamssw.standalone.legacy_descriptor import cluster_descriptor
from pamssw.standalone.recovered_direction import (
    RecoveredDirectionController, RecoveredDirectionSettings)

out = Path(__file__).parent
protocol = json.loads((out / 'protocol.json').read_text())
plan = json.loads(Path(protocol['parent_protocol']).read_text())
initial = read(protocol['input'], ':')
bonds = {(int(a), int(b)): v for a, b, v in plan['bond_lengths']}
refs = [cluster_descriptor(a.numbers, a.positions, bonds, plan['neighbor_range']) for a in initial]
kwargs = dict(initial=initial, groups=None, references=refs,
    descriptor_bonds=bonds, descriptor_weights=plan['descriptor_weights'],
    neighbor_range=plan['neighbor_range'], proposal_bond_limits={},
    descriptor_row_order='full_fingerprint', config=PaperGAConfig(**protocol['ga']),
    ssw_config=SSWConfig(**protocol['ssw']), max_evaluations=protocol['max_evaluations'])
settings = RecoveredDirectionSettings(**protocol['recovered_direction'])
original_initialize = RecoveredDirectionController.initialize
original_begin = RecoveredDirectionController.begin_escape


def run(name, *, direction, checkpoint=None, pause=False):
    surface = ASESurface(EMT())
    controllers, starts, escapes = [], [], []
    def initialize(self, *args, **kw):
        controllers.append(self)  # Retain identities; no id reuse in diagnostic.
        starts.append(dict(controller=len(controllers)-1,
            fresh=self._pair is None and self._group is None))
        return original_initialize(self, *args, **kw)
    def begin(self, *args, **kw):
        escapes.append(next(i for i, c in enumerate(controllers) if c is self))
        return original_begin(self, *args, **kw)
    start = time.monotonic()
    with patch.object(RecoveredDirectionController, 'initialize', initialize), \
         patch.object(RecoveredDirectionController, 'begin_escape', begin):
        result = run_ga_ssw(**kwargs, surface=surface, rng=np.random.default_rng(3),
            recovered_direction=direction, checkpoint=checkpoint,
            checkpoint_callback=(lambda state: True) if pause else None)
    (out / f'{name}.pkl').write_bytes(pickle.dumps(result, protocol=4))
    # Check all observed geometries, including rejected/noneligible ones.
    checks = []
    for obs in result.observations:
        atoms = obs.result.atoms
        e, f = ASESurface(EMT()).evaluate(atoms)
        fmax = float(np.linalg.norm(f, axis=1).max())
        checks.append(dict(id=obs.id, phase=obs.phase, eligible=obs.eligible_for_archive,
            energy=e, fmax=fmax, force_pass=fmax <= kwargs['config'].quench_fmax,
            composition_pass=bool(len(atoms)==13 and np.all(atoms.numbers==29)),
            boundary_pass=bool(not atoms.pbc.any() and np.array_equal(atoms.cell, initial[0].cell)),
            energy_agreement=abs(e-obs.result.energy), parent_ids=obs.parent_ids))
    row = dict(name=name, status=result.status, cumulative_requests=result.evaluation_requests,
        segment_requests=surface.requests, fresh_requests=len(checks),
        seconds=time.monotonic()-start, checks=checks, starts=starts, escapes=escapes,
        stages=[dict(phase=s.phase, status=s.status, requests=s.evaluation_requests)
                for s in result.stages], failures=[asdict(f) for f in result.failures])
    (out / f'{name}.json').write_text(json.dumps(row, indent=2))
    if result.observations:
        write(out / f'{name}.extxyz', [o.result.atoms for o in result.observations])
    print(name, result.status, surface.requests, 'requests', len(checks), 'checks', flush=True)
    return result

baseline = run('baseline', direction=None)
full = run('full', direction=settings)
first = run('first', direction=settings, pause=True)
resumed = run('resumed', direction=settings, checkpoint=first.checkpoint)
# Analysis is separate so a readout defect never requires rerunning the PES.
from analyze import analyze
analyze(out)
