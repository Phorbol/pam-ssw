"""Bounded TYPE0 GA/LS lifecycle audit on two ASE G2 C4H6 references.

This is workflow evidence on GFN2-xTB, not an efficiency or LS advantage study.
"""
import argparse, hashlib, json, shutil, time, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import networkx as nx
from ase.collections import g2
from ase.geometry import distance
from ase.io import write
from tblite.ase import TBLite


def dump(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def molecular_components(atoms, bonds):
    graph = nx.Graph(); graph.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pair = tuple(sorted((int(atoms.numbers[i]), int(atoms.numbers[j]))))
            if np.linalg.norm(atoms.positions[i] - atoms.positions[j]) <= bonds[pair] + .1:
                graph.add_edge(i, j)
    return nx.number_connected_components(graph), [sorted(c) for c in nx.connected_components(graph)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree('pamssw', out/'source'/'pamssw', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out/'runner.py')
    sys.path.insert(0, str(out/'source'))
    from pamssw.standalone import ASESurface, PaperGAConfig, SSWConfig, LSSettings, NativeLSSettings, run_ga_ssw
    from pamssw.standalone.legacy_descriptor import cluster_descriptor
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    global serial
    from research.ga_ssw.compare_vc_arms import serial
    molecules = {name: g2[name].copy() for name in ("butadiene", "cyclobutene")}
    expanded = molecules["butadiene"].copy(); expanded.positions *= 1.1
    references = [molecules["butadiene"], molecules["cyclobutene"], expanded]
    bond_limits = {k: 0.7 * v for k, v in HC_BOND_LENGTHS.items()}
    cfg = PaperGAConfig(quick_steps=1, generations=1, generation_steps=1,
        fine_steps=1, ga_candidates=1, regions=1, fine_regions=1,
        quench_fmax=.01, quench_steps=400, proposal_max_batches=1,
        proposal_max_cut_attempts=100, proposal_max_pair_attempts=100,
        partition_max_draws=100, projection_tolerance=1e-5, energy_window=100.,
        proposal_type=0, proposal_max_insertion_attempts=100)
    ssw = SSWConfig(width=.1, rotation_bias=100., max_gaussians=3,
        temperature_K=150., fmax=.01, bias_fmax=.1, relax_steps=400,
        fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        rotation_solver='dimer', cluster_frame='direction_only',
        direction_sampling='global', quench_optimizer='safe-lbfgs-total')
    arms = {
        'ssw': None,
        'paper_ls': LSSettings(HC_BOND_ENERGIES,
            {k: v + .1 for k, v in HC_BOND_LENGTHS.items()}, target_per_atom=.7),
        'native_derived_ls': NativeLSSettings(HC_BOND_ENERGIES, HC_BOND_LENGTHS,
            target_mev_per_atom=700.),
    }
    dump(out/'plan.json', dict(systems=list(molecules), seeds=[11, 29], arms=list(arms),
        config=asdict(cfg), ssw_config=asdict(ssw), max_search_evaluations=4000,
        per_arm_wall_seconds=90, fresh_maxima=10, descriptor='HC_BOND_LENGTHS',
        references='butadiene, cyclobutene, and butadiene uniformly expanded 1.1; '
                   'third reference is a descriptor reference only, not an independent physical sample',
        collision='proposal cutoff is fixed at 0.7*bond length; heuristic retained from existing runner',
        backend='GFN2-xTB accuracy .001; CPU; single-thread environment',
        scope='small-molecule GA+LS workflow development validation; no LS advantage or GA efficiency claim'))
    for name, atoms in molecules.items(): write(out/(name+'.extxyz'), atoms)
    write(out/'reference-expanded-1.1.extxyz', expanded)
    dump(out/'source-manifest.json', {'git_head': __import__('subprocess').check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}})
    dump(out/'ls-settings.json', arms)
    results = []
    for name, atoms in [('c4h6_population', list(molecules.values()))]:
        bonds = {k: v for k, v in HC_BOND_LENGTHS.items()}
        refs = [cluster_descriptor(a.numbers, a.positions, bonds, 1.2) for a in references]
        for arm, ls in arms.items():
            for seed in (11, 29):
                runout = out/f'{name}-{arm}-seed{seed}'; runout.mkdir()
                started = time.monotonic(); row = dict(system=name, arm=arm, seed=seed)
                ledger = runout/'evaluations.jsonl'
                class Bounded(ASESurface):
                    denied = 0
                    def evaluate(self, a):
                        if self.requests >= 4000 or time.monotonic()-started >= 90:
                            self.denied += 1; raise RuntimeError('bounded search cap or wall limit')
                        try:
                            e, f = super().evaluate(a)
                        except Exception as error:
                            with ledger.open('a') as h:
                                h.write(json.dumps(serial(dict(request=self.requests, error=repr(error), atoms=a)))+'\n')
                            raise
                        with ledger.open('a') as h: h.write(json.dumps(serial(dict(request=self.requests, energy=e, forces=f, atoms=a)))+'\n')
                        return e, f
                surface = Bounded(TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0))
                try:
                    result = run_ga_ssw([a.copy() for a in atoms],
                        surface, groups=None, references=refs, descriptor_bonds=bonds,
                        descriptor_weights=(1.,)*6, neighbor_range=1.2,
                        proposal_bond_limits=bond_limits, config=cfg, ssw_config=ssw,
                        rng=np.random.default_rng(seed), ls=ls, max_evaluations=4000,
                        structure_matcher=lambda a, b: bool(distance(a, b, permute=True) / np.sqrt(len(a)) <= .1))
                    dump(runout/'result.json', result)
                    stage_rows = [dict(phase=x.phase, generation=x.generation, cycle=x.cycle,
                        status=x.status, requests=x.evaluation_requests, details=x.details) for x in result.stages]
                    row.update(status=result.status, search_requests=surface.requests,
                        archive=len(result.archive), observations=len(result.observations),
                        accounted=result.evaluation_requests==surface.requests,
                        failures=[x.reason for x in result.failures], stages=stage_rows,
                        ls_updates=sum(1 for w in result.walks for r in w.records if r.ls_update is not None),
                        ls_responses=sum(1 for w in result.walks for r in w.records if r.energy_response is not None),
                        walk_statuses=[w.status for w in result.walks])
                    fresh = ASESurface(TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0)); checks=[]
                    for item in sorted(result.archive, key=lambda x:x['energy'])[:10]:
                        e, f = fresh.evaluate(item['atoms']); checks.append(dict(id=item['id'], energy=e,
                            energy_error=e-item['energy'], fmax=float(np.linalg.norm(f, axis=1).max()),
                            force_qualified=bool(np.linalg.norm(f, axis=1).max() <= .01),
                            molecular_components=molecular_components(item['atoms'], bonds)))
                    dump(runout/'fresh-checks.json', checks); row['fresh_requests']=fresh.requests; row['fresh_checks']=checks; row['unverified_archive']=max(0,len(result.archive)-len(checks))
                except Exception as exc:
                    row.update(status='exception', error=repr(exc), search_requests=surface.requests)
                row['wall_seconds'] = time.monotonic()-started
                dump(runout/'summary.json', row); results.append(row); dump(out/'summary.json', results)
                print(name, arm, seed, row.get('status'), row.get('search_requests'), flush=True)


if __name__ == '__main__': main()
