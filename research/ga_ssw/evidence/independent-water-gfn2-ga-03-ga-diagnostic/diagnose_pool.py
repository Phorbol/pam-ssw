"""Pure geometry/RNG replay, no calculator imports or evaluations."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import ast
import json
import numpy as np
from ase import Atoms
from pamssw.standalone.ga_operators import _molecular_pool, propose_type3
from pamssw.standalone.paper_reference import sample_initial_direction
from pamssw.standalone.population import partition

here = Path(__file__).resolve().parent
source = here.parent / 'independent-water-gfn2-ga-03'
run = json.loads((source / 'result.json').read_text())
config = json.loads((source / 'config.json').read_text())
groups = tuple(tuple(range(i, i+3)) for i in range(0, 45, 3))
observations = run['observations']
def atoms(data):
    return Atoms(**data)

rng = np.random.default_rng(config['seed'])
errors = []
for walk in run['walks'][:3]:
    direction = sample_initial_direction(atoms(walk['initial']['atoms']), rng, mode='paper')
    errors.append(float(np.max(np.abs(direction - np.asarray(walk['records'][0]['initial_direction'])))))
# Archive after three quick walks is observations1,0,5: lower energy replacements.
rows = [dict(id=i, energy=observations[i]['result']['energy'], sims=observations[i]['projection'],
             atoms=atoms(observations[i]['result']['atoms'])) for i in (1, 0, 5)]
regions = partition(rows, 1, rng, max_draws=10000)
parents = [rows[i] for region in regions for i in region]
state_before_pool = deepcopy(rng.bit_generator.state)
sons, daughters = _molecular_pool([r['atoms'].copy() for r in parents],
    np.array([r['energy'] for r in parents]), groups, rng, 100)

matches = []
for i, (sp, son) in enumerate(sons):
    required = set(range(15)) - set(son.group_ids)
    for j, (dp, daughter) in enumerate(daughters):
        if set(daughter.group_ids) == required:
            matches.append((i, j, sp, dp))
matching = {(i, j) for i, j, _, _ in matches}
after_pool = deepcopy(rng.bit_generator.state)
first = None
for attempt in range(1, 10001):
    i = int(np.floor(rng.random() * len(sons)))
    j = int(np.floor(rng.random() * len(daughters)))
    if (i, j) in matching:
        first = dict(attempt=attempt, son_index=i, daughter_index=j,
                     son_parent=sons[i][0], daughter_parent=daughters[j][0])
        break
p = len(matches) / (len(sons) * len(daughters))
summary = dict(source=str(source), calculator_requests=0, initial_direction_replay_max_errors=errors,
    actual_parent_observation_ids=[r['id'] for r in parents], son_pool=len(sons), daughter_pool=len(daughters),
    compatible_pairs=len(matches), all_possible_pairs=len(sons)*len(daughters), exact_success_probability=p,
    expected_draws=1/p, probability_zero_matches_in_100=(1-p)**100,
    same_parent_compatible_pairs=sum(sp==dp for _,_,sp,dp in matches),
    different_parent_compatible_pairs=sum(sp!=dp for _,_,sp,dp in matches),
    unique_son_subsets=len(set(g.group_ids for _,g in sons)),
    unique_daughter_subsets=len(set(g.group_ids for _,g in daughters)), first_match=first,
    config_issue='G=4 produces only one crossover and zero mutations per batch; max_batches=2 cannot reach four candidates',
    budget_trials=[])
limits = {ast.literal_eval(k):v for k,v in config['proposal_bond_limits'].items()}
for batches, pairs in ((2,100),(2,10000),(4,10000)):
    replay = np.random.default_rng(); replay.bit_generator.state=deepcopy(state_before_pool)
    out = propose_type3([r['atoms'].copy() for r in parents], [r['energy'] for r in parents], groups,
        (0,)*15, replay, min_ga=4, bond_limits=limits, max_batches=batches,
        max_cut_attempts=100, max_pair_attempts=pairs)
    summary['budget_trials'].append(dict(max_batches=batches,max_pair_attempts=pairs,status=out.status,
        candidates=len(out.candidates),batches=out.batches,rejected=out.rejected_by_bond_limit,reason=out.reason,
        operators=dict(Counter(c.operation for c in out.candidates)),
        child_min_pair_distances=[float(c.atoms.get_all_distances()[np.triu_indices(45,1)].min()) for c in out.candidates]))
# Independent check on the three initial-quench structures, without using quick replacements.
replay=np.random.default_rng(config['seed'])
initial=sorted(observations[:3],key=lambda o:o['result']['energy'])
s,d=_molecular_pool([atoms(o['result']['atoms']) for o in initial],
    np.array([o['result']['energy'] for o in initial]),groups,replay,100)
freq=Counter(tuple(sorted(set(range(15))-set(g.group_ids))) for _,g in s)
compatible=sum(freq[g.group_ids] for _,g in d)
summary['initial_quench_only_pool']=dict(seed=config['seed'], parent_ids=[o['id'] for o in initial],
    compatible_pairs=compatible, all_possible_pairs=len(s)*len(d), success_probability=compatible/(len(s)*len(d)))
(here/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
