"""Read-only summaries of preregistered molecular trial; no PES evaluation."""
import json
from pathlib import Path
import numpy as np
import networkx as nx
from ase import Atoms

BASE=Path('research/ga_ssw/evidence/c4h6-safe-ssw-ls')


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    rows=[]
    full=[r for r in summary['runs'] if r['phase']=='full']
    common=min((r['search_requests'] for r in full),default=0)
    for row in full:
        folder=BASE/f'full-{row["seed"]}-{row["arm"]}'
        result=json.loads((folder/'result.json').read_text())
        checks=json.loads((folder/'fresh-checks.json').read_text())
        initial=np.array(checks[0]['atoms']['positions']);initial-=initial.mean(axis=0)
        for check in checks:
            x=np.array(check['atoms']['positions']);x-=x.mean(axis=0)
            u,_,vt=np.linalg.svd(x.T@initial);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
            atoms=Atoms(**check['atoms'])
            g=nx.Graph();g.add_nodes_from(range(len(atoms)));g.add_edges_from(check['edges'])
            check['fragment_formulas']=[atoms[sorted(c)].get_chemical_formula() for c in nx.connected_components(g)]
            check['fixed_index_kabsch_rmsd']=float(np.sqrt(np.mean(np.sum((x@u@d@vt-initial)**2,axis=1))))
        valid=[c for c in checks if c['force_pass']]
        prefix=[c for c in valid if c['cumulative_search_requests']<=common]
        preparations=[json.loads(line) for line in (folder/'ls-preparation.jsonl').read_text().splitlines()]
        data=dict(**row,common_request_budget=common,prefix_certified_records=len(prefix),
            completed_certified_records=len(valid),
            all_energy_relative_initial=[c['energy']-checks[0]['energy'] for c in checks],
            all_carbon_dihedrals=[c['carbon_chain_dihedral'] for c in checks],
            max_kabsch_rmsd=max(c['fixed_index_kabsch_rmsd'] for c in checks),
            graph_changes=sum(not c['initial_connectivity_isomorphic'] for c in checks),
            min_threshold_margin=min(abs(d['threshold_margin']) for c in checks for d in c['distances']),
            components=[c['components'] for c in checks],
            disconnected_records=[dict(index=c['index'],formula=c['fragment_formulas'],energy_relative_initial=c['energy']-checks[0]['energy'],force_max=c['force_max'],requests=c['cumulative_search_requests']) for c in checks if c['components']>1],
            first_connected_cis_requests=next((c['cumulative_search_requests'] for c in checks if c['components']==1 and abs(c['carbon_chain_dihedral']-180)>90),None),
            prefix_connected_records=sum(c['components']==1 for c in prefix),
            prefix_carbon_dihedrals=[c['carbon_chain_dihedral'] for c in prefix],
            prefix_energy_relative_initial=[c['energy']-checks[0]['energy'] for c in prefix],
            ls_preparation=preparations)
        rows.append(data)
        (folder/'geometry-summary.json').write_text(json.dumps(data,indent=2)+'\n')
    report=dict(common_request_budget=common,runs=rows,
        caveat='certified records are not distinct minima; fixed-index Kabsch is not permutation-aware identity; connectivity may miss torsional isomers',
        search_requests_including_pilots=summary['search_requests'],validation_requests=summary['validation_requests'])
    (BASE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ['ls_preparation']},indent=2))

if __name__=='__main__':main()
