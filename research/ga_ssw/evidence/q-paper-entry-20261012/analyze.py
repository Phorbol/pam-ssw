"""Derive a coverage readout without changing raw runs or evaluating a PES."""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
run=Path(sys.argv[1]); output=Path(sys.argv[2])
provenance=json.loads((run/'provenance.json').read_text())
rows=[]
for row in provenance['runs']:
 folder=run/(row['case']+'-'+row['arm'])
 result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
 ledger=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
 paid=[x for x in ledger if x['kind'] in ('search','search_failure')]
 stages=[]; landings=[]
 if result:
  for rec in result['records']:
   for stage in rec['climb']:
    q=stage.get('recovered_direction',{}).get('descriptor_direction')
    stages.append({'outer':rec['index'],'gaussian':stage['index'],'status':stage.get('status'),'requests':stage.get('requests'),'Q':q,'has_gaussian': 'width' in stage})
   if rec['landing'] is not None: landings.append(rec['landing'])
  assert row['search_requests']==result['evaluation_requests'], (folder,'request closure')
 assert len(paid)==row['search_requests'],(folder,'ledger mismatch')
 force_checks=row.get('fresh_checks',[])
 periodic_images=[]
 if (folder/'minima.extxyz').exists():
  for k,a in enumerate(read(folder/'minima.extxyz',':')):
   if a.pbc.any():
    i,j,s,d=neighbor_list('ijSd',a,3.0,self_interaction=False)
    counts=np.bincount(i,minlength=len(a))
    periodic_images.append({'minimum':k,'cutoff_A':3.,'full_image_neighbor_count':counts.tolist(),'minimum_image_distance_A':float(min(d)) if len(d) else None,'self_images':int(np.sum(i==j))})
 rows.append({'case':row['case'],'arm':row['arm'],'run_status':row['status'],'failure_class':row.get('failure_class'),'outer_steps':len(result['records']) if result else 0,'outer_statuses':Counter(r['status'] for r in result['records']) if result else {},'gaussians_constructed':sum(s['has_gaussian'] for s in stages),'descriptor_stages':sum(s['Q'] is not None for s in stages),'descriptor_families':sorted({s['Q']['family'] for s in stages if s['Q']}),'landings':len(landings),'search_requests':row['search_requests'],'search_calculate_calls':row['search_calculate_calls'],'fresh_requests':row.get('fresh_requests',0),'fresh_force_pass':sum(bool(c.get('force_qualified')) for c in force_checks),'fresh_count':len(force_checks),'max_fresh_energy_error_eV':max([abs(c['energy_eV']-c['reported_energy_eV']) for c in force_checks if c.get('reported_energy_eV') is not None] or [0]),'fresh_geometry':[c.get('geometry_diagnostics') for c in force_checks],'full_periodic_images':periodic_images,'stages':stages,'resume_status':row.get('resume_status'),'initial_energy':result['initial']['energy'] if result else None,'landing_energies':[x['energy'] for x in landings]})
assert sum(x['search_requests'] for x in rows)==provenance['search_requests_total']
readout={'raw_run':str(run),'status':provenance['status'],'search_requests':sum(x['search_requests'] for x in rows),'fresh_requests':sum(x['fresh_requests'] for x in rows),'rows':rows,'scope':'one seed, fixed-cell lifecycle verification; no efficiency or phase-search claim','periodic_diagnostic_note':'raw runner coordination uses MIC pairs; this readout additionally enumerates all images at3Angstrom without changing raw data'}
output.mkdir(exist_ok=False)
(output/'analysis.json').write_text(json.dumps(readout,indent=2))
print(json.dumps({k:v for k,v in readout.items() if k!='rows'},indent=2))
for x in rows:print(json.dumps({k:x[k] for k in ['case','arm','outer_steps','gaussians_constructed','descriptor_stages','landings','search_requests','fresh_force_pass','fresh_count','outer_statuses']}))
