"""Inspect actual stored restart boundaries; no PES calls."""
import json,pickle
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for seed in (3,17):
 folder=root/'runs'/f'c60-seed{seed}-ga-v2'
 with (folder/'ga-stage.pkl').open('rb') as f:cp=pickle.load(f)
 with (folder/'result.pkl').open('rb') as f:result=pickle.load(f)
 rows.append(dict(seed=seed,result_requests=result.evaluation_requests,
  saved_phase=cp.phase,saved_requests=cp.evaluation_requests,
  requests_after_saved_boundary=result.evaluation_requests-cp.evaluation_requests,
  last_walk_requests=result.walks[-1].evaluation_requests,
  last_walk_has_checkpoint=result.walks[-1].checkpoint is not None,
  result_has_checkpoint=result.checkpoint is not None,
  last_walk_records=len(result.walks[-1].records)))
(root/'checkpoint-gap.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
