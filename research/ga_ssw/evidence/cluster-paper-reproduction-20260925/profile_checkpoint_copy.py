"""Zero-PES engineering timing on real saved history, not search performance."""
import sys,json,time,gc
from pathlib import Path
from dataclasses import replace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]))
from pamssw.standalone.paper_reference import load_ssw_checkpoint,_checkpoint_copy
source=Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/ssw-seed61/checkpoint.pkl')
checkpoint=load_ssw_checkpoint(source)
rows=[]
for n in (10,50,100,200,400):
 if n>len(checkpoint.records):continue
 # Prefix payloads isolate copy growth; these are not valid restart artifacts.
 prefix=replace(checkpoint,records=checkpoint.records[:n],minima=checkpoint.minima[:n+1])
 samples=[]
 for _ in range(3):
  gc.collect();start=time.perf_counter();copied=_checkpoint_copy(prefix)
  samples.append(time.perf_counter()-start)
  assert len(copied.records)==n
  del copied
 rows.append(dict(records=n,minima=len(prefix.minima),seconds=samples))
result=dict(scope='copy timing only; real C4H6 history prefixes are not search runs or valid truncated restart files',source=str(source),source_bytes=source.stat().st_size,source_records=len(checkpoint.records),rows=rows,PES_requests=0)
(HERE/'checkpoint-copy-profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
