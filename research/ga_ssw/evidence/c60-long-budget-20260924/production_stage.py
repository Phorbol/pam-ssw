"""Execute one preallocated stage. Terminal arms skip; interrupted arms stop."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--stage',type=int,choices=(1,2,3,4),required=True)
parser.add_argument('--task',type=int,choices=range(4),required=True)
a=parser.parse_args()
qualification=json.loads((root/'preflight-qualification.json').read_text())
if qualification['status']!='passed':raise RuntimeError('preflight not passed')
name=('ssw-17101','native-ls-17101','ssw-17102','native-ls-17102')[a.task]
folder=root/name
state=json.loads((folder/'budget.json').read_text()) if (folder/'budget.json').exists() else None
if state and state['status'] not in ('ready','paused'):
    print(json.dumps(dict(arm=name,stage=a.stage,skipped_status=state['status'])))
    sys.exit(0 if state['status'] in ('search_exhausted','wall_exhausted','completed','failed') else 1)
expected_segments=a.stage-1
if (len(state['segments']) if state else 0)!=expected_segments:
    raise RuntimeError('missing or duplicate stage; no automatic recovery')
seconds=(43200,43200,43200,28800)[a.stage-1]
subprocess.run([sys.executable,str(root/'production_runner.py'),'--run-dir',str(folder),
                '--segment-seconds',str(seconds),'--execute'],check=True)
