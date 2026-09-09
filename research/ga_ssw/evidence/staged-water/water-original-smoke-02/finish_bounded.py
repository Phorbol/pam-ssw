import os,subprocess,time,json,psutil
from pathlib import Path
start=time.monotonic()
with open('finish-stdout.txt','w') as f:
 p=subprocess.Popen(['java','-cp','.:sgn.jar','FinishWaterProbe'],cwd='soft',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try: code=p.wait(timeout=90);state='completed'
 except subprocess.TimeoutExpired:
  children=psutil.Process(p.pid).children(recursive=True)
  for c in reversed(children):
   try:c.kill()
   except psutil.NoSuchProcess:pass
  p.kill();code=p.wait();psutil.wait_procs(children,timeout=3);state='timeout'
Path('finish-status.json').write_text(json.dumps(dict(state=state,returncode=code,wall_seconds=time.monotonic()-start,timeout_seconds=90),indent=2)+'\n')
