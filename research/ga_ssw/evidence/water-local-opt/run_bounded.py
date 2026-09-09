import os,subprocess,time,json,signal
from pathlib import Path
start=time.monotonic()
with open('stdout.txt','w') as f:
    p=subprocess.Popen(['/lib64/ld-linux-x86-64.so.2','../../GA-SSW_program/lasp'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=p.wait(timeout=20);status='completed'
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);code=p.wait();status='timeout'
Path('run-status.json').write_text(json.dumps(dict(status=status,returncode=code,wall_seconds=time.monotonic()-start,timeout_seconds=20,threads=1),indent=2)+'\n')
