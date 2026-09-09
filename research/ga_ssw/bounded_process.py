"""Linux-only isolated supervisor: reap MPI descendants across process groups.

Run as a dedicated subprocess, never inside a host with unrelated children.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil


def cleanup_children():
    owner=psutil.Process();deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        children=owner.children(recursive=True)
        for child in children:
            try:child.suspend()
            except psutil.NoSuchProcess:pass
        for child in reversed(children):
            try:child.kill()
            except psutil.NoSuchProcess:pass
        while True:
            try:
                pid,_=os.waitpid(-1,os.WNOHANG)
                if pid==0:break
            except ChildProcessError:break
        if not owner.children(recursive=True):return []
        time.sleep(.02)
    return [dict(pid=p.pid,created=p.create_time()) for p in owner.children(recursive=True)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd',required=True);parser.add_argument('--timeout',required=True,type=float)
    parser.add_argument('--log',required=True,type=Path);parser.add_argument('--status',required=True,type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.timeout<=0:parser.error('command and positive timeout required')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot enable child subreaper')
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    start=time.monotonic();proc=None;code=None;state='launch_failed'
    try:
        with args.log.open('w') as log:
            proc=subprocess.Popen(command,cwd=args.cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=args.timeout);state='completed'
            except subprocess.TimeoutExpired:state='timeout'
    except KeyboardInterrupt:state='interrupted'
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        survivors=cleanup_children()
        result=dict(state=state,returncode=code,wall_seconds=time.monotonic()-start,
                    timeout_seconds=args.timeout,cleanup_survivors=survivors)
        args.status.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    if survivors:raise SystemExit(2)


if __name__=='__main__':main()
