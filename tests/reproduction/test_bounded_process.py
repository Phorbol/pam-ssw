"""Real process-tree cleanup tests; no atomistic/energy calculations."""
import json
from pathlib import Path
import subprocess
import sys
import psutil
import pytest

SUPERVISOR=Path(__file__).resolve().parents[2]/'research/ga_ssw/bounded_process.py'

@pytest.mark.parametrize('early_exit',[True,False])
def test_orphan_in_new_session_is_reaped_on_failure_or_timeout(tmp_path,early_exit):
    parent=tmp_path/'parent.py'
    parent.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                      "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n"
                      "Path('child.pid').write_text(str(p.pid))\n"+
                      ("sys.exit(7)\n" if early_exit else "time.sleep(30)\n"))
    subprocess.run([sys.executable,str(SUPERVISOR),'--cwd',str(tmp_path),'--timeout','1',
                    '--log',str(tmp_path/'stdout'),'--status',str(tmp_path/'status.json'),
                    '--',sys.executable,str(parent)],check=True,timeout=10,capture_output=True)
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']==('completed' if early_exit else 'timeout')
    if early_exit:assert status['returncode']==7
    assert status['cleanup_survivors']==[]
    assert not psutil.pid_exists(int((tmp_path/'child.pid').read_text()))


def test_preparation_wrapper_sigint_preserves_supervisor_cleanup(tmp_path):
    import os
    import signal
    import time
    root=tmp_path/'reference'
    example=root/'GA-SSW_examples_run/global_exploration'
    template=example/'input-templates/TYPE3-(H2O)15';template.mkdir(parents=True)
    keys='CPU Memory TaskNum SSWTaskNum CombineMultiUnitNum GANum OPTSSWStep QuickSSWStep SSWStep QuickSSWIterations FineSSWIterations PopClassifyNum LaspPath'
    (template/'configure.non').write_text(''.join(k+'=1\n' for k in keys.split()))
    common=example/'GA-SSW/input';(common/'ssw_gaussian').mkdir(parents=True)
    (common/'nna1.jar').write_bytes(b'process-test-only')
    (root/'GA-SSW_program').mkdir()
    for name in ['sgn.jar','lasp']:(root/'GA-SSW_program'/name).write_bytes(b'process-test-only')
    java=root/'tools/jdk-17.0.2/bin/java';java.parent.mkdir(parents=True)
    java.write_text('#!'+sys.executable+'\nimport subprocess,sys,time\nfrom pathlib import Path\n'
                    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)\n"
                    "Path('child.pid').write_text(str(p.pid))\ntime.sleep(30)\n")
    java.chmod(0o755)
    run=tmp_path/'run';wrapper=SUPERVISOR.with_name('run_original_smoke.py')
    proc=subprocess.Popen([sys.executable,str(wrapper),'--reference-root',str(root),'--run-dir',str(run),'--execute'],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    child=None
    try:
        deadline=time.monotonic()+5
        while not (run/'soft/child.pid').exists() and time.monotonic()<deadline:time.sleep(.02)
        child=int((run/'soft/child.pid').read_text())
        proc.send_signal(signal.SIGINT);proc.communicate(timeout=10)
        assert proc.returncode==0
        status=json.loads((run/'status.json').read_text())
        assert status['state']=='interrupted' and status['cleanup_survivors']==[]
        assert not psutil.pid_exists(child)
    finally:
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if child is not None and psutil.pid_exists(child):psutil.Process(child).kill()
