"""Real LASP -> external script -> persistent ASE/MACE interface qualification.

Fixed-cell nonperiodic C60 only for this bounded research probe. No binary edits.
"""
import json,os,sys,socketserver,threading,subprocess,tempfile,time,hashlib
from pathlib import Path
import numpy as np


def main(out):
    from ase import Atoms
    from mace.calculators import MACECalculator
    plan=json.loads((out/'plan.json').read_text())
    for key in ('model','binary'):
        if hashlib.sha256(Path(plan[key]).read_bytes()).hexdigest()!=plan[key+'_sha256']:raise ValueError(key+' changed')
    calc=MACECalculator(model_paths=plan['model'],device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    requests=[];current_case=None
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request=json.loads(self.rfile.readline());raw=request['coord'];lines=raw.splitlines();rows=[line.split() for line in lines[3:] if line.strip()]
                if len(rows)!=60 or any(row[0]!='C' or len(row)<4 for row in rows):raise ValueError('expected 60 C coordinate rows after three header lines')
                x=np.array([[float(v) for v in row[1:4]] for row in rows]);a=Atoms('C60',positions=x,pbc=False)
                if len(requests)>=plan['request_cap']:raise RuntimeError('probe request cap')
                record=dict(request=len(requests)+1,case=current_case,external_coord=raw,positions=x.tolist())
                requests.append(record)
                a.calc=calc;e=float(a.get_potential_energy());f=a.get_forces()
                if not np.isfinite(e) or not np.isfinite(f).all():raise ValueError('nonfinite MACE result')
                record.update(energy=e,forces=f.tolist());response=dict(ok=True,energy=e,forces=f.tolist())
            except Exception as error:
                response=dict(ok=False,error=repr(error))
            with (out/'requests.jsonl').open('a') as log:log.write(json.dumps(dict(response=response,**(record if 'record' in locals() else {})))+'\n')
            self.wfile.write((json.dumps(response)+'\n').encode())
    # Slurm epilog removes this user's files under /tmp and /dev/shm. Keep the
    # live endpoint in a unique short HOME directory so another job cleanup
    # cannot unlink the socket; Unix socket paths also have a short limit.
    with tempfile.TemporaryDirectory(dir=Path.home(), prefix='.lm-') as temp:
        address=str(Path(temp)/'eval.sock')
        if len(os.fsencode(address)) >= 108:
            raise RuntimeError(f'Unix socket path is too long: {address}')
        with socketserver.UnixStreamServer(address,Handler) as server:
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            env=os.environ.copy();env['LASP_MACE_SOCKET']=address
            env['LD_LIBRARY_PATH']=plan['mpi_lib']+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            env.update(I_MPI_FABRICS='shm',I_MPI_PIN='0')
            results=[]
            try:
                for name in plan['cases']:
                    current_case=name;folder=out/name;before=len(requests)
                    cmd=[sys.executable,str(out/'bounded_process.py'),'--cwd',str(folder),'--timeout','180',
                         '--log',str(folder/'stdout.txt'),'--status',str(folder/'process.json'),'--',
                         '/lib64/ld-linux-x86-64.so.2',plan['binary']]
                    subprocess.run(cmd,env=env,check=True)
                    status=json.loads((folder/'process.json').read_text());lasp=(folder/'lasp.out').read_text() if (folder/'lasp.out').exists() else ''
                    row=dict(case=name,process=status,requests=len(requests)-before,ssw_done='SSW all done' in lasp)
                    if (folder/'allfor.arc').exists():
                        row['allfor_arc']=(folder/'allfor.arc').read_text()
                    results.append(row);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(name,status,row['requests'],flush=True)
            finally:server.shutdown();worker.join()


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
