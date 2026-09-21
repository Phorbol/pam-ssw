"""Bounded GFN2-xTB HVP finite-difference diagnostic for the S22 water dimer."""
from pathlib import Path
import json, signal, time
import numpy as np
import ase
from ase.data.s22 import create_s22_system

OUT = Path("research/ga_ssw/evidence/hvp-step-diagnostic-water-20260912")
STEPS = (1e-4, 1e-3, 5e-3, 1e-2)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        from tblite.ase import TBLite
        import tblite
    except Exception as exc:
        payload = dict(status="missing_dependency", dependency="tblite",
                       error=repr(exc), requests=0,
                       limits="No installation attempted; no PES evaluated.")
        (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        return
    atoms = create_s22_system('Water_dimer')
    calc_kwargs = dict(method='GFN2-xTB', accuracy=.001, verbosity=0)
    requests = 0
    def force(positions):
        nonlocal requests
        probe = atoms.copy(); probe.positions[:] = positions
        probe.calc = TBLite(**calc_kwargs)
        requests += 1
        return np.asarray(probe.get_forces(), dtype=float)
    seed = 41; rng = np.random.default_rng(seed)
    direction = rng.normal(size=atoms.positions.shape); direction /= np.linalg.norm(direction)
    x = atoms.positions.copy(); rows=[]; centered={}
    start=time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120 second wall limit')))
    signal.alarm(120)
    try:
        f0 = force(x)
        for h in STEPS:
            fp=force(x+h*direction); fm=force(x-h*direction)
            forward=(f0-fp)/h; centre=(fm-fp)/(2*h); centered[h]=centre.copy()
            u=forward.ravel();v=centre.ravel(); nu=np.linalg.norm(u);nv=np.linalg.norm(v)
            cos=float(np.dot(u,v)/(nu*nv)) if nu and nv else None
            rows.append(dict(step=h, relative_difference=float(np.linalg.norm(u-v)/nv) if nv else None,
                angle_rad=float(np.arccos(np.clip(cos,-1,1))) if cos is not None else None,
                forward_norm=float(nu), centered_norm=float(nv)))
    except Exception as exc:
        payload=dict(status="failed", error=repr(exc), requests=requests,
                     elapsed_seconds=time.monotonic()-start, rows=rows,
                     ase_version=ase.__version__, tblite_version=getattr(tblite, '__version__', 'unknown'))
    else:
        ref=centered[STEPS[0]].ravel()
        for row,h in zip(rows,STEPS): row['centered_vs_small_h_relative']=float(np.linalg.norm(centered[h].ravel()-ref)/np.linalg.norm(ref))
        payload=dict(status="complete", model="tblite GFN2-xTB accuracy .001",
            ase_version=ase.__version__, tblite_version=getattr(tblite, '__version__', 'unknown'),
            input="ASE S22 create_s22_system('Water_dimer')", symbols=atoms.get_chemical_symbols(),
            positions=atoms.positions.tolist(), direction=direction.tolist(), seed=seed,
            steps=list(STEPS), actual_force_requests=requests, rows=rows,
            elapsed_seconds=time.monotonic()-start,
            limits="Fixed geometry and random direction; no optimization; numerical diagnostic only.")
    finally:
        signal.alarm(0)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "script.py").write_text(Path(__file__).read_text())
    print(json.dumps({"status":payload["status"],"requests":payload["requests"] if "requests" in payload else payload["actual_force_requests"]}))

if __name__ == '__main__': main()
