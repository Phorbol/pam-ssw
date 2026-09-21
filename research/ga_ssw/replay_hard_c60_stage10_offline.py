"""Replay C60 paper-LS stage 10 rotation from recorded E/F only."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.cluster_frame import ClusterFrame

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step/results/paper-seed3"
out = P / "offline-stage10"
out.mkdir(exist_ok=True)
r = json.loads((P / "result.json").read_text())
record = r["records"][0]
events = record["climb"][:9]
start = Atoms(**record["last_atoms"])
initial = Atoms(**r["initial"]["atoms"])
prep = json.loads((P / "preparations.json").read_text())[0]
soft = FrozenBondSoftening(
    tuple(map(int, initial.numbers)),
    tuple(map(tuple, initial.cell.array)), tuple(map(bool, initial.pbc)),
    tuple(map(tuple, prep["pairs"])),
    tuple(initial.get_distance(i, j, mic=True) for i, j in prep["pairs"]),
    tuple(prep["strengths"]), .2)
# Rotation callback in paper_reference uses only frozen LS terms; Gaussian
# terms are introduced after a direction converges, for height/quench only.
terms = [soft]
rows = [json.loads(line) for line in (P / "evaluations.jsonl").read_text().splitlines()]
start_call = 917  # 915 is last quench; 916 is its true-E check.
tail = [x for x in rows if start_call <= int(x["call"]) <= 1016]
if len(tail) != 100 or [int(x["call"]) for x in tail] != list(range(917, 1017)):
    raise RuntimeError("stage-10 cache must be ordered calls 917..1016")
errors = []

class RecordedSurface:
    def __init__(self, records): self.records, self.i = records, 0
    def evaluate(self, atoms):
        row = self.records[self.i]
        err = float(np.max(np.abs(atoms.positions - np.asarray(row["atoms"]["positions"]))))
        errors.append(dict(call=int(row["call"]), coordinate_error=err))
        if err > 1e-9: raise RuntimeError(f"ordered trajectory mismatch at call {row['call']}: {err}")
        self.i += 1
        return float(row["energy"]), np.asarray(row["forces"], dtype=float)

def evaluate(candidate):
    candidate = candidate.copy()
    candidate.positions = frame.positions(candidate.positions)
    e, f = surface.evaluate(candidate)
    for term in terms:
        de, df = term.evaluate(candidate)
        e += de; f += df
    return e, frame.project(f)

def trace_dimer(atoms, n0, *, rotation_bias, fd_step, max_hvp, tol, evaluate):
    shape = atoms.positions.shape; n0 = n0.ravel().copy(); center = atoms.positions.copy()
    def force_at(pos):
        trial = atoms.copy(); trial.positions = pos.reshape(shape)
        return np.asarray(evaluate(trial)[1]).ravel().copy()
    f0 = force_at(center); n = n0.copy(); history=[]; hvp_calls=0; force_calls=1
    def hvp(v):
        nonlocal hvp_calls, force_calls
        f1 = force_at(center + fd_step*v.reshape(shape)); hvp_calls += 1; force_calls += 1
        return (f0-f1)/fd_step-rotation_bias*np.dot(n0,v)*n0
    hn=hvp(n); force_calls=1+hvp_calls
    while True:
        curvature=float(n@hn); residual=hn-curvature*n; res=float(np.linalg.norm(residual))
        history.append(dict(hvp_calls=hvp_calls, curvature=curvature, residual=res,
                           direction=n.reshape(shape).tolist()))
        if res<=tol or hvp_calls+2>max_hvp: break
        t=-residual; t-=np.dot(t,n)*n; t/=np.linalg.norm(t); ht=hvp(t)
        projected=np.array([[curvature,n@ht],[t@hn,t@ht]])
        history[-1].update(a01=float(projected[0, 1]), a10=float(projected[1, 0]),
                           projected_antisymmetry=float(np.linalg.norm(projected-projected.T)))
        _,vectors=np.linalg.eigh((projected+projected.T)/2)
        n=vectors[0,0]*n+vectors[1,0]*t; n/=np.linalg.norm(n)
        if np.dot(n,n0)<0:n=-n
        hn=hvp(n)
    return dict(direction=n.reshape(shape),curvature=curvature,residual=res,hvp_calls=hvp_calls,force_calls=force_calls,converged=res<=tol,history=history)

anchor = np.asarray(record["initial_direction"], dtype=float)
norm = np.linalg.norm(anchor)
frame = ClusterFrame(start)
anchor = frame.project(anchor)
anchor /= np.linalg.norm(anchor)
surface = RecordedSurface(tail)
mode = trace_dimer(start, anchor, rotation_bias=100., fd_step=1e-4,
                   max_hvp=100, tol=.02, evaluate=evaluate)
summary = dict(start_call=start_call, matched_start_error=float(np.max(np.abs(start.positions - np.asarray(tail[0]["atoms"]["positions"])))),
                recorded_tail=len(tail), consumed=surface.i, max_coordinate_error=max(x["coordinate_error"] for x in errors),
                coordinate_match_tolerance=1e-9,
                ordered_replay_valid=bool(max(x["coordinate_error"] for x in errors) <= 1e-9),
                curvature=float(mode["curvature"]), residual=float(mode["residual"]),
                hvp_calls=mode["hvp_calls"], force_calls=mode["force_calls"],
                converged=bool(mode["converged"]), direction=mode["direction"].tolist(), extra_PES=0,
                residual_history=mode["history"])
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "evaluation-match.json").write_text(json.dumps(errors, indent=2))
(out / "frozen-objective.json").write_text(json.dumps(dict(
    center=start.positions.tolist(), anchor=anchor.tolist(), rotation_bias=100.0,
    fd_step=1e-4, max_hvp=100, tol=.02, ls_pairs=prep["pairs"],
    ls_strengths=prep["strengths"], ls_xi=.2, source_calls="917..1016",
    extra_PES=0), indent=2))
(out / "plane-tail.json").write_text(json.dumps([
    {k: x[k] for k in ("hvp_calls", "residual", "a01", "a10", "projected_antisymmetry")}
    for x in mode["history"] if "a01" in x][-10:], indent=2))
print(json.dumps({k: summary[k] for k in ('start_call','consumed','max_coordinate_error','ordered_replay_valid','curvature','residual','hvp_calls','force_calls','converged')}, indent=2))
