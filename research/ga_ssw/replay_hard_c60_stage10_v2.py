"""Strict 0-PES replay of C60 paper-LS stage 10 using the public dimer."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.dimer import paper_dimer_direction
from pamssw.standalone.softening import FrozenBondSoftening
from research.ga_ssw.frozen_rotation_helper import make_frozen_rotation_evaluator

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step/results/paper-seed3"
out = P / "offline-stage10"
r = json.loads((P / "result.json").read_text()); rec = r["records"][0]
start = Atoms(**rec["last_atoms"]); initial = Atoms(**r["initial"]["atoms"])
prep = json.loads((P / "preparations.json").read_text())[0]
refs = tuple(initial.get_distance(i, j, mic=True) for i, j in prep["pairs"])
soft = FrozenBondSoftening(tuple(map(int, initial.numbers)), tuple(map(tuple, initial.cell.array)),
    tuple(map(bool, initial.pbc)), tuple(map(tuple, prep["pairs"])), refs,
    tuple(prep["strengths"]), .2)
rows = [json.loads(x) for x in (P / "evaluations.jsonl").read_text().splitlines()]
tail = [x for x in rows if 917 <= int(x["call"]) <= 1016]
if [int(x["call"]) for x in tail] != list(range(917, 1017)): raise RuntimeError("calls 917..1016 required")
errors = []
class Cached:
    def __init__(self, rows): self.rows, self.i = rows, 0
    def evaluate(self, atoms):
        row = self.rows[self.i]; err = float(np.max(np.abs(atoms.positions - np.asarray(row["atoms"]["positions"]))))
        errors.append({"call": int(row["call"]), "coordinate_error": err})
        if err > 1e-9: raise RuntimeError(f"coordinate mismatch call {row['call']}: {err}")
        self.i += 1; return float(row["energy"]), np.asarray(row["forces"], float)
surface = Cached(tail)
center, raw_anchor, anchor, evaluate = make_frozen_rotation_evaluator(start, np.asarray(rec["initial_direction"]), soft, surface)
mode = paper_dimer_direction(center, anchor, rotation_bias=100., fd_step=1e-4, max_hvp=100, tol=.02, evaluate=evaluate)
result = dict(source_calls="917..1016", consumed=surface.i, max_coordinate_error=max(x["coordinate_error"] for x in errors),
              coordinate_match_tolerance=1e-9, ordered_replay_valid=True, curvature=float(mode.curvature),
              residual=float(mode.residual_norm), hvp_calls=mode.hvp_calls, force_calls=mode.force_calls,
              converged=bool(mode.converged), extra_PES=0)
(out / "v2-result.json").write_text(json.dumps(result, indent=2)); (out / "v2-evaluation-match.json").write_text(json.dumps(errors, indent=2))
(out / "frozen-objective-v2.json").write_text(json.dumps(dict(
    atoms=dict(numbers=[int(x) for x in center.numbers], positions=center.positions.tolist(), cell=center.cell.array.tolist(), pbc=[bool(x) for x in center.pbc]),
    raw_anchor=raw_anchor.tolist(), projected_normalized_anchor=anchor.tolist(),
    ls=dict(numbers=[int(x) for x in soft.numbers], cell=[list(x) for x in soft.cell], pbc=list(soft.pbc), pairs=[[int(i) for i in x] for x in soft.pairs],
            reference_distances=list(soft.reference_distances), strengths=list(soft.strengths), xi=soft.xi),
    frame_contract="direction_only: ClusterFrame(center); project positions and complete force",
    objective_contract="physical E/F + frozen LS only; no Gaussian terms", source_calls="917..1016", extra_PES=0), indent=2))
print(json.dumps(result, indent=2))
