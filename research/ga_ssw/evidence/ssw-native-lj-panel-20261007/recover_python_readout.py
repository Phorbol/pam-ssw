#!/usr/bin/env python3
"""Recover only final exports from trusted, already completed local checkpoints.

Never resumes search. Raw failed outputs remain untouched; recovered files link
back to their parent run. Fresh E/F work requires --fresh explicitly.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("panel_export", HERE / "run_panel.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def recover(raw, output, fresh_enabled):
    from ase.io import read, write
    from pamssw.standalone.paper_reference import load_ssw_checkpoint
    cp = load_ssw_checkpoint(raw / "checkpoint.pkl")
    rows = [json.loads(line) for line in (raw / "ef-ledger.jsonl").read_text().splitlines()]
    charged = sum(bool(r.get("charged")) for r in rows)
    assert charged == cp.evaluation_requests, (charged, cp.evaluation_requests)
    assert cp.config == panel.settings()[0]
    provenance = json.loads((raw / "provenance.json").read_text())
    assert provenance["imports"]["head_pamssw_tree"] == "c572a1cc0766f8d3ff534a467ad64613aaa78ce0"
    assert panel.sources()["tracked_pamssw_changes"] == []
    progress = [json.loads(line) for line in (raw / "progress.jsonl").read_text().splitlines()]
    assert len(cp.records) == sum(p["kind"] == "outer_step" for p in progress)
    output.mkdir(parents=True, exist_ok=False)
    for path in raw.iterdir():
        if path.name != "provenance.json":
            (output / path.name).symlink_to(path.resolve())
    shutil.copy2(__file__, output / "recovery_script.py")
    provenance.update(recovery={"raw_parent": str(raw.resolve()),
        "method": "checkpoint export only; no resumed or repeated search",
        "raw_failure": json.loads((raw / "failure.json").read_text()),
        "exporter_imports": panel.sources(), "fresh_enabled": fresh_enabled})
    panel.json_write(output / "provenance.json", provenance)
    minima = []
    for i, minimum in enumerate(cp.minima):
        path = output / f"minimum-{i:04d}.extxyz"
        # Stored observer snapshots must agree with trusted checkpoint states.
        if path.exists():
            np.testing.assert_allclose(read(path).positions, minimum.atoms.positions, rtol=0., atol=5.1e-9)
        else:
            write(path, minimum.atoms)
        minima.append({"index": i, "energy_eV": float(minimum.energy),
            "fmax_eV_A": float(minimum.max_force), "converged": bool(minimum.converged),
            "geometry_gate": panel.geometry_gate(minimum.atoms)})
    write(output / "best.extxyz", cp.best.atoms)
    checks = []
    fresh = None
    if fresh_enabled:
        from research.ga_ssw.full_pair_lj import FullPairLJ
        fresh = panel.CountedSurface(FullPairLJ(epsilon=1., sigma=2.7), output / "fresh-ef.jsonl", cap=2)
        for label, m in (("initial", cp.initial), ("best", cp.best)):
            assert m.converged and m.max_force <= .05
            energy, force = fresh.evaluate(m.atoms)
            fmax = float(np.linalg.norm(force, axis=1).max())
            checks.append({"label": label, "energy_eV": float(energy), "fmax_eV_A": fmax,
                "energy_error_eV": float(energy-m.energy),
                "qualified": bool(fmax <= .05 and abs(energy-m.energy) < 1e-8)})
    panel.json_write(output / "result.json", {"status": cp.status,
        "evaluation_requests": cp.evaluation_requests, "surface_requests": charged,
        "initial": {"energy_eV": cp.initial.energy, "fmax_eV_A": cp.initial.max_force,
                    "converged": cp.initial.converged},
        "best_energy_eV": cp.best.energy, "minima": minima, "outer_events": progress,
        "fresh": checks, "fresh_requests": 0 if fresh is None else fresh.requests,
        "best_geometry_gate": panel.geometry_gate(cp.best.atoms),
        "recovery_scope": "final export failure; original search and checkpoint unchanged"})
    print(output, charged, cp.status, cp.best.energy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    recover(a.raw.resolve(), a.output.resolve(), a.fresh)


if __name__ == "__main__":
    main()
