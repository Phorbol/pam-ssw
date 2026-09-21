"""Prepare a matched-cost, single-basin SSW control for the GA study.

Execution is deliberately opt-in.  The default source is the frozen copy used
by fixed-ga-single-basin-20260912, so this control can be run after review
without silently importing a newer working tree.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.collections import g2


def dump(path, value, serial):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--source-root",
        type=Path,
        default=Path("research/ga_ssw/evidence/fixed-ga-single-basin-20260912/source"),
        help="frozen pamssw source tree from the matched GA study",
    )
    args = ap.parse_args()
    out = args.output.resolve()
    source = args.source_root.resolve()
    out.mkdir(parents=False, exist_ok=False)
    if not (source / "pamssw").is_dir():
        raise FileNotFoundError(f"frozen source tree missing: {source / 'pamssw'}")
    shutil.copytree(source, out / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(Path(__file__), out / "runner.py")

    sys.path.insert(0, str(out / "source"))
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    import pamssw
    assert str(out / "source") in str(pamssw.__file__)
    from research.ga_ssw.compare_vc_arms import serial
    from tblite.ase import TBLite

    initial = g2["bicyclobutane"].copy()
    config = SSWConfig(
        width=0.1, rotation_bias=None, max_gaussians=25, temperature_K=150.0,
        fmax=0.01, bias_fmax=0.1, relax_steps=400, fd_step=1e-4,
        rotation_hvp=100, rotation_tol=0.02, pre_rotation_hvp=5,
        direction_sampling="global", rotation_solver="ritz",
        cluster_frame="direction_only", quench_optimizer="safe-lbfgs-total",
    )
    caps = {11: 12000, 29: 11154}
    ga_root = source.parent
    assert asdict(config) == json.loads((ga_root/'plan.json').read_text())['ssw_config']
    for seed, cap in caps.items():
        folder = ga_root/f'bicyclobutane-seed{seed}-plain'
        assert json.loads((folder/'summary.json').read_text())['search_requests'] == cap
        with (folder/'evaluations.jsonl').open() as stream: first = json.loads(next(stream))
        assert np.array_equal(initial.positions, np.array(first['atoms']['positions']))
        assert np.array_equal(initial.numbers, np.array(first['atoms']['numbers']))
    plan = dict(
        system="bicyclobutane", initial="one G2 bicyclobutane only",
        seeds=[11, 29], steps=100, steps_guard="100 is a defensive maximum",
        caps=caps, wall_seconds=120, config=asdict(config), ls=None,
        backend="GFN2-xTB accuracy .001; CPU single thread",
        source_root=str(source),
        matched_cost="caps equal the realized GA search-request counts: seed11=12000, seed29=11154",
        purpose="single-basin SSW control for lifecycle/cost comparison; not a pre-registered performance benchmark",
        controls="same initial, SSWConfig, source snapshot, seeds and fresh protocol as GA arm; no GA/proposal/LS",
    )
    dump(out / "plan.json", plan, serial)
    dump(out / "initial.json", initial, serial)
    dump(out / "source-manifest.json", {
        "source_root": str(source),
        "sha256": {str(p.relative_to(out / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (out / "source").rglob("*.py")},
    }, serial)
    if not args.execute:
        return

    for seed, cap in caps.items():
        folder = out / f"bicyclobutane-seed{seed}-ssw"
        folder.mkdir()
        started = time.monotonic()
        ledger = folder / "evaluations.jsonl"

        class Counted(ASESurface):
            denied = 0
            boundary = None

            def evaluate(self, atoms):
                if self.requests >= cap or time.monotonic() - started >= 120:
                    self.denied += 1
                    self.boundary = "request_cap" if self.requests >= cap else "wall_cap"
                    with ledger.open("a") as stream:
                        stream.write(json.dumps(serial({"kind": "denied", "request_attempt": self.requests + 1,
                                                        "error": self.boundary, "atoms": atoms})) + "\n")
                    raise RuntimeError(self.boundary)
                try:
                    energy, forces = super().evaluate(atoms)
                    row = dict(kind="search", request=self.requests, energy=energy,
                               forces=forces, atoms=atoms)
                except Exception as error:
                    row = dict(kind="search", request=self.requests,
                               error=repr(error), atoms=atoms)
                    with ledger.open("a") as stream:
                        stream.write(json.dumps(serial(row)) + "\n")
                    raise
                with ledger.open("a") as stream:
                    stream.write(json.dumps(serial(row)) + "\n")
                return energy, forces

        surface = Counted(TBLite(method="GFN2-xTB", accuracy=0.001, verbosity=0))
        row = dict(system="bicyclobutane", seed=seed, arm="matched_single_basin_ssw")
        fresh_checks = []
        fresh = None
        try:
            result = run_ssw(initial.copy(), surface, steps=100, config=config,
                             rng=np.random.default_rng(seed), ls=None)
            dump(folder / "result.json", result, serial)
            fresh = ASESurface(TBLite(method="GFN2-xTB", accuracy=0.001, verbosity=0))
            for index, minimum in enumerate(result.minima):
                try:
                    fresh.calculator = TBLite(method="GFN2-xTB", accuracy=0.001, verbosity=0)
                    energy, forces = fresh.evaluate(minimum.atoms)
                    fresh_checks.append(dict(index=index, energy=energy,
                        energy_error=energy - minimum.energy,
                        fmax=float(np.linalg.norm(forces, axis=1).max()),
                        force_qualified=bool(np.linalg.norm(forces, axis=1).max() <= 0.01)))
                except Exception as error:
                    fresh_checks.append(dict(index=index, error=repr(error)))
            dump(folder / "fresh-checks.json", fresh_checks, serial)
            row.update(status=result.status, search_requests=surface.requests,
                       minima=len(result.minima), accepted=sum(r.accepted for r in result.records),
                       records=len(result.records), nonaccepted_record_statuses=[r.status for r in result.records if not r.accepted],
                       best_energy=min((m.energy for m in result.minima), default=None),
                       search_accounted=result.evaluation_requests == surface.requests,
                       fresh_requests=fresh.requests, fresh_checks=fresh_checks)
        except Exception as error:
            row.update(status="exception", error=repr(error), search_requests=surface.requests,
                       fresh_requests=0 if fresh is None else fresh.requests, fresh_checks=fresh_checks)
        row.update(boundary=surface.boundary, denied=surface.denied,
                   ledger_count=sum(1 for _ in ledger.open()) if ledger.exists() else 0,
                   wall_seconds=time.monotonic() - started)
        row["ledger_search_rows"] = sum(1 for line in ledger.open()
                                         if json.loads(line).get("kind") == "search") if ledger.exists() else 0
        row["ledger_accounted"] = row["ledger_search_rows"] == surface.requests
        dump(folder / "summary.json", row, serial)


if __name__ == "__main__":
    main()
