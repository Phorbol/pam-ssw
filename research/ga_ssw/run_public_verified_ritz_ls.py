"""Prepare a bounded public verified-Ritz SSW/LS lifecycle comparison.

Preparation-only unless ``--execute`` is supplied.  The runner freezes only
pamssw and uses the public SSW, paper-LS, and native-LS entry points without a
research monkeypatch.
"""
import argparse, hashlib, json, shutil, subprocess, sys, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.collections import g2


def dump(path, value, serial):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def serial(value):
    if isinstance(value, Atoms):
        return dict(numbers=value.numbers.tolist(), positions=value.positions.tolist(),
                    cell=value.cell.array.tolist(), pbc=value.pbc.tolist())
    if is_dataclass(value):
        return {key: serial(item) for key, item in vars(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--source-root", type=Path, default=Path(
        "."))
    args = ap.parse_args()
    out, source = args.output.resolve(), args.source_root.resolve()
    out.mkdir(parents=False, exist_ok=False)
    if not (source / "pamssw").is_dir():
        raise FileNotFoundError(f"frozen source tree missing: {source / 'pamssw'}")
    # Freeze only the production package; research scripts and the rest of the
    # checkout are deliberately outside the runnable source snapshot.
    (out / "source").mkdir()
    shutil.copytree(source / "pamssw", out / "source" / "pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(Path(__file__), out / "runner.py")
    sys.path.insert(0, str(out / "source"))
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out / "source")
    from pamssw.standalone import (ASESurface, SSWConfig, LSSettings,
                                   NativeLSSettings, run_ssw, run_ls_ssw,
                                   run_native_ls_ssw)
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS

    initial = g2["butadiene"].copy()
    # This is the explicit table used by the existing staged integration run.
    arms = {
        "ssw": None,
        "paper_ls": LSSettings(HC_BOND_ENERGIES,
            {k: v + 0.1 for k, v in HC_BOND_LENGTHS.items()}, target_per_atom=0.7),
        "native_ls": NativeLSSettings(HC_BOND_ENERGIES, HC_BOND_LENGTHS,
            target_mev_per_atom=700.0),
    }
    config = SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=5,
        max_gaussians=25, temperature_K=150., fmax=.01, bias_fmax=.1,
        relax_steps=400, fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        rotation_solver="ritz", cluster_frame="direction_only",
        direction_sampling="global", quench_optimizer="safe-lbfgs-total")
    plan = dict(
        system="trans-butadiene (ASE G2 butadiene)", initial_source="ase.collections.g2['butadiene']",
        seeds=[11, 29], variants=list(arms), steps=100, max_evaluations=6000,
        wall_seconds=90, config=asdict(config), ls_settings=arms,
        target_context="C4H6 reaction-space connectivity coverage only; paper oracle is DFT GGA-PBE, while this runner uses GFN2-xTB as an algorithm/coverage comparison, not a numerical paper reproduction",
        entry_points={"ssw": "run_ssw", "paper_ls": "run_ls_ssw", "native_ls": "run_native_ls_ssw"},
        backend="GFN2-xTB accuracy .001; CPU single thread",
        source_root=str(source),
        scope="bounded public verified-Ritz lifecycle comparison; 100 is an outer-step limit, not paper numerical reproduction; no global-minimum or LS-advantage claim",
        accounting="failed E/F calls and cap denials are retained in evaluations.jsonl; denial is not counted as a calculator request; every returned landing gets fresh cold E/F",
    )
    dump(out / "plan.json", plan, serial)
    dump(out / "initial.json", initial, serial)
    dump(out / "source-manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_root": str(source),
        "sha256": {str(p.relative_to(out / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (out / "source").rglob("*.py")},
    }, serial)
    if not args.execute:
        return

    from tblite.ase import TBLite
    dispatch = {"ssw": run_ssw, "paper_ls": run_ls_ssw, "native_ls": run_native_ls_ssw}
    for arm, ls in arms.items():
        for seed in (11, 29):
            folder = out / f"butadiene-{arm}-seed{seed}"; folder.mkdir()
            started = time.monotonic(); ledger = folder / "evaluations.jsonl"
            class Counted(ASESurface):
                denied = 0; boundary = None
                def evaluate(self, atoms):
                    if self.requests >= 6000 or time.monotonic() - started >= 90:
                        self.denied += 1
                        self.boundary = "request_cap" if self.requests >= 6000 else "wall_cap"
                        with ledger.open("a") as h:
                            h.write(json.dumps(serial(dict(kind="search_denial", request=self.requests,
                                error=self.boundary, atoms=atoms))) + "\n")
                        raise RuntimeError(self.boundary)
                    try:
                        e, f = super().evaluate(atoms)
                        row = dict(kind="search", request=self.requests, energy=e, forces=f, atoms=atoms)
                    except Exception as error:
                        row = dict(kind="search_failure", request=self.requests,
                                   error=repr(error), atoms=atoms)
                        with ledger.open("a") as h: h.write(json.dumps(serial(row)) + "\n")
                        raise
                    with ledger.open("a") as h: h.write(json.dumps(serial(row)) + "\n")
                    return e, f
            surface = Counted(TBLite(method="GFN2-xTB", accuracy=.001, verbosity=0))
            row = dict(system="trans-butadiene", arm=arm, seed=seed)
            fresh = None; checks = []
            try:
                kwargs = dict(atoms=initial.copy(), surface=surface, steps=100,
                              config=config, rng=np.random.default_rng(seed))
                if arm != "ssw": kwargs["ls"] = ls
                result = dispatch[arm](**kwargs)
                dump(folder / "result.json", result, serial)
                fresh = ASESurface(TBLite(method="GFN2-xTB", accuracy=.001, verbosity=0)); checks=[]
                for i, minimum in enumerate(result.minima):
                    try:
                        fresh.calculator = TBLite(method="GFN2-xTB", accuracy=.001, verbosity=0)
                        e, f = fresh.evaluate(minimum.atoms)
                        checks.append(dict(index=i, energy=e, energy_error=e-minimum.energy,
                            fmax=float(np.linalg.norm(f, axis=1).max()),
                            force_qualified=bool(np.linalg.norm(f, axis=1).max() <= .01)))
                    except Exception as error: checks.append(dict(index=i, error=repr(error)))
                    dump(folder / "fresh-checks.json", checks, serial)
                dump(folder / "fresh-checks.json", checks, serial)
                rotation_requests = [event["rotation_force_requests"]
                                     for record in result.records for event in record.climb
                                     if "rotation_force_requests" in event]
                assert all(value <= 101 for value in rotation_requests)
                initial_requests = result.initial.evaluation_requests
                record_requests = sum(record.evaluation_requests for record in result.records)
                row.update(status=result.status, search_requests=surface.requests,
                    minima=len(result.minima), records=len(result.records),
                    accepted=sum(r.accepted for r in result.records),
                    nonaccepted_record_statuses=[r.status for r in result.records if not r.accepted],
                    ls_updates=sum(r.ls_update is not None for r in result.records),
                    ls_responses=sum(r.energy_response is not None for r in result.records),
                    rotation_force_requests=rotation_requests,
                    max_rotation_force_requests=max(rotation_requests, default=0),
                    initial_requests=initial_requests, record_requests=record_requests,
                    best_energy=min((m.energy for m in result.minima), default=None),
                    fresh_requests=fresh.requests, fresh_checks=checks,
                    accounted=(initial_requests + record_requests == surface.requests == result.evaluation_requests))
            except Exception as error:
                row.update(status="exception", error=repr(error), search_requests=surface.requests,
                           fresh_requests=fresh.requests if fresh is not None else 0, fresh_checks=checks)
            ledger_rows = [json.loads(line) for line in ledger.open()] if ledger.exists() else []
            search_rows = [r for r in ledger_rows if r["kind"] in ("search", "search_failure")]
            row.update(ledger_search_requests=len(search_rows),
                ledger_sequential=[r["request"] for r in search_rows] == list(range(1, surface.requests + 1)))
            row.update(boundary=surface.boundary, denied=surface.denied,
                ledger_count=sum(1 for _ in ledger.open()) if ledger.exists() else 0,
                wall_seconds=time.monotonic()-started)
            dump(folder / "summary.json", row, serial)


if __name__ == "__main__": main()
