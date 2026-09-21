"""Bounded real EMT check of the GA ``offspring_ssw`` branch.

The existing GA walker-options runner owns the protocol and audit helpers. This
wrapper freezes that runner and the PAM source first, then changes only
``PaperGAConfig.offspring_steps`` from zero to one. It is an interface
qualification, not a GA performance experiment.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/ga_ssw/evidence/ga-offspring-options-emt-20260920"
HELPER = "run_ga_walker_options_emt"


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _freeze():
    if OUT.exists():
        execution_artifacts = ("source", "manifest.json", "quick.chk",
                               "result.json", "full-snapshot.json",
                               "quick-snapshot.json", "resumed-snapshot.json")
        present = [name for name in execution_artifacts if (OUT / name).exists()]
        if present:
            raise FileExistsError(f"refusing to overwrite execution artifacts: {present}")
    else:
        OUT.mkdir(parents=True)
    source = OUT / "source"
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2(ROOT / "research/ga_ssw/run_ga_walker_options_emt.py",
                 source / "run_ga_walker_options_emt.py")
    shutil.copy2(ROOT / "research/ga_ssw/run_ga_checkpoint_emt.py",
                 source / "run_ga_checkpoint_emt.py")
    shutil.copytree(ROOT / "pamssw", source / "pamssw")
    return source


def _json(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {k: _json(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def main():
    source = _freeze()
    os.chdir(ROOT)
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(ROOT / "research/ga_ssw"))

    import importlib
    helper = importlib.import_module(HELPER)
    helper.OUT = OUT
    import pamssw
    pamssw_path = Path(pamssw.__file__).resolve()
    helper_path = Path(helper.__file__).resolve()
    if source not in pamssw_path.parents or source not in helper_path.parents:
        raise AssertionError(f"imports escaped frozen source: {pamssw_path}, {helper_path}")
    from pamssw.standalone import RecoveredRotationSettings, run_ga_ssw
    from pamssw.standalone.ga_checkpoint import GACheckpoint
    from pamssw.standalone.native_mc import NativeMCSettings
    from run_ga_checkpoint_emt import _inputs

    initial, bonds, refs, input_source, groups = _inputs()
    mc = NativeMCSettings(.1, 99999)
    recovered = RecoveredRotationSettings(
        pre_rotmax=5, rotmax=15, pre_ftol=1., ftol=.1,
        metric="euclidean", max_force_calls=40)
    manifest = {
        "protocol": "GA offspring_ssw branch interface qualification",
        "input_source": str(input_source), "groups": groups, "seed": 3,
        "max_evaluations_per_trajectory": helper.MAX_EVALUATIONS,
        "fresh_max_frames": helper.FRESH_MAX,
        "wall_seconds_per_trajectory": helper.WALL_SECONDS,
        "only_config_change": "PaperGAConfig.offspring_steps=1",
        "mc": _json(mc), "recovered_rotation": _json(recovered),
        "source_snapshot": str(source),
        "runner_sha256": _sha256(Path(__file__)),
        "pamssw_import_path": str(pamssw_path),
        "helper_import_path": str(helper_path),
        "source_manifest": {
            str(path.relative_to(source)): _sha256(path)
            for path in sorted(source.rglob("*.py"))
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run_one(label, *, checkpoint=None, stop_quick=False, seconds=None):
        surface = helper.LedgerSurface()
        captured = []
        callback = None
        if stop_quick:
            def callback(state):
                captured.append(state)
                return state.phase == "quick_complete"
        kwargs = helper._ga_kwargs(initial, bonds, refs, 3, surface,
                                   mc=mc, recovered_rotation=recovered)
        kwargs["config"] = dataclasses.replace(kwargs["config"], offspring_steps=1)
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        result = helper._run_with_wall(
            label, surface,
            lambda: run_ga_ssw(**kwargs, checkpoint_callback=callback),
            helper.WALL_SECONDS if seconds is None else seconds)
        return result, surface, captured

    full, full_surface, _ = run_one("full")
    split_started = time.monotonic()
    partial, split_surface, captured = run_one("quick", stop_quick=True)
    if partial.checkpoint is None or not captured or captured[-1].phase != "quick_complete":
        raise AssertionError("quick_complete checkpoint was not produced")
    checkpoint_path = OUT / "quick.chk"
    partial.checkpoint.save(checkpoint_path)
    resumed, resumed_surface, _ = run_one(
        "resumed", checkpoint=GACheckpoint.load(checkpoint_path),
        seconds=helper.WALL_SECONDS - (time.monotonic() - split_started))

    audits = {"full": helper._audit_walk_options(full),
              "quick_resumed": helper._audit_walk_options(resumed)}
    def audit_offspring(result):
        stages = [stage for stage in result.stages if stage.phase == "offspring_ssw"]
        if not stages:
            raise AssertionError("no offspring_ssw stage")
        records = []
        for stage in stages:
            records.extend(stage.details.get("records", ()))
        if not records:
            raise AssertionError("offspring_ssw stage has no walk records")
        if not any(record.mc_telemetry is not None for record in records):
            raise AssertionError("offspring_ssw records lack MC telemetry")
        if not any(any(event.get("rotation_solver") == "recovered-cbd"
                       for event in record.climb) for record in records):
            raise AssertionError("offspring_ssw records lack recovered-CBD events")
        return {"stage_count": len(stages), "record_count": len(records),
                "requests": sum(record.evaluation_requests for record in records)}
    offspring_audits = {"full": audit_offspring(full),
                        "quick_resumed": audit_offspring(resumed)}
    if full.evaluation_requests > helper.MAX_EVALUATIONS:
        raise AssertionError("full trajectory exceeded request budget")
    if split_surface.requests + resumed_surface.requests > helper.MAX_EVALUATIONS:
        raise AssertionError("quick plus resume exceeded request budget")
    for label, result, audit in (("full", full, audits["full"]),
                                 ("quick_resumed", resumed, audits["quick_resumed"])):
        if audit["mc_telemetry_count"] == 0 or audit["recovered_cbd_event_count"] == 0:
            raise AssertionError(f"{label} lacks MC/CBD events")

    if helper._observation_fingerprint(full) != helper._observation_fingerprint(resumed):
        raise AssertionError("full and quick-resume observations differ")
    if helper._ledger_fingerprint(full_surface.ledger) != helper._ledger_fingerprint(
            split_surface.ledger + resumed_surface.ledger):
        raise AssertionError("full and quick-resume surface ledger differs")
    payload = {
        "protocol": manifest["protocol"],
        "full": {**helper._state_rows(full), "ledger_requests": full_surface.requests},
        "quick_partial": {**helper._state_rows(partial), "ledger_requests": split_surface.requests},
        "quick_resumed": {**helper._state_rows(resumed), "ledger_requests": resumed_surface.requests},
        "split_resume_total_requests": split_surface.requests + resumed_surface.requests,
        "full_vs_resume_observations_match": True,
        "full_vs_resume_surface_ledger_match": True,
        "full_walk_audit": audits["full"],
        "resumed_walk_audit": audits["quick_resumed"],
        "offspring_stage_audit": offspring_audits,
        "fresh_validation": helper._fresh_archive(resumed),
        "status_is_integration_check_only": True,
    }
    (OUT / "result.json").write_text(json.dumps(_json(payload), indent=2) + "\n")
    print(json.dumps(_json(payload)))


if __name__ == "__main__":
    main()
