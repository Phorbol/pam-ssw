"""Zero-PES diagnosis of cell-cycle and atomic-boundary geometries."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms

ROOT = Path("research/ga_ssw/fe7c3-block-baseline")


def atoms(d):
    return Atoms(numbers=d["numbers"], positions=d["positions"],
                cell=d["cell"], pbc=d["pbc"])


def metrics(row):
    a = atoms(row["atoms"])
    f = np.asarray(row["forces"], dtype=float)
    s = np.asarray(row["stress"], dtype=float)
    ds = a.get_all_distances(mic=True)
    ds[np.diag_indices(len(a))] = np.inf
    singular = np.linalg.svd(a.cell.array, compute_uv=False)
    return {"request": row.get("request"), "energy": float(row["energy"]),
            "fmax": float(np.linalg.norm(f, axis=1).max()),
            "stress_max": float(np.abs(s).max()),
            "stress_frobenius": float(np.linalg.norm(s)),
            "volume": float(a.get_volume()),
            "cell_singular_values": singular.tolist(),
            "cell_condition_number": float(singular.max() / singular.min()),
            "minimum_mic_distance": float(ds.min()), "atoms": row["atoms"]}


def charged_rows(path):
    return [json.loads(line) for line in open(path / "evaluations.jsonl")
            if line.strip() and json.loads(line).get("charged") and
            json.loads(line).get("stage") == "search"]


def row_at_exact(rows_by_request, request):
    row = rows_by_request.get(int(request))
    if row is None:
        raise AssertionError(f"missing charged search evaluation at request {request}")
    return metrics(row)


def audit(seed):
    path = ROOT / "comparison" / f"safe_total-seed{seed}"
    data = json.loads((path / "result.json").read_text())
    rows = charged_rows(path)
    rows_by_request = {int(row["request"]): row for row in rows}
    records = data["records"]
    cumulative = 0
    outer = []
    for rec in records:
        if rec.get("stage") == "initial":
            cumulative += int(rec["requests"])
            continue
        start = cumulative
        cycle_rows = []
        for cycle in rec.get("cell_cycles", []):
            cumulative += int(cycle["requests"])
            end = row_at_exact(rows_by_request, cumulative)
            cell_after = np.asarray(cycle["cell_after"], dtype=float)
            if not np.allclose(np.asarray(end["atoms"]["cell"], dtype=float), cell_after,
                               rtol=0., atol=1e-10):
                raise AssertionError(f"cycle cell mismatch at request {cumulative}")
            mode = cycle.get("mode") or {}
            direction = np.asarray(mode.get("direction", []), dtype=float)
            cycle_rows.append({"cycle": int(cycle["index"]),
                "requests": int(cycle["requests"]),
                "status": cycle.get("status"),
                "partial_status": cycle.get("partial_status"),
                "partial_steps": cycle.get("partial_steps"),
                "partial_error": cycle.get("partial_error"),
                "mode": {"converged": mode.get("converged"),
                         "residual_norm": mode.get("residual_norm"),
                         "curvature": mode.get("curvature"),
                         "force_calls": mode.get("force_calls"),
                         "direction_norm": (None if direction.size == 0 else
                                             float(np.linalg.norm(direction))),
                         "displacement_length": cycle.get("distance")},
                "end": end})
        atomic = rec.get("atomic")
        atomic_boundaries = []
        if atomic:
            atomic_request = cumulative
            atomic_events = atomic.get("checkpoint", {}).get("climb", [])
            if atomic_events:
                # The first Gaussian center is the exact post-cell-cycle
                # structure.  This catches an off-by-one ledger alignment
                # before checking subsequent completed boundaries.
                entry = row_at_exact(rows_by_request, atomic_request)
                if not np.allclose(np.asarray(entry["atoms"]["positions"]),
                                   np.asarray(atomic_events[0]["center"]),
                                   rtol=0., atol=1e-10):
                    raise AssertionError("atomic entry does not match first event.center")
            for event in atomic_events:
                rotation_requests = int(event["rotation_force_requests"])
                quench_requests = int(event["quench_requests"])
                atomic_request += rotation_requests + 1 + quench_requests + 1
                boundary = row_at_exact(rows_by_request, atomic_request)
                if abs(float(boundary["energy"]) - float(event["true_energy"])) > 1e-10:
                    raise AssertionError(f"atomic energy mismatch at request {atomic_request}")
                atomic_boundaries.append({"index": event["index"],
                    "true_energy": event["true_energy"],
                    "bias_weight": event["weight"],
                    "request": atomic_request,
                    "rotation_requests": rotation_requests,
                    "height_requests": 1,
                    "quench_requests": quench_requests,
                    "true_energy_requests": 1,
                    "geometry": boundary})
            if atomic_boundaries:
                # Each accepted true-energy boundary is the work structure
                # used as the next Gaussian center.  Compare exact charged
                # records rather than an energy-based nearest-row heuristic.
                for previous, following in zip(atomic_boundaries, atomic_events[1:]):
                    if not np.allclose(
                            np.asarray(previous["geometry"]["atoms"]["positions"]),
                            np.asarray(following["center"]), rtol=0., atol=1e-10):
                        raise AssertionError(
                            f"atomic boundary {previous['index']} does not match "
                            f"next event.center {following['index']}")
                final_boundary = atomic_boundaries[-1]["geometry"]["atoms"]
                checkpoint_atoms = atomic["checkpoint"]["atoms"]
                if not np.allclose(np.asarray(final_boundary["positions"]),
                                   np.asarray(checkpoint_atoms["positions"]), rtol=0., atol=1e-10):
                    raise AssertionError("last atomic boundary positions do not match checkpoint atoms")
                if not np.allclose(np.asarray(final_boundary["cell"]),
                                   np.asarray(checkpoint_atoms["cell"]), rtol=0., atol=1e-10):
                    raise AssertionError("last atomic boundary cell does not match checkpoint atoms")
        outer.append({"index": rec["index"], "status": rec.get("status"),
                      "atomic_scheduled": bool(rec.get("atomic_scheduled")),
                      "requests": int(rec["requests"]), "start_request": start,
                      "cell_cycles": cycle_rows,
                      "entering_atomic": cycle_rows[-1]["end"] if atomic and cycle_rows else None,
                      "atomic_boundaries": atomic_boundaries,
                      "atomic_status": None if atomic is None else atomic.get("status"),
                      "atomic_requests": None if atomic is None else atomic.get("requests")})
        # A record also contains the landing/atomic work after the cell-cycle
        # prefix. Advance from the record boundary, rather than only summing
        # the cycle sub-requests, before locating the next outer record.
        cumulative = start + int(rec["requests"])
    return {"seed": seed, "initial": metrics(next(r for r in rows if int(r["request"]) == 2)),
            "outer": outer}


def main():
    out = {"status": "audited", "zero_pes": True,
           "runs": [audit(7), audit(101)],
           "interpretation": "Large energy or stress values are diagnostics; no physical-failure label is inferred from energy alone."}
    path = ROOT / "cell-energy-diagnosis.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
