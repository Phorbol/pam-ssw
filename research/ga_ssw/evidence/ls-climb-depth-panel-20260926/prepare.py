"""Zero-PES preflight for the archived LS climb-depth mechanism panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence")
SOURCES = (
    {
        "environment": "c60_mh1_omol",
        "source_dir": EVIDENCE / "mh1-native-ls-equal-budget-20260920",
        "files": {
            "seed17093": ("c60_17093-first-native_ls/result.json", (0, 5, 10)),
            "seed17094": ("c60_17094-first-native_ls/result.json", (0, 5, 10)),
        },
    },
    {
        "environment": "tio2_omat_pbe",
        "source_dir": EVIDENCE / "native-ls-tio2-lifecycle-20260920",
        "files": {
            "rutile": ("rutile/result.json", (0, 1, 2)),
            "anatase": ("anatase/result.json", (0, 1, 2)),
        },
    },
)
DEPTH_QUENCH_CAP = 1000
FRESH_PER_ENDPOINT = 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atoms(value, expected_n: int, label: str) -> dict:
    if not isinstance(value, dict) or not all(k in value for k in ("numbers", "positions", "cell", "pbc")):
        raise ValueError(f"{label}: missing serialized ASE geometry fields")
    if len(value["numbers"]) != expected_n or len(value["positions"]) != expected_n:
        raise ValueError(f"{label}: atom count mismatch")
    if any(len(x) != 3 for x in value["positions"]):
        raise ValueError(f"{label}: malformed position array")
    return {k: value[k] for k in ("numbers", "positions", "cell", "pbc")}


def extract() -> dict:
    rows = []
    source_hashes = {}
    for group in SOURCES:
        for label, (relative, selected) in group["files"].items():
            path = group["source_dir"] / relative
            data = json.loads(path.read_text())
            source_hashes[str(path)] = digest(path)
            records = data.get("records")
            if not isinstance(records, list):
                raise ValueError(f"{path}: no records list")
            system_rows = 0
            for index in selected:
                if index >= len(records):
                    raise ValueError(f"{path}: selected record {index} absent")
                record = records[index]
                climb = record.get("climb")
                prep = record.get("ls_preparation")
                landing = record.get("landing")
                if not isinstance(climb, list) or not climb:
                    raise ValueError(f"{path}: record {index} has no Gaussian centers")
                if not isinstance(prep, dict) or not isinstance(landing, dict):
                    raise ValueError(f"{path}: record {index} lacks LS prep/landing")
                center_index = (len(climb) - 1) // 2
                n_atoms = len(landing["atoms"]["numbers"])
                soft = atoms(prep["soft_quench"]["atoms"], n_atoms, f"{path}:{index}:soft")
                center = atoms(climb[center_index].get("center"), n_atoms,
                               f"{path}:{index}:center{center_index}")
                full = atoms(landing.get("atoms"), n_atoms, f"{path}:{index}:landing")
                prefix_requests = [int(x["requests"]) for x in climb]
                if any(value < 0 for value in prefix_requests):
                    raise ValueError(f"{path}: negative climb request count")
                case_id = f"{group['environment']}-{label}-record{index:03d}"
                rows.append({
                    "case_id": case_id,
                    "environment": group["environment"],
                    "source_file": str(path),
                    "source_sha256": source_hashes[str(path)],
                    "record_index": index,
                    "record_status": record.get("status"),
                    "accepted": record.get("accepted"),
                    "n_atoms": n_atoms,
                    "pbc": full["pbc"],
                    "n_climb_centers": len(climb),
                    "midpoint_center_index": center_index,
                    "ls_preparation_requests": int(prep.get("evaluation_requests", 0)),
                    "climb_requests_by_center": prefix_requests,
                    "climb_prefix_requests_to_midpoint": sum(prefix_requests[:center_index + 1]),
                    "full_record_requests": int(record.get("evaluation_requests", 0)),
                    "endpoint_geometries": {
                        "prequench": soft,
                        "midpoint_center": center,
                        "completed_physical_landing": full,
                    },
                })
                system_rows += 1
            if system_rows != len(selected):
                raise ValueError(f"{path}: selected row count mismatch")
    if len(rows) != 12:
        raise ValueError(f"expected 12 fixed attempts, found {len(rows)}")
    return {
        "status": "prepared_not_executed",
        "purpose": "zero-PES extraction of fixed LS depth checkpoints across C60 cluster and periodic TiO2",
        "source_hashes": source_hashes,
        "fixed_sample": [
            {"environment": g["environment"], "cases": {k: list(v[1]) for k, v in g["files"].items()}}
            for g in SOURCES
        ],
        "new_work_ceiling": {
            "new_true_quenches": len(rows) * 2,
            "requests_per_new_quench": DEPTH_QUENCH_CAP,
            "fresh_endpoint_checks": len(rows) * 3 * FRESH_PER_ENDPOINT,
            "max_total_ef_requests": len(rows) * 2 * DEPTH_QUENCH_CAP + len(rows) * 3 * FRESH_PER_ENDPOINT,
        },
        "rows": rows,
    }


def main() -> None:
    output = HERE / "prepared.json"
    payload = (json.dumps(extract(), indent=2, allow_nan=False) + "\n").encode()
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite changed manifest: {output}")
    output.write_bytes(payload)
    print(f"prepared_not_executed: {len(json.loads(payload)['rows'])} attempts; "
          f"max new requests={json.loads(payload)['new_work_ceiling']['max_total_ef_requests']}; zero PES")


if __name__ == "__main__":
    main()
