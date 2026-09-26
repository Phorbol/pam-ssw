#!/usr/bin/env python3
"""Zero-PES readout of one saved seven-image C60 NEB run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HESSIAN_PATH = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/curvature/isomer-2.npz"
ENDPOINT_ANALYSIS_PATH = ROOT / "research/ga_ssw/evidence/c60-displacement-spectrum-20260927/analysis.json"
FMAX_THRESHOLD = 0.05
RECONSTRUCTION_TOLERANCE = 1e-8


def write_unqualified(out_dir: Path, run_dir: Path, status: str, reason: str,
                      summary: dict | None = None) -> None:
    result = {
        "execution_status": status,
        "qualification": "unqualified",
        "reason": reason,
        "run_dir": str(run_dir),
        "available_summary_fields": sorted(summary) if isinstance(summary, dict) else [],
        "no_path_or_barrier_claim": True,
    }
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# C60 NEB 路径诊断",
        "",
        f"执行状态：`{status}`；资格：未通过。",
        "",
        f"原因：{reason}",
        "",
        "当前没有可用的完整七图像终态，因此不报告能量剖面、NEB 虚力或势垒。此状态不触发补算。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def graph_summary(xyz: np.ndarray, cutoff: float) -> dict:
    from scipy.spatial.distance import cdist
    import networkx as nx
    from collections import Counter

    distances = cdist(xyz, xyz)
    np.fill_diagonal(distances, np.inf)
    edges = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(distances < cutoff, 1)))]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xyz)))
    graph.add_edges_from(edges)
    return {
        "cutoff_A": cutoff,
        "edge_count": len(edges),
        "component_count": nx.number_connected_components(graph),
        "connected": nx.is_connected(graph),
        "degree_counts": {str(k): int(v) for k, v in sorted(Counter(dict(graph.degree()).values()).items())},
    }


def analyze(run_dir: Path, out_dir: Path) -> dict:
    from ase.io import read
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.mep import NEB

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        write_unqualified(out_dir, run_dir, "missing_summary", "summary.json is absent")
        return json.loads((out_dir / "analysis.json").read_text())
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        write_unqualified(out_dir, run_dir, "invalid_summary", f"cannot parse summary.json: {exc}")
        return json.loads((out_dir / "analysis.json").read_text())
    if summary.get("status") != "completed":
        write_unqualified(out_dir, run_dir, str(summary.get("status", "unknown")),
                          "run did not finish with status=completed; no image data were qualified", summary)
        return json.loads((out_dir / "analysis.json").read_text())

    rows = summary.get("images")
    if not isinstance(rows, list) or len(rows) != 7:
        write_unqualified(out_dir, run_dir, "completed_but_incomplete", "completed summary lacks exactly seven image records", summary)
        return json.loads((out_dir / "analysis.json").read_text())
    try:
        initial = read(summary["initial_path"], index=0)
        final = read(summary["final_path"], index=0)
        energies = np.asarray([row["energy_eV"] for row in rows], dtype=float)
        positions = [np.asarray(row["positions_A"], dtype=float) for row in rows]
        forces = [np.asarray(row["forces_eV_A"], dtype=float) for row in rows]
    except (KeyError, OSError, ValueError, TypeError) as exc:
        write_unqualified(out_dir, run_dir, "completed_but_invalid_images", f"image/input parsing failed: {exc}", summary)
        return json.loads((out_dir / "analysis.json").read_text())

    n = len(initial)
    if (n != 60 or len(final) != n or any(x.shape != (n, 3) for x in positions + forces)
            or not np.isfinite(energies).all()
            or not all(np.isfinite(x).all() for x in positions + forces)):
        write_unqualified(out_dir, run_dir, "completed_but_invalid_images",
                          "image arrays are non-finite or do not have the expected C60 dimensions", summary)
        return json.loads((out_dir / "analysis.json").read_text())
    if not np.array_equal(initial.numbers, final.numbers) or not np.all(initial.numbers == 6):
        write_unqualified(out_dir, run_dir, "completed_but_invalid_endpoints",
                          "endpoint species/order is not the expected 60-carbon sequence", summary)
        return json.loads((out_dir / "analysis.json").read_text())

    if not (np.allclose(positions[0], initial.positions, rtol=0, atol=1e-10)
            and np.allclose(positions[-1], final.positions, rtol=0, atol=1e-10)):
        write_unqualified(out_dir, run_dir, "changed_endpoints",
                          "saved path endpoints differ from frozen inputs", summary)
        return json.loads((out_dir / "analysis.json").read_text())

    # Reconstruct ASE's NEB forces with frozen per-image E/F; no calculator
    # evaluation or model loading occurs in this analysis.
    images = []
    for xyz, energy, force in zip(positions, energies, forces):
        image = initial.copy()
        image.positions[:] = xyz
        image.calc = SinglePointCalculator(image, energy=float(energy), forces=force)
        images.append(image)
    neb = NEB(images, k=float(summary.get("spring_k_eV_A2", 0.1)),
              climb=True, method="improvedtangent")
    reconstructed = np.asarray(neb.get_forces(), dtype=float)
    reconstructed_fmax = float(np.linalg.norm(reconstructed.reshape((-1, 3)), axis=1).max())
    reported_fmax = float(summary["neb_projected_fmax_eV_A"])
    fmax_difference = abs(reconstructed_fmax - reported_fmax)
    reported_qualified = bool(np.isfinite(reported_fmax) and reported_fmax <= FMAX_THRESHOLD)
    reconstructed_qualified = bool(np.isfinite(reconstructed_fmax) and reconstructed_fmax <= FMAX_THRESHOLD)

    physical_fmax = [float(np.linalg.norm(force, axis=1).max()) for force in forces]
    physical_rms = [float(np.sqrt(np.mean(force**2))) for force in forces]
    delta = energies - energies[0]
    endpoint_delta = float(energies[-1] - energies[0])
    saved_endpoint_delta = float(summary["endpoint_delta_eV"])
    endpoint_delta_difference = abs(endpoint_delta - saved_endpoint_delta)
    distances = []
    graph_rows = []
    for xyz in positions:
        from scipy.spatial.distance import cdist
        pair_distances = cdist(xyz, xyz)
        np.fill_diagonal(pair_distances, np.inf)
        distances.append(float(pair_distances.min()))
        graph_rows.append({str(c): graph_summary(xyz, c) for c in (1.64, 1.8)})

    hessian = np.load(HESSIAN_PATH)
    h_positions = hessian["positions"]
    if h_positions.shape != (n, 3) or not np.allclose(positions[0], h_positions, rtol=0.0, atol=1e-8):
        write_unqualified(out_dir, run_dir, "completed_but_unmatched_hessian",
                          "NEB initial positions do not directly match saved Hessian positions within 1e-8 A; no alignment or reordering was applied", summary)
        return json.loads((out_dir / "analysis.json").read_text())
    basis = hessian["internal_basis"]
    eigenvectors = hessian["eigenvectors"]
    if basis.shape != (180, 174) or eigenvectors.shape != (174, 174):
        write_unqualified(out_dir, run_dir, "completed_but_invalid_hessian",
                          "saved Hessian internal basis or eigenvector dimensions are unexpected", summary)
        return json.loads((out_dir / "analysis.json").read_text())

    def modal_projection(displacement: np.ndarray) -> dict:
        vector = displacement.reshape(-1)
        internal = basis.T @ vector
        internal_norm = float(np.linalg.norm(internal))
        vector_norm = float(np.linalg.norm(vector))
        if internal_norm == 0 or not np.isfinite(internal_norm):
            raise ValueError("zero or non-finite internally projected displacement")
        unit_internal = internal / internal_norm
        weights = (eigenvectors.T @ unit_internal) ** 2
        return {
            "cartesian_norm_A": vector_norm,
            "internal_projected_norm_A": internal_norm,
            "internal_retained_fraction": internal_norm / vector_norm if vector_norm else None,
            "cumulative_squared_projection": {str(m): float(weights[:m].sum()) for m in (1, 5, 10, 20, 50, 174)},
            "projection_weight_sum": float(weights.sum()),
        }

    first_segment_projection = modal_projection(positions[1] - positions[0])
    endpoint_projection = modal_projection(positions[-1] - positions[0])
    endpoint_analysis = None
    if ENDPOINT_ANALYSIS_PATH.is_file():
        prior = json.loads(ENDPOINT_ANALYSIS_PATH.read_text())
        prior_weights = prior["endpoint_displacement"]["cumulative_projection_fraction"]
        endpoint_weights = endpoint_projection["cumulative_squared_projection"]
        endpoint_analysis = {
            "source": str(ENDPOINT_ANALYSIS_PATH),
            "previous_endpoint_cumulative_squared_projection": prior_weights,
            "path_endpoint_Rend_minus_R0_cumulative_squared_projection": endpoint_weights,
            "absolute_projection_difference": {
                str(m): abs(float(endpoint_weights[str(m)]) - float(prior_weights[str(m)]))
                for m in (1, 5, 10, 20, 50, 174)
            },
        }

    result = {
        "execution_status": "completed",
        "qualification": {
            "runtime_reported_neb_projected_fmax_eV_A": reported_fmax,
            "runtime_reported_neb_qualified_at_0.05": reported_qualified,
            "reconstructed_neb_projected_fmax_eV_A": reconstructed_fmax,
            "reconstructed_neb_qualified_at_0.05": reconstructed_qualified,
            "difference_eV_A": fmax_difference,
            "reconstruction_agrees_with_reported_within_1e-8": fmax_difference <= RECONSTRUCTION_TOLERANCE,
            "threshold_eV_A": FMAX_THRESHOLD,
            "endpoint_delta_saved_difference_eV": endpoint_delta_difference,
            "endpoint_delta_consistent_with_summary_within_1e-8_eV": endpoint_delta_difference <= 1e-8,
        },
        "run": {key: summary.get(key) for key in ("calculator", "model", "backend_head", "dtype", "device", "git_head", "n_images", "spring_k_eV_A2", "neb_method", "interpolation", "fmax_eV_A", "total_calculator_calls_started", "total_calculator_calls_completed", "phases")},
        "path": {
            "endpoint_delta_eV": endpoint_delta,
            "relative_image_energies_eV": delta.tolist(),
            "highest_image_index": int(np.argmax(energies)),
            "per_image_physical_fmax_eV_A": physical_fmax,
            "per_image_physical_force_rms_eV_A": physical_rms,
            "per_image_shortest_C_C_A": distances,
            "per_image_graphs": graph_rows,
            "graph_note": "1.64 and 1.8 A cutoff graphs are geometric summaries, not chemical truth.",
        },
        "hessian_projection": {
            "initial_position_max_abs_difference_A": float(np.max(np.abs(positions[0] - h_positions))),
            "first_segment_R1_minus_R0": first_segment_projection,
            "previous_endpoint_displacement_analysis": endpoint_analysis,
            "interpretation": "R1-R0 is a finite first-segment secant, not an exact reaction tangent; endpoint displacement is a separate geometric comparison.",
        },
        "scope_limit": "Zero new PES evaluations. A converged discretized CI-NEB is not a certified transition state and its image-energy span is not an independently validated DFT barrier.",
    }
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out_dir / "report.md").write_text(render_report(result))
    return result


def render_report(result: dict) -> str:
    q = result["qualification"]
    p = result["path"]
    h = result["hessian_projection"]
    lines = [
        "# C60 NEB 路径诊断",
        "",
        f"运行状态：`{result['execution_status']}`。这是保存轨迹的零新 PES 读出。",
        "",
        f"ASE 独立重建的 CI-NEB 最大虚力为 {q['reconstructed_neb_projected_fmax_eV_A']:.8g} eV/Å（运行记录 {q['runtime_reported_neb_projected_fmax_eV_A']:.8g} eV/Å，差 {q['difference_eV_A']:.3g} eV/Å）。按 0.05 eV/Å 阈值，运行记录资格为 {q['runtime_reported_neb_qualified_at_0.05']}，重建资格为 {q['reconstructed_neb_qualified_at_0.05']}；两值在 1e-8 eV/Å 内一致：{q['reconstruction_agrees_with_reported_within_1e-8']}。",
        "",
        f"普通 NEB 阶段收敛：{result['run']['phases'][0]['converged']}（{result['run']['phases'][0]['steps']} 步）；爬升阶段收敛：{result['run']['phases'][1]['converged']}（{result['run']['phases'][1]['steps']} 步）；计算器调用 {result['run']['total_calculator_calls_completed']} 次。",
        "",
        f"端点能差为 {p['endpoint_delta_eV']:.8g} eV；相对初态能量剖面（eV）：`{[round(x, 6) for x in p['relative_image_energies_eV']]}`。逐图像真实原子力最大值（eV/Å）：`{[round(x, 6) for x in p['per_image_physical_fmax_eV_A']]}`。最高能图像索引为 {p['highest_image_index']}。",
        "",
        "每图像最短 C–C 距离（Å）及 1.64/1.8 Å 图连通分量数：",
    ]
    for i, (distance, graphs) in enumerate(zip(p["per_image_shortest_C_C_A"], p["per_image_graphs"])):
        lines.append(f"- 图像 {i}: {distance:.6f} Å；1.64 Å: {graphs['1.64']['component_count']} 个分量；1.8 Å: {graphs['1.8']['component_count']} 个分量。")
    first = h["first_segment_R1_minus_R0"]
    endpoint_comparison = h["previous_endpoint_displacement_analysis"]
    lines += [
        "",
        f"首段有限位移 R1−R0 在初态 174 维内部 Hessian 的累计平方投影（m=1,5,10,20,50,174）为 `{first['cumulative_squared_projection']}`；去刚体后保留范数 {first['internal_projected_norm_A']:.8g}/{first['cartesian_norm_A']:.8g} Å。与既有端点诊断相比，NEB输入的 Rend−R0 累计投影为 `{endpoint_comparison['path_endpoint_Rend_minus_R0_cumulative_squared_projection']}`，逐项绝对差为 `{endpoint_comparison['absolute_projection_difference']}`。",
        "",
        "R1−R0 是有限段割线，不是精确反应切向；它不能充当 SSW 的强制 target。图连通性仅是指定距离阈值下的几何描述。CI-NEB 收敛不认证过渡态，图像能量剖面也不等同于独立验证的 DFT 势垒。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    result = analyze(run_dir, out_dir)
    print(json.dumps({"execution_status": result.get("execution_status"),
                      "qualification": result.get("qualification")}, indent=2))


if __name__ == "__main__":
    main()
