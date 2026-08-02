"""Reusable batched-MACE feature, PCA, and UMAP atlas workflow.
MACE and plotting dependencies are optional.  The public API consumes an
ordered :class:`AtlasRecord` manifest, performs one graph-batched inference
pass, and writes an auditable directory containing features, mappings, tables,
static figures, and an optional GIF.
"""
from __future__ import annotations
import csv
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np

@dataclass(frozen=True)
class AtlasRecord:
    record_id: str
    structure: str
    phase: str = "search"
    kind: str = "accepted_minimum"
    phase_trial: int | None = None
    campaign_trial: int | None = None
    archive_energy_eV: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def as_dict(self) -> dict[str, Any]:
        result = {
            "record_id": self.record_id,
            "phase": self.phase,
            "kind": self.kind,
            "phase_trial": self.phase_trial,
            "campaign_trial": self.campaign_trial,
            "archive_energy_eV": self.archive_energy_eV,
            "structure": self.structure,
        }
        result.update(self.metadata)
        return result

@dataclass(frozen=True)
class AtlasConfig:
    model: str
    output_dir: str
    batch_size: int = 128
    device: str = "cuda"
    dtype: str = "float32"
    enable_cueq: bool = True
    elements: tuple[int, ...] | None = None
    pca_components: int = 10
    umap_enabled: bool = True
    umap_neighbors: int = 30
    umap_min_dist: float = 0.12
    umap_metric: str = "euclidean"
    random_state: int = 0
    video_frames: int = 72
    render_gif: bool = True
    overwrite: bool = False

def _numeric(value: Any, integer: bool = False):
    if value in (None, "", "None", "null"):
        return None
    return int(value) if integer else float(value)

def _make_record(raw: dict[str, Any], base: Path) -> AtlasRecord:
    if not raw.get("record_id") or not raw.get("structure"):
        raise ValueError("every atlas record requires record_id and structure")
    path = Path(str(raw["structure"]))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    known = {"record_id", "structure", "phase", "kind", "phase_trial", "campaign_trial", "archive_energy_eV"}
    return AtlasRecord(
        record_id=str(raw["record_id"]),
        structure=str(path),
        phase=str(raw.get("phase", "search")),
        kind=str(raw.get("kind", "accepted_minimum")),
        phase_trial=_numeric(raw.get("phase_trial"), True),
        campaign_trial=_numeric(raw.get("campaign_trial"), True),
        archive_energy_eV=_numeric(raw.get("archive_energy_eV")),
        metadata={key: value for key, value in raw.items() if key not in known},
    )

def load_manifest(path: str | Path) -> list[AtlasRecord]:
    """Load JSONL, JSON-array, or CSV records and resolve relative paths."""
    manifest = Path(path).resolve()
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="") as handle:
            raw = list(csv.DictReader(handle))
    elif manifest.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(manifest.read_text())
        raw = payload.get("records", payload) if isinstance(payload, dict) else payload
    records = [_make_record(row, manifest.parent) for row in raw]
    ids = [row.record_id for row in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("manifest is empty or contains duplicate record_id values")
    return sorted(records, key=lambda row: (row.campaign_trial is None, row.campaign_trial or 0, row.record_id))

def records_from_archive(
    archive_dir: str | Path,
    *,
    phase: str = "search",
    campaign_offset: int = 0,
    start_structure: str | Path | None = None,
    start_energy_eV: float | None = None,
) -> list[AtlasRecord]:
    """Build a manifest from an SSW accepted-structures directory."""
    root = Path(archive_dir).resolve()
    log = root / "accepted_structures.jsonl"
    if not log.is_file():
        raise FileNotFoundError(log)
    result: list[AtlasRecord] = []
    if start_structure is not None:
        start = Path(start_structure)
        if not start.is_absolute():
            start = (root / start).resolve()
        result.append(AtlasRecord(f"{phase}:start", str(start), phase, "restart_start", 0, campaign_offset, start_energy_eV))
    for line in log.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
            trial, entry, energy = int(row["trial_index"]), int(row["discovered_entry_id"]), float(row["energy"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        structure = root / "accepted_minima" / f"trial{trial:04d}_entry{entry:04d}_accepted.xyz"
        if structure.is_file():
            result.append(AtlasRecord(
                f"{phase}:t{trial}:e{entry}", str(structure), phase, "accepted_minimum", trial,
                campaign_offset + trial, energy, {"seed_entry_id": row.get("seed_entry_id")},
            ))
    if not result:
        raise ValueError(f"no usable structures in {root}")
    return sorted(result, key=lambda row: (row.campaign_trial or 0, row.record_id))

def write_manifest(records: Sequence[AtlasRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row.as_dict(), sort_keys=True) + "\n" for row in records))

class MACEFeatureExtractor:
    """MACE graph-batched inference with element-resolved mean/std pooling."""
    def __init__(self, model: str, *, batch_size: int = 128, device: str = "cuda", dtype: str = "float32", enable_cueq: bool = True, elements: Sequence[int] | None = None, force_pbc: Sequence[bool] | None = None):
        self.model, self.batch_size, self.device = str(model), int(batch_size), device
        self.dtype, self.enable_cueq = dtype, bool(enable_cueq)
        self.elements = tuple(int(z) for z in elements) if elements is not None else None
        self.force_pbc = tuple(bool(value) for value in force_pbc) if force_pbc is not None else None
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
    def _graphs(self, records: Sequence[AtlasRecord]):
        try:
            from ase.io import read
            from mace import data as mace_data
            from mace.calculators import MACECalculator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MACE feature extraction requires ASE and MACE") from exc
        calculator = MACECalculator(model_paths=self.model, device=self.device, default_dtype=self.dtype, enable_cueq=self.enable_cueq)
        atoms_list = []
        for record in records:
            atoms = read(record.structure)
            if self.force_pbc is not None:
                atoms.pbc = self.force_pbc
            atoms_list.append(atoms)
        if self.elements is None:
            self.elements = tuple(sorted({int(z) for atoms in atoms_list for z in atoms.numbers}))
        z_to_column = {int(z): i for i, z in enumerate(calculator.z_table.zs)}
        missing = [z for z in self.elements if z not in z_to_column]
        if missing:
            raise ValueError(f"model does not contain atomic numbers {missing}")
        graphs = []
        for atoms in atoms_list:
            config = mace_data.config_from_atoms(atoms, head_name=calculator.head)
            graphs.append(mace_data.AtomicData.from_config(config, z_table=calculator.z_table, cutoff=calculator.r_max, heads=calculator.available_heads))
        return calculator, graphs, [z_to_column[z] for z in self.elements]
    @staticmethod
    def _pool(node_features, batch_ids, node_attrs, n_graphs, columns):
        import torch
        chunks = []
        width = node_features.shape[1]
        for column in columns:
            mask = node_attrs[:, column] > 0.5
            ids, selected = batch_ids[mask], node_features[mask]
            sums = torch.zeros((n_graphs, width), device=node_features.device, dtype=node_features.dtype)
            sums.index_add_(0, ids, selected)
            squared = torch.zeros_like(sums)
            squared.index_add_(0, ids, selected.square())
            counts = torch.bincount(ids, minlength=n_graphs).to(node_features.dtype).unsqueeze(1).clamp_min(1.0)
            means = sums / counts
            chunks.extend((means, (squared / counts - means.square()).clamp_min(0).sqrt()) )
        return torch.cat(chunks, dim=1)
    def infer(self, records: Sequence[AtlasRecord]):
        import torch
        from mace.tools.torch_geometric import Batch
        calculator, graphs, columns = self._graphs(records)
        model = calculator.models[0].eval()
        features, energies = [], []
        for start in range(0, len(graphs), self.batch_size):
            batch = Batch.from_data_list(graphs[start : start + self.batch_size]).to(self.device)
            with torch.inference_mode():
                output = model(batch.to_dict(), compute_force=False)
                features.append(self._pool(output["node_feats"], batch.batch, batch.node_attrs, batch.num_graphs, columns).detach().cpu().numpy())
                energies.append(output["energy"].detach().reshape(-1).cpu().numpy())
            print(f"[feature-atlas] MACE {min(start + self.batch_size, len(graphs))}/{len(graphs)}", flush=True)
        return np.concatenate(features), np.concatenate(energies), self.elements or ()

def _density(features: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    if len(features) < 2:
        return np.ones(len(features))
    distances = NearestNeighbors(n_neighbors=min(26, len(features))).fit(features).kneighbors(features)[0][:, -1]
    values = -np.log(distances + 1e-12)
    low, high = np.quantile(values, (.02, .98))
    return np.clip((values - low) / max(high - low, 1e-12), 0, 1)

def _dump_mapping(payload: dict[str, Any], path: Path) -> None:
    try:
        import joblib
        joblib.dump(payload, path)
    except ImportError:  # pragma: no cover
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

def _save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def _render(output: Path, prefix: str, records: Sequence[AtlasRecord], coords: np.ndarray, density: np.ndarray, energies: np.ndarray, video_frames: int, render_gif: bool, explained: Sequence[float] | None = None) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False})
    relative = energies - np.nanmin(energies)
    first = min(range(len(records)), key=lambda i: records[i].campaign_trial if records[i].campaign_trial is not None else i)
    best = int(np.nanargmin(energies))
    axis_names = (f"{prefix.upper()} 1", f"{prefix.upper()} 2") if explained is None else (f"PC1 ({explained[0]*100:.1f}% feature variance)", f"PC2 ({explained[1]*100:.1f}% feature variance)")
    def marks(axis):
        for index, label in ((first, "first retained"), (best, "current global best")):
            axis.scatter(*coords[index], marker="*", s=88, c="white", edgecolor="black", linewidth=.7, zorder=5)
            axis.annotate(label, coords[index], xytext=(8, 8), textcoords="offset points", fontsize=7)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    artist = ax.scatter(coords[:, 0], coords[:, 1], c=density, cmap="cividis", s=8, linewidths=0, rasterized=True)
    fig.colorbar(artist, ax=ax, pad=.02, label="high-dimensional MACE feature density (quantile-scaled)")
    marks(ax); ax.set(xlabel=axis_names[0], ylabel=axis_names[1], title=f"{prefix.upper()} structural coverage"); ax.grid(alpha=.15)
    _save_figure(fig, output / f"{prefix}_density_coverage")
    cap = max(2., float(np.quantile(relative, .98)))
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    artist = ax.scatter(coords[:, 0], coords[:, 1], c=np.minimum(relative, cap), cmap="magma_r", norm=Normalize(0, cap), s=8, linewidths=0, rasterized=True)
    fig.colorbar(artist, ax=ax, pad=.02, extend="max", label="MACE energy above current best (eV)")
    marks(ax); ax.set(xlabel=axis_names[0], ylabel=axis_names[1], title=f"{prefix.upper()} relative energy"); ax.grid(alpha=.15)
    _save_figure(fig, output / f"{prefix}_relative_energy")
    phases = list(dict.fromkeys(record.phase for record in records)); colors = ["#606060", "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#dd8452"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.1, 1.]})
    for number, phase in enumerate(phases):
        ids = np.asarray([i for i, record in enumerate(records) if record.phase == phase])
        left.scatter(coords[ids, 0], coords[ids, 1], s=8, alpha=.5, color=colors[number % len(colors)], label=phase.replace("_", " "), rasterized=True)
    marks(left); left.set(xlabel=axis_names[0], ylabel=axis_names[1], title=f"{prefix.upper()} coverage by phase"); left.legend(fontsize=7); left.grid(alpha=.15)
    order = np.asarray(sorted(range(len(records)), key=lambda i: records[i].campaign_trial if records[i].campaign_trial is not None else i)); trials = np.asarray([records[i].campaign_trial if records[i].campaign_trial is not None else i for i in order]); values = energies[order]
    right.plot(trials, values, color="#999", alpha=.32, linewidth=.45, label="recorded minima"); right.step(trials, np.minimum.accumulate(values), where="post", color="#c44e52", linewidth=1.8, label="global best"); right.set(xlabel="campaign trial", ylabel="MACE energy (eV)", title="Energetic evolution"); right.legend(fontsize=7); right.grid(alpha=.15)
    _save_figure(fig, output / f"{prefix}_phase_evolution")
    if not render_gif:
        return
    frame_dir = output / f"{prefix}_evolution_frames"; frame_dir.mkdir(parents=True, exist_ok=True); images = []
    counts = np.unique(np.linspace(1, len(order), min(video_frames, len(order)), dtype=int)); xpad = max(.05, .08*np.ptp(coords[order, 0])); ypad = max(.05, .08*np.ptp(coords[order, 1])); limits = (coords[order, 0].min()-xpad, coords[order, 0].max()+xpad, coords[order, 1].min()-ypad, coords[order, 1].max()+ypad)
    for number, count in enumerate(counts):
        selected = order[:count]; selected_trials, selected_values = trials[:count], energies[selected]; current = selected[int(np.argmin(selected_values))]
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.1, .9]})
        left.scatter(coords[selected, 0], coords[selected, 1], c=np.minimum(relative[selected], cap), cmap="magma_r", norm=Normalize(0, cap), s=8, linewidths=0, rasterized=True); left.scatter(*coords[current], marker="*", s=100, c="white", edgecolor="black", linewidth=.7, zorder=5); left.set(xlim=limits[:2], ylim=limits[2:], xlabel=axis_names[0], ylabel=axis_names[1], title=f"Coverage through campaign trial {selected_trials[-1]}"); left.grid(alpha=.15)
        right.plot(selected_trials, selected_values, color="#999", alpha=.32, linewidth=.45); right.step(selected_trials, np.minimum.accumulate(selected_values), where="post", color="#c44e52", linewidth=1.8); right.set(xlabel="campaign trial", ylabel="MACE energy (eV)", title="Sampled minima and incumbent"); right.grid(alpha=.15); fig.tight_layout()
        frame = frame_dir / f"frame_{number:03d}.png"; fig.savefig(frame, dpi=300); plt.close(fig); images.append(Image.open(frame).convert("P", palette=Image.ADAPTIVE))
    if images:
        images[0].save(output / f"{prefix}_campaign_evolution.gif", save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)

def _write_points(output: Path, records: Sequence[AtlasRecord], energies: np.ndarray, density: np.ndarray, pca: np.ndarray | None, umap: np.ndarray | None) -> None:
    fields = ["record_id", "phase", "kind", "phase_trial", "campaign_trial", "archive_energy_eV", "mace_energy_eV", "relative_mace_energy_eV", "feature_density_quantile", "structure", "pca1", "pca2", "umap1", "umap2"]
    with (output / "atlas_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); best = float(np.nanmin(energies))
        for i, record in enumerate(records):
            writer.writerow({"record_id": record.record_id, "phase": record.phase, "kind": record.kind, "phase_trial": record.phase_trial, "campaign_trial": record.campaign_trial, "archive_energy_eV": record.archive_energy_eV, "mace_energy_eV": float(energies[i]), "relative_mace_energy_eV": float(energies[i]-best), "feature_density_quantile": float(density[i]), "structure": record.structure, "pca1": None if pca is None else float(pca[i, 0]), "pca2": None if pca is None else float(pca[i, 1]), "umap1": None if umap is None else float(umap[i, 0]), "umap2": None if umap is None else float(umap[i, 1])})

def run_feature_atlas(records: Sequence[AtlasRecord], config: AtlasConfig) -> dict[str, Any]:
    """Run batched MACE extraction and export PCA/UMAP artifacts."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise ValueError("feature atlas requires at least one structure record")
    if config.pca_components not in (0,) and (config.pca_components < 2 or len(records) < 2):
        raise ValueError("PCA plotting requires pca_components >= 2 and at least two records")
    if config.umap_enabled and (len(records) < 3 or config.umap_neighbors < 2):
        raise ValueError("UMAP requires at least three records and umap_neighbors >= 2")
    extractor = MACEFeatureExtractor(config.model, batch_size=config.batch_size, device=config.device, dtype=config.dtype, enable_cueq=config.enable_cueq, elements=config.elements)
    features, energies, elements = extractor.infer(records); scaler = StandardScaler().fit(features); standardized = scaler.transform(features); density = _density(standardized)
    np.save(output / "mace_pooled_features.npy", features); np.save(output / "mace_predicted_energy_eV.npy", energies); write_manifest(records, output / "atlas_manifest.jsonl")
    pca_coords = None; pca_report = None
    if config.pca_components > 0:
        pca = PCA(n_components=min(config.pca_components, *standardized.shape), random_state=config.random_state).fit(standardized); pca_coords = pca.transform(standardized); np.save(output / "pca_coordinates.npy", pca_coords[:, :2]); _dump_mapping({"scaler": scaler, "pca": pca, "elements": elements}, output / "pca_mapping.joblib"); _render(output, "pca", records, pca_coords[:, :2], density, energies, config.video_frames, config.render_gif, pca.explained_variance_ratio_); pca_report = {"components": pca.n_components_, "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    umap_coords = None; umap_report = None
    if config.umap_enabled:
        try:
            import umap
            from sklearn.manifold import trustworthiness
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("UMAP output requires umap-learn") from exc
        umap_neighbors = min(config.umap_neighbors, len(records) - 1)
        mapper = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_state=config.random_state,
            n_jobs=1,
        )
        umap_coords = mapper.fit_transform(standardized)
        np.save(output / "umap_coordinates.npy", umap_coords)
        _dump_mapping({"scaler": scaler, "umap": mapper, "elements": elements}, output / "umap_mapping.joblib")
        _render(output, "umap", records, umap_coords, density, energies, config.video_frames, config.render_gif)
        if len(records) >= 4:
            trust_k = min(15, len(records) // 2 - 1)
            trust_score = float(trustworthiness(standardized, umap_coords, n_neighbors=trust_k))
        else:
            trust_k = None
            trust_score = None
        umap_report = {
            "n_neighbors": umap_neighbors,
            "min_dist": config.umap_min_dist,
            "metric": config.umap_metric,
            "random_state": config.random_state,
            "trustworthiness_k": trust_k,
            "trustworthiness": trust_score,
            "warning": "UMAP is a local-neighborhood visualization; global distances and area are not physical metrics.",
        }
    _write_points(output, records, energies, density, pca_coords[:, :2] if pca_coords is not None else None, umap_coords)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "n_records": len(records), "feature_width": int(features.shape[1]), "elements": list(elements), "model": config.model, "dtype": config.dtype, "enable_cueq": config.enable_cueq, "batch_size": config.batch_size, "phase_counts": {phase: sum(record.phase == phase for record in records) for phase in sorted({record.phase for record in records})}, "best_mace_energy_eV": float(np.nanmin(energies)), "best_record_id": records[int(np.nanargmin(energies))].record_id, "pca": pca_report, "umap": umap_report, "density_semantics": "high-dimensional standardized MACE-feature density"}
    (output / "atlas_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); (output / "atlas_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"); (output / "figure_contract.md").write_text("# MACE feature atlas\n\nMACE inference is batched. PCA and UMAP are fitted once on the standardized feature matrix. Density is calculated in high-dimensional feature space; UMAP distances are local-neighborhood visualization coordinates.\n")
    return report

def transform_features(mapping_path: str | Path, features: np.ndarray) -> np.ndarray:
    """Project new pooled features with a saved PCA or UMAP mapping."""
    try:
        import joblib
        mapping = joblib.load(mapping_path)
    except ImportError:  # pragma: no cover
        with Path(mapping_path).open("rb") as handle:
            mapping = pickle.load(handle)
    scaled = mapping["scaler"].transform(np.asarray(features))
    model = mapping.get("pca", mapping.get("umap"))
    if model is None:
        raise ValueError("mapping does not contain a PCA or UMAP model")
    return model.transform(scaled)

__all__ = ["AtlasConfig", "AtlasRecord", "MACEFeatureExtractor", "load_manifest", "records_from_archive", "run_feature_atlas", "transform_features", "write_manifest"]
