# Batched MACE feature atlas

`pamssw.feature_atlas` is the reusable analysis workflow for structure
coverage. It performs graph-batched MACE inference, element-resolved mean/std
pooling of the final `node_feats`, StandardScaler + PCA, optional UMAP, and
exports source tables, fitted mappings, static PNG/SVG/PDF/600-dpi TIFF
figures, and a fixed-axis GIF.

The plotting colors called “density” are calculated from the standardized
high-dimensional MACE features. UMAP is a local-neighborhood visualization;
its global distances and areas are not physical metrics.

## Install/runtime

The base `pamssw` package intentionally does not depend on MACE or plotting
packages. Run the tool in the configured MACE environment with `ase`,
`mace-torch`, `scikit-learn`, `matplotlib`, `pillow`, and `umap-learn`.

## Structure manifest

JSONL is the preferred audit format. Required fields are `record_id` and
`structure`; the other fields preserve phase and provenance:

```json
{"record_id":"restart:t1:e1","phase":"k8_restart","kind":"accepted_minimum","phase_trial":1,"campaign_trial":5001,"archive_energy_eV":-1151.2,"structure":"accepted_minima/trial0001_entry0001_accepted.xyz"}
```

Relative structure paths are resolved relative to the manifest file. CSV and
JSON arrays with the same field names are also accepted.

An SSW archive can be used directly, including its restart start structure:

```bash
python -m pamssw.feature_atlas_cli \
  --archive-dir runs/pdo_restart \
  --phase k8_restart --campaign-offset 5000 \
  --start-structure starting_structure.xyz \
  --start-energy -1151.8212 \
  --model /path/to/maceomat0smallmodel \
  --output-dir runs/pdo_atlas \
  --batch-size 128 --device cuda --dtype float32 \
  --umap-neighbors 30 --umap-min-dist 0.12 --video-frames 72
```

For multiple campaigns, concatenate their JSONL records into one manifest and
assign non-overlapping `campaign_trial` ranges before running the tool.

## Python API

```python
from pamssw.feature_atlas import AtlasConfig, load_manifest, run_feature_atlas

records = load_manifest("campaign.jsonl")
report = run_feature_atlas(records, AtlasConfig(
    model="/path/to/maceomat0smallmodel",
    output_dir="runs/atlas",
    batch_size=128,
    device="cuda",
    dtype="float32",
    enable_cueq=True,
))
```

The output directory contains `mace_pooled_features.npy`,
`mace_predicted_energy_eV.npy`, `atlas_points.csv`, `pca_mapping.joblib`,
`umap_mapping.joblib`, coordinates, `atlas_summary.json`, and the rendered
figures. The mappings are fitted once and can be reused to project future
pooled features without refitting the atlas.
