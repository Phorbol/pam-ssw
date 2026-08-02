"""Command-line entry point for :mod:`pamssw.feature_atlas`."""

from __future__ import annotations

import argparse
import json

from .feature_atlas import AtlasConfig, load_manifest, records_from_archive, run_feature_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched MACE feature + PCA/UMAP atlas")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="JSONL/JSON/CSV structure manifest")
    source.add_argument("--archive-dir", help="SSW output directory containing accepted_structures.jsonl")
    parser.add_argument("--model", required=True, help="MACE model file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", default="search")
    parser.add_argument("--campaign-offset", type=int, default=0)
    parser.add_argument("--start-structure")
    parser.add_argument("--start-energy", type=float)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--no-cueq", action="store_true")
    parser.add_argument("--elements", type=int, nargs="+")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.12)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=72)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        records = load_manifest(args.manifest)
    else:
        records = records_from_archive(
            args.archive_dir,
            phase=args.phase,
            campaign_offset=args.campaign_offset,
            start_structure=args.start_structure,
            start_energy_eV=args.start_energy,
        )
    config = AtlasConfig(
        model=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        enable_cueq=not args.no_cueq,
        elements=None if args.elements is None else tuple(args.elements),
        pca_components=args.pca_components,
        umap_enabled=not args.no_umap,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        random_state=args.random_state,
        video_frames=args.video_frames,
        render_gif=not args.no_gif,
        overwrite=args.overwrite,
    )
    print(json.dumps(run_feature_atlas(records, config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
