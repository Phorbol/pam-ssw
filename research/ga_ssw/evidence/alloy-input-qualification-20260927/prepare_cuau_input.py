#!/usr/bin/env python3
"""Prepare the separate Cu30Au30 source structure without touching Ag output."""
from __future__ import annotations

from pathlib import Path

from prepare_input import HERE, prepare_target


def main() -> None:
    prepare_target(
        target_composition={"Cu": 30, "Au": 30},
        formula="Cu30Au30",
        output=HERE / "Cu30Au30_gupta_source.extxyz",
        metadata_path=HERE / "cu30au30-input-metadata.json",
    )


if __name__ == "__main__":
    main()
