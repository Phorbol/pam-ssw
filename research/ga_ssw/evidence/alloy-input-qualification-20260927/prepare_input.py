#!/usr/bin/env python3
"""Extract the Ag30Au30, 60-atom block from the archived CC-BY SI.

This is input preparation only.  The source energy is a Gupta-potential label
from the paper and must not be interpreted as an OMAT energy or target.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import write


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rsos190342_si_001.txt"
FIGSHARE_METADATA = HERE / "figshare-metadata.json"
OUTPUT = HERE / "Ag30Au30_gupta_source.extxyz"
METADATA = HERE / "input-metadata.json"

ARTICLE_DOI = "10.1098/rsos.190342"
DATASET_DOI = "10.6084/m9.figshare.9164993.v2"
DATASET_URL = (
    "https://rs.figshare.com/articles/dataset/"
    "Structures_of_Ag-Au_and_Cu-Au_clusters_from_Theoretical_study_of_the_"
    "structures_of_bimetallic_Ag_Au_and_Cu_Au_clusters_up_to_108_atoms/9164993"
)
LICENSE = "CC BY 4.0"


def read_target_block(lines: list[str]):
    """Return the unique N=60, Ag30Au30 data block and its 1-based source lines."""
    matches = []
    i = 0
    while i < len(lines) - 1:
        size_match = re.fullmatch(r"\s*(\d+)\s*", lines[i])
        if not size_match or not lines[i + 1].lstrip().startswith("energy:"):
            i += 1
            continue

        n_atoms = int(size_match.group(1))
        energy_match = re.search(
            r"energy:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*eV",
            lines[i + 1],
        )
        if not energy_match:
            raise ValueError(f"unparsed energy field at source line {i + 2}")
        atoms_rows = []
        j = i + 2
        while j < len(lines) and len(atoms_rows) < n_atoms:
            fields = lines[j].split()
            if len(fields) == 4 and fields[0] in {"Ag", "Au", "Cu"}:
                try:
                    xyz = [float(x) for x in fields[1:]]
                except ValueError as exc:
                    raise ValueError(f"bad coordinate at source line {j + 1}") from exc
                if not np.isfinite(xyz).all():
                    raise ValueError(f"non-finite coordinate at source line {j + 1}")
                atoms_rows.append((fields[0], xyz, j + 1))
            elif fields:
                raise ValueError(f"unexpected non-atom row at source line {j + 1}")
            j += 1
        if len(atoms_rows) != n_atoms:
            raise ValueError(
                f"block at source line {i + 1}: expected {n_atoms} atom rows, "
                f"found {len(atoms_rows)}"
            )

        composition = Counter(symbol for symbol, _, _ in atoms_rows)
        if n_atoms == 60 and composition == Counter({"Ag": 30, "Au": 30}):
            matches.append(
                {
                    "header_line": i + 1,
                    "energy_line": i + 2,
                    "first_atom_line": atoms_rows[0][2],
                    "last_atom_line": atoms_rows[-1][2],
                    "n_atoms": n_atoms,
                    "composition": dict(sorted(composition.items())),
                    "source_energy_eV": float(energy_match.group(1)),
                    "rsuc": (
                        int(m.group(1))
                        if (m := re.search(r"Rsuc:\s*(\d+)", lines[i + 1]))
                        else None
                    ),
                    "symbols": [symbol for symbol, _, _ in atoms_rows],
                    "positions_A": [xyz for _, xyz, _ in atoms_rows],
                }
            )
        i = j

    if len(matches) != 1:
        raise ValueError(f"expected exactly one Ag30Au30, N=60 block; found {len(matches)}")
    return matches[0]


def main() -> None:
    if not SOURCE.is_file() or not FIGSHARE_METADATA.is_file():
        raise FileNotFoundError("expected archived SI and Figshare metadata beside script")
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite existing prepared input or metadata")

    source_bytes = SOURCE.read_bytes()
    lines = source_bytes.decode("utf-8").splitlines()
    block = read_target_block(lines)
    positions = np.asarray(block["positions_A"], dtype=float)
    atoms = Atoms(
        symbols=block["symbols"],
        positions=positions,
        pbc=False,
    )
    distances = atoms.get_all_distances(mic=False)
    distances[np.diag_indices_from(distances)] = np.inf
    bbox = np.ptp(positions, axis=0)

    # Keep extxyz self-describing, while retaining the full trace in JSON.
    atoms.info.update(
        source_article_doi=ARTICLE_DOI,
        source_dataset_doi=DATASET_DOI,
        source_block_header_line=block["header_line"],
        source_energy_eV_gupta=block["source_energy_eV"],
        source_license=LICENSE,
    )
    write(OUTPUT, atoms, format="extxyz")

    figshare = json.loads(FIGSHARE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        "purpose": "input preparation and geometric audit only; no PES calls",
        "source": {
            "article_title": (
                "Theoretical study of the structures of bimetallic Ag-Au and "
                "Cu-Au clusters up to 108 atoms"
            ),
            "authors": [
                "Rongbin Du", "Sai Tang", "Xia Wu", "Yiqing Xu", "Run Chen", "Tao Liu"
            ],
            "article_doi": ARTICLE_DOI,
            "dataset_doi": DATASET_DOI,
            "dataset_url": DATASET_URL,
            "dataset_file": SOURCE.name,
            "dataset_file_bytes": len(source_bytes),
            "dataset_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "figshare_metadata_license": figshare.get("license", {}).get("name"),
            "license": LICENSE,
            "method_as_reported": "modified adaptive immune optimization algorithm with Gupta potential",
            "block_source_lines_1_based": {
                "header": block["header_line"],
                "energy": block["energy_line"],
                "first_atom": block["first_atom_line"],
                "last_atom": block["last_atom_line"],
            },
            "source_energy_eV": block["source_energy_eV"],
            "source_energy_model": "Gupta potential; source label only, not OMAT energy/target",
            "source_Rsuc_over_100_runs": block["rsuc"],
        },
        "structure": {
            "formula": "Ag30Au30",
            "n_atoms": len(atoms),
            "element_counts": dict(sorted(Counter(atoms.get_chemical_symbols()).items())),
            "pbc": [False, False, False],
            "cell_A": atoms.cell.array.tolist(),
            "minimum_pair_distance_A": float(np.min(distances)),
            "bounding_box_extent_A": bbox.tolist(),
            "coordinates_transformed": False,
        },
        "interpretation_boundary": (
            "The paper's Gupta energy and putative-global-minimum designation apply to its "
            "reported Gupta search. They are not an OMAT energy, OMAT minimum certificate, "
            "or target label for a changed-potential qualification."
        ),
        "output_extxyz": OUTPUT.name,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"structure": OUTPUT.name, "metadata": METADATA.name,
                      "n_atoms": len(atoms), "composition": metadata["structure"]["element_counts"],
                      "source_lines": metadata["source"]["block_source_lines_1_based"],
                      "source_energy_eV_gupta": block["source_energy_eV"],
                      "minimum_pair_distance_A": metadata["structure"]["minimum_pair_distance_A"],
                      "bounding_box_extent_A": bbox.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
