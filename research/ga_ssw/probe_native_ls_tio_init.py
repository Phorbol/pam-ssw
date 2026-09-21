"""Bounded Ti/O native-LS initialization comparison.

The existing leaf lookup probe and initialization Oracle are reused.  This
stops at bond_info_init_'s 0x6c7530 boundary and never enters LASP's main
program, PES, custom-file path, or adaptive controller.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.standalone.native_ls import initialize_native_ls
from research.ga_ssw.probe_native_ls_initialization import Oracle, load_elf
import research.ga_ssw.probe_native_ls_pair_table_hco as leaf_probe


ELF_SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"


def raw_tio(elf):
    """Execute the existing isolated leaf probe for Z=8,22."""
    leaf_probe.ELEMENTS = (8, 22)
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        old = sys.argv
        try:
            sys.argv = ["probe_native_ls_pair_table_hco", "--elf", elf,
                        "--output", handle.name]
            leaf_probe.main()
        finally:
            sys.argv = old
        rows = json.loads(Path(handle.name).read_text())["rows"]
    return {
        row["function"] + ":" + ",".join(map(str, row["pair"])):
        row["raw_return"] for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256:
        raise ValueError("unsupported ELF; addresses are version-specific")

    raw = raw_tio(args.elf)
    energies = {(22, 22): raw["bondeneval_:22,22"],
                (22, 8): raw["bondeneval_:22,8"],
                (8, 8): raw["bondeneval_:8,8"]}
    lengths = {(22, 22): raw["bondlenval_:22,22"],
               (22, 8): raw["bondlenval_:22,8"],
               (8, 8): raw["bondlenval_:8,8"]}
    atoms = Atoms(numbers=[22, 8, 8],
                  positions=[[0., 0., 0.], [1.943, 0., 0.],
                             [-1.943, 0., 0.]],
                  cell=np.diag([50., 50., 50.]), pbc=False)
    native = Oracle(segments).run(atoms)
    python = initialize_native_ls(atoms, bond_energies=energies,
                                  bond_lengths=lengths)
    element_order = [22, 8]
    python_energy = [[python.table[tuple(sorted((a, b)))]
                      for b in element_order] for a in element_order]
    python_length = [[python.lengths[tuple(sorted((a, b)))]
                      for b in element_order] for a in element_order]
    f32_factor = float(np.float32(len(atoms)) * np.float32(.02))
    native_energy = np.asarray(native["energy_matrix"])
    native_length = np.asarray(native["length_matrix"])
    py_energy = np.asarray(python_energy)
    py_length = np.asarray(python_length)
    output = {
        "elf_path": str(Path(args.elf).resolve()),
        "elf_sha256": digest,
        "scope": {
            "entry": "bond_info_init_ 0x6c6a30",
            "stop_boundary": "0x6c7530",
            "elements": {"Ti": 22, "O": 8},
            "custom_file": "not used; no-custom initialization branch",
            "pseudopotential_or_pes": False,
            "main_program": False,
            "adaptive_controller": False,
        },
        "raw_lookup": {
            "energy_eV": {"Ti-Ti": energies[(22, 22)],
                          "Ti-O": energies[(22, 8)],
                          "O-O": energies[(8, 8)]},
            "length_A": {"Ti-Ti": lengths[(22, 22)],
                         "Ti-O": lengths[(22, 8)],
                         "O-O": lengths[(8, 8)]},
            "source": "existing probe_native_ls_pair_table_hco.py leaf execution",
        },
        "input": {"numbers": atoms.numbers.tolist(),
                  "positions_A": atoms.positions.tolist(),
                  "cell_A": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist()},
        "native": {
            "bond_count": native["bond_count"],
            "bond_ener_scale": native["bond_ener_scale"],
            "energy_filter": native["energy_filter"],
            "length_filter": native["length_filter"],
            "len_toller_A": native["len_toller"],
            "energy_matrix_B": native["energy_matrix"],
            "length_matrix_L_A": native["length_matrix"],
            "runtime_stubs": native["runtime_stubs"],
        },
        "python": {"bond_count": python.bond_count,
                   "energy_matrix_B": python_energy,
                   "length_matrix_L_A": python_length},
        "formula": {
            "scale": 5.0,
            "f32_N_times_0_02": f32_factor,
            "reference_CC_energy": 3.4468400478363037,
            "TiO_expected_B": energies[(22, 8)] * 5.0 * f32_factor /
                              (native["bond_count"] * 3.4468400478363037),
            "native_python_max_abs_energy_error": float(np.max(np.abs(native_energy - py_energy))),
            "native_python_max_abs_length_error": float(np.max(np.abs(native_length - py_length))),
        },
        "interpretation": (
            "raw lookup values are recovered release reference parameters, "
            "not physical optima or a production default table"
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
