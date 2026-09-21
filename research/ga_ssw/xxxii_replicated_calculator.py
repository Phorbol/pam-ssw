"""Research-only explicit periodic representation for the audited XXXII model.

The calculator evaluates an explicitly supplied ``N * prod(repetitions)``
lifted structure in one replicated LAMMPS engine, then returns energy per
replica and the mean force over corresponding 172-atom images.  It does not
auto-replicate caller coordinates and is not connected to an SSW walker.
"""

import ctypes
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammps.coordinatetransform import Prism
from ase.stress import full_3x3_to_voigt_6_stress

from research.ga_ssw.convert_xxxii_amber import DATA_SHA, INPUT_SHA, sections
from research.ga_ssw.xxxii_lammps_calculator import (
    ENERGY_TO_EV,
    REAL_NKTV2P,
    TYPE_NUMBERS,
)


class ReplicatedDomainError(ValueError):
    """Raised before engine evaluation when a frozen graph pair crosses a half box."""


class XXXIIReplicatedCalculator(Calculator):
    """Evaluate explicit XXXII periodic replicas with one persistent engine.

    ``repetitions`` is a required positive integer tuple.  ``reference_atoms``
    supplies the ordered 172-atom identity and primitive cell; the caller must
    pass already lifted coordinates in the corresponding repeated-cell order.
    The original data/input/manifest are required and must be the audited
    converted files.  This is an experimental numerical representation check,
    not a new physical model.  Table-0 Ewald accuracy (1e-12) and gewald
    0.47570069 come from the existing adapter/input contract.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, *, data_path, input_path, model_manifest,
                 reference_atoms, repetitions):
        super().__init__()
        self.data_path = Path(data_path).resolve()
        self.input_path = Path(input_path).resolve()
        self.model_manifest = Path(model_manifest).resolve()
        if (not isinstance(repetitions, tuple) or len(repetitions) != 3
                or any(isinstance(x, (bool, np.bool_))
                       or not isinstance(x, (int, np.integer)) or x <= 0
                       for x in repetitions)):
            raise ValueError("repetitions must be a tuple of three positive integers")
        self.repetitions = tuple(int(x) for x in repetitions)
        self.replicas = int(np.prod(self.repetitions))
        manifest = json.loads(self.model_manifest.read_text())
        if manifest["source"]["data_sha256"] != DATA_SHA or manifest["source"]["input_sha256"] != INPUT_SHA:
            raise ValueError("audited XXXII source manifest required")
        for name, path in (("data", self.data_path), ("input", self.input_path)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["output_sha256"][name]:
                raise ValueError(f"converted {name} SHA mismatch")
        data = sections(self.data_path.read_text())
        atom_rows = sorted(data["Atoms"], key=lambda row: int(row[0]))
        if [int(row[0]) for row in atom_rows] != list(range(1, 173)):
            raise ValueError("172 contiguous original atom IDs required")
        self._numbers = np.array([TYPE_NUMBERS[int(row[2])] for row in atom_rows])
        self._types = np.array([int(row[2]) for row in atom_rows], dtype=np.int32)
        self._charges = np.array([float(row[3]) for row in atom_rows])
        self._graph_pairs = self._distance_three_pairs(data)
        self._reference_cell = np.array(reference_atoms.cell.array, dtype=float, copy=True)
        self._reference_numbers = np.array(reference_atoms.numbers, dtype=int, copy=True)
        if self._reference_numbers.shape != (172,) or not np.array_equal(self._reference_numbers, self._numbers):
            raise ValueError("reference_atoms must contain the ordered 172-atom XXXII species")
        if not np.isfinite(self._reference_cell).all() or np.linalg.det(self._reference_cell) <= 0:
            raise ValueError("reference_atoms must have a finite positive primitive cell")
        self._lmp = None
        self.closed = False
        self.api_calls = 0
        self.engine_calls = 0
        self.atoms_evaluated = 0
        self.requests = 0
        self.last_force_image_max_difference = None

    @staticmethod
    def _distance_three_pairs(data):
        adjacency = defaultdict(set)
        for row in data["Bonds"]:
            i, j = int(row[2]) - 1, int(row[3]) - 1
            adjacency[i].add(j)
            adjacency[j].add(i)
        pairs = set()
        for start in range(172):
            distances = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                if distances[current] == 3:
                    continue
                for neighbor in adjacency[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            pairs.update((min(start, end), max(start, end))
                         for end, distance in distances.items()
                         if 0 < distance <= 3)
        return tuple(sorted(pairs))

    def _check_identity(self, atoms):
        if len(atoms) != 172:
            raise ValueError("public XXXII input must contain the original 172 atoms")
        if not np.array_equal(atoms.numbers, self._numbers):
            raise ValueError("public XXXII input must preserve the ordered species")
        if not np.asarray(atoms.pbc, dtype=bool).all():
            raise ValueError("replicated XXXII representation requires full PBC")
        if (not np.isfinite(atoms.positions).all()
                or not np.isfinite(atoms.cell.array).all()
                or np.linalg.det(atoms.cell.array) <= 0):
            raise ValueError("public positions and cell must be finite with positive volume")

    def _check_graph_domain(self, atoms):
        prism = Prism(atoms.cell.array, pbc=True, reduce_cell=False)
        restricted = np.asarray(prism.vector_to_lammps(atoms.positions, wrap=False))
        box = np.asarray(prism.get_lammps_prism()[:3], dtype=float)
        half = box / 2.0
        for image in range(self.replicas):
            offset = image * 172
            for i, j in self._graph_pairs:
                delta = restricted[offset + j] - restricted[offset + i]
                if np.any(np.abs(delta) >= half):
                    raise ReplicatedDomainError(
                        f"graph pair ({i}, {j}) image {image} crosses restricted half-box: "
                        f"abs(delta)={np.abs(delta).tolist()}, half_box={half.tolist()}")

    def _new_engine(self):
        from lammps import lammps
        return lammps(cmdargs=["-log", "none", "-screen", "none"])

    def _initialize(self):
        engine = self._new_engine()
        try:
            for raw in self.input_path.read_text().splitlines():
                command = raw.split("#", 1)[0].strip()
                if not command:
                    continue
                if command.startswith("read_data "):
                    command = f'read_data "{self.data_path}"'
                engine.command(command)
            if self.replicas > 1:
                engine.command("replicate " + " ".join(map(str, self.repetitions)))
            engine.command("pair_modify table 0")
            engine.command("kspace_style ewald 1e-12")
            engine.command("kspace_modify gewald 0.47570069")
            engine.command("compute pam_virial all pressure NULL virial")
            engine.command("thermo_style custom step pe c_pam_virial[1] c_pam_virial[2] c_pam_virial[3] c_pam_virial[4] c_pam_virial[5] c_pam_virial[6]")
            engine.command("thermo 1")
            self._verify_engine_identity(engine)
        except BaseException:
            engine.close()
            raise
        self._lmp = engine

    def _verify_engine_identity(self, engine):
        count = 172 * self.replicas
        if int(engine.get_natoms()) != count:
            raise RuntimeError("replicated LAMMPS atom count changed")
        types = np.ctypeslib.as_array(engine.gather_atoms("type", 0, 1), shape=(count,))
        charges = np.ctypeslib.as_array(engine.gather_atoms("q", 1, 1), shape=(count,))
        if not np.array_equal(types, np.tile(self._types, self.replicas)):
            raise RuntimeError("replicated per-ID force-field types changed")
        if not np.array_equal(charges, np.tile(self._charges, self.replicas)):
            raise RuntimeError("replicated per-ID charges changed")

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        if self.closed:
            raise RuntimeError("calculator explicitly closed")
        self.results = {}
        self.api_calls += 1
        self._check_identity(atoms)
        expanded = atoms.repeat(self.repetitions)
        self._check_graph_domain(expanded)
        super().calculate(atoms, properties, system_changes)
        if self._lmp is None:
            self._initialize()
        prism = Prism(expanded.cell.array, pbc=True, reduce_cell=False)
        xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
        self.requests += 1
        lmp = self._lmp
        lmp.command(f"change_box all x final 0 {xhi:.17g} y final 0 {yhi:.17g} z final 0 {zhi:.17g} xy final {xy:.17g} xz final {xz:.17g} yz final {yz:.17g} units box")
        lmp.command("set atom * image 0 0 0")
        positions = np.ascontiguousarray(prism.vector_to_lammps(expanded.positions, wrap=False), dtype=np.float64)
        lmp.scatter_atoms("x", 1, 3, positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.engine_calls += 1
        self.atoms_evaluated += len(expanded)
        lmp.command("run 0 post no")
        self._verify_engine_identity(lmp)
        count = 172 * self.replicas
        force_lmp = np.ctypeslib.as_array(lmp.gather_atoms("f", 1, 3), shape=(count * 3,)).copy().reshape(count, 3)
        force = prism.vector_to_ase(force_lmp) * ENERGY_TO_EV
        blocks = force.reshape(self.replicas, 172, 3)
        self.last_force_image_max_difference = float(np.max(np.abs(blocks - blocks[0])))
        force = blocks.mean(axis=0)
        pv = lmp.extract_compute("pam_virial", 0, 1)
        xx, yy, zz, xyv, xzv, yzv = [float(pv[i]) for i in range(6)]
        pressure = np.array([[xx, xyv, xzv], [xyv, yy, yzv], [xzv, yzv, zz]])
        stress = prism.tensor2_to_ase(-pressure * ENERGY_TO_EV / REAL_NKTV2P)
        energy = float(lmp.get_thermo("pe")) * ENERGY_TO_EV / self.replicas
        if not np.isfinite(energy) or not np.isfinite(force).all() or not np.isfinite(stress).all():
            raise RuntimeError("nonfinite replicated XXXII energy/forces/stress")
        self.results = dict(energy=energy, forces=force,
                           stress=full_3x3_to_voigt_6_stress(stress))

    def close(self):
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
