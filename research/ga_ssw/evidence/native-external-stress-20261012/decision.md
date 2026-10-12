# Native external tensor handoff qualified in a bounded EMT diagnostic

CPU1757580, source1065e56, 2026-10-12. This is an implementation diagnostic,
not a Cu search benchmark. [Protocol](protocol.md), [readout](readout-1757580.json).
The original uploaded ELF was executed unchanged; no date/identity alteration,
register or memory writes, or protection bypass. Three class entries were
observed before the planned debugger stop. Three external requests cost three
actual EMT calculations. The supervisor completed in 2.50 seconds, reported
no surviving descendants, and the callback had no failure/denial.

With the explicit parsed `external.dedcell F`, the private callback supplied
energy, Cartesian force and nine Fortran-ordered entries of minus ASE full
stress in eV/Angstrom^3. The intermediate legacy stress exactly equals the
ASE tensile-positive tensor on all three requests. At the class consumer,
maximum absolute differences are 4.26e-9, 1.55e-8 and 7.10e-8 eV/Angstrom^3.
These are far below the existing .001 physical stress gate; they are retained,
not investigated as a precision optimization target. No extra unit scale or
opposite sign appears. All three consumed flags and the parsed input flag are
false; stored external pressure is zero. The native stored cell transposes to
the callback cell. Cells genuinely change: maximum component differences from
the first request are .00650 and .02039 Angstrom for requests two and three.

The three class events link to the nearest completed otherpot return with the
same callback request ID. Class atom coordinates were not read, so this is not
a full atom-coordinate identity proof; symmetric physical stress also cannot
discriminate row versus column tensor flattening. This is evidence for this
ELF's external/EMT path, not dedcell=T, all backends, full enthalpy-gradient
correctness, local cell convergence, MACE qualification or VC search efficacy.

Decision: the sign/unit transport prerequisite is sufficiently resolved to
propose an explicit optional E/F/stress bridge mode. Do not silently enable
variable cells in the existing E/F-only interface. A reusable bridge contract
and MACE finite-strain qualification are next; no native TiO2 search has run.
The old fixed-cell C60 evidence remains limited to its own supported boundary.

Root repeated zero-PES preparation and syntax checks before submission; an
independent reviewer checked the original callback/trace/provenance artifacts
and the cost/linkage/cleanup conclusions. Reanalysis uses `analyze.py`, costs
zero PES and keeps raw files unchanged. Raw run1757580, logs and copied binary
remain untracked at the local evidence path; compact results are versioned.
