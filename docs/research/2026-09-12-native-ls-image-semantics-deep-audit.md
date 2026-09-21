# Native LS image semantics: `pot_bond_add` audit

**Scope:** static ELF disassembly only. The LASP main program, protection path,
and PES were not executed. The binary is
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
This supplements, rather than replaces, the existing native-LS table and
initialization audits.

## Call and image evidence

The recovered symbol table identifies
`pot_bond_mod_mp_pot_bond_add_` at `0x6c5c50` and
`mirror_min_dist_` at `0x6c5930` (`analysis/lasp-symbols.txt`). In
`analysis/ls-pot-bond-add.asm`, the bond loop starts at `0x6c5fc6`: it loads
atom indices from the `allbonds` record (`0x6c5fd1–0x6c5fdb`), constructs the
corresponding Cartesian and fractional pointers (`0x6c5fe4–0x6c609b`), and
calls `mirror_min_dist_` at `0x6c60aa`. The same helper is called by the native
bond counter at `0x6c4fea` in `analysis/ls-bond-counter.asm`, showing that the
selection/count and potential path share this distance primitive.

The helper body is present in the ELF disassembly for `0x6c5930–0x6c5c34`; a reproducible copy is saved at `research/ga_ssw/evidence/native-ls-image-semantics-20260912/mirror_min_dist.objdump`.

- It first subtracts the two Cartesian pointers and forms the squared norm
  (`0x6c5930–0x6c5987`). It compares the direct norm with the global
  `pot_bond_var_def_mp_bond_len_max_` at `0x6c598b–0x6c5990`; the pinned ELF
  `.data` symbol is at `0x5520448` and its raw bytes are
  `00 00 00 e0 4d 62 50 3f`, which decode as the double
  `0.0010000000474974513` in the file image. No writer for this symbol was
  identified in the existing textual analysis, so this is an initializer value,
  not a claim that runtime configuration cannot overwrite it. If the direct
  distance is within that bound, it
  writes the direct vector/scalar and returns (`0x6c5c35–0x6c5c44`). This
  short-circuit means the routine is not unconditionally an exact MIC call.
- Otherwise it constructs image candidates from the lattice data passed in
  the `r8` argument. Each loop counter starts at zero and is converted with
  `lea -1(counter)` (`0x6c5a21`, `0x6c5a93`, `0x6c5b02`), then increments to
  three (`0x6c5a8e`, `0x6c5bc1`, `0x6c5c05`): the tested image offsets are
  therefore `-1, 0, +1` in each of three lattice directions, not `0,1,2`.
  The candidate Cartesian image vector is assembled using lattice-vector
  products and additions (`0x6c5a42–0x6c5b32`). The squared candidate norm
  is compared with the current minimum (`0x6c5b37–0x6c5b86`). At `0x6c5af0`,
  the code loads the `.rodata` double at `0x4a4c158`, whose bytes decode to
  `0.001`; `0x6c5b6a–0x6c5b73` computes
  `current_best_squared - 0.001`, and the `jbe` at `0x6c5b78` skips replacement
  when the candidate is not smaller than that margin. Replacement therefore
  requires `candidate_squared < current_best_squared - 0.001`. This is a
  squared-distance selection margin in the helper's input units, not a generic
  exact-minimum or tie rule. The winning vector is written to the output
  pointer at `rsi`; the scalar minimum is returned through the pointer passed at
  `0x10(%rbp)` (`0x6c5c0b–0x6c5c2b`).
- The result is therefore a per-call bounded image search over a `3-by-3-by-3`
  candidate set only when the direct vector is outside the early bound. It must
  not be summarized as “always returns the nearest MIC”.
  The disassembly does not show a stored image label being carried into later
  `pot_bond_add` calls. `pot_bond_add` consumes the returned vector and scalar
  immediately (`0x6c6107–0x6c6187`) to form the exponential bond contribution
  and updates the two force records (`0x6c61ed–0x6c6284`).

This is stronger evidence than the Python implementation: the original
potential path recomputes a minimum image for every bond evaluation. It is not
a frozen-image sum. The exact numerical value and units of
`bond_len_max` are not assigned here from the symbol name alone; its static
value and the caller's descriptor units require a separate data/ABI reading.

## Initialization and scope

`bond_info_init_` loads/normalizes the bond-energy and bond-length lists at
`0x6c6cc2` and `0x6c6d86` in `analysis/ls-bond-info-init.asm`. The dynamic
geometric thresholding/count path calls the same `mirror_min_dist_` helper
before comparing the returned distance with a table length plus tolerance
(`analysis/ls-bond-counter.asm`, `0x6c4fea–0x6c5050`). This establishes shared
image-distance machinery, but does not by itself prove that every initialization
branch uses identical list filtering or tolerance values.

## Consequence for the Cu111 observation

The isolated replay of saved Cu111 requests 132 and 152 shows the consequence
of this margin. Native `mirror_min_dist_` returned the same `[0,1,0]` image for
pair `[8,11]`; the vector change was about `6.0e-8` A and the unit-direction
change about `2.4e-8`. ASE `find_mic` selected the other near-degenerate image
at request 152. The two squared distances differ by about `3.5e-8` A2, below
the native `0.001` margin, so this pair does not evidence a native image jump.
A frozen `periodic-images` potential would be a mathematical alternative, not
a native-parity correction. The ELF evidence does not prove that the observed
prequench failure is caused solely by image switching, nor that the bounded
27-candidate search covers all possible cells. Those remain explicit boundaries
for any future isolated oracle.


## Isolated callee probe

`research/ga_ssw/probe_native_mirror_min_dist.py` executes only `mirror_min_dist_`
(and, in v3, the already-called `reclat_loc_` primitive) under Unicorn. The
inspected ELF hash, entry address, interpreter and outputs are saved in
`research/ga_ssw/evidence/native-ls-image-semantics-20260912/` as
`mirror_min_dist-results-v2.json` and `mirror_min_dist-results-v3.json`.

The direct-call v2 cases are retained as superseded because they did not call
`reclat_loc_`. v3/v4 are likewise superseded probe variants. In v8, experiment
A passes the ASE Cartesian vectors and the real cell as three C-order vectors
directly to `mirror_min_dist_`; all A cases match the explicit `-1,0,+1`
27-image calculation, including skew cell. Experiment B passes the reciprocal
lattice in the caller structure layout to `reclat_loc_`, then asserts its output
bytes equal the direct real-cell representation before the helper call; this
assertion passes for the orthogonal B case. The skew A result differs from ASE
exact MIC because ASE finds a shorter image outside the native bounded 27
candidate convention for that non-reduced skew cell. The remaining producer-side
`xfrac` descriptor/unit contract is still not fully reconstructed, so the
probe establishes the helper's bounded-cell semantics rather than a universal
claim about every native caller. This is why the static conclusion is limited
to the fast path, bounded image loop and explicit squared-distance margin; no
production change follows from the probe.

The boundary probe `mirror_min_dist-epsilon-v9.json` reads the `0.001` literal
from `0x4a4c158` and confirms the strict behavior: with a wrapped candidate
improvement of `0.0008` A2 the old image is retained, while improvements of
`0.0010000000000012` and `0.0012` A2 replace it. The near-threshold value is
reported with its measured floating-point value; it is not treated as an exact
equality case.
