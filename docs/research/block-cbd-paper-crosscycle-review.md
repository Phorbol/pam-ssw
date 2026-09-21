# Python block CBD-cell: cross-cycle review against SSW-crystal 2014

2026-09-11. Read-only comparison of the current independent Python block
implementation with Shang, Zhang and Liu, PCCP 2014, Section 2.2 and Section
2.3. No PES calculation was run. This review distinguishes confirmed code
behavior from an unresolved paper interpretation.

## The lattice-length reference is the current cycle lattice

The paper's Eq. (3), p. 17847 (PDF p. 3), writes

```text
L_(n+1) = L_n + DeltaL * N_cell
DeltaL = 0.15 * sqrt(sum_ij L_(n,ij)^2)
```

and explicitly describes `L_n` as the current lattice for cycle `n` (text at
`vc2014-author.txt:180-199`; the same parameter statement is at
`:297-308`). The Python loop rebuilds `CellChart(work)` for every cycle and
computes

```text
distance = config.cell_step_fraction * norm(work.cell.array)
```

at `pamssw/standalone/block_ssw.py:98-110`. Since `work` is replaced by the
latest displaced/partially relaxed structure at line 121, this uses the
current `L_n`, not one frozen initial `L_0`. For the default `.15`, the
implemented scalar therefore matches the paper's printed Frobenius formula
at this point. There is no confirmed `DeltaL`/`L_0` discrepancy to fix.

The cell chart stores the current unwrapped fractional coordinates and maps
the new nine-entry lattice back with `positions = fractional @ lattice` at
`pamssw/standalone/cbd_cell.py:63-68`. The paper requires fixed-lattice atom
relaxation after each displacement (`vc2014-author.txt:185-187` and
`:329-336`), but does not specify this remapping convention in enough detail
to call it either a paper mismatch or exact parity.

## Direction lifetime: definite Python state, but no definite paper conflict

Each Python cycle draws a new nine-component random input at
`block_ssw.py:100-107`; no previous `mode.direction` is passed to the next
cycle. `cell_direction()` projects that input using the current cycle center
(`cbd_cell.py:122-140`), and `generalized_dimer()` normalizes the initial
anchor and every proposed mode in `generalized_numerics.py:151-175`. Thus the
implemented policy is **fresh random anchor, then a newly solved unit
direction, per cell cycle**.

The paper says that a few CBD-cell cycles repeatedly displace the lattice
“along one particular lattice mode direction” (`vc2014-author.txt:172-179`),
but its cycle procedure also says to “utilize the CBD method to identify a soft
mode `N_cell`” in each pass (`:329-336`). It gives a random starting vector
for the CBD rotation (`:205-218`) without stating whether that random vector
is redrawn or whether the previous mode is used as the next anchor. The
phrase “one particular” could describe a smoothly followed mode within the
block, while the numbered procedure permits a fresh identification each
cycle. Therefore the fresh-anchor behavior is a concrete independent-policy
difference from a possible continuation implementation, but **not a
source-proven paper deviation**. It must remain labeled as an explicit
choice, rather than being advertised as recovered cross-cycle direction
retention or as disproving the paper's wording.

## Review outcome

The two requested checks find one confirmed implementation fact and no
confirmed algorithmic defect: `DeltaL` uses the current cycle's `L_n` and the
raw nine-entry Frobenius norm, as required by Eq. (3). The only live parity
question is direction lifetime. Resolving it requires source/SI or a native
state trace; changing the Python schedule based only on “one particular” would
add an unsupported heuristic.

Source: `literature/benchmark-sources/vc2014/vc2014-author.txt`, the two
standalone files and `generalized_numerics.py` cited above. This is an
independent implementation review, not native-behavior evidence.
