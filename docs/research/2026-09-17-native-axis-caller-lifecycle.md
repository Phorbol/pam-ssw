# Fixed-cell axis caller and pair refresh lifecycle

This is a bounded static audit of the uploaded Run_type 5 ELF. It does not run
LASP or a PES and does not claim that the complete direction controller is
restored.

## `copy_str` producer and callers

The fixed-cell structure table at `0x53ca680` has `+0x28 = 0x5a6b70`,
`class_struc_mp_copy_str_`. Its coordinate-copy branch is explicit:

* `0x5a6c33` passes `&str+0x6d8` (`cart_copy`) to `for_realloc_lhs`;
* `0x5a6c3f` forms `str+0x170` (`cart`/current coordinates);
* `0x5a6cad–0x5a6e48` copies current coordinates into the reallocated array.

The normal fixed-cell `ssw_move` caller uses this slot at `0x5bc8ad`. The
preceding status descriptor is the literal `NewStart` at `0x4a45ec0`, and the
call is guarded by `str+0x2298 != -1` (`0x5bc897–0x5bc8a3`). Thus NewStart is a
real reference-snapshot boundary. `make_decision` also calls the same slot at
`0x5d34b0`, after resetting `str+0x2298` and its counter (`0x5d3495–0x5d34a6`),
so the reference can be refreshed at a later periodic stage boundary. The
static evidence does not justify identifying every refresh with a previous
landing.

## Judge gate and `findleast/getpair` order

`ssw_fixlat_mp_allopt_` starts at `0x5d46e0`, but is a repeatable state callback.
Its relevant control flow is:

1. `0x5d47d5` calls the fixed-cell judge slot, resolved to
   `ssw_fixlat_mp_allopt_judge_converg_` (`0x5cf6f0`);
2. `0x5d47dc` tests the returned flag;
3. `0x5d47e0` jumps to `0x5d4a58` when the flag is false, bypassing automatic
   selection and continuing the optimization path;
4. only the terminating/judge-true branch reaches
   `find_leastmoveatoms_` at `0x5d481f`, then `get_atompair_` at `0x5d48fc`.

Consequently axis/group selection is post-judge bookkeeping on the ending
branch, not an unconditional allopt-entry operation and not, by itself, a
certificate of an independent true-PES quench.

## Pair overwrite and second-candidate controls

At `0x5d47f8–0x5d4818`, allopt passes `str+0x6d8` as reference, `str+0x170` as
current, `str+0x8a8` as the mask, `&str+0x1ad8` as one output, and the value at
`str+0x1ae0` as the other output. It then passes `&str+0x1ad8` to
`get_atompair_` as its output at `0x5d48ee–0x5d48fc`.

Inside `get_atompair_` (`0x57f990`), the output address is retained in `r13`
(`0x57f9f8`). The first pair scalar is conditionally rewritten at
`0x57fcda`: the tests at `0x57fc92–0x57fcd8` use the whole-system
minimum and maximum atomic numbers. When both are below 10 and the first
random draw exceeds the recovered threshold, the second draw selects a new
first atom. These are not two candidate-neighbor tests. The alternate path
preserves the existing first value. This is a conditional overwrite, not a
guaranteed replacement of `findleast` output.

The second candidate is selected in the retry loop beginning at `0x58046b`.
Each trial draws a random index and reads a candidate from the distance array
(`0x58046b–0x58049d`); the static slice does not justify renaming that value
as a neighbor quantity. It then applies the `fixatom` array test:

```text
fixatom[index] = [para+0x2df30 + 8*(index - para+0x2df70)]
```

The candidate is rejected when `fixatom[index] >=` the constant loaded at
`0x5804b8`; `0x5804c0–0x5804c5` therefore permits the strict
`fixatom[index] < constant` case, after which the second value is written at
`0x5804c7`. The loop has a visible `r15 < 150` bound at
`0x58052e–0x580535`, but `r15` is incremented only on the `0x58052b` retry edge.
Thus 150 bounds that particular distance/fixatom rejection loop, not every
candidate attempt: element and forbidden checks can reject without consuming
this counter. The failure edge exits through surrounding fallback paths and no
universal new pair value should be invented from the static code.

The value read at `0x5804d8` is `atomic_numbers[index]`. If it is `<= 20`,
control falls through to `check_forbiden_` preparation at `0x5804ef`. For an
atomic number `> 20`, the random value is compared with the literal at
`0x4a438b0` (`0x5804e0–0x5804ed`): the comparison can bypass the forbidden
check; it is not a gate that must pass before calling it. The ordinary call is
at `0x580507`, and its logical result is tested at `0x580520–0x580529`. The
call passes the address of integer literal 1 at `0x4a434bc`, as confirmed
by the subsequent instruction probe. Likewise, `para+0x2df30` is DWARF-labeled
`fixatom`; its input/default population is separate from this control flow.

Thus `get_atompair_` can overwrite the first pair scalar after `findleast`,
while the separate group address supplied to `findleast` is not passed to
`get_atompair_`. The final pair/group consumer for every Run_type 5 branch
still requires a runtime or branch-specific trace.

Evidence: `analysis/allopt-convergence.asm`,
`analysis/kernel-ssw_fixlat_mp_ssw_move_.asm`,
`analysis/kernel-ssw_fixlat_mp_make_decision_.asm`, and the uploaded ELF
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.

## Canonical getpair verification update (2026-09-17)

Python `refresh_native_pair` now matches the canonical 24-case set: 24/24 native
runs completed. The edge set covers heavy-first C/Cu, empty and repeated pairs,
free/one-fixed/all-fixed masks, RNG branches and 90-degree rotation: 24/24
completed, with pair, draw count, accepted status and all three rejection counts
matching Python. Evidence: `native-getpair-canonical-20260917.json` and
`native-getpair-edges-20260917.json`.

The earlier centered-coordinate pipeline (6/9) and fixed-pair statistics (8/9)
remain historical artifacts; their normal-behavior interpretation is withdrawn.
`native-pair-coordinate-chart-20260917.json` records that `get_dist` wraps
coordinates while `check_forbiden` subtracts original endpoint coordinates.

The first-atom random branch uses whole-system `min(Z)<10` and `max(Z)<10`, not
the selected atom's Z. Caller literal `0x4a434bc` is integer 1.
