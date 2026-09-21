# Native `CBD.fact` provenance

Static audit of the frozen ELF only; no LASP main program, protection path, or PES evaluation was run. The inspected executable is
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`, SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.

## Recovered source and call chain

The DWARF member map identifies `ssw_parameters::cbdfact` at `para + 0x2dc98` (`analysis/kernel-dwarf-member-offsets.txt:86–102`; repeated in `analysis/selected-dwarf.txt`, for example lines 769–770). The `cbd_rotation` formal `fact` is the ninth stack argument at `+0x50` in the DWARF signature (`/tmp/lasp-dwarf.info`, `newssw_basics_mp_cbd_rotation_`, entry `0x6e5590`).

In the fixed-cell unbiased caller, `0x5c4069–0x5c4070` computes `lea 0x2dc98(%rdx),%rax` with `rdx = &ssw_parameters_mp_para_`, then pushes that pointer; the call is at `0x5c40ca`. Therefore the actual `fact` passed to `cbd_rotation`, and onward to `rotate_dimer`, is the scalar stored in `para.cbdfact`. The biased caller uses the same parameter field at `0x5c4c9a` (see `analysis/kernel-ssw_fixlat_mp_unbiasedrot_.asm` and `..._biasedrot_.asm`).

The parser setup at `0x6898ff–0x689962` and call at `0x68996a` invokes `readinput_mp_get_real_` with destination `para+0x2dc98`; the rodata at `0x4a4a2bc` is the key string `CBD.fact` (the next key, `CBD.cellfact`, begins at `0x4a4a2c8`). Thus the input-controlled source is concretely `CBD.fact -> para.cbdfact -> cbd_rotation.fact`.

The parser's `rcx` argument is also `para+0x2dc98` (`0x689906–0x68990d`). At `readinput_mp_get_real_` entry (`0x60f390–0x60f3a5`), it dereferences that pointer and copies the value into the destination before scanning the input. Consequently, the missing-key value is whatever initialization has already placed in `cbdfact`; it is not a literal supplied by this call.

## Runtime distinction

At `rotate_dimer` `0x6e5e76–0x6e5e86`, the supplied `fact` is copied into the evolving static `FACT1` on the first rotation. Later retry/rotation logic uses `FACT1`; the documented `.8` retry scaling is runtime state (`docs/research/native-rotation-followup.md:7–22`), not a second input default. BRIONS4's `G0/STEP1` is a separate argument/value and must not be substituted for `CBD.fact`.

The saved native `allkeys.log` files are effective-parameter dumps, not proof that the corresponding `lasp.in` explicitly contains these keys. They report `CBD.fact 0.0500` and `CBD.cellfact 0.5000` (for example `.../runs/water-ase-equivariance/eval-00001/allkeys.log:125–126`), establishing values effective in those archived runs only.

## Default-value boundary

The default initialization is recoverable in `desw_options_` (`0x6c4700`, DWARF formal `level`). `readsswpara_` reads `DESW.quick_setting` into `ssw_parameters::desw_setting` at `para+0x10c` (`0x686b58–0x686bce`) and calls `desw_options_` at `0x686c99`. Its five-level jump table at `0x4a4bf48` dispatches to these stores:

| `DESW.quick_setting` level | `cbdfact` store | value |
|---:|---:|---:|
| 0 | `0x6c4b4d–0x6c4c15` | 0.05 |
| 1 | `0x6c4a50–0x6c4b17` | 0.25 |
| 2 | `0x6c4956–0x6c4a1a` | 0.15 |
| 3 | `0x6c4847–0x6c4926` | 0.05 |
| 4 | `0x6c4736–0x6c4818` | 0.02 |

If `CBD.fact` is present, the later `readinput_mp_get_real_` call overwrites that initialized value. If absent, the table value is the recovered default, conditional on `DESW.quick_setting`; values outside 0–4 are clamped to level 1 at `0x6c4709–0x6c4711`. The research helper's `initial_factor=1` remains a caller choice and is not native parity.

`cbdcellfact` at `para+0x2dca0` is a distinct field; it is not the `fact` pointer established by these fixed-cell caller snippets.


Root byte verification: `ssw_parameters_mp_para_` is initialized data at
0x53ed7a0. Its `cbdfact` field at0x541b438 contains bytes
`9a9999999999a93f` (double0.05); `cbdcellfact` at0x541b440 is0.5, and the
static `desw_setting` at para+0x10c is integer0. The default parse therefore
starts from level0, unless input overrides the level or explicit CBD.fact.
The preceding get_real call0x6898fa belongs to CBD.maxdist, not CBD.fact.

The next bounded fixed-cell experiment selects FACT0.05 based on this source
provenance, without searching its values against performance. Earlier FACT1
fixed-geometry results remain a separately labeled provisional comparison.
