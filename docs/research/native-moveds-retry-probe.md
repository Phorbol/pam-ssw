# Native `moveds` retry decision probe

2026-09-11. This is a bounded Unicorn execution of the uploaded ELF bytes
from `0x5c7452` through the three branch targets of the retry decision. It does
not call LASP main, `present_tooshort`, PES, allocation, or protection code.

The probe supplies the preceding `present_tooshort` logical flag and scalar as
synthetic locals, plus the `para` fields `disp_perstep` at `+0x2e1f0` and
`bonddisp_perstep` at `+0x2e1f8`. The retry slice itself executes the native
comparisons at `0x5c7452–0x5c748e`, the `ds_n *= 0.95` update at
`0x5c7494–0x5c74a3`, and the bounds at `0x5c74be–0x5c74cb`.

`short_pres_factor` is a different field: the source-backed parser record
stores it at **`para+0x2db40`**, with `1/(1+0.005*externaltp)`; it is not an
`object+0x2db40` field. It is not needed by this slice because
`present_tooshort` is stubbed at its output boundary.

The four cases stop at `0x5ca780` (accepted), `0x5c6ddc` (shortened retry),
and `0x5c74d1` (retry exhausted by either `ds_n < 0.1` or retry count > 50).
The probe has a PC guard: any instruction outside `0x5c7452–0x5c74cb` or the
three declared stop targets fails, rather than being reported as a normal row.
The JSON records every executed address and labels all synthetic dependencies.
The three scalar locals are deliberately distinct: `rbp-0x40` is the preceding
candidate scalar, `rbp-0xd8` is the local used in the `disp*(1-bonddisp)`
expression, and `rbp-0xd0` is the scalar returned by `present_tooshort`.
Their exact physical meanings are not inferred here. These results close only
the local branch arithmetic and retry routing. They do not close `present_tooshort` semantics,
the full move record, width write, or post-`Allopt` state.

Run with:

```bash
python -m research.ga_ssw.probe_native_moveds_retry \
  --output research/ga_ssw/evidence/native-moveds-retry/result.json
```
