# Native `climb_convg_` and lower-energy exit (bounded static audit)

This note answers one narrow question: can the first Gaussian stop the climb when its true energy is below the initial energy? It is a static comparison of the archived ELF disassembly and the current paper reference loop. No PES call or production change was made.

## Python reference fact

In [`pamssw/standalone/paper_reference.py`](../../pamssw/standalone/paper_reference.py), `current_energy` is initialized from the certified initial quench (lines 201–203). After each biased quench, the code evaluates the unmodified surface at `work` (lines 370–379). At lines 381–382 it tests `true_energy < current_energy`, sets `status='lower_true_energy'`, and breaks the Gaussian loop. Therefore, on the first Gaussian (`gaussian_index == 0`) this condition compares against the initial true-quench energy and can terminate that climb immediately. It is a strict comparison with no 0.1-eV margin. The subsequent lines 401–420 perform the unrestricted true landing quench and only then apply MC acceptance; the lower-energy flag itself is not an acceptance decision.

This is the paper-level Python interpretation documented in the module header, not a claim of native execution parity.

## Native ELF evidence

Raw selected instructions are preserved in [`research/ga_ssw/evidence/native-climb-convergence.asm`](../../research/ga_ssw/evidence/native-climb-convergence.asm); the complete archived disassembly is `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/kernel-ssw_fixlat_mp_climb_convg_.asm`.

The native routine is `ssw_fixlat_mp_climb_convg_` at `0x5cd130`. It loads two object fields at `0x5cd27f–0x5cd291`: `[r15+0x1ac0]` into the local at `[rbp-0x1a8]`, and `[r15+0x1ac8]` into `[rbp-0x40]`. At `0x5cd44a–0x5cd469` it computes

```
    candidate = [r15+0x1ac8]
    reference_minus_margin = [r15+0x1ac0] - 0.1
    energy_lower_flag = (candidate < reference_minus_margin)
                       & ~(control+0x1bc)
```

The `0.1` literal is loaded from ELF address `0x4a45e08`; its bytes decode to 0.1. This flag is stored at `[rbp-0x160]`. It is combined with other convergence/status flags at `0x5cd610–0x5cd658`, rather than returning at the comparison. The final common exit at `0x5cd9a8–0x5cd9bc` writes the accumulated mask to `[control+0x78]` and the derived status to `[control+0x7c]`.

There is a separate `r12d == 1` path: `0x5cd7a5–0x5cd7a9` branches to `0x5cd9ce`; `0x5cd9ce–0x5cd9eb` compares an integer local with `[para+0x2dd24]` and ORs that result into the status. This path does not add a direct energy return. The surrounding comparisons at `0x5cd4e4–0x5cd542` and `0x5cd610–0x5cd655` are mode/threshold flag aggregation, not evidence that every energy-lower event alone exits the outer Gaussian loop.

## Outer status consumer

The outer caller is `ssw_fixlat_mp_climb_` at `0x5ca8e0`; its indirect call at `0x5cb03d` reaches `climb_convg`. It tests `control+0x78` (`lclimb_allstop`) at `0x5cb377`. A clear bit branches to `0x5cb9ae`, where `control+0x7c` (`lclimbstop`) is tested. Both clear returns through `0x5cc554`; the two set-bit paths have distinct actions described below. The selected disassembly is [`native-climb-outer-status.asm`](../../research/ga_ssw/evidence/native-climb-outer-status.asm).

The energy-lower bit is ORed into `ebx` at `0x5cd63a`, after the `r12d != 1` mask is applied to a different bit at `0x5cd638`. Thus that earlier index/mode mask does not suppress the energy bit. The explicit `multi_pes` mask still applies. These source facts supersede the earlier caller-independent uncertainty over whether the energy bit reaches the outer completion path.

## Action boundary from the outer consumer

The status polarity can be narrowed, although the entire state machine is still not a one-line early return. The `lclimb_allstop` set path is the fall-through from `0x5cb377` into the work-array handling beginning at `0x5cb383`. On that path, after the indexed work-buffer copies, `0x5cb8dd–0x5cb8e4` writes `structure+0x230` from `[structure+0x1ac8]` (`tene0`). The path then reaches `0x5cb910–0x5cb928`, which calls the object status setter with the `Allopt` string. If soft-mode cleanup is enabled, `0x5cb8ee–0x5cb904` first calls `del_pot_bond_`; otherwise it goes directly to the setter. This is direct evidence for restoring the saved scalar energy and handing control to the `Allopt` status path after `lclimb_allstop` is set.

If `lclimb_allstop` is clear but `lclimbstop` is set, `0x5cb9ba` enters a different indexed-buffer path. It performs reallocations and copies through `0x5cb9cb–0x5cbc43`, then continues into later record/state handling; it does not call the `Allopt` setter in the inspected interval. The clear-`lclimbstop` edge at `0x5cb9ae–0x5cb9b4` returns through `0x5cc554`. The force-buffer copies on these paths are indexed work/record transfers; this slice does not prove that a particular copy is the final physical-force certificate for the post-climb structure.

Accordingly, the 0.1-eV comparison is sufficient to set the native `lclimb_allstop` bit when its comparison is true and the `multi_pes` mask permits it, because that comparison is ORed into `ebx` before `control+0x78` is written. The outer set-bit path does perform saved-energy restoration and enters `Allopt`, which is the strongest static evidence here for bias-removal/post-climb handling. It is still not evidence that a true-PES quench has already occurred at that point: the status setter and subsequent optimizer/energy evaluation are separate calls. The remaining unclosed point is the action inside `Allopt` after the setter and the exact condition under which `lclimbstop` rather than `lclimb_allstop` is selected.

## Remaining interpretation limits

`tene0` is saved from structure `+0x230` at `0x5caf55–0x5caf71`, before the subsequent added-Gaussian call sequence (see `native-gaussian-caller.md`). The energy at that callback can include the active LS contribution; its exact physical/softened scope cannot be inferred from the field name alone. A complete native-like stop policy therefore still needs the `energy0` initialization and relevant energy-callback lifetime, plus release-coordinate selection. No 0.1 eV parameter has been added to the Python kernel. The source supports a concrete native distinction, not a causal proof that changing one scalar margin would improve global search or cure the ordinary C60 same-basin return.

The normal `ssw_move` entry provides an initialization slice: after matching the eight-byte status string at `0x4a45ec0`, and provided structure `+0x2298 != -1`, `0x5bcf1c–0x5bcf23` copies structure energy `+0x230` to `energy0`. This precedes `bond_counter` (`0x5bcf43`) and the `l_softmode` branch (`0x5bcf4f`). `make_decision` also writes this field in specialized branches, so it is not globally immutable. The normal entry supports a starting-state energy snapshot, while the special-status and callback energy scopes remain explicit limits.
