# VC trajectory `+0x230` producer follow-up

2026-09-11. Bounded static data-flow review for the candidate scalar consumed
by VC `climb_convg` at `0x5f4970–0x5f49bd`. No PES or LASP main execution.

## Consumer index

At `0x5f4970–0x5f498f`, the consumer computes

```text
record = object+0x1668
       + ([object+0x1660] - [object+0x16a8]) * 0x690
```

and at `0x5f4996–0x5f49a9` reads `record+0x230`. The loaded scalar is
compared after subtracting the constant at `0x4a46d30`. The formula shows
that the record selected by the consumer is the value held at
`object+0x1660` relative to the lower index `object+0x16a8`; it is not an
implicit read of `tene0` or `energy0`. The DWARF table identifies
`object+0x1660` as the trajectory descriptor in this VC object layout; an
enclosing counter name is not assumed here.

## `climb_` writes

The VC climb body has a direct next-record write:

1. At `0x5f322d–0x5f3245`, it loads `[object+0x1b28]` (`tene0`) into
   `[object+0x230]`, then loads `[object+0x1660]` and increments that value
   before multiplying by `0x690`.
2. At `0x5f3271–0x5f3286`, it loads the trajectory base
   `[object+0x1668]` and subtracts
   `[object+0x16a8]*0x690`, producing the record origin
   `base + (N+1-lower)*0x690`, where `N` is the value loaded from
   `object+0x1660`.
3. At `0x5f34b6–0x5f34d1`, it reaches that origin and writes
   `[record+0x230] <- [object+0x1b28]` (`tene0`).

Relative to the same value `N` used by the consumer, this path writes the
**next indexed** trajectory record `base+(N+1-lower)*0x690`, while
`climb_convg` reads `base+(N-lower)*0x690`. This is a precise index offset,
without assigning an unproven `ng` name to the descriptor field. The code
does not show that the write and the subsequent consumer call occur without
an intervening status/trajectory update; that caller ordering remains outside
this slice.

The same climb function also snapshots the current working scalar at
`0x5f241f–0x5f243b` (`object+0x230 -> object+0x1b28`) and restores
`object+0x230 <- object+0x1b28` at `0x5f2bf9–0x5f2c02`. Those are object-local
copies, not trajectory writes.

## `set_status_` initialization write

An independent trajectory write occurs in `ssw_crystal_basic_mp_set_status_`:

* `0x5eabc4–0x5eabd9` forms `base-lower*0x690` from
  `[object+0x1668]` and `[object+0x16a8]`.
* `0x5eabed–0x5eac0a` sets `rcx=0x690`, loads `[object+0x230]`, and writes
  `[base+(1-lower)*0x690+0x230] <- [object+0x230]`.

This is a first-slot-relative-to-lower write. The available instructions do
not establish that this slot is the current `object+0x1660` or the next one
at every caller; it must not be relabeled as `ng` or `ng+1` without the
enclosing status call condition.

## `moveds_` boundary

The VC `ssw_crystal_basic_mp_moveds_` body (`0x5eccf0`) reads the working
object scalar at `0x5ee217–0x5ee21e` and uses trajectory bases/indices for
coordinate and mode work, but the inspected body has no store to a
trajectory `record+0x230`. Therefore this slice provides no direct evidence
that `moveds_` itself publishes the scalar consumed by `climb_convg`; the
publication is in `climb_` and `set_status_` paths above.

This evidence concerns the scalar at trajectory `+0x230`. It does not identify
that scalar as `energy0`; DWARF separately maps `object+0x1b20` to `energy0`
and `object+0x1b28` to `tene0`.

Sources: `native-cell-reference-evidence/climb.asm`,
`native-cell-reference-evidence/set_status.asm`,
`analysis/kernel-ssw_fixlat_mp_moveds_.asm`, and
`analysis/kernel-dwarf-member-offsets.txt`.
