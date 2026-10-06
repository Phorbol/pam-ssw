# M80 paired transfer: closed, no continuation or default promotion

CPU array1663302 (source8204fd9, core treec572a1cc) and readout1663308 completed.
The isolated Morse evaluator and shared 80G reference qualification passed.
The initial/outer sums and raw charged/calculator ledgers close in all four arms.
The first rotation arm ends with `evaluation_failed` because request80001 is
denied at the predeclared80k cap, **not a failed calculator call**. The other
three arms complete100 outer attempts; there are zero charged evaluator errors.

| Fixed input | Rotation best connected E | Full best connected E | Common paid horizon | Interpretation |
|---|---:|---:|---:|---|
|25092501, initially connected80|−328.162129|−334.887924|75827|Full is lower by6.725795eV on this pair|
|25092502, initially73+6+1|−332.853608|−332.897723|64199|Both reconnect; small endpoint difference, no strong efficiency claim|

All four best endpoints are independently cold force-qualified (fmax .04065,
.04831,.04120,.04466eV/A) and connected80 at1.3r0. None passes the predeclared
80G target E≤−340.811371+.001; joint hits0/4. This is not evidence against the
paper's8/100 within2000steps: the present fixed-start development panel runs
at most100 attempts, has different qualified-input provenance and operational
stopping/height controls. Nor is the second pair's .0441eV difference a measured
general advantage; the force tolerance and basin identity are not refined to
establish significance. Rotation reaches its eventual best in18152requests on
that pair, full in53452. Qualified observations74/97/34/93 can repeat basins.

Search286056 requests /273687 actual calculator calls; independent initial/best
8 plus one shared reference; reused historical preparation1674 separately.
Current panel286065; full preparation+panel287739. Times perarm396.4/367.4/
308.0/321.4seconds include algorithm/recording, and are not standalone optimizer
or calculator benchmarks. No input/seed/penalty/temperature/force-window tuning,
no retries and no hidden fragmented-input removal occurred.

**Decision:** the complete direction implementation functions on this shorter-
range PES and has a scoped paired improvement; it remains an explicit candidate,
not a universal default or proof of superiority to LASP. This panel has answered
its transfer question and closes. Do not expand it automatically into the paper's
100-trajectory success-rate campaign. Next mainline work is a qualified TiO2
phase target for VC, with model, geometry and search evidence distinguished.

Evidence: [frozen protocol](protocol.md),
[raw-cost readout](readout-1663308/analysis.json), four `run-1663302-*` folders,
and scalar search/fresh ledgers plus initial/best/outer structures therein.
