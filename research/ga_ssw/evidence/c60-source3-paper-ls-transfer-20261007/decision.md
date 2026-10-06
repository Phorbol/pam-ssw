# C60 source #3: local paper-LS transfer result

2026-10-07. Development component transfer on MH-1/omol, one published
non-Ih geometry and two search RNG seeds; not a reproduction of the LS
paper's potential or long random-start success rate.

## Executed and qualified

Frozen runtime source `4fec3e3f7d09009f0808dfc43cc1c2c2e78900d6`;
GPU array1664389, CPU readout1664393. Inputs, source snapshots, plans and
raw ledgers are in [prepared-20261007-a](prepared-20261007-a/); derived
observations, LS replay and cost curves are in
[readout-1664393/analysis.json](readout-1664393/analysis.json).
[Protocol](protocol.md) fixes the sole LS factor and its parameter sources.

| Search seed | Arm | Paid search requests | Actual search calculate | Completed outer steps | Fresh checks | Ih target |
|---|---|---:|---:|---:|---:|---:|
|26100791|SSW|15791|14449|37|2|0|
|26100791|paper-LS|15917|14464|37|2|0|
|26100792|SSW|15615|14348|35|2|0|
|26100792|paper-LS|15754|14419|34|2|0|

All four stopped at the runner's reserved wall boundary, about915–929s;
none reached the20000-request prefix. Total63077 search plus8 fresh
requests;57680 actual search calculate calls. Fresh calculate counts were
not independently instrumented by the frozen runner and remain unknown.
The initial eligibility's13 calls and failed preparation's2 wrapper
attempts/0 actual calculations remain separate. No failed PES requests,
denials or uncertain reservations occurred in the four search arms.

All142 completed outer steps reached the12-Gaussian limit. The resulting
142 landings plus4 initial observations met the true-force gate; three
landings were fragmented at at least one recorded bond cutoff. The8 cold
initial/best checks passed force/energy consistency, but no observation
met the Ih graph+energy+force target. No positive target therefore exists
to certify as a three-dimensional cage. New connected graph classes
27/27/26/24 are topology counts, not deduplicated minima or kinetics.

## What changed our interpretation

The LS arm actually performed37/34 force-qualified soft prequenches, at
362/330 requests, with no prequench or controller domain failure. Replaying
the response equation including MC-rejected outer steps reproduced the
saved frozen pairs and strengths. From initial total strength9.747eV and
response.00227857eV/atom, the final response approached.01983/.01973eV/atom
against the preset.02 target, and strength approached29.55/29.48eV.
Thus this result is not explained by LS never executing or a failed soft
prequench. It does not prove this controller is optimal on MH-1.

At matched5000/10000 requests, both arms' best connected energies remained
within.0001eV of the initial cage, which is1.84216eV above Ih. Those tiny
differences are not ranked. MC accepted9/10/6/8 completed steps in table
order, yet this produced no meaningful improvement; the readout includes
rejected qualified discoveries, so acceptance alone does not hide an Ih
hit. No Hessian, transition-state or global-minimum claim is made.

The root independently checked sequential raw ledgers against budget,
segment and checkpoint-lineage costs; all117 frozen core files, input and
plan hashes; every fresh numerical qualification; completed status counts
and MC decisions. The existing readout independently replays LS and
retains charged versus observable horizons. These are numerical/provenance
checks; they do not turn the negative target result into physical success.

## Decision and next discriminating comparison

Keep paper-LS experimental; do not promote defaults, retune this cage or
automatically extend this panel. Current evidence supports no local repair
benefit within these attained costs, not rejection of the paper's long-run
claim. The already planned native reference uses the same MH-1 source#3
geometry and comparable paid cost to distinguish a native-stack advantage
from a difficult local barrier under both implementations. Native periodic
vacuum geometry and seeded input have separate qualification; different
RNG, optimizer and MC details mean this will be a stack comparison, not a
single-component causal ablation. The independently ongoing random-C60
direction panel remains unchanged.
