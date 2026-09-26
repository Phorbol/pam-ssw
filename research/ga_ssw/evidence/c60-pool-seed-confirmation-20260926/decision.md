# C60 seed confirmation: stop expansion, retain optional pool mode

Question: did the seed181 PAM Ih repair persist with frozen settings and new
mainseed197/selector199 on the same saved defect? No settings/model/input changes.
GPU1501032 completed all three process runs; all searches reached the search
wall deadline. CPU1501037 produced the readout and returned2 to flag censoring,
not a failed analysis or a numerical optimizer defect.

|Mode|Paid search|Fresh|Certified initial+landings|Committed restarts|
|---|---:|---:|---:|---:|
|MC|30017|68|68/68|0|
|uniform|27709|73|73/73|63|
|PAM|26141|68|68/68|39|

Total83867search+209fresh. Root independently checked initial+outer cost sums,
full fresh coverage, qualification and wall_deadline in all three raw summaries.
At exact common26141 search requests, all three modes have zero Ih graph matches
at1.64/1.7/1.8 Angstrom and zero reference-energy-window hits. The best energies
differ by less than0.00012eV and do not resolve a useful energy ranking.
The previous seed181 PAM repair at21055 requests remains valid; this new seed
did not reproduce it. Two seeds on one input are not a success-rate estimate.

Decision follows the prospectively frozen protocol: do not promote PAM, tune
weights, add a third seed, or launch random-C60 production from these results.
Keep the approved ASE identity option as an interface correctness improvement,
separate from the unresolved search advantage. C4H6 also showed no pool benefit.
Retain MC default and optional uniform/PAM research modes.

Next mainline work: qualify traceable SiO2 input on OMAT-small before any
held-out joint-cell optimizer comparison. This tests whether the prior Safe-total
cost signal transfers; it does not claim BKS phase ranking or broader global
search success. No new GA/RC/Q feature development is motivated by this screen.

Sources: plan.json/protocol.md, run-1501032-{0,1,2},
readout-1501032/{report.md,analysis.json}; all raw failed/censored outcomes retained.
