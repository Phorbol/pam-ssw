# Morse initial ensemble affects the task before any SSW move

Both ensembles use the same two random seeds at N29 and80; the second only changes the sphere radius according to the frozen source-based volume rule. No SSW or LS move was run. All8 terminal structures meet the0.01eV/Angstrom force condition; this is not a connected-cluster or Hessian certificate.

At neighbor cutoff1.5r0 (the same main components also hold at1.3r0):

| Initial ensemble | M29,seed01 | M29,seed02 | M80,seed01 | M80,seed02 |
|---|---|---|---|---|
| radius5.5r0,largest component |3/29|5/29|18/80|13/80|
| volume-scaled radius,largest component |28/29|27/29|80/80|73/80|

Diffuse energies: -5.004,-12.004,-109.899,-107.142epsilon. Scaled energies: -89.271,-77.001,-271.529,-249.649epsilon. These differences precede the search and cannot be credited to a direction/optimizer change. Original SSW2013 does not specify this initial distribution. The diffuse ensemble remains valid for an assembly-plus-search task; the compact ensemble shifts effort toward rearrangement, but still contains isolated atoms/fragments in3/4 cases. Do not discard those starts, resample, or shrink radius further after observing them.

CPU1488321: search counts48+94+745+754=1641, plus4fresh,11s. CPU1488349:708+498+1006+666=2878search+4fresh,15s. Code for the first run is8a0e15d, for the size-scaled run da9f566. Exact inputs and output coordinates remain in `morse-initials/` and `morse-compact-initials/`; JSON summaries are separate. Radius has only been changed once under a prospectively recorded rule, not scanned to make all inputs pass.

Decision: no more initial-radius adjustments. Report both ensembles explicitly in any future Morse study. Confinement is not enabled; adding it would change the search objective inside the boundary and requires a separate controlled protocol. This qualification neither demonstrates nor falsifies global-search efficiency.
