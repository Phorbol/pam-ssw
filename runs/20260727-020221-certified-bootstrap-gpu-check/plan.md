# Certified bootstrap GPU check

This is a two-call validation of commit
`01bcdc47a002bdf00f0d568f966be06169080756`.

It reuses the exact C60/PdO state, MACE calculator, quench tolerance, quench
iteration cap, and total budget from the frozen G1 driver. Each system is run
once in a separate process. No proposal action is dispatched.

Success requires:

- the per-atom force certificate passes without relaxing `fmax=0.03 eV/A`;
- every charged call is assigned a purpose;
- any failure persists its exact charged ledger.
