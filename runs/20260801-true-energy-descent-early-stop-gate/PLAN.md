# G-E0: true-PES descent stopping certificate

G-E0 audits the first completed outer SSW micro step whose true-PES energy is
lower than its macro starter by more than the existing `dedup_energy_tol`.
It distinguishes a sufficient lower-basin certificate from an optimal stopping
rule by comparing the first crossing with the original terminal landing.

The immutable cohort is the 24 C60/PdO D0/K4 paths and 82 accepted endpoints
from the current-action first-passage corpus. Existing energy and quench rows
are reused; only missing true energies and unreused crossing/terminal quenches
are evaluated.

Limits and exclusions:

- at most 10,000 new force evaluations;
- at most 300 seconds GPU kernel wall time;
- zero direction HVP;
- zero biased proposal relaxation;
- zero unattributed evaluations;
- no production-code, selector, direction, Gaussian, optimizer or profile
  change.
