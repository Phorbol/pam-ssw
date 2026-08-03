# Raw-PdO direction-continuation repeat gate

- Decision: `repeat_stable_transport_support`
- Paired transported landing wins: 6/6
- Certificates (control / transported): 6 / 6
- Action FE (control / transported / saved): 3011 / 2282 / 729
- Direction FE (control / transported / saved): 696 / 26 / 670
- Action wall time (control / transported / saved): 67.828 / 51.267 / 16.561 s
- Non-direction action FE (control / transported / saved): 2315 / 2256 / 59
- Microsteps (control / transported): 35 / 19
- Bootstrap state hashes identical across repeats: False

Mechanism: 670 of 729 saved action evaluations come directly from replacing repeated 12-HVP Ritz solves with one-HVP transport checks. The remaining 59 evaluations come from fewer downstream microsteps.
Numerical caveat: the two float32 GPU bootstrap states have the same reported energy but different coordinate hashes. The mechanism conclusion is repeat-stable, not bitwise trajectory-stable.

Claim ceiling: two complete raw-input PdO repeats with three paired seeds per arm; combined with C60 this supports direction transport as a candidate mechanism, not yet as a 200-step production default.
