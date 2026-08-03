# G-E1: online first-descent paired action gate

Run 24 fresh paired C60/PdO D0/K4 actions at seeds 45--47.  The only
difference is whether the walk stops when an already-computed true endpoint
energy first falls below the starter by the existing 0.001 eV deduplication
tolerance.

Limits:

- at most 25,000 newly executed force evaluations;
- at most 300 seconds GPU kernel wall time;
- no public configuration field or production-default change;
- no selector, direction, Gaussian, trust, optimizer, quench or matcher
  change;
- no scalarized cost/energy reward.
