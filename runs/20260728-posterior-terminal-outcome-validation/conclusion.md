# Posterior terminal-outcome validation

Source commit: `cc7efc16907ae8dc319020c348edd40f79507525`

The analytic ThreadPool campaign committed 6 actions
in batch widths [3, 2, 1]. Bootstrap, action, and total
force-evaluation counts were 2,
104, and 106;
unattributed evaluations were zero.

All real-campaign and terminal-matrix invariants passed. The committed event
log reconstructed the same starter-productivity posterior as the live
controller.

This validation does not compare starter-policy performance, establish
thermodynamic or stationary-distribution unbiasedness, validate GPU/MLIP
execution, or make a C60/PdO performance claim.
