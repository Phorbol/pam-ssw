# Saved-center return diagnosis

All 12 C60 arms have 12 saved Gaussian centers. Four feedback-only landings
retain the initial indexed C-C edges at 1.64/1.70/1.80 A. No feedback-only
center differs at 1.70 A; three trajectories have only one-edge excursions at
one of the other thresholds. The full/local modes have larger changes.
These cutoff-sensitive excursions do not establish a useful chemical transition.

Decision: retain the previous no-promotion decision. The archived centers do
not support attributing return specifically to final true quenching: the final
biased endpoint was not saved. Do not modify quench release/acceptance using
this incomplete causal evidence, and do not launch another return-only panel.
For a future independently justified experiment, preserve the last biased
endpoint using an existing trajectory hook if available. No new API is justified
solely by this archive limitation.

Evidence: classification.json and classify.py, 12/12 slots, 144 saved centers;
no PES calls, no numerical search or parameter changes. Root checked all slots
and the reported feedback counts against the compact output.
