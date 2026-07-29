# Shared-cap-400 fixed replay

The production-cap replay is right-censored: most FIRE/FIRE2 tasks reached the
80/120 step cap before satisfying the common 0.05 eV/A certificate.

This follow-up loads the exact same frozen task payloads and changes only the
shared observation cap to 400 for all three backends. The biased PES, initial
state, force tolerance, trust radius, model, precision, and backend algorithms
remain unchanged.

The cap is an observation horizon, not a backend-specific tuning parameter.
Results are still separated by endpoint; a lower evaluation count at a
different endpoint is not labeled same-endpoint acceleration.
