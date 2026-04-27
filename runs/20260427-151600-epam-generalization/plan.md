# Approved Plan

1. Refactor E-PAM-SSW so default proposal generation is generic.
2. Add an explicit benchmark helper switch for LJ magic-cluster motif proposals.
3. Add tests proving default mode does not use LJ-specific motifs and periodic/slab states do not receive cluster-only proposal pools.
4. Run `pytest -q tests/unit tests/integration`.
5. Run generic-only quick LJ benchmark.
6. Run optional LJ-helper quick LJ benchmark as a diagnostic only.
7. Attempt larger LJ benchmark with `LJ55` and, if target support is added safely, `LJ75`.
8. Persist a report summarizing generic versus helper evidence and residual risks.
