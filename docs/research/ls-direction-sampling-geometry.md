# LS initial-direction geometry (2026-09-12)

Problem: `run_ssw` soft-quenched `work`, then sampled the geometry-dependent
paper pair direction from the still-accepted true minimum `current`. A pair
threshold/vector could therefore refer to different coordinates than those
entering the rotation surface. Global sampling only uses masses and is unaffected.

Source: Guan, Shang and Liu, JCTC 2024, 20, 11093,
DOI [10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081),
section 2.4 steps 2–3: soft-surface optimization precedes mode generation.
Author [PDF](https://www.lasphub.com/publication/215.pdf), existing local
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/215.txt:337–351`.
The text was re-read from the archive; publisher retrieval returned403 and the
current author PDF request502. Native caller ordering independently places
BFGS/LS before get_random_mode0; see native-ls-prequench-exit-pair.md. That
static ordering alone does not prove all native optimizer exit geometries.

Implementation: sample from `work` after soft quenching; keep the same sampled
anchor throughout this independent climb. No distribution constants, random
calls or budgets change. This is the explicit interpretation of a geometrical
proposal generated on the softened search state, not full native random-mode
parity. The separate question of anchor lifetime across Gaussian stages is
unchanged and must not be inferred from this one-line correction.

Verification: real Cu13/EMT with paper pair sampling captures the actual
soft-quench coordinates and confirms they differ from the true initial minimum;
the sampler must receive exactly that softened geometry. Recent frozen campaigns
used global sampling and thus retain their numerical provenance; they do not
validate the search-effect size of this correction for pair sampling.
