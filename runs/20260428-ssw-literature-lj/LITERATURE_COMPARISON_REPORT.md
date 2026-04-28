# LJ Cluster Literature Comparison And Current Production Quick Config

Date: 2026-04-28

## Scope

This report compares the current E-PAM-SSW LJ cluster quick configuration against:

- the in-repo lightweight BH/GA baselines,
- literature-level expectations for Lennard-Jones cluster global optimization,
- Cambridge Cluster Database target energies.

The current quick configuration is:

```text
max_steps_per_walk = 8
proposal_relax_steps = 80
target_uphill_energy = 1.2
quench_fmax = 1e-3
dedup_rmsd_tol = 0.2
rigid-body projection = enabled for free non-periodic clusters
L-BFGS-B gtol = fmax
```

## Literature Context

The relevant literature baselines are not the same as the in-repo quick BH/GA scripts.

- Wales and Doye's basin-hopping work is the standard literature reference for LJ clusters and reports lowest-energy LJ structures up to 110 atoms: https://doi.org/10.1021/jp970984n
- Deaven and Ho introduced the cut-and-splice genetic algorithm idea for molecular/cluster geometry optimization: https://doi.org/10.1103/PhysRevLett.75.288
- The LJ target energies used here come from the Cambridge Cluster Database LJ table: https://doye.chem.ox.ac.uk/jon/structures/LJ/

Implication: our BH/GA scripts are lightweight internal baselines, not literature-grade BH/GA implementations. Literature comparison should therefore be made mainly through known global-minimum target energies and qualitative algorithm class, not by claiming our BH/GA numbers reproduce published BH/GA performance.

## Are The Current BH/GA Baselines ASE?

Energy and force evaluation:

- All three algorithms use ASE's `LennardJones` calculator through `ASECalculator`.

Local minimization:

- All three algorithms use this repo's SciPy L-BFGS-B based `Relaxer`, not ASE optimizers.
- After M20, L-BFGS-B receives `gtol=fmax`, strict `ftol`, and `maxls=50`.

Algorithmic moves:

- BH is a simple in-repo basin-hopping-like loop: Gaussian Cartesian displacement, center removal, L-BFGS-B quench, Metropolis accept/reject.
- GA is a simple in-repo population/cut-and-splice baseline: tournament selection, cut-and-splice child, small random jitter, L-BFGS-B quench.
- These are not ASE's basinhopping implementation and not tuned literature codes.

Therefore, SSW beating these BH/GA baselines is evidence against the current in-repo baselines, not a direct claim of beating published BH/GA.

## Results

Production quick run:

```text
runs/20260428-ssw-literature-lj/lj13_38_55_prod_h8_prop80.json
```

| System | SSW mean gap | BH mean gap | GA mean gap |
|---|---:|---:|---:|
| LJ13 | 0.42739603752985644 | 0.4273960381856696 | 0.4273960381612305 |
| LJ38 | 5.471893746304644 | 7.269735555874135 | 5.903696103794346 |
| LJ55 | 10.968849326606573 | 17.780370065155097 | 14.967247873689018 |

Hard-case smoke:

```text
runs/20260428-ssw-literature-lj/lj75_seed0_prod_h8_prop80.json
```

| System | SSW gap | BH gap | GA gap |
|---|---:|---:|---:|
| LJ75 seed0 | 13.337300455975821 | 27.63912252353998 | 20.85496870885811 |

Interpretation:

- LJ13 reaches the same nonzero gap for all three methods under this workload.
- LJ38 and LJ55 favor SSW over the current in-repo BH/GA.
- LJ75 seed0 favors SSW over current in-repo BH/GA but does not reach the global minimum; this is expected for a short single-seed run on a hard funnel system.

## Energy-Lowering Curves

The curve plot is:

```text
runs/20260428-ssw-literature-lj/lj13_38_55_prod_h8_prop80_curves.png
```

Trace JSON:

```text
runs/20260428-ssw-literature-lj/lj13_38_55_prod_h8_prop80_traces.json
```

Observed trend:

- LJ38: SSW decreases energy faster early and ends below both in-repo BH and GA.
- LJ55: SSW continues improving after BH/GA plateau earlier.
- LJ13: all methods converge to the same practical gap under this quick workload.

## Minima Files

All minima files are written under:

```text
runs/20260428-ssw-literature-lj/minima_xyz/
```

There are 21 XYZ files:

- `ssw_LJ13_seed0_minima.xyz`, `ssw_LJ13_seed1_minima.xyz`
- `ssw_LJ38_seed0_minima.xyz`, `ssw_LJ38_seed1_minima.xyz`
- `ssw_LJ55_seed0_minima.xyz`, `ssw_LJ55_seed1_minima.xyz`
- `ssw_LJ75_seed0_minima.xyz`
- corresponding `bh_*` and `ga_*` files.

For BH/GA, files contain the sequence of quenched local minima encountered during the run. For SSW, files contain the archive minima.

Each XYZ frame comment includes:

```text
algorithm=<name> size=<N> seed=<seed> index=<i> energy=<E> ...
```

## Current Conclusion

The current production-level LJ quick configuration is:

```text
H = 8
proposal_relax_steps = 80
target_uphill_energy = 1.2
```

This configuration is strong enough for LJ13/LJ38/LJ55 quick testing and gives better aggregate results than the current in-repo lightweight BH/GA baselines.

It is not yet a literature-grade claim. Before paper-level comparison, we need:

1. A stronger BH baseline closer to Wales-Doye basin hopping, including tuned step size/temperature and multiple independent starts.
2. A stronger GA baseline closer to literature cut-and-splice GA, with population tuning and duplicate handling.
3. Robust SSW runs over more seeds and larger budgets for LJ38/LJ55/LJ75.
4. A direct success-rate table against Cambridge target energies, not only mean gaps.
