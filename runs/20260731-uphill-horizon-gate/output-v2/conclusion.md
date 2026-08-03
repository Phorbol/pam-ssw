# Single-Trajectory H=8 versus H=14 Capacity Gate

## Protocol

Each comparison quenches checkpoint 8 and checkpoint 14 from one shared C60
uphill trajectory. It therefore does not compare independently rerun prefixes.
Two fixed starters (`late` and `mid`) and seeds 42-44 were used.

The standard production walk ball was retained first. A single fixed
`walk_trust_radius=1e6 Å` diagnostic was run only when the standard trajectory
was radius-censored before checkpoint 14. No radius scan was performed.

## Evidence

- Execution commit: `8ed75ca72042dada8f61ce5c4ef93d8655ffbe9f`
- Complete checkpoint pairs: 5/6
- Same strict landing basin: 2/5
- Checkpoint 14 lower/higher landing counts: 2/1
- Additional uphill force evaluations from checkpoint 8 to 14: 1,087
- Standard radius-censored trajectories: 2
- Fixed no-walk-ball diagnostic trajectories: 2
- Unresolved before checkpoint 14: 1
- Aggregate force evaluations: 5,278
- Unattributed force evaluations: 0
- Recorded walk/validation time: 82.45/14.30 s

All ten checkpoint quenches in the five complete pairs had strict force
convergence certificates.

## Paired Outcomes

- `late/seed42`, no-walk-ball diagnostic: same basin, -0.00009 eV.
- `late/seed43`, no-walk-ball diagnostic: new basin, -1.56110 eV.
- `late/seed44`, standard: new basin, -0.08994 eV.
- `mid/seed43`, standard: new basin, +0.68970 eV.
- `mid/seed44`, standard: same basin, -0.00012 eV.
- `mid/seed42`, standard: explicit geometry invalidity before checkpoint 8;
  unresolved and not counted as an H8/H14 pair.

The extra six microsteps cost 158-319 uphill force evaluations per complete
trajectory.

## Mechanism Interpretation

The longer horizon changes the landing basin in 3/5 complete trajectories, so
the serial uphiller does have additional capacity beyond eight microsteps.
That capacity is not monotonic improvement: two longer trajectories land
lower, one lands substantially higher, and two return to the same basin.

Across all trajectories, requested and executed sigma means were equal and
`sigma_capped_steps=0`. The observed horizon limit is therefore not caused by
the configured per-step RMS cap suppressing requested motion. Bias base-weight
saturation occurred only intermittently and does not explain the mixed
landing outcomes.

The walk ball can hide a productive longer route: one of the two censored
standard trajectories found a basin 1.561 eV lower after the fixed no-walk-ball
diagnostic. It is not safe to remove the ball globally, because the other
diagnostic returned to the same basin and one uncensored trajectory already
ended in explicit geometry invalidity.

## Decision

Keep `max_steps_per_walk=8` and the current walk radius as production defaults.
Do not promote `H=14`: it costs 1,087 extra uphill evaluations across five
complete pairs and has a mixed 2-win/1-loss/2-neutral landing outcome.

The evidence does justify one clean extension of the later action model:
represent a longer-horizon continuation as a discrete policy arm, activated
only when its posterior value exceeds its measured cost. Do not add a
continuous horizon scheduler, radius scan, or stronger generic bias from this
gate.
