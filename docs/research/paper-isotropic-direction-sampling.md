# Paper reference isotropic direction sampling

`pamssw.standalone.paper_reference` now exposes the explicit
`direction_sampling="isotropic"` option. It draws `rng.normal(size=(N, 3))`
and normalizes the Cartesian vector, without the existing `1/sqrt(mass)`
factor. The default `paper` and explicit `global` paths are unchanged.

This is an experimental proposal-chart choice for energy-surface exploration:
mass weighting can overemphasize light atoms when the objective is configurational
energy search. It is not an exact reconstruction of the native RNG or mixture;
the native probe only supports the distinction between the mass-free draw and
the later mean-subtraction/projection steps. Existing rotation-frame projection
is still applied by the caller after sampling and is not duplicated here.

For three-dimensional PBC, the public validator permits `translation_only` with
either `global` or `isotropic`; periodic geometry still handles the translation
projection. Other kernels retain their existing explicit `global` restrictions.
No default, reward, temperature, or other search parameter changes are implied.
