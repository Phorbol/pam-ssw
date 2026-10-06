# Saved-state rotation diagnosis: iteration limitation, not unreachable residual

GPU1663620/source196d502 completed424paid E/F/stress:320iterative,84Hessian,
20independent directional checks; raw request/call ledgers allclose, failures0,
denials0. Initial CPU1663621 readout failed because its glob mixed fresh-records
summaries into raw ledgers. Preserve that error. Source1d0da71 excludes that
summary in runner/analysis; CPU1663637 then passed without any PES rerun. The
original result.json aggregate ledger map retains its spurious nonledger entry;
derived readout uses only actual evaluation ledgers and verifies total424.

Both methods receive the same exact anchor, chart, image distance, rank-one
rotation term and40EFS per direction. Four pairs yield the following independent
central residuals (eV/A^2); none reaches the unchanged .02gate:

| Size / anchor | Plane dimer | Central Ritz | Full12atom reference |
|---|---:|---:|---:|
|12/0|1.071791|.065652|.000016095|
|12/1|2.017085|.067662|.000016091|
|48/0|1.185999|.538498|not computed|
|48/1|1.397869|.534199|not computed|

Ritz's returned biased Rayleigh curvature is lower in all four pairs; directions
differ45–49degrees. This suggests a retained subspace helps on these states,
not that it is already a better global-search kernel. Central direct checks
agree with its measured residuals. The full12atom Hessian reference satisfies
the same direct test by a large margin; projected asymmetry norm.00022262 is
recorded. Reference is local, rotation-biased and finite-difference-based, not
a physical minimum-stability certificate or a proposed dense production solver.

**Decision:** current40-request two-vector rotation is inadequate for the strict
eigenresidual gate on these states. For12atoms the gate is demonstrably attainable;
we should not blame model noise or line search.48atoms remain an iterative-budget
and metric question, without a dense reference. Do not promote central Ritz,
change tolerances, scale the metric or extend the same four-arm search from this
local observation. First test the already approved finite-budget proposal policy
at the periodic atomic entry, keeping numerical labels and downstream physical
certificates distinct. Joint strict driver remains explicit experimental baseline.
