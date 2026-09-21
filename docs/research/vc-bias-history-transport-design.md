# Deferred draft: history transport across VC Gaussian stages

**Not in the execution plan.** User directed focus exclusively on existing
Safe-total after rejecting renewed bias-separated work. No history-transport
implementation or real-system job was started; the preliminary unimplemented
test scaffold was removed. This draft is retained only as a record, not as
authorized queued work.

2026-09-11. User requested adapting PAM Safe-LBFGS to cell coordinates and
using native evidence to address VC biased-quench failures. Existing
`generalized_numerics.safe_lbfgs` already implements total-objective Armijo,
positive secant screening and common whole-direction step scaling in flat
3N+6 coordinates. The original `safe-lbfgs-total` also uses total gradients;
`bias-separated-lbfgs` is a different optional method, explicitly unsupported
with LS. Neither should be described as the other.

The concrete remaining opportunity is loss of all secant information whenever
a new Gaussian changes the objective. Within one outer VC attempt, the chart,
physical enthalpy and frozen LS stay fixed; only the analytic Gaussian sum
changes. Let F_k(q)=A(q)+B_k(q), where A=E+pV+frozen LS. For previously
accepted endpoints a,b, retain s=b-a and

    y_A = [grad F_k(b)-grad F_k(a)]-[grad B_k(b)-grad B_k(a)].

At a new Gaussian stage j reconstruct

    y_j = y_A + grad B_j(b)-grad B_j(a).

This is algebraically the secant difference of the **current total objective**
at those same endpoints, subject to floating-point arithmetic. It is not a
claim of an exact Hessian or a globally reliable old local approximation.
Re-screen every reconstructed pair with the existing relative positive-curvature
criterion; build ordinary Safe-total inverse products from the retained pairs.
Retain the existing bounded memory m, with no new age threshold, damping,
reward, fallback or numerical tuning parameter. New accepted steps populate
the history; rejected line-search trials never do. Clear all stored pairs at
the outer chart/LS reset. It is invalid to transfer across different charts,
cell metrics, physical constraints or LS strengths without another derivation.

Implementation is an explicit `quench_history='transport_bias'` option; the
reset default stays byte-for-byte numerically unchanged. Generic Safe-total
needs only a validated optional initial secant list. Analytic transport lives
in a separate small history object, using existing accepted traces, and must
make zero physical calls. When native-derived height rewrites previous weights,
recompute against the complete new Gaussian sum, not merely the newest term.
The same Armijo objective, gradient, convergence and step caps remain in force.

Relation to literature: Anitescu, Chiang and Petra, 'A Structured Quasi-Newton
Algorithm for Optimizing with Incomplete Hessian Information', SIAM J. Optim.,
2019, DOI10.1137/18M1167942, develops methods separating known and unavailable
second-order information. Author preprint record:
https://optimization-online.org/2018/02/6451/ . That motivates examining known
bias structure; its algorithm and convergence theorem are **not** the
history-transport construction above. This project's identity is derived
explicitly here and does not inherit their convergence claims.

Validation: exact secants after added/rewritten Gaussians, invalid curvature
filtering, memory bounds and zero-PES transport; unchanged reset trajectories;
then matched end-to-end MACE joint VC runs on qualified Fe7C3-80 and CuO64,
seeds7/101, original no-LS forward-height/history10 parameters,2000 total EFS
per arm including3 reserved final checks,2 requested outer attempts. Existing
complete reset runs are frozen controls; no parameter selection from outcomes.
At most8000 new EFS, one V100/30min,480s per arm. Count all initial, failed,
trial and fresh work and report all8 requested attempts. No new biased
stationary point alone establishes a physical minimum; qualify any actual
physical landing independently. If no useful improvement appears, keep the
option experimental and end this test rather than add more knobs.
