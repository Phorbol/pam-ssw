# Full request-budget LJ38 readout

Question resolved: removing repeated full-history copying permits this frozen pilot to execute its original request cap within the existing wall limit. It does not resolve target discovery. CPU1492861/1492862 completed in28m00/27m42 across two sequential seeds each; CPU1492863 completed geometry/prefix/cost analysis in24s. No GPU. Total3,200,000 search requests plus4 fresh final-best E/F checks, below the3,200,000+8 bound. No cap extension or new search settings.

| Direction | Seed | Completed outer boundaries | Paid terminal partial requests | Best energy/eV | Fresh fmax/eV A^-1 | Search wall/s |
|---|---:|---:|---:|---:|---:|---:|
|global|25092501|905|20|-173.2523738915|0.008762|797.54|
|global|25092502|903|362|-173.2523692919|0.005923|866.18|
|paper|25092501|1162|575|-173.2523497139|0.006032|799.42|
|paper|25092502|1148|470|-173.2523650043|0.007455|847.41|

Every row used800000 search requests. Generic driver status is `evaluation_failed` because the bounded oracle deliberately rejects the next request; `boundary=arm_request_cap` identifies budget censoring, not an observed calculator numerical failure. Partial attempts remain charged and are excluded from completed-boundary counts. `len(records)` includes that final partial attempt and must not be used as the completed count.

The replay agrees with all1570 old common scalar boundaries, including4 initial quenches: no status, acceptance, cost or energy mismatches. Identical generated input/RNG was checked prospectively; the later common-prefix analysis additionally matches the archived initial coordinates. New and old histories are the same experiments continued by replay, not independent replicates. The wall figures describe actual runs, not a controlled identical-hardware timing speedup.

Target discovery is0/4 trajectories across two shared seeds and two direction settings, not four independent random inputs. None satisfies GM energy -173.928427 within0.001eV. All four final minima pass the fixed fresh force criterion0.01eV/Angstrom and geometrically match the separately qualified second-lowest reference: RMS0.000147,0.000280,0.000573,0.000284 Angstrom, respectively. Small energy differences inside that same minimum are not useful algorithm ranking. Paper direction produces more completed steps per equal request budget without finding a different best structure; step count is not success.

Decision: do not extend these trajectories or promote either direction. The previous wall-censoring explanation is removed for the specified800000-request test. These results neither reproduce the paper's1000-run statistics nor establish universal failure. Retain LJ55 positive controls and real C60/C4H6/material results separately. The next bounded diagnostic captures only three early outer attempts in each arm, selected to cover the already observed first fragmented paper proposals. It asks where rejected fragmentation arises; it does not change the search or fit parameters. See `stage-probe-plan.md`.

Evidence: `compact-comparison.json`; both compact run roots' summaries and `geometry-analysis.json`; `lj38-competitor-geometry.json`; raw per-run minima/outer-step ledgers. Analysis commands: `analyze_lj38_compact.py`, `analyze_lj_pilot.py <run-root>`, and `analyze_lj38_competitors.py` (all zeroPES).
