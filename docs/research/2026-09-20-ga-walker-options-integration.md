# GA walker options integration

This record covers interface and checkpoint integration only. It does not
measure GA search performance.

The public GA entry point now accepts optional `mc` and
`recovered_rotation` settings and forwards them to each new SSW walk. The
checkpoint contract preserves those settings, and each resumed walk restores
the complete GA boundary state. The 50-case interface/checkpoint suite passed
in [`pytest-1414746.log`](../../research/ga_ssw/evidence/ga-walker-options-tests-20260920/pytest-1414746.log).
The mock `offspring_ssw` forwarding contract is covered by that suite; it was
not exercised by the real EMT run.

The real case is
[`ga-walker-options-emt-20260920`](../../research/ga_ssw/evidence/ga-walker-options-emt-20260920/).
The recorded result uses the frozen source import shown in `manifest.json`.
The full run used 1287 EMT E/F requests. The quick-boundary split used 388
requests before the `quick_complete` checkpoint and 899 after resume, again
1287 total. Full and resumed observation coordinates/energies matched, and
their complete surface ledgers matched including forces. Total search cost
across the full and split/resume trajectories was therefore 2574 E/F calls;
12 independent fresh EMT checks brought the recorded total to 2586.

The real trajectory exercised `quick`, `generation_short`, and `fine` walks,
with `initial_quench`, `offspring_quench`, and proposal stages also recorded.
It did not exercise `offspring_ssw`; the real case used only
`offspring_quench`, so no claim is made for that path beyond the mock contract
test.

Both full and resumed paths recorded five eligible MC decisions and 45
`recovered-cbd` rotation events. The checkpoint was taken at the complete
`quick_complete` boundary. MC state is reset per SSW walk as specified by the
GA interface; it is not shared between walks. The 12 fresh frames were all
energy-consistent, passed the 0.01 eV/A force check, and were connected at the
3 Å diagnostic cutoff.

The recovered-CBD settings in this integration case were `rotation_bias=1`,
`pre_rotmax=5`, `rotmax=15`, `pre_ftol=1`, `ftol=.1`, Euclidean metric, and
40 force calls. These are the GA interface test settings and are distinct from
the C60 settings `.2/.02`; the result is an interface/checkpoint qualification,
not an effects comparison or a GA performance result.

## TYPE0 BLLimit contract check

The uploaded Java TYPE0 path does not apply the configured BLLimit to its
returned candidates. `sgn/ga_Interface/TYPE0.java:39-67` constructs crossover
and mutation candidates and returns them directly. The dispatcher
`sgn/app_ssw_ga/SSWGaSupport.java:675-711` only selects TYPE0; it does not add
a post-filter. The generic default table is defined by
`sgn/configure/ConfigureRead.java:127-151` as
`(ElementPara.getAtomR(Z1)+ElementPara.getAtomR(Z2))*d`, with the element table
and final division by two in `sgn/other/ElementPara.java:243-357`.
`sgn/other/CooHandle.java:298-315` is the pair-distance filter used by other
GA types. For C60, `getAtomR(6)=0.639278` Å, but no TYPE0 default cutoff is
implied by this chain.

The Python port preserves this boundary. `pamssw/standalone/atomic_ga.py:296-305`
documents that TYPE0 has no JAR BLLimit filter, while
`pamssw/standalone/paper_ga.py:622-629` passes the caller's explicit
`proposal_bond_limits` to `propose_type0`; the loop in
`pamssw/standalone/ga_operators.py:613-635` accepts an empty mapping without
rejecting candidates. Thus `proposal_bond_limits={}` is a valid explicit
no-cutoff input to `run_ga_ssw`; it is not an omitted or automatically derived
default. The Cu13/Al13 runner's `.7*nearest` cutoff is an experiment-specific
caller choice (`research/ga_ssw/run_fixed_ga_multicase.py:53-55`), not a Java
TYPE0 default.

The previously untested real `offspring_ssw` branch is now exercised by
`run_ga_offspring_options_emt.py`. It freezes and reuses the original Cu13
runner and changes only `offspring_steps=1`; `ga_candidates=8`, empty proposal
cutoff, initial states, seed and all walker options remain unchanged.

CPU job 1415944 completed in 13 seconds. Full and quick-boundary-resumed
trajectories each cost 4034 search E/F (resume split: 388+3646). They produce
identical complete E/F ledgers and observation sequences. Each trajectory
executes 13 offspring stages (the candidate generator can overshoot its batch
target), with MC and recovered-CBD events verified from offspring-specific
records. All stages completed without recorded failures. The 2721 requests in
the offspring *outer-step records* exclude their initial quench costs and must
not replace full stage or total trajectory accounting.

All 13 retained structures passed fresh EMT energy, force <=0.01 eV/A and
3 A connectivity checks. Total experiment cost is 8068 search +13 fresh =8081
E/F. This qualifies the execution, option propagation and resume contract; it
does not establish GA global-search superiority or C60 success.

[Fixed protocol and frozen source](../../research/ga_ssw/evidence/ga-offspring-options-emt-20260920/plan.json),
[result](../../research/ga_ssw/evidence/ga-offspring-options-emt-20260920/result.json).


## TYPE1/TYPE4 与 NativeLS 的既有回调组合

无需增加公共参数或改写控制器；通过 `functools.partial(run_ssw, ls=settings)`
传给固定胞 `run_periodic_ga(..., walker=walker)`，表面分支则绑定
`run_constrained_ssw`。新增 `test_ga_ls_composition.py` 检查每一次快速、子代和
精细行走确实进入LS路径；此接口检查使用注入的proposal，允许明确的LS失败，
不作为真实GA成功证据。CPU作业1416793相关11项测试通过。

同作业的Al31/EMT真实TYPE1实验使用既有三个保存父体、原有参数和去重规则，
4次行走（快速3、精细1）的LS预淬火及响应更新均成功。726搜索加8独立复核
共734次E/F，落点最大力均低于0.01 eV/Å，能量、组分、固定胞和PBC检查通过。
但归档只保留一个结构代表，控制器返回no_proposal，未产生子代；
因此真实offspring+LS仍未验收，不通过放松去重或强造父体补足覆盖。

[冻结协议、原始结果及限制](../../../ga-ls-composition/research/ga_ssw/evidence/ga-ls-composition-al31-20260920/prepared-v3/RESULT.md)。
控制器核心未改动，仅集成接口测试；TYPE1/TYPE4尚无全局checkpoint，
表面GA只支持其声明的FixAtoms，不将独立SSW的Hookean支持外推。


后续Cu31复用既有三个TYPE1父体，未改去重或参数。CPU1417067完成22秒：
7次LS行走（quick3、真实offspring_quick3、fine1），7次预淬火和响应更新通过；
1739搜索+14独立复核=1753 E/F，14/14帧能量、力≤0.01 eV/Å、cell/PBC/组分
核验通过（最大力0.00984 eV/Å），无缺失或预算截断。
主agent独立核对1739条paid ledger、所有walk成本和fresh原始记录。
这补齐了固定胞TYPE1真实子代与LS组合验收；不证明GA/LS效率优势，
不外推到TYPE4表面真实子代或VC。
[Cu31冻结协议和原始产物](../../../ga-ls-composition/research/ga_ssw/evidence/ga-ls-composition-cu31-20260920/prepared/)。
该支线已达到本次验收，暂停扩算，继续SSW阶段机制主线。
