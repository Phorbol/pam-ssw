#!/usr/bin/env python3
"""Read-only summarizer for the four bounded M80 direction-transfer slots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ledger_cost(path):
    cost = dict(charged=0, actual=0, denied=0, failed=0, denial_reasons={}, present=path.is_file())
    if path.is_file():
        for line in path.open():
            row=json.loads(line)
            cost['charged']+=bool(row.get('charged'))
            cost['actual']+=int(row.get('actual_calculator_calls',0))
            cost['denied']+=row.get('status')=='denied'
            cost['failed']+=row.get('status')=='failed'
            if row.get('status')=='denied':
                reason=row.get('reason','unspecified')
                cost['denial_reasons'][reason]=cost['denial_reasons'].get(reason,0)+1
    return cost


def read_arm(path: Path, slot: int):
    result_path = path / "result.json"
    costs={kind:ledger_cost(path/name) for kind,name in
        [('search','search-ef-ledger.jsonl'),('fresh','fresh-ef-ledger.jsonl'),
         ('reference','shared-reference-fresh-ledger.jsonl')]}
    if not result_path.is_file():
        return {"run_dir": str(path), "slot":slot, "result_present":False,
                "status": "missing_result", "raw_costs":costs}
    result = json.loads(result_path.read_text())
    assert result['slot']==slot
    assert result['head_pamssw_tree']=='c572a1cc0766f8d3ff534a467ad64613aaa78ce0'
    assert costs['search']['charged']==result['search_requests']
    assert costs['search']['actual']==result['search_actual_calculator_calls']
    assert costs['search']['denied']==result['search_denied_requests']
    assert costs['fresh']['charged']==result['fresh_requests']
    assert costs['reference']['charged']==result['shared_reference_requests']
    if result['request_closure'] is not None:
        assert result['request_closure'], f'initial/outer cost mismatch {path}'
    qualified = result.get("connected_force_qualified_minima") or []
    best_q = min(qualified, key=lambda row: row["energy_eV"]) if qualified else None
    fresh = {row.get("role"): row for row in result.get("fresh", [])}
    best_fresh = fresh.get("best")
    candidate = result.get("candidate")
    return {
        "run_dir": str(path), "slot": result.get("slot"), "result_present":True,
        "raw_costs":costs, "request_closure":result['request_closure'],
        "git_head":result['git_head'],"core_tree":result['head_pamssw_tree'],
        "outer_status_counts":result['outer_status_counts'],
        "input_seed": result.get("input_seed"), "input": result.get("input"),
        "input_sha256": result.get("input_sha256"), "arm": result.get("arm"),
        "paired_rng_seed": result.get("paired_rng_seed"),
        "prior_random_qualification": result.get("prior_random_qualification"),
        "execution_status": result.get("result_status") or ("failed" if result.get("error") else "unknown"),
        "error": result.get("error"), "search_requests": result.get("search_requests"),
        "search_actual_calculator_calls": result.get("search_actual_calculator_calls"),
        "search_wall_seconds": result.get("search_wall_seconds"),
        "outer_callbacks": result.get("outer_callbacks"),
        "budget_denials": costs['search']['denial_reasons'],
        "best_connected_force_qualified_minimum": best_q,
        "connected_force_qualified_minimum_count": len(qualified),
        "connected_force_qualified_curve": qualified,
        "target_candidate_seen": bool(candidate and candidate.get("qualification", {}).get("qualified_candidate")),
        "fresh_initial": fresh.get("initial"), "fresh_best": best_fresh,
        "fresh_best_target_qualified": bool(best_fresh and best_fresh.get("qualification", {}).get("qualified_candidate")),
        "confirmed_80G_hit": bool(result.get('candidate_pause_confirmed') and
            result.get('result_status')=='paused' and candidate and candidate.get("qualification", {}).get("qualified_candidate") and
            best_fresh and best_fresh.get("qualification", {}).get("qualified_candidate")),
        "fresh_requests": result.get("fresh_requests"),
        "shared_reference": result.get("shared_reference_qualification"),
        "shared_reference_requests": result.get("shared_reference_requests"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=4, type=Path,
        help="slot directories in order 0, 1, 2, 3; this readout performs no PES calls")
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    arms = [read_arm(p,i) for i,p in enumerate(args.run_dirs)]
    by_seed = {}
    for arm in arms:
        if arm.get("input_seed") is not None:
            by_seed.setdefault(str(arm["input_seed"]), {})[arm.get("arm", "unknown")] = arm
    pairs = []
    for seed, pair in sorted(by_seed.items()):
        rot, full = pair.get("rotation"), pair.get("full_per_atom")
        qrot = None if not rot else rot.get("best_connected_force_qualified_minimum")
        qfull = None if not full else full.get("best_connected_force_qualified_minimum")
        pairs.append({"input_seed": int(seed), "rotation_present": rot is not None,
            "full_per_atom_present": full is not None,
            "rotation_best_qualified_energy_eV": None if not qrot else qrot.get("energy_eV"),
            "full_best_qualified_energy_eV": None if not qfull else qfull.get("energy_eV"),
            "paired_energy_delta_full_minus_rotation_eV": (None if not qrot or not qfull else
                float(qfull["energy_eV"] - qrot["energy_eV"])),
            "rotation_requests_at_best_qualified": None if not qrot else qrot.get("cumulative_paid_requests"),
            "full_requests_at_best_qualified": None if not qfull else qfull.get("cumulative_paid_requests"),
            "rotation_confirmed_80G_hit": None if not rot else rot.get("confirmed_80G_hit"),
            "full_confirmed_80G_hit": None if not full else full.get("confirmed_80G_hit")})
    for pair in pairs:
        grouped=by_seed[str(pair['input_seed'])]
        horizon=min((a['raw_costs']['search']['charged'] for a in grouped.values()),default=0)
        pair['common_paid_horizon']=horizon
        pair['common_cost_connected_best_eV']={method:min((q['energy_eV'] for q in a['connected_force_qualified_curve']
            if q['cumulative_paid_requests']<=horizon),default=None) for method,a in grouped.items()}
    summary = {"scope": "bounded development transfer only; not paper-rate replication or arm success-rate estimate",
        "real_pes_requests": 0, "arms_received": len(arms), "arms": arms, "paired_comparison": pairs,
        "planned_slots": 4, "missing_slots": [a['slot'] for a in arms if not a['result_present']],
        "observed_search_requests_sum": sum(int(a["search_requests"]) for a in arms if a.get("search_requests") is not None),
        "observed_fresh_requests_sum": sum(int(a.get("fresh_requests") or 0) +
            int(a.get("shared_reference_requests") or 0) for a in arms),
        "confirmed_80G_hits": sum(bool(a.get("confirmed_80G_hit")) for a in arms),
        "censoring_note": "missing results, errors, request/wall caps, and partial callback tails remain visible; no status is treated as scientific success"}
    summary['reference_fresh_qualified']=bool(arms[0].get('shared_reference',{}).get('passed'))
    summary['costs']={kind:sum(a['raw_costs'][kind]['charged'] for a in arms) for kind in ('search','fresh','reference')}
    prior={a['input_seed']:a['prior_random_qualification'] for a in arms if a.get('prior_random_qualification')}
    summary['reused_initial_preparation_requests']=sum(p['search_requests']+p['fresh_requests'] for p in prior.values())
    if not summary['reference_fresh_qualified']:
        for a in arms: a['confirmed_80G_hit']=False
        for p in pairs:
            p['rotation_confirmed_80G_hit']=p['full_confirmed_80G_hit']=False
    summary['scope']+='; qualified minimum observations may repeat a basin, not deduplicated basin counts'
    lines=['# M80 bounded direction transfer', '',
           '| Input seed | Arm | Paid search | Calculator calls | Connected force-qualified observations | Best connected E | 80G joint hit | Execution |',
           '|---:|---|---:|---:|---:|---:|---|---|']
    for a in arms:
        q=a.get('best_connected_force_qualified_minimum') or {}
        status=a.get('execution_status',a.get('status'))
        if a['raw_costs']['search']['denial_reasons']:
            status += ' (budget censored: '+', '.join(a['raw_costs']['search']['denial_reasons'])+')'
        lines.append(f"| {a.get('input_seed')} | {a.get('arm')} | {a['raw_costs']['search']['charged']} | {a['raw_costs']['search']['actual']} | {a.get('connected_force_qualified_minimum_count')} | {q.get('energy_eV')} | {a.get('confirmed_80G_hit')} | {status} |")
    lines += ['',f"Current panel requests: {summary['costs']}; reused historical input preparation {summary['reused_initial_preparation_requests']} (reported separately).",
        'Saved noninitial landings use the algorithm force certificate. Cold E/F checks cover initial and best only.',
        'Two fixed inputs, including one fragmented input; not paper success-rate replication, general superiority or default promotion.']
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == "__main__":
    main()
