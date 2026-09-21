"""Read-only summary of the two Fe7C3 block cell-metric campaigns."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ase import Atoms

TOLERANCES = {"strict": (.05, .10, 2.), "default": (.20, .30, 5.), "loose": (.30, .50, 10.)}

def atoms(d):
    if "atoms" in d: d=d["atoms"]
    return Atoms(numbers=d["numbers"], positions=d["positions"], cell=d["cell"], pbc=d["pbc"])

def evaluations(path):
    p=path/"evaluations.jsonl"
    if not p.is_file(): return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def counts(rows):
    return {"rows":len(rows), "charged":sum(bool(x.get("charged")) for x in rows),
      "denied":sum(not bool(x.get("charged")) for x in rows),
      "search_charged":sum(x.get("stage")=="search" and bool(x.get("charged")) for x in rows),
      "fresh_charged":sum(x.get("stage")=="fresh" and bool(x.get("charged")) for x in rows),
      "search_denied":sum(x.get("stage")=="search" and not bool(x.get("charged")) for x in rows),
      "fresh_denied":sum(x.get("stage")=="fresh" and not bool(x.get("charged")) for x in rows)}

def final_metrics(rows, request):
    search=[x for x in rows if x.get("stage")=="search" and x.get("charged") and int(x.get("request",-1))==request]
    if not search: return {"status":"missing_exact_request","request":request}
    x=search[0]; a=atoms(x["atoms"]); f=np.asarray(x.get("forces",[]),float); s=np.asarray(x.get("stress",[]),float)
    return {"request":int(x["request"]),"exact":True,"last_evaluated":True,"accepted_endpoint":False,"energy":x.get("energy"),"volume":float(a.get_volume()),
      "fmax":None if f.size==0 else float(np.linalg.norm(f,axis=1).max()),"stress_max":None if s.size==0 else float(np.abs(s).max()),"atoms":x["atoms"]}

def arm(path, requested_attempts=2):
    rp=path/"result.json"; ep=path/"evaluations.jsonl"
    if not rp.is_file() or not ep.is_file(): return {"status":"pending","missing":[x.name for x in (rp,ep) if not x.is_file()]}
    d=json.loads(rp.read_text()); rows=evaluations(path); c=counts(rows); errors=[]
    if d.get("requests") is not None and int(d["requests"])!=c["charged"]: errors.append("reported_requests_vs_charged")
    records=d.get("records") or []; attempts=[r for r in records if isinstance(r,dict) and "index" in r]
    paid=[r for r in attempts if int(r.get("requests",0) or 0)>0]
    fresh=d.get("fresh") if isinstance(d.get("fresh"),dict) else {}; checks=fresh.get("checks") or []
    landings=d.get("landings") or []; new=[]
    for ch in checks:
        i=int(ch.get("index",-1))
        if 0<=i<len(landings) and int(landings[i].get("index",-1))>=0:
            new.append({"landing_index":int(landings[i]["index"]),"check_index":i,"certified":ch.get("certified") is True,"accepted":bool(landings[i].get("accepted",False)),"energy":landings[i].get("energy")})
    for ch in checks:
        i=int(ch.get("index",-1))
        if 0<=i<len(landings) and isinstance(ch.get("atoms"),dict) and isinstance(landings[i].get("atoms"),dict):
            ca,la=ch["atoms"],landings[i]["atoms"]
            if ca.get("numbers")!=la.get("numbers") or not np.allclose(ca.get("positions"),la.get("positions"),rtol=0.,atol=1e-10) or not np.allclose(ca.get("cell"),la.get("cell"),rtol=0.,atol=1e-10): errors.append(f"fresh_landing_atoms_mismatch:{i}")
    cumulative=0; outer=[]
    for r in records:
        if not isinstance(r,dict) or "requests" not in r: continue
        start=cumulative; cumulative+=int(r.get("requests",0) or 0)
        atomic=r.get("atomic") or {}; cp=atomic.get("checkpoint") or {}; climb=atomic.get("climb") or cp.get("climb") or r.get("climb") or []
        outer.append({"index":r.get("index"),"status":r.get("status"),"start_request":start,"requests":r.get("requests"),"end_request":cumulative,
                      "last_evaluated":final_metrics(rows,cumulative),"atomic_status":atomic.get("status"),"completed_gaussians":len([x for x in climb if isinstance(x,dict) and "true_energy" in x])})
    return {"status":"audited","counts":c,"reported_requests":d.get("requests"),"consistency_errors":errors,
      "requested_attempts":requested_attempts,"record_attempts":len(attempts),"paid_attempts":len(paid),
      "not_entered_attempts":max(0,requested_attempts-len(attempts)),"initial_certified":(checks[0].get("certified") if checks and int(checks[0].get("index",-1))<1 else None),
      "fresh":{"requested":fresh.get("requested"),"checked":len(checks),"certified":sum(x.get("certified") is True for x in checks),"checks":checks},
      "new_landing_count":len(new),"new_certified_count":sum(x["certified"] for x in new),"new_landings":new,"outer":outer,
      "reported_status":d.get("status"),"records":len(records)}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--output",type=Path); args=ap.parse_args(argv)
    root=args.root; arms={}; structures=[]
    for metric in sorted(root.iterdir() if root.is_dir() else []):
        plan=metric/"plan.json"
        if not plan.is_file(): continue
        pd=json.loads(plan.read_text()); solver=(pd.get("research_solvers") or ["safe_total"])[0]
        for seed in pd.get("research_seeds",pd.get("seeds",[7,101])):
            key=f"{metric.name}/{solver}-seed{seed}"; path=metric/"comparison"/f"{solver}-seed{seed}"
            arms[key]=arm(path,requested_attempts=int(pd.get("steps_per_arm",2)))
            if arms[key].get("status")=="audited":
                d=json.loads((path/"result.json").read_text()); land=d.get("landings") or []; checks=(d.get("fresh") or {}).get("checks") or []
                for ch in checks:
                    i=int(ch.get("index",-1))
                    if 0<=i<len(land) and int(land[i].get("index",-1))>=0 and ch.get("certified") is True:
                        structures.append({"id":f"{key}/landing{land[i]['index']}","initial":atoms(land[0]),"terminal":atoms(land[i])})
    tol={k:[*v] for k,v in TOLERANCES.items()}; matrices={}
    try:
        from pamssw.standalone.periodic_ga_reference import pymatgen_identity
        for name,vals in TOLERANCES.items():
            fn=pymatgen_identity(ltol=vals[0],stol=vals[1],angle_tol=vals[2]); all_s=[x["initial"] for x in structures]+[x["terminal"] for x in structures]
            matrices[name]={"structure_ids":[x["id"] for x in structures],"initial_terminal_rows":[x["id"] for x in structures],"all_structure_labels":[f"{x['id']}:initial" for x in structures]+[f"{x['id']}:terminal" for x in structures],"initial_vs_terminal":[bool(fn(x["initial"],x["terminal"])) for x in structures],"all_pairwise":[[bool(fn(a,b)) for b in all_s] for a in all_s]}
    except Exception as exc: matrices={"status":"unavailable","error":f"{type(exc).__name__}: {exc}"}
    out={"status":"audited" if arms and all(x.get("status")=="audited" for x in arms.values()) else "pending","zero_pes":True,"arms":arms,"identity":{"tolerances":tol,"count":len(structures),"matrices":matrices},"scope":"read-only metric comparison; certified fresh landing identity is approximate and not energy-only"}
    outp=args.output or root/"metric-comparison-summary.json"; outp.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n"); print(outp)
if __name__=="__main__": main()
