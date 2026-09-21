"""Zero-PES frequency audit of completed atomic energy-boundary triggers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit_file(path):
    d=json.loads(Path(path).read_text()); rows=[]
    for run in d.get("runs",[]):
        initial=float(run["initial"]["energy"]); seed=run["seed"]
        for outer in run.get("outer",[]):
            if int(outer.get("index",-1))!=1: continue
            entry=outer.get("entering_atomic")
            previous=None if not entry else float(entry["energy"])
            for b in outer.get("atomic_boundaries",[]):
                e=float(b["true_energy"]); delta_center=e-previous if previous is not None else None
                row={"seed":seed,"outer_index":outer["index"],"gaussian_index":b["index"],"request":b.get("request"),
                     "true_energy":e,"center_energy":previous,"delta_from_gaussian_center":delta_center,
                     "trigger_local_1eV":bool(delta_center is not None and delta_center < -1.0),
                     "delta_from_outer_initial":e-initial,
                     "trigger_outer_initial_minus_0.1":bool(e < initial-0.1)}
                rows.append(row); previous=e
    return rows

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--output",type=Path); args=ap.parse_args(argv)
    root=args.root; rows=[]; sources=[]
    for p in sorted(root.glob("*/cell-energy-diagnosis.json")):
        sources.append({"path":str(p),"sha256":sha(p)}); rows.extend([{**x,"metric":p.parent.name} for x in audit_file(p)])
    n=len(rows); native=sum(x["trigger_local_1eV"] for x in rows); outer=sum(x["trigger_outer_initial_minus_0.1"] for x in rows)
    by_metric={}
    for metric in sorted({x["metric"] for x in rows}):
        sub=[x for x in rows if x["metric"]==metric]; nn=sum(x["trigger_local_1eV"] for x in sub); oo=sum(x["trigger_outer_initial_minus_0.1"] for x in sub)
        by_metric[metric]={"completed_boundaries":len(sub),"local_center_drop_count":nn,"local_center_drop_frequency":nn/len(sub) if sub else None,"outer_initial_drop_count":oo,"outer_initial_drop_frequency":oo/len(sub) if sub else None}
    out={"status":"audited" if sources else "pending","zero_pes":True,"sources":sources,"completed_boundary_count":n,
         "local_center_drop_threshold_eV":-1.0,"outer_initial_threshold_eV":-0.1,
         "local_center_drop_trigger_count":native,"local_center_drop_trigger_frequency":(native/n if n else None),
         "outer_initial_drop_trigger_count":outer,"outer_initial_drop_trigger_frequency":(outer/n if n else None),"by_metric":by_metric,
         "rows":rows,"scope":"Completed Gaussian true-energy boundaries only; trial evaluations and acceptance are not inferred.",
         "interpretation":"These bare-energy inequalities omit native masks and unresolved energy-scope differences; they are not a native-policy replay. Frequencies do not establish early stopping, saved cost, or performance."}
    outp=args.output or root/"energy-trigger-summary.json"; outp.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");
    detail="\n".join(f"- `{m}`: {v['local_center_drop_count']}/{v['completed_boundaries']} local, {v['outer_initial_drop_count']}/{v['completed_boundaries']} outer" for m,v in by_metric.items())
    md=outp.with_name("energy-trigger-summary.md"); md.write_text("# Completed atomic energy trigger audit\n\nZero-PES readout of completed Gaussian true-energy boundaries. The first boundary uses the entering-atomic energy as its center; later boundaries use the preceding completed true energy.\n\n- Completed boundaries: `%d`\n- Local center-drop diagnostic (`delta < -1.0 eV`): `%d/%d` (`%s`)\n- Outer-initial trigger (`true energy < initial - 0.1 eV`): `%d/%d` (`%s`)\n\nBy metric:\n%s\n\nThese are bare-energy inequalities over completed boundaries only, not a native-policy replay: native masks and energy-scope differences are not reproduced. They do not imply that a trial was accepted, that an inner stopping rule would save cost, or that either criterion improves the search.\n"%(n,native,n,("n/a" if n==0 else f"{native/n:.6f}"),outer,n,("n/a" if n==0 else f"{outer/n:.6f}"),detail))
    print(outp); print(md)
if __name__=="__main__": main()
