#!/usr/bin/env python3
"""Offline C60 recovered-rotation readout; no calculator calls."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import networkx as nx
from ase.io import read

EREF = -62215.39337032347
ETOL = 0.01
FMAX = 0.03
GRAPH = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-vacuum-geometry/research/ga_ssw/evidence/c60-native-periodic-readout-20260920/analyze_c60_random_development.py")

def graph_module():
    spec = importlib.util.spec_from_file_location("c60_graph_validator", GRAPH)
    if spec is None or spec.loader is None: raise ImportError(str(GRAPH))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def load(path):
    return json.loads(path.read_text())

def ledger(path, summary):
    if not path.exists(): return {"present": False}
    ids=[]; searches=failures=denials=rows=0; errors=[]
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try: x=json.loads(line)
            except Exception as exc: errors.append({"line":line_no,"error":repr(exc)}); continue
            rows += 1
            if not isinstance(x, dict): continue
            if x.get("kind") in ("search", "search_failure"):
                searches += 1; failures += x.get("kind") == "search_failure"
                if isinstance(x.get("request"), int): ids.append(x["request"])
            elif x.get("kind") == "search_denial": denials += 1
    return {"present":True,"rows":rows,"search_rows":searches,"search_failures":failures,
            "denials":denials,"ids_unique":len(ids)==len(set(ids)),
            "ids_contiguous":ids==list(range(1, searches+1)),
            "summary_request_match":summary.get("search_requests")==searches,
            "summary_denial_match":summary.get("denials")==denials,"errors":errors}

def atom_match(a, b):
    try:
        return (a.get("numbers") == b.get("numbers") and
                np.array_equal(np.asarray(a.get("positions")), np.asarray(b.get("positions"))) and
                np.array_equal(np.asarray(a.get("cell")), np.asarray(b.get("cell"))) and
                np.array_equal(np.asarray(a.get("pbc")), np.asarray(b.get("pbc"))))
    except Exception: return False

def minima_with_cost(result):
    minima=result.get("minima",[]); records=result.get("records",[])
    if not minima or not records or not isinstance(result.get("initial"),dict):
        raise ValueError("result lacks initial/minima/records")
    if minima[0].get("energy") != result["initial"].get("energy") or not atom_match(minima[0].get("atoms",{}), result["initial"].get("atoms",{})):
        raise ValueError("minimum zero is not result initial")
    costs={0:result["initial"].get("evaluation_requests")}; used={0}
    if not isinstance(costs[0],int): raise ValueError("initial request count missing")
    cumulative=costs[0]
    next_index=1
    for record in records:
        n=record.get("evaluation_requests")
        if not isinstance(n,int): raise ValueError("record request count missing")
        cumulative += n
        landing=record.get("landing")
        if (not isinstance(landing,dict) or not landing.get("converged") or
                not isinstance(landing.get("energy"),(int,float)) or not np.isfinite(landing["energy"])): continue
        if next_index >= len(minima): raise ValueError(f"too many converged landings at record {record.get('index')}")
        minimum=minima[next_index]
        if minimum.get("energy") != landing.get("energy") or not atom_match(minimum.get("atoms",{}),landing.get("atoms",{})):
            raise ValueError(f"ordered landing/minimum mismatch at record {record.get('index')}")
        costs[next_index]=cumulative; used.add(next_index); next_index += 1
    if next_index != len(minima): raise ValueError("unmapped minima remain")
    return costs

def fresh(summary, folder):
    if "fresh" in summary:
        rows=summary["fresh"]; out={x.get("label"):x for x in rows if isinstance(x,dict)}
        def norm(x): return {"status":x.get("status"),"energy_eV":x.get("energy_eV"),"fmax_eV_A":x.get("fmax_eV_A"),"energy_error_eV":x.get("energy_error_eV"),"numerical_qualified":x.get("numerical_qualified")}
    elif "fresh_checks" in summary:
        rows=summary["fresh_checks"]; valid=[x for x in rows if isinstance(x,dict)]
        out={"initial":next((x for x in valid if x.get("index")==0),None),"best":min(valid,key=lambda x:x["energy_eV"]) if valid else None}
        def norm(x): return None if x is None else {"status":"completed","energy_eV":x.get("energy_eV"),"fmax_eV_A":x.get("fmax_eV_per_A"),"energy_error_eV":x.get("energy_error_eV"),"numerical_qualified":x.get("qualified")}
    else: raise ValueError("summary lacks exact fresh schema")
    return {label:(None if out.get(label) is None else norm(out[label]))
            for label in ("initial","best")}

def arm(root, case, seed, new, graph, reference_graph):
    folder=root/(f"{case}-seed{seed}" if new else case)
    sp=folder/"summary.json"
    if not sp.exists(): return {"case":case,"seed":seed,"folder":str(folder),"missing":True}
    summary=load(sp); lp=folder/"requests.jsonl"; result_path=folder/"result.json"
    frows=fresh(summary,folder)
    fi,fb=frows["initial"],frows["best"]
    fdelta=(fb["energy_eV"]-fi["energy_eV"] if fi and fb and fi.get("energy_eV") is not None and fb.get("energy_eV") is not None else None)
    row={"case":case,"seed":seed,"folder":str(folder),"missing":False,
         "status":summary.get("status"),"search_requests":summary.get("search_requests"),
         "actual_calculate":summary.get("search_calculator_calls"),"fresh_requests":summary.get("fresh_requests"),
         "fresh_actual_calculate":summary.get("fresh_calculator_calls"),"boundary":summary.get("boundary"),
         "denials":summary.get("denials"),"ledger":ledger(lp,summary),"fresh":frows,"fresh_energy_delta_eV":fdelta}
    if not result_path.exists(): row["result"]={"present":False}; return row
    result=load(result_path); costs=minima_with_cost(result); rows=[]
    for i,m in enumerate(result["minima"]):
        z=m.get("atoms",{}).get("numbers",[]); pos=m.get("atoms",{}).get("positions",[])
        fmax=m.get("max_force"); qualified=bool(len(z)==60 and all(x==6 for x in z) and isinstance(m.get("energy"),(int,float)) and np.isfinite(m["energy"]) and m.get("converged") and
                                                  isinstance(fmax,(int,float)) and np.isfinite(fmax) and fmax<=FMAX)
        graphs={str(c):graph.graph_row(z,pos,c,reference_graph if c==1.8 else None) for c in (1.8,1.64,1.7)}
        rows.append({"index":i,"cost_request":costs[i],"energy_eV":m.get("energy"),"converged":m.get("converged"),
                     "fmax_eV_A":fmax,"numerical_qualified":qualified,"graph":graphs,
                     "energy_target":bool(qualified and m.get("energy")<=EREF+ETOL),
                     "graph_cage_candidate":bool(graphs["1.8"]["graph_cage_candidate"])})
    best=None; curve=[]
    for x in sorted(rows,key=lambda x:x["cost_request"]):
        if x["numerical_qualified"] and (best is None or x["energy_eV"] < best): best=x["energy_eV"]
        curve.append({"cost_request":x["cost_request"],"best_energy_eV":best})
    row["result"]={"present":True,"status":result.get("status"),"evaluation_requests":result.get("evaluation_requests"),
                    "minima":rows,"best_energy_curve":curve,"best_qualified_energy_eV":min((x["energy_eV"] for x in rows if x["numerical_qualified"]),default=None),
                    "best_qualified_cage":any(x["numerical_qualified"] and x["graph_cage_candidate"] for x in rows),
                    "any_target_cage":any(x["energy_target"] and x["graph_cage_candidate"] for x in rows)}
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument("--new",type=Path,required=True); p.add_argument("--old",type=Path,required=True); p.add_argument("--reference",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); graph=graph_module(); plan=load(a.new/"plan.json"); reference=read(a.reference); d=graph.np.linalg.norm(reference.positions[:,None]-reference.positions[None,:],axis=2); reference_graph=graph.nx.Graph(); reference_graph.add_nodes_from(range(60)); reference_graph.add_edges_from(zip(*graph.np.where(graph.np.triu((d<1.8)&(d>0),1)))); rows=[]
    for case,seed in plan["seeds"].items():
        rows.append({"case":case,"seed":seed,"new":arm(a.new,case,seed,True,graph,reference_graph),"old":arm(a.old,case,seed,False,graph,reference_graph)})
    a.output.write_text(json.dumps({"scope":"offline C60 validator readout; no PES and no automatic success claim","reference_energy_eV":EREF,"energy_tolerance_eV":ETOL,"fmax_eV_A":FMAX,"graph_cutoffs_A":[1.8,1.64,1.7],"reference_path":str(a.reference),"pairs":rows},indent=2)+"\n")
if __name__=="__main__": main()
