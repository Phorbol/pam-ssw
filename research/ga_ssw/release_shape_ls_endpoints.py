"""Release the eight saved shape-bias downhill endpoints on physical V."""
import argparse
from pathlib import Path
from release_ls_channel_endpoints import main_run

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("finite", "torsion", "ring", "out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    endpoints=tuple((amplitude,branch,sign) for amplitude in (.125,1.)
                    for branch in ("easy_ts","ring_ts") for sign in ("minus","plus"))
    result=main_run(a.finite,a.torsion,a.ring,a.out,endpoints=endpoints)
    if result["status"]!="completed_all_endpoint_diagnostics":
        raise SystemExit(1)
