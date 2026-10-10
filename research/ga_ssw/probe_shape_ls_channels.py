"""Bounded shape-normalized LS channel experiment, not a public SSW variant."""
import argparse
from pathlib import Path
from probe_ls_channel_continuation import main_run

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("torsion",type=Path)
    p.add_argument("ring",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    result=main_run(a.torsion,a.ring,a.out,amplitudes=(.125,1.),
                    bias_geometry="shape",request_cap=3000,wall_seconds=600)
    if result["status"]!="completed_stationary_branch_diagnostics":
        raise SystemExit(1)
