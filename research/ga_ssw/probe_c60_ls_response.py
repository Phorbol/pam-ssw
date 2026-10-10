"""Two finite LS loading endpoints on archived C60 #2; no search or tuning."""
import argparse
from pathlib import Path
import torch
from mace.calculators import MACECalculator
from probe_ls_response import run_case, ledger, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
    if ledger.sha256(model) != "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47":
        raise ValueError("model changed")
    calc = MACECalculator(model_paths=str(model), head="omol", device="cuda",
                          default_dtype="float64", enable_cueq=False, enable_oeq=False)
    source = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-2/final.extxyz"
    # a=1 is the published 3% initialization; a=16 is the frozen C4 probe,
    # not a C60 recommendation. An unresolved stationary point stops the run.
    result = run_case(source, args.out, calc, amplitudes=(1., 16.),
                      bond_tables=({(6, 6): 3.61}, {(6, 6): 1.64}),
                      request_cap=4000, wall_seconds=600)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
