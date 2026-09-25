"""Matched direction-only ablation of the existing SSW2013 Eq1-2 option."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lj_pilot_base',HERE/'run_lj_pilot.py')
runner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runner
spec.loader.exec_module(runner)
base_settings=runner.make_settings

def paper_settings():
    config,rotation=base_settings()
    return replace(config,direction_sampling='paper'),rotation

runner.make_settings=paper_settings
runner.PLAN=HERE/'lj-paper-direction-plan.md'
if __name__=='__main__': runner.main()
