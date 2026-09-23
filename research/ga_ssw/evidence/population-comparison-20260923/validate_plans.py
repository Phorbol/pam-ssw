import json, tempfile
from pathlib import Path
from unittest.mock import patch
from ase.calculators.emt import EMT
from research.ga_ssw import run_population_comparison as runner
root=Path(__file__).resolve().parent
for name in ("c60-seed3-v2.json", "c60-seed17-v2.json", "cu13-seed3.json", "cu13-seed17.json"):
    plan=json.loads((root/name).read_text())
    options=runner.walker_options(plan)
    with tempfile.TemporaryDirectory() as folder, patch.object(runner,"build_backend",return_value=EMT()):
        prepared=runner._prepare(plan,Path(folder))
        assert prepared[1].requests==0
        runner.initial_quench_options(prepared[2])
    if "native_mc" in plan:
        assert options["mc"].energy_tol==0.1 and options["mc"].maxtrap==99999
    print(name,"PASS: config objects, raw descriptors, walker options; zero PES, model loading excluded")
