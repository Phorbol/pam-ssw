"""Replay frozen outputs from original ELF instructions, with host acos."""
import json
from pathlib import Path
import numpy as np
from pamssw.standalone.gaussian import adjust_native_weight


def test_frozen_original_instruction_outputs():
    path=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/native-weight-emulated/result.json'
    report=json.loads(path.read_text())
    assert report['total']==report['passed']==54
    for case in report['cases']:
        args=case['input']
        actual=adjust_native_weight(**args)
        for key in ('weight','energy','angle_degrees','force','updates'):
            np.testing.assert_allclose(getattr(actual,key),case['emulated'][key],rtol=0,atol=1e-10,
                                       err_msg=str({k:v for k,v in case.items() if k in ('label','natoms','trial')}))
