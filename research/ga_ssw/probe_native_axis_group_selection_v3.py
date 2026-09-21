"""Differential checks of the recovered geometry; no native main or PES."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.native_local_group import select_native_local_group
from research.ga_ssw.probe_native_axis_group_selection_v2 import run, ENTRY, STOP, SHA


def main():
    rows = []
    cur = np.array([[0.,0.,0.],[1.,.2,.1],[1.8,0.,.4],[4.,1.,.3],
                    [6.,-1.,.7],[8.,.4,-.2],[10.,2.,1.],[13.,-1.,2.],[16.,1.,-1.]])
    ref = cur.copy()
    ref[:,0] += [5.,5.,5.,0.,-.4,1.2,-.7,.3,0.]
    masks = [np.ones((9,3), bool), np.array([[1,0,0],[0,1,1],[1,1,0],
             [1,0,1],[0,1,1],[1,1,1],[1,0,0],[0,0,1],[1,1,0]], bool)]
    inputs = []
    for mask in masks:
        for shift in ([0.,0.,0.], [100.,0.,0.], [-100.,4.,-3.]):
            for draw in (0., .5, np.nextafter(1.,0.)):
                inputs.append(('noncollinear', ref+shift, cur+shift, mask, draw))
    # Exact near cutoff (2), exact outer floor (3), and an empty group.
    for label, x in [('cutoffs', [[0.,0.,0.],[2.,0.,0.],[1.5,0.,0.],[3.,0.,0.]]),
                     ('empty', [[0.,0.,0.],[.5,0.,0.]])]:
        x = np.array(x)
        a = x + np.arange(len(x))[:,None] * [.1,.02,.03]
        for rotation in (np.eye(3), np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])):
            inputs.append((label, a@rotation, x@rotation, np.ones_like(x, bool), .5))
    for label, a, b, mask, draw in inputs:
        native = run(len(b), mask[:,0].astype(int), reference=a, current=b,
                     mask_rows=mask.astype(int), pair_init=(1,2), rng=draw)
        expected = select_native_local_group(a, Atoms('C'*len(b), positions=b),
                                            iter([draw]), mask)
        pair = tuple(i-1 if i else None for i in native['pair'])
        count = native['calls'].count('random')
        matched = (pair == expected.pair and np.array_equal(native['group'], expected.group_mask)
                   and count == expected.draw_count)
        rows.append(dict(label=label, reference=a.tolist(), current=b.tolist(), mask=mask.tolist(),
                         rng=draw, pair_init=[1,2], native_pair=pair, expected_pair=expected.pair,
                         native_group=native['group'], expected_group=expected.group_mask.tolist(),
                         native_draw_count=count, expected_draw_count=expected.draw_count,
                         matched=matched))
    report = dict(scope=__doc__, sha256=SHA, entry=hex(ENTRY), stop=hex(STOP),
                  case_count=len(rows), failure_count=sum(not r['matched'] for r in rows), cases=rows)
    Path('research/ga_ssw/evidence/native-axis-group-selection-v3-20260917.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('case_count','failure_count')}))
    assert not report['failure_count']


if __name__ == '__main__':
    main()
