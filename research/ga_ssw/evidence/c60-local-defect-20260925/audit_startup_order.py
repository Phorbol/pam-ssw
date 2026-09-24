"""Zero-PES check of exact startup displacement ties and physical axis identity."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from ase.io import read
    from pamssw.standalone.native_local_group import select_native_local_group
    atoms = read(HERE / 'qualification/isomer-2/final.extxyz')
    initialization = []
    for name in ('ssw_without_ls-1101','ssw_without_ls-1102','native_ls-1101','native_ls-1102'):
        result = json.loads((HERE / 'direction-probe/runs' / name / 'result.json').read_text())
        delta = np.asarray(result['initial']['atoms']['positions'])-atoms.positions
        initialization.append(dict(arm=name, max_displacement_A=float(np.linalg.norm(delta,axis=1).max()),
            exactly_unchanged=bool(np.array_equal(delta,np.zeros_like(delta)))))
    rows=[]
    for shift in (0,1,7,19,31):
        order=np.roll(np.arange(60),shift)
        permuted=atoms[order]
        # The selector's first axis is deterministic; second axis is random.
        selected=select_native_local_group(permuted.positions.copy(),permuted,np.random.default_rng(1101))
        rows.append(dict(cyclic_shift=shift,new_to_source_index=order.tolist(),
            selected_pair_new=list(selected.pair),
            selected_pair_source=[int(order[i]) if i is not None else None for i in selected.pair],
            selected_group_source=sorted(int(order[i]) for i in np.flatnonzero(selected.group_mask)),
            draw_count=selected.draw_count))
    out=HERE/'direction-probe/startup-order.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(scope='Geometric selector startup only, no PES or search rerun; does not show that label dependence caused observed Ih hits',actual_initialization=initialization,rows=rows),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
