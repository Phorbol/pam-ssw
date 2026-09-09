"""Finite-difference force sign/scale check at the supplied nonstationary water input."""
import argparse
import json
from pathlib import Path
import numpy as np
from validate_water_archive import Oracle,read_arc,displaced_arc


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();oracle=Oracle(args.reference_root,args.output)
    source=oracle.template/'addition/add.arc';lines,indices,symbols,xyz,stored=read_arc(source)
    e,f,_=oracle.evaluate(source.read_text(),'original');direction=f/np.linalg.norm(f);rows=[]
    for h in [.01,.005,.002,.001]:
        ep,_,_=oracle.evaluate(displaced_arc(lines,indices,xyz+h*direction),f'plus-{h}')
        em,_,_=oracle.evaluate(displaced_arc(lines,indices,xyz-h*direction),f'minus-{h}')
        derivative=(ep-em)/(2*h);expected=-float(np.linalg.norm(f))
        rows.append(dict(h=h,derivative=derivative,negative_force_projection=expected,relative_difference=abs(derivative-expected)/abs(expected)))
    repeat,fr,_=oracle.evaluate(source.read_text(),'repeat')
    result=dict(source=str(source),energy=e,max_force_component=float(np.abs(f).max()),directional_checks=rows,
                identical_repeat_energy_difference=repeat-e,identical_repeat_max_force_difference=float(np.abs(fr-f).max()),
                scope='Same original oracle, finite-difference numerical check on uploaded real input, not physical PES validation')
    (oracle.output/'derivative-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
