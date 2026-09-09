"""Compare Python reconstruction against the last actual SGN->NNA exchange."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.ga_ssw.behavior import cluster_descriptor, descriptor_similarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',required=True,type=Path);p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args();root=args.reference_root.resolve();run=args.run_dir.resolve()
    exchange=run/'output/nna_tem';jdk=root/'tools/jdk-17.0.2/bin';jar=exchange/'nna1.jar'
    config=dict(line.split('=',1) for line in (exchange/'c.p').read_text().splitlines() if '=' in line)
    if config['ifCys'].strip()!='false':raise ValueError('Only nonperiodic reconstruction is implemented')
    original=np.loadtxt(exchange/'allSim.txt',ndmin=2)
    with tempfile.TemporaryDirectory(prefix='gassw-runtime-projection-') as tmp:
        source=Path(__file__).resolve().parent/'java/NnaProbe.java'
        subprocess.run([str(jdk/'javac'),'-cp',str(jar),'-d',tmp,str(source)],check=True,timeout=30)
        cmd=[str(jdk/'java'),'-Xmx512m','-XX:ActiveProcessorCount=1','-cp',tmp+':'+str(jar),'NnaProbe',
             str(exchange/'all.arc'),'0',config['neiR'],config['type'],config['gridSize'],str(len(original)),
             str(run/'runtime-projection-fixture.json'),'references',str(exchange/'coninfo.base')]
        subprocess.run(cmd,check=True,timeout=30)
    data=json.loads((run/'runtime-projection-fixture.json').read_text())
    bonds={(int(a),int(b)):v for a,b,v in data['bond_lengths']};weights=list(map(float,config['wei'].split()))
    frames=[f for f in data['frames'] if f['label'].endswith(':original')]
    assert len(frames)==len(original)
    calculated=[]
    for frame in frames:
        desc=cluster_descriptor(frame['numbers'],frame['positions'],bonds,float(config['neiR']))
        calculated.append([descriptor_similarity(desc,ref,weights) for ref in data['reference_descriptors']])
    calculated=np.array(calculated)
    np.testing.assert_allclose(calculated,original[:,:3],rtol=0,atol=2e-12)
    np.testing.assert_allclose([f['energy'] for f in frames],original[:,-1],rtol=0,atol=1e-10)
    result=dict(rows=len(frames),max_projection_error=float(np.max(np.abs(calculated-original[:,:3]))),
                matching_energy_order=True,scope='Last actual SGN/NNA exchange only; no scientific performance claim')
    (run/'runtime-projection-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
