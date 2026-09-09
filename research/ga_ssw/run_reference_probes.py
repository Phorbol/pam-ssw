"""Regenerate bounded original-JAR behavior fixtures; no LASP/search calls.

The supplied archives must already be extracted under --reference-root.
The frozen water-coninfo.base is reused; its original generation was unseeded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'tests/fixtures/ga_ssw')
    args=parser.parse_args(); root=args.reference_root.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent/'java'
    frozen=Path(__file__).resolve().parents[2]/'tests/fixtures/ga_ssw/water-coninfo.base'
    if not (out/'water-coninfo.base').exists(): (out/'water-coninfo.base').write_bytes(frozen.read_bytes())
    jdk=root/'tools/jdk-17.0.2/bin'; sgn=root/'GA-SSW_program/sgn.jar'
    examples=root/'GA-SSW_examples_run/global_exploration'; nna=examples/'GA-SSW/input/nna1.jar'
    records=[]
    def write(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
    def run(cmd,timeout=30):
        subprocess.run([str(x) for x in cmd],check=True,timeout=timeout,capture_output=True,text=True)
    with tempfile.TemporaryDirectory(prefix='gassw-probes-') as tmp:
        build=Path(tmp); (build/'nna').mkdir(); (build/'sgn').mkdir()
        run([jdk/'javac','-cp',nna,'-d',build/'nna',source/'NnaProbe.java'])
        run([jdk/'javac','-cp',sgn,'-d',build/'sgn',*[source/x for x in ['CoreProbe.java','ProjectionProbe.java','VirtualReferenceProbe.java']]])
        def java(jar,kind,klass,*args):
            return [str(x) for x in [jdk/'java','-Xmx512m','-XX:ActiveProcessorCount=1','-cp',str(build/kind)+':'+str(jar),klass,*args]]
        cases=[('water','TYPE3-(H2O)15',0,2.0,2,3.0,1),('aloh','TYPE1-AlOH',1,1.5,1,3.0,3),('molecular_crystal','TYPE2-XXXII',1,2.5,1,6.0,1)]
        for case,template,periodic,ran,kind,grid,limit in cases:
            arc=examples/'input-templates'/template/'addition/add.arc'
            cmd=java(nna,'nna','NnaProbe',arc,periodic,ran,kind,grid,limit,out/(case+'.json'))
            if case=='water': cmd+=['references',str(out/'water-coninfo.base')]
            run(cmd)
            records.append(dict(case=case,command=cmd,source_sha256=sha(arc),jar_sha256=sha(nna)))
        write('provenance.json',records)
        al=json.loads((out/'aloh.json').read_text()); rows=[]
        for i,f in enumerate(al['frames']):
            rows.append(dict(id=i,energy=f['energy'],sims=[al['similarity_matrix'][i][j] for j in (0,2,4)]))
        for i in (0,4): rows.append(dict(rows[i],id=len(rows)))
        # Deliberately artificial contract probes: replacement and nontransitive tolerance.
        for offset,energy in [(.00075,rows[4]['energy']),(.0015,rows[2]['energy'])]:
            rows.append(dict(id=len(rows),energy=energy,sims=[s+offset for s in rows[0]['sims']]))
        (out/'core-input.tsv').write_text(''.join(' '.join(map(str,[r['energy'],*r['sims']]))+'\n' for r in rows))
        cmd=java(sgn,'sgn','CoreProbe',out/'core-input.tsv',out/'core-output.json',.001,.1);run(cmd)
        write('core-input.json',dict(rows=rows,tolerance=.001,window=.1,command=cmd,sgn_jar_sha256=sha(sgn),
            projection_source='AlOH actual-structure anchors 0,2,4, not virtual runtime references',
            artificial_contract_probe_ids=[8,9],contract_probe_note='Shifted projections with reassigned energies; not physical structures or scientific evidence'))
        water=json.loads((out/'water.json').read_text())
        (out/'water-projection.tsv').write_text(''.join(' '.join(map(str,[f['energy'],*s]))+'\n' for f,s in zip(water['frames'],water['projections'])))
        cmd=java(sgn,'sgn','ProjectionProbe',out/'water-projection.tsv',out/'water-classification.json',.0001);run(cmd)
        config=examples/'input-templates/TYPE3-(H2O)15/configure.non'
        write('water-classification-provenance.json',dict(command=cmd,sgn_jar_sha256=sha(sgn),reference_sha256=sha(out/'water-coninfo.base'),
              template=str(config),template_sha256=sha(config),tolerance=.0001,scope='Original NNA descriptors plus original SGN classifier, frozen virtual references; not a complete search trajectory'))
        cmd=java(nna,'nna','NnaProbe',water['source'],0,2.0,2,3.0,1,out/'unused.json','batch')
        start=time.monotonic()
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=3)
            status=dict(status='completed',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired as e:
            status=dict(status='timeout',stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout)
        write('batch-probe.json',dict(command=cmd,wall_seconds=time.monotonic()-start,timeout_seconds=3,jar_sha256=sha(nna),**status))
    print('Original-JAR fixtures written to',out)


if __name__=='__main__': main()
