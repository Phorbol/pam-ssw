"""Regenerate TYPE0 oracle fixtures by executing uploaded Java bytecode."""
import argparse,json,subprocess,hashlib,tempfile
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--reference-root', type=Path, required=True)
root=parser.parse_args().reference_root
work=Path(__file__).resolve().parents[2]
java=root/'tools/jdk-17.0.2/bin'
jar=root/'GA-SSW_program/sgn.jar'
build=Path(tempfile.mkdtemp(prefix='pam-type0-probe-'))
source=work/'research/ga_ssw/java/Type0CutProbe.java'
subprocess.run([java/'javac','-cp',jar,'-d',build,source],check=True)
numbers=[29,47,29,47,29,47]
positions=[[3.1,2.2,4.3],[1.2,1.5,2.6],[2.1,4.2,1.3],[4.4,3.2,2.8],[3.6,1.2,1.1],[1.3,2.1,4.2]]
infile=build/'input.txt';infile.write_text('\n'.join(' '.join(map(str,[z,*p])) for z,p in zip(numbers,positions)))
result={'source':'original uploaded sgn.jar Cut bytecode; Type0CutProbe.java','jar_sha256':hashlib.sha256(jar.read_bytes()).hexdigest(),'cases':[]}
for seed in [17,71,555]:
    raw=subprocess.check_output([java/'java','--add-opens','java.base/java.lang=ALL-UNNAMED','-Xmx128m','-XX:ActiveProcessorCount=1','-cp',f'{build}:{jar}','Type0CutProbe',infile,str(seed)],text=True,timeout=20)
    case={'seed':seed,'numbers':numbers,'positions':positions,'draws':[],'son':{'numbers':[],'positions':[]},'daughter':{'numbers':[],'positions':[]}}
    for line in raw.splitlines():
        r=line.split()
        if r[0]=='SLOPE':case['slope']=float(r[1])
        elif r[0]=='DRAW':case['draws'].append(float(r[1]))
        else:
            part=case[r[0].lower()];part['numbers'].append(int(r[1]));part['positions'].append(list(map(float,r[2:])))
    result['cases'].append(case)
(work/'tests/standalone/fixtures/type0_cut.json').write_text(json.dumps(result,indent=2)+'\n')
result={'source':'original uploaded sgn.jar Cross bytecode; Type0CutProbe.java cross','jar_sha256':hashlib.sha256(jar.read_bytes()).hexdigest(),'seed':71,'numbers':numbers,'positions':positions,'scales':[1.,1.02,.98],'energies':[0.,1.,2.],'children':[],'draws':[]}
raw=subprocess.check_output([java/'java','--add-opens','java.base/java.lang=ALL-UNNAMED','-Xmx128m','-XX:ActiveProcessorCount=1','-cp',f'{build}:{jar}','Type0CutProbe',infile,'71','cross'],text=True,timeout=30)
for line in raw.splitlines():
    r=line.split()
    if r[0]=='DRAW':result['draws'].append(float(r[1]));continue
    index=int(r[0][5:])
    if len(result['children']) <= index:result['children'].append({'numbers':[],'positions':[]})
    child=result['children'][index];child['numbers'].append(int(r[1]));child['positions'].append(list(map(float,r[2:])))
(work/'tests/standalone/fixtures/type0_cross.json').write_text(json.dumps(result,indent=2)+'\n')
