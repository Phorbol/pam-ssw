"""Execute uploaded Java mutation bytecode, preserving its anomaly as evidence."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
from ase.cluster.icosahedron import Icosahedron
root=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909');java=root/'tools/jdk-17.0.2/bin';jar=root/'GA-SSW_program/sgn.jar';build=Path(tempfile.mkdtemp(prefix='pam-mutation-probe-'))
subprocess.run([java/'javac','-cp',jar,'-d',build,'research/ga_ssw/java/Type0MutationProbe.java'],check=True)
a=Icosahedron('Cu',2);a.numbers[::2]=47
inp=build/'input.txt';inp.write_text('\n'.join(' '.join(map(str,[z,*p])) for z,p in zip(a.numbers,a.positions)))
report=dict(jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(),numbers=a.numbers.tolist(),positions=a.positions.tolist(),cases=[])
for method in ['disturbance','exchange','interMu']:
 for seed in [17,71]:
  raw=subprocess.check_output([java/'java','--add-opens','java.base/java.lang=ALL-UNNAMED','-Xmx128m','-XX:ActiveProcessorCount=1','-cp',f'{build}:{jar}','Type0MutationProbe',inp,str(seed),method],text=True,timeout=15)
  row=dict(method=method,seed=seed,numbers=[],positions=[])
  for s in raw.splitlines():
   r=s.split()
   if r[0]=='ATOM':row['numbers'].append(int(r[1]));row['positions'].append(list(map(float,r[2:])))
   elif r[0]=='NO_CLOSE_PAIR':row['no_close_pair']=r[1]=='true'
  report['cases'].append(row)
Path('tests/standalone/fixtures/type0_mutation.json').write_text(json.dumps(report,indent=2)+'\n')
print([(r['method'],r['seed'],r['no_close_pair']) for r in report['cases']])
