"""Trace only two already recorded failed kernels, reusing stored E/F (zero calls)."""
import json,struct
from pathlib import Path
import numpy as np
from .probe_native_lbfgs_emt import Oracle,ELF,load_elf
class FailureOracle(Oracle):
    def hook(self,u,pc,size,data):
        if pc in (0x6eac23,0x6eac29,0x6eaf94):
            self.events.append(dict(pc=hex(pc),request=self.current['request'],dginit=struct.unpack('<d',u.mem_read(0x791b940,8))[0],lp=self.integer(0x5521968)))
        super().hook(u,pc,size,data)

def main():
    out=Path('research/ga_ssw/evidence/cu13-failed-quench-native-lbfgs');summary=json.loads((out/'summary.json').read_text());_,segments=load_elf(ELF);rows=[]
    for r in summary['runs']:
        if r['status']!='native_failure':continue
        d=json.loads((out/f'{Path(r["source"]).stem}-step{r["step"]}.json').read_text());o=FailureOracle(segments,d['evaluations'][0]['positions']);o.events=[]
        for v in d['evaluations']:o.advance(v['energy'],np.array(v['forces']),v)
        points=[d['evaluations'][0]]+d['accepted'];ys=[]
        for a,b in zip(points,points[1:]):ys.append(float(np.sum((np.array(a['forces'])-np.array(b['forces']))*(np.array(b['positions'])-np.array(a['positions'])))))
        rows.append(dict(source=r['source'],step=r['step'],events=o.events,secant_curvatures=ys));del o
    (out/'failure-trace.json').write_text(json.dumps(dict(rows=rows,extra_ef_requests=0),indent=2));(out/'failure-trace-script.py').write_text(Path(__file__).read_text());print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
