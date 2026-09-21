"""Replay recorded native E/g to locate failure, with zero new PES requests."""
import json,struct
from pathlib import Path
import numpy as np
from research.ga_ssw.probe_native_lbfgs_vc import FlatOracle,load_elf,ELF,FLAG_ADDR

class Trace(FlatOracle):
    def hook(self,u,pc,size,data):
        if pc in (0x6eac23,0x6eac29,0x6eaf94):
            self.events.append(dict(pc=hex(pc),request=self.current['request'],
                dginit=struct.unpack('<d',u.mem_read(0x791b940,8))[0]))
        super().hook(u,pc,size,data)

if __name__=='__main__':
    p=Path('research/ga_ssw/evidence/fe7c3-frozen-native-lbfgs/diagnostic/ls_filter-seed101.json')
    d=json.loads(p.read_text());_,segments=load_elf(ELF)
    o=Trace(segments,np.array(d['q_start']),accepted_step_cap=300,joint_gtol=.001);o.events=[]
    for r in d['evaluations']:
        assert np.array_equal(o.vector(),r['q'])
        flag=o.advance(r['objective'],np.array(r['gradient']),r)
    points=[d['evaluations'][0]]+d['accepted']
    secants=[dict(request=b['request'],sy=float((np.array(b['q'])-a['q'])@(np.array(b['gradient'])-a['gradient']))) for a,b in zip(points,points[1:])]
    out=dict(IFLAG=flag,MCSRCH_INFO=o.integer(FLAG_ADDR),events=o.events,
             secants=secants,new_PES_requests=0)
    (p.parent/'native-failure-trace.json').write_text(json.dumps(out,indent=2)+'\n')
    print(dict(IFLAG=flag,MCSRCH_INFO=out['MCSRCH_INFO'],last_events=o.events[-4:],nonpositive_secants=sum(s['sy']<=0 for s in secants),last_secants=secants[-2:],new_PES_requests=0))
