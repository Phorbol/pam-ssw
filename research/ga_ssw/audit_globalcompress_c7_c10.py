"""Zero-PES raw output matrix for C=7..10 globalcompress branches."""
import json
from pathlib import Path
import numpy as np
from oracle_compress_mode import run

def main(out='research/ga_ssw/evidence/native-cluster-control-generator/compress-c7-c10-audit.json'):
    x=np.array([[.3,.8,1.4],[2.1,-.7,.5],[-1.2,1.7,.2],[3.2,.4,-.9],[.1,-2.,1.1],[-.8,.9,2.7]],float)
    axes={'IMAX1':[30.,0,0,0,1.,0,0,0,1.],'IMAX2':[1.,0,0,0,30.,0,0,0,1.],'IMAX3':[1.,0,0,0,1.,0,0,0,30.]}
    rows=[]
    for axis,cell in axes.items():
        for c,u in [(7,(7.1/11)),(8,(8.1/11)),(9,(9.1/11)),(10,.99)]:
            r=run(x,cell=cell,randoms=[u,.89],cache='paired',pair=(0,0)); rows.append({'axis':axis,'C_requested':c,'status':r['status'],'globals':r.get('globals'),'input':r.get('input'),'uncached_output':r.get('uncached_output'),'cached_output':r.get('output'),'max_cache_delta':None if r['status']!='ok' else float(np.max(np.abs(np.asarray(r['output'])-np.asarray(r['uncached_output']))))})
    result={'scope':'zero-PES isolated native compress_mode output matrix; C7-C10','source_addresses':{'C7_C8':'0x6e3d8d-0x6e3f41','C9':'0x6e3ca2-0x6e3d88','C10':'0x6e3fbc-0x6e3ffc'},'confirmed_algebra':{'C9':'base=COM-R, then branch-specific tested IMAX component can be overwritten by -2.0 marker; exact per-atom gate is retained in asm','C10':'output=COM-R+5*e_IMAX for observed mask-positive rows','C7_C8':'raw centered vector plus IMAX-dependent scalar overwrite and sign masks; conditional gate requires full branch-level symbolic reduction'},'rows':rows}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2)); print(json.dumps({'output':out,'rows':len(rows),'max_cache_delta':max(r['max_cache_delta'] for r in rows)}))
if __name__=='__main__':main()
