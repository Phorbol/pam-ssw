"""Offline formula audit for the recovered compress_mode branches."""
import json
from pathlib import Path
import numpy as np
from oracle_compress_mode import run

def main(out='research/ga_ssw/evidence/native-cluster-control-generator/compress-pairtail-audit.json'):
    x=np.array([[.3,.8,1.4],[2.1,-.7,.5],[-1.2,1.7,.2],[3.2,.4,-.9],[.1,-2.,1.1],[-.8,.9,2.7]],float)
    cell=[30.,0.,0.,0.,25.,0.,0.,0.,20.]
    rows=[]
    for pair,randoms,label in [((1,2),(.5,.5),'valid_pair_C2_le5'),((1,2),(.5,.89),'valid_pair_C2_gt5'),((0,0),(.5,.5),'disabled_pair_C2_le5'),((1,6),(.5,.5),'nonaxis_pair_C2_le5')]:
        r=run(x,cell=cell,randoms=randoms,cache='paired',pair=pair)
        com=np.asarray(r['globals']['COM']); base=com-x
        expected=base
        if pair[0]>0 and pair[1]>0 and r['globals']['C2']<=5:
            v=x[pair[0]-1]-x[pair[1]-1]; expected=v*(base@v)[:,None]
        got=np.asarray(r['output'])
        rows.append({'label':label,'pair':pair,'randoms':randoms,'status':r['status'],'globals':r.get('globals'),'max_abs_error':float(np.max(np.abs(got-expected))) if r['status']=='ok' else None,'output_kind':'v*(v dot (COM-R))' if expected is not base else 'COM-R','oracle':r})
    result={'scope':'zero-PES isolated ELF formula audit','formula_C_le6':'output_i = COM - R_i','formula_pairtail_C2_le5':'v = R[p0]-R[p1]; output_i = v*(v dot (COM-R_i))','pair_index_contract':'positive one-based indices activate pairtail; nonpositive pair disables it in observed probe','rows':rows}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2));print(json.dumps({'output':out,'errors':[r['max_abs_error'] for r in rows]}))
if __name__=='__main__':main()
