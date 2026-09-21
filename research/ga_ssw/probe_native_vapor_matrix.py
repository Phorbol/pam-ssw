"""No-PES input matrix for check_vapor_new_ plus strict mode-0 reference."""
import argparse,json
from pathlib import Path
import numpy as np
from research.ga_ssw.probe_native_vapor_oracle import load_elf,run,ELF_DEFAULT

def ref0(x,cri):
 x=np.asarray(x,float); d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2); n=len(x); p=list(range(n))
 def f(i):
  while p[i]!=i:p[i]=p[p[i]];i=p[i]
  return i
 for i in range(n):
  for j in range(i):
   if d[i,j]<cri:
    a,b=f(i),f(j)
    if a!=b:p[a]=b
 g={}
 for i in range(n):g.setdefault(f(i),[]).append(i)
 gs=list(g.values()); cross=[d[i,j] for a in range(len(gs)) for b in range(a) for i in gs[a] for j in gs[b]]
 return {'components':sorted(map(len,gs),reverse=True),'return':float(min(cross) if cross else 0.)}

def rot(x):
 a=.37; c,s=np.cos(a),np.sin(a);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 return np.asarray(x)@R.T+np.array([2.3,-1.1,.7])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--elf',default=ELF_DEFAULT);a=ap.parse_args();_,segs=load_elf(a.elf)
 base=np.array([[0.,0,0],[.8,0,0],[.8,.8,0],[8.,0,0]])
 cases=[]
 for name,x,cri in [('three_component',[[0,0,0],[.8,0,0],[4,0,0],[8,0,0]],1.7),('three_component_noncollinear',[[0,0,0],[.8,.2,0],[3.5,2,0],[8,-1,1]],1.7),('anchor_vs_global_min',[[0,0,0],[10,0,0],[11,0,0],[13,0,0]],1.7),('equal',[[0,0,0],[1.7,0,0]],1.7),('below',[[0,0,0],[1.7-1e-8,0,0]],1.7),('above',[[0,0,0],[1.7+1e-8,0,0]],1.7),('reordered',base[[2,0,3,1]],1.7),('rotated_translated',rot(base),1.7)]:cases.append((name,np.asarray(x,float),cri))
 rows=[]
 for name,x,cri in cases:
  for mode in (0,1):
   r=run(segs,x.tolist(),mode,cri);r.update({'name':name,'reference_mode0':ref0(x,cri) if mode==0 else None});rows.append(r)
 out={'elf_sha256':'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704','scope':'oracle matrix; no PES','mode0_reference':'strict distance graph and minimum cross-component distance','rows':rows}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'ok':sum(r['status']=='ok' for r in rows)}))
if __name__=='__main__':main()
