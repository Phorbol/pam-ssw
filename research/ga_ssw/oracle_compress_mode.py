"""Zero-PES Unicorn probe for native compress_mode.

This is an instruction probe, not a production implementation.  The caller
contract is the one recovered at gen_randommode 0x5d6b94: N pointer, 3N
Cartesian records, 9-double cell record, 3N int mask, 3N output records,
pair-index record, and stack arg7 cache flag.
"""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ENTRY,FUNC_END,STOP=0x6e36f0,0x6e4490,0x700000000000
RANDOM=0x580640; MEMSET=0x4a10430
STACK,DATA=0x710000000000,0x720000000000
IMAX,C,C2,COM=0x791b668,0x791b66c,0x791b670,0x791b630

def _q(u,p): return struct.unpack('<Q',u.mem_read(p,8))[0]
def _ret(u):
    sp=u.reg_read(UC_X86_REG_RSP); u.reg_write(UC_X86_REG_RIP,_q(u,sp)); u.reg_write(UC_X86_REG_RSP,sp+8)

def run(pos, *, cell=None, mask=None, randoms=(.5,.5), cache=False, pair=(1,2)):
    x=np.asarray(pos,dtype='<f8'); n=len(x)
    blob,segs=load_elf(ELF); u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segs:
        lo=va&~4095; u.mem_map(lo,((va+ms+4095)&~4095)-lo)
        if chunk: u.mem_write(va,chunk)
    for p,s in ((STOP,0x2000),(STACK,0x200000),(DATA,0x300000)):u.mem_map(p,s)
    nptr=DATA; coords=DATA+0x1000; cellp=DATA+0x3000; maskp=DATA+0x3100; out=DATA+0x5000; pairp=DATA+0x7000; cachep=DATA+0x7100
    if cell is None: cell=np.zeros(9,dtype='<f8')
    if mask is None: mask=np.ones(3*n,dtype='<i4')
    cell=np.asarray(cell,dtype='<f8'); mask=np.asarray(mask,dtype='<i4')
    if cell.shape!=(9,) or mask.shape!=(3*n,): raise ValueError('cell must be 9 doubles and mask 3N int32')
    u.mem_write(nptr,struct.pack('<i',n)); u.mem_write(coords,x.tobytes()); u.mem_write(cellp,cell.tobytes()); u.mem_write(maskp,mask.tobytes())
    paired=(cache=='paired'); cache_flag=bool(cache) and not paired
    u.mem_write(out,bytes(24*n)); u.mem_write(pairp,struct.pack('<ii',*pair)); u.mem_write(cachep,struct.pack('<i',int(cache_flag)))
    sp=STACK+0x1ff00; u.mem_write(sp,struct.pack('<QQ',STOP,cachep)); u.reg_write(UC_X86_REG_RSP,sp)
    draws=iter(randoms); calls=[]
    def hook(uc,address,size,user):
        if ENTRY <= address < FUNC_END: return
        calls.append(address)
        if address==RANDOM:
            try: uval=float(next(draws))
            except StopIteration: raise RuntimeError('random draw sequence exhausted')
            uval=max(0.0,min(np.nextafter(1.0,0.0),uval)); uc.mem_write(uc.reg_read(UC_X86_REG_RDI),struct.pack('<d',uval)); _ret(uc); return
        if address==MEMSET:
            dst=uc.reg_read(UC_X86_REG_RDI); val=uc.reg_read(UC_X86_REG_RSI); size=uc.reg_read(UC_X86_REG_RDX); uc.mem_write(dst,bytes([val&255])*size); _ret(uc); return
        raise RuntimeError(f'unexpected external {address:#x} at {uc.reg_read(UC_X86_REG_RIP):#x}')
    u.hook_add(UC_HOOK_CODE,hook)
    def invoke(flag):
        u.mem_write(cachep,struct.pack('<i',int(flag)))
        u.mem_write(sp,struct.pack('<QQ',STOP,cachep)); u.reg_write(UC_X86_REG_RSP,sp)
        for reg,val in ((UC_X86_REG_RDI,nptr),(UC_X86_REG_RSI,coords),(UC_X86_REG_RDX,cellp),(UC_X86_REG_RCX,maskp),(UC_X86_REG_R8,out),(UC_X86_REG_R9,pairp)):u.reg_write(reg,val)
        u.emu_start(ENTRY,STOP,count=2000000)
        if u.reg_read(UC_X86_REG_RIP) != STOP: raise RuntimeError(f'entry returned at {u.reg_read(UC_X86_REG_RIP):#x}, expected STOP')
        return np.frombuffer(u.mem_read(out,24*n),dtype='<f8').reshape(n,3).copy()
    try:
        uncached=invoke(False) if paired else None
        output=invoke(True) if paired else invoke(cache_flag)
        def rd(addr,count,fmt): return list(struct.unpack('<'+fmt,u.mem_read(addr,count)))
        return {'status':'ok','output':output.tolist(), 'uncached_output':None if uncached is None else uncached.tolist(), 'calls':[hex(a) for a in calls], 'random_calls':sum(a==RANDOM for a in calls), 'cache':bool(cache), 'paired_cache':paired, 'n':n, 'globals':{'IMAX':rd(IMAX,4,'i')[0],'C':rd(C,4,'i')[0],'C2':rd(C2,4,'i')[0],'COM':rd(COM,24,'3d')}, 'input':{'positions':x.tolist(),'cell':cell.tolist(),'mask':mask.tolist()}, 'elf_sha256':hashlib.sha256(blob).hexdigest()}
    except Exception as e:
        return {'status':'blocked','error':f'{e} rip={u.reg_read(UC_X86_REG_RIP):#x}','calls':[hex(a) for a in calls], 'n':n, 'elf_sha256':hashlib.sha256(blob).hexdigest()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--c60-json'); ap.add_argument('--output',default='research/ga_ssw/evidence/native-cluster-control-generator/compress-oracle.json'); a=ap.parse_args()
    base=np.array([[0,0,0],[1,2,0],[2,0,1],[4,1,0],[0,3,2],[2,2,4.]])
    cases=[{'name':'n4_nonzero_box_mask','pos':np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3.]]),'cell':np.array([30.,0.,0.,0.,25.,0.,0.,0.,20.]),'mask':np.array([1,1,1,1,0,1,1,1,1,1,1,1],dtype='<i4'),'pair':(0,0)},
           {'name':'n6_largebox_C0','pos':base,'cell':np.array([30.,0.,0.,0.,25.,0.,0.,0.,20.]),'mask':np.ones(18,dtype='<i4'),'randoms':(.01,.01),'pair':(1,2)},
           {'name':'n6_largebox_C10_C2high','pos':base,'cell':np.array([30.,0.,0.,0.,25.,0.,0.,0.,20.]),'mask':np.ones(18,dtype='<i4'),'randoms':(.99,.89),'pair':(1,2)},
           {'name':'n6_axis_x','pos':base,'cell':np.array([30.,0.,0.,0.,1.,0.,0.,0.,1.]),'mask':np.ones(18,dtype='<i4'),'randoms':(.5,.55),'pair':(1,2)},
           {'name':'n6_axis_y','pos':base,'cell':np.array([1.,0.,0.,0.,30.,0.,0.,0.,1.]),'mask':np.ones(18,dtype='<i4'),'randoms':(.5,.55),'pair':(1,2)},
           {'name':'n6_axis_z','pos':base,'cell':np.array([1.,0.,0.,0.,1.,0.,0.,0.,30.]),'mask':np.ones(18,dtype='<i4'),'randoms':(.5,.55),'pair':(1,2)}]
    if a.c60_json:
        d=json.load(open(a.c60_json)); z=d.get('initial',d); z=z.get('atoms',z); p=z.get('positions') if isinstance(z,dict) else None
        if p is None: raise ValueError('C60 JSON has no initial.atoms.positions')
        cases.append({'name':'c60_saved_initial','pos':np.asarray(p,float),'cell':np.zeros(9),'mask':np.ones(3*len(p),dtype='<i4')})
    rows=[]
    for c in cases:
        # One VM per case: cache=True must reuse the globals produced by the
        # immediately preceding uncached call.  A fresh VM would be invalid.
        paired=run(c['pos'],cell=c.get('cell'),mask=c.get('mask'),randoms=c.get('randoms',[.5,.5]),cache='paired',pair=c.get('pair',(1,2))); paired['name']=c['name']; rows.append(paired)
    out={'scope':'static ELF oracle; no PES','entry':hex(ENTRY),'source_elf':ELF,'rows':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':a.output,'rows':[(r['name'],r['cache'],r['status']) for r in rows]}))
if __name__=='__main__': main()
