"""Isolated Unicorn probe for LASP localatompair_mode (no PES)."""
import hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ENTRY, END, STOP = 0x6e4490, 0x6e4b00, 0x700000000000
STACK, DATA, HEAP = 0x710000000000, 0x720000000000, 0x730000000000
SPECIES, NEIGH, NEIGH_END, RANDOM = 0x58f680, 0x580c50, 0x580def, 0x580640
RET = 0x700000000100

def q(u, p): return struct.unpack('<Q', u.mem_read(p, 8))[0]
def ret(u):
    sp=u.reg_read(UC_X86_REG_RSP); u.reg_write(UC_X86_REG_RIP,q(u,sp)); u.reg_write(UC_X86_REG_RSP,sp+8)

def run(pos, pair=(0,1), marker=0, randoms=(), neighbors=(3,),
        neighbor_sequences=None, freedom_mask=None, native_neighbors=False,
        atomic_numbers=None, native_species=False):
    blob,segs=load_elf(ELF); u=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segs:
        lo=va&~4095; u.mem_map(lo,((va+ms+4095)&~4095)-lo)
        if chunk: u.mem_write(va,chunk)
    for p,s in ((STOP,0x2000),(STACK,0x200000),(DATA,0x200000),(HEAP,0x400000)):u.mem_map(p,s)
    pos=np.asarray(pos,dtype='<f8'); n=len(pos); count=DATA; coords=DATA+0x1000; species=DATA+0x2000; radii=DATA+0x2800; out=DATA+0x3000; pairp=DATA+0x4000; markerp=DATA+0x5000; logical=DATA+0x6000; neighbor_count=DATA+0x6800; neighbor_data=DATA+0x7000; mask=DATA+0x7800
    if freedom_mask is None: freedom_mask=np.ones(3*n,dtype='<i4')
    if neighbor_sequences is None:
        neighbor_sequences=(tuple(neighbors), tuple(neighbors))
    else:
        neighbor_sequences=tuple(tuple(s) for s in neighbor_sequences)
    initial_neighbors = tuple(neighbor_sequences[0])
    if atomic_numbers is None: atomic_numbers=np.ones(n,dtype='<i4')
    u.mem_write(count,struct.pack('<i',n));u.mem_write(coords,pos.tobytes());u.mem_write(species,np.asarray(atomic_numbers,dtype='<i4').tobytes());u.mem_write(out,bytes(8*3*n));u.mem_write(pairp,struct.pack('<ii',*pair));u.mem_write(markerp,struct.pack('<i',marker));u.mem_write(logical,struct.pack('<i',0));u.mem_write(neighbor_count,struct.pack('<i',len(initial_neighbors)));u.mem_write(neighbor_data,np.asarray(initial_neighbors,dtype='<i4').tobytes());u.mem_write(mask,np.asarray(freedom_mask,dtype='<i4').tobytes())
    # arg7 is the logical branch flag and arg8 is the marker, matching the
    # caller's push r14(marker), push r10(logical) order.
    sp=STACK+0x1ff00;u.mem_write(sp,struct.pack('<QQQ',STOP,logical,markerp));u.reg_write(UC_X86_REG_RSP,sp)
    # ABI: n descriptor, Cartesian coordinates, species indices, radius table,
    # mode output and pair/marker descriptor pointers.
    for reg,val in ((UC_X86_REG_RDI,count),(UC_X86_REG_RSI,coords),(UC_X86_REG_RDX,species),(UC_X86_REG_RCX,mask),(UC_X86_REG_R8,out),(UC_X86_REG_R9,pairp)):u.reg_write(reg,val)
    calls=[]; ri=iter(randoms); neighbor_call=0
    def hook(uc,address,size,user):
        if ENTRY <= address < END: return
        calls.append(address)
        if address==SPECIES:
            if native_species:
                return
            # species_radius returns through its second (output) argument.
            uc.mem_write(uc.reg_read(UC_X86_REG_RSI),struct.pack('<d',1.0));ret(uc);return
        if native_species and SPECIES < address < 0x58f800:
            return
        if address==RANDOM:
            x=float(next(ri,0.5));uc.mem_write(uc.reg_read(UC_X86_REG_RDI),struct.pack('<d',x));ret(uc);return
        if address==NEIGH:
            if native_neighbors:
                # Let the archived neighboringlist execute; only its nested
                # species-radius calls are emulated below.
                return
            # The helper visits pair[1] then pair[0].  Each endpoint receives
            # a fresh neighboring-list result.
            nonlocal neighbor_call
            seq=neighbor_sequences[min(neighbor_call, len(neighbor_sequences)-1)]
            neighbor_call += 1
            uc.mem_write(uc.reg_read(UC_X86_REG_RCX),struct.pack('<i',len(seq)))
            uc.mem_write(uc.reg_read(UC_X86_REG_R8),np.asarray(seq,dtype='<i4').tobytes());ret(uc);return
        if native_neighbors and NEIGH < address < NEIGH_END:
            return
        if address==RET: return
        raise RuntimeError(f'unexpected external {address:#x} at {uc.reg_read(UC_X86_REG_RIP):#x}')
    u.hook_add(UC_HOOK_CODE,hook)
    try:
        u.emu_start(ENTRY,STOP,count=1000000)
        got=np.frombuffer(u.mem_read(out,8*3*n),dtype='<f8').reshape(n,3).copy()
        return {'status':'ok','output':got.tolist(),'calls':[hex(x) for x in calls], 'elf_sha256':hashlib.sha256(blob).hexdigest(), 'randoms':list(randoms), 'neighbors':list(neighbors), 'neighbor_sequences':[list(s) for s in neighbor_sequences], 'random_calls':sum(x == RANDOM for x in calls)}
    except Exception as e:
        return {'status':'blocked','error':f'{e} rip={u.reg_read(UC_X86_REG_RIP):#x}','calls':[hex(x) for x in calls]}

def reference(pos,pair=(0,1),marker=0,neighbors=None,neighbor_sequences=None,
             randoms=(),freedom_mask=None):
    """Finite reference for the observed localatompair_mode branch."""
    x=np.asarray(pos,float); i,j=pair[0]-1,pair[1]-1
    d=x[i]-x[j]; dist=np.linalg.norm(d)
    if dist < max(.6*(1.0+1.0), .7):
        return np.zeros_like(x)
    v=d/dist
    out=np.zeros_like(x); out[i]=-v; out[j]=v
    if neighbor_sequences is None:
        if neighbors is None: neighbors=range(1,len(x)+1)
        neighbor_sequences=(tuple(neighbors),tuple(neighbors))
    rng=iter(randoms)
    # The assembly processes pair[1], then pair[0].  Selected slots are
    # cleared; attempts continue up to n_neighbor, with <=4 accepted.
    for endpoint, seq in ((j,neighbor_sequences[0]), (i,neighbor_sequences[1])):
        slots=[int(z) for z in seq]; accepted=0
        for _ in range(len(slots)):
            r=float(next(rng,.5))
            if not (0.0 <= r < 1.0): raise ValueError('random draw must satisfy 0<=u<1')
            idx=int(len(slots)*r)
            k=slots[idx]-1
            if k == -1: continue
            if k in (i,j): continue
            delta=x[k]-x[endpoint]; norm=np.linalg.norm(delta)
            if norm <= 3.0: continue
            slots[idx]=0
            out[k] += -.8*delta/norm
            accepted += 1
            if accepted == 4: break
    if marker == -1: out=-out
    if freedom_mask is not None: out *= np.asarray(freedom_mask,float).reshape(x.shape)
    return out

if __name__=='__main__':
    cases=[([[0,0,0],[4,0,0],[0,4,0]],(1,2),0),([[0,0,0],[0,4,0],[0,0,4]],(1,2),-1)]
    rows=[]
    for p,pair,m in cases:
        row=run(p,pair,m,randoms=[.25,.75]);row['reference']=reference(p,pair,m).tolist();rows.append(row)
    print(json.dumps({'entry':hex(ENTRY),'scope':'isolated helper; no PES','rows':rows},indent=2))
