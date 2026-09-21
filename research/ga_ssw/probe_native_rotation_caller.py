"""Bounded post-cbd_rotation caller probe (no LASP main or PES).

This starts at the return address immediately after unbiasedrot's first
``cbd_rotation`` call.  It is intentionally a diagnostic slice: allocator
calls are equivalent memory hooks and the biasedrot callback is stopped before
entering the rotation body.
"""
import argparse, hashlib, json, struct
from pathlib import Path
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import (UC_X86_REG_R13, UC_X86_REG_R14,
    UC_X86_REG_R15, UC_X86_REG_RBP, UC_X86_REG_RSP, UC_X86_REG_RDI,
    UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RIP, UC_X86_REG_RAX)
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256

DATA=0x720000010000; STACK=0x720000200000; STOP=0x720000300000
OBJ=DATA+0x1000; DESC=DATA; START=0x5c40cf; BIASED=0x5c4887

def run(segments, mode, curvature, lconverge, *, n=2):
    if n != 2 or mode != 'CBD_PreRot':
        raise ValueError('probe supplies explicit N=2 PreRot buffers')
    uc=Uc(UC_ARCH_X86, UC_MODE_64)
    for va,size,chunk in segments:
        lo=va & ~4095; uc.mem_map(lo,((va+size+4095)&~4095)-lo); uc.mem_write(va,chunk)
    uc.mem_map(DATA,0x100000); uc.mem_map(STACK,0x10000); uc.mem_map(STOP,0x10000)
    # Object handle and fixed-width mode/status fields.
    uc.mem_write(DESC, struct.pack('<Q', OBJ))
    uc.mem_write(OBJ+0x1b52, mode.encode().ljust(30,b' ')[:30])
    uc.mem_write(OBJ+0x18a8, struct.pack('<d', curvature))
    uc.mem_write(OBJ+0x18b0, struct.pack('<d', curvature))
    uc.mem_write(OBJ+0x18b8, struct.pack('<d', 99.))
    uc.mem_write(0x53ed5c4, struct.pack('<i', 9))
    uc.mem_write(OBJ+0x1e8, struct.pack('<Q', 0))
    uc.mem_write(OBJ+0x9e0, bytes([1]))
    # Distinct force/coordinate arrays and direction/anchor arrays.  The
    # post-return branch copies work1(+0x9c8) into fa(+0x1d0) only on LCONVERGE.
    def arr(addr, value):
        uc.mem_write(addr, struct.pack('<'+'d'*(3*n), *value))
    arr(DATA+0x20000, [1.,2.,3.,4.,5.,6.]); arr(DATA+0x30000,[11.,12.,13.,14.,15.,16.])
    arr(DATA+0x40000,[21.,22.,23.,24.,25.,26.]); arr(DATA+0x50000,[31.,32.,33.,34.,35.,36.])
    arr(DATA+0x60000,[41.,42.,43.,44.,45.,46.]); arr(DATA+0x70000,[51.,52.,53.,54.,55.,56.])
    # Fortran descriptors used by the copied work arrays and n/anchor.
    def desc(addr, target):
        # Fortran descriptor layout used by the existing UpdateOracle.
        uc.mem_write(addr, struct.pack('<QQQQQ', target, 8, 0, 1, 2))
        uc.mem_write(addr+0x30, struct.pack('<QQQ', 3, 8, 1))
        uc.mem_write(addr+0x48, struct.pack('<QQQ', n, 24, 1))
    desc(OBJ+0x9c8, DATA+0x20000); desc(OBJ+0x1d0, DATA+0x30000)
    desc(OBJ+0x1788, DATA+0x40000); desc(OBJ+0x17e8, DATA+0x50000)
    desc(OBJ+0x1a60, DATA+0x60000); desc(OBJ+0x1a00, DATA+0x70000)
    # Local LCONVERGE byte and saved control pointer are consumed by the
    # caller.  The exact stack byte is found by the existing source slice.
    rbp=STACK+0x8000
    uc.mem_write(STACK+0x8000, struct.pack('<Q',STOP))
    uc.mem_write(rbp-0x68, struct.pack('<Q',0x53ed5c0))
    # LCONVERGE is the function-local logical stored in the ELF BSS literal
    # address tested at 0x5c420f, rather than the temporary stack byte.
    uc.mem_write(0x78eaea8, bytes([1 if lconverge else 0]))
    uc.mem_write(rbp-0x80, struct.pack('<Q',DESC)); uc.mem_write(rbp-0x90, struct.pack('<Q',OBJ))
    uc.reg_write(UC_X86_REG_R13,DESC); uc.reg_write(UC_X86_REG_R14,OBJ)
    uc.reg_write(UC_X86_REG_R15,OBJ); uc.reg_write(UC_X86_REG_RBP,rbp); uc.reg_write(UC_X86_REG_RSP,STACK+0x9000)
    hits=[]; calls=[]; last=[]
    def hook(m,a,size,u):
        last.append(a)
        if a in (BIASED,0x5c48b4): hits.append('biasedrot' if a==BIASED else 'return'); m.emu_stop()
        elif a==0x5c4896: hits.append('set_status'); m.emu_stop()
        elif a==0x4970360:
            lhs=m.reg_read(UC_X86_REG_RDI); rhs=m.reg_read(UC_X86_REG_RSI)
            for offset in (8, 0x20, 0x30, 0x48):
                assert m.mem_read(lhs+offset,8)==m.mem_read(rhs+offset,8)
            assert struct.unpack('<Q',m.mem_read(lhs,8))[0]
            calls.append('same_shape_realloc_noop')
            rsp=m.reg_read(UC_X86_REG_RSP)
            ret=struct.unpack('<Q',m.mem_read(rsp,8))[0]
            m.reg_write(UC_X86_REG_RSP,rsp+8); m.reg_write(UC_X86_REG_RIP,ret)
        elif a==0x4a101c0:
            size=m.reg_read(UC_X86_REG_RDX)
            left=bytes(m.mem_read(m.reg_read(UC_X86_REG_RDI),size))
            right=bytes(m.mem_read(m.reg_read(UC_X86_REG_RSI),size))
            value=next((x-y for x,y in zip(left,right) if x != y),0)
            m.reg_write(UC_X86_REG_RAX,value & 0xffffffff)
            calls.append('memcmp')
            rsp=m.reg_read(UC_X86_REG_RSP)
            ret=struct.unpack('<Q',m.mem_read(rsp,8))[0]
            m.reg_write(UC_X86_REG_RSP,rsp+8); m.reg_write(UC_X86_REG_RIP,ret)
        elif a==0x4a102b0:
            # Equivalent memcpy hook, preserving the native caller's actual
            # source/destination/length and returning through its stack.
            dst=m.reg_read(UC_X86_REG_RDI); src=m.reg_read(UC_X86_REG_RSI); size=m.reg_read(UC_X86_REG_RDX)
            m.mem_write(dst,m.mem_read(src,size)); calls.append(('memcpy',hex(src),hex(dst),size))
            rsp=m.reg_read(UC_X86_REG_RSP); ret=struct.unpack('<Q',m.mem_read(rsp,8))[0]
            m.reg_write(UC_X86_REG_RSP,rsp+8); m.reg_write(UC_X86_REG_RIP,ret)
        elif a==0x49a8420: calls.append('for_cpstr')
        elif not (START<=a<=0x5c48b4 or 0x49a8420<=a<0x49a85a0):
            raise RuntimeError(f'unexpected {a:#x}')
    uc.hook_add(UC_HOOK_CODE,hook)
    try: uc.emu_start(START,STOP,timeout=1_000_000,count=300_000)
    except Exception as exc: raise RuntimeError(f'{exc}; last={list(map(hex,last[-12:]))}') from exc
    mode_out=bytes(uc.mem_read(OBJ+0x1b52,30)).decode('ascii','replace')
    copied=struct.unpack('<'+'d'*(3*n), uc.mem_read(DATA+0x30000,8*3*n))
    work=struct.unpack('<'+'d'*(3*n), uc.mem_read(DATA+0x20000,8*3*n))
    threshold=-1e-6
    expected_action=('set_status' if lconverge and curvature <= threshold else
                     'biasedrot' if lconverge and curvature > threshold else 'return')
    expected_mode=('CBD_biasedRot' if expected_action == 'biasedrot' else
                   'CBD_UnbiasedRot' if curvature < threshold else 'CBD_PreRot')
    expected_fa=list(work) if lconverge else [11.,12.,13.,14.,15.,16.]
    copy_ok=all(abs(a-b)<1e-14 for a,b in zip(copied,expected_fa))
    mode_ok=mode_out.rstrip() == expected_mode
    action_ok=hits == [expected_action]
    anchor=list(struct.unpack('<6d', uc.mem_read(DATA+0x50000,48)))
    center=list(struct.unpack('<6d', uc.mem_read(DATA+0x60000,48)))
    direction=list(struct.unpack('<6d', uc.mem_read(DATA+0x40000,48)))
    weight=struct.unpack('<d',uc.mem_read(OBJ+0x18b8,8))[0]
    rotstep=struct.unpack('<i',uc.mem_read(0x53ed5c4,4))[0]
    state_ok=(anchor == (direction if expected_action=='biasedrot' else [31.,32.,33.,34.,35.,36.])
              and center == [41.,42.,43.,44.,45.,46.]
              and direction == [21.,22.,23.,24.,25.,26.]
              and weight == (curvature if expected_action=='biasedrot' else 99.)
              and rotstep == (1 if expected_action=='biasedrot' else 0 if lconverge else 9))
    return dict(mode=mode,curvature=curvature,lconverge=lconverge,hits=hits,
                expected_action=expected_action,mode_out=mode_out,expected_mode=expected_mode,
                fa=list(copied),work1=list(work),copy_ok=copy_ok,mode_ok=mode_ok,
                anchor=anchor,center_force=center,weight=weight,rotstep=rotstep,state_ok=state_ok,
                calls=calls,passed=bool(action_ok and copy_ok and mode_ok and state_ok))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--elf',default=ELF_DEFAULT); ap.add_argument('--output',required=True); a=ap.parse_args()
    blob,segs=load_elf(a.elf); digest=hashlib.sha256(blob).hexdigest()
    if digest!=ELF_SHA256: raise ValueError('unsupported ELF')
    # The literal is recorded by the established weight probe; use its exact value.
    threshold=-1e-6
    cases=[]
    for conv in (False,True):
        for curv in (-2e-6, threshold, -5e-7, 0.5):
            try: cases.append(run(segs,'CBD_PreRot',curv,conv))
            except Exception as exc: cases.append(dict(curvature=curv,lconverge=conv,error=str(exc),passed=False))
    report=dict(elf=a.elf,sha256=digest,entry=hex(START),cases=cases,
                threshold=threshold,scope='post-cbd_rotation caller slice; allocator and callback hooks; no rotation body, main or PES',
                passed=all(c['passed'] for c in cases))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)
if __name__=='__main__': main()
