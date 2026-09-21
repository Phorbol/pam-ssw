"""Run only the native unbiasedrot mode/curvature -> weight assignment slice.

The ELF's ``for_cpstr`` is executed as uploaded.  No LASP entry point,
allocator, calculator, protection path, or PES is entered.
"""
import argparse, hashlib, json, struct
from pathlib import Path
from research.ga_ssw.probe_native_weight_emulated import load_elf
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_R14, UC_X86_REG_R15, UC_X86_REG_RBP, UC_X86_REG_RSP, UC_X86_REG_RIP, UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_R8, UC_X86_REG_RBX

ELF_SHA256='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
ELF_DEFAULT='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
OBJ=0x720000010000; STACK=0x720000020000; STOP=0x720000030000
START=0x5c44b7; ASSIGNED=0x5c45fe; SKIP=0x5c4896

def run(segments, mode, curvature, old_weight, rotstep=0):
    uc=Uc(UC_ARCH_X86,UC_MODE_64)
    for va,ms,chunk in segments:
        lo=va&~4095; uc.mem_map(lo,((va+ms+4095)&~4095)-lo); uc.mem_write(va,chunk)
    for base in (OBJ,STACK,STOP): uc.mem_map(base,0x10000)
    # r14/r15 are the two aliases used by this routine; only referenced fields
    # are initialized.  r14+0x1b52 is the fixed-width mode buffer.
    uc.mem_write(OBJ+0x1b52, mode.encode('ascii').ljust(30,b' ')[:30])
    uc.mem_write(OBJ+0x18a8, struct.pack('<d', curvature))
    uc.mem_write(OBJ+0x18b8, struct.pack('<d', old_weight))
    uc.mem_write(OBJ+0x1e8, struct.pack('<Q', 0))
    control=0x53ed5c0; uc.mem_write(control+4, struct.pack('<I', rotstep))
    rbp=STACK+0x8000; uc.mem_write(rbp-0x68, struct.pack('<Q',control))
    # The surrounding kernel reloads r15 from [rbp-0x80] and r14 from
    # [rbp-0x90] before entering this mode/curvature block.
    uc.mem_write(rbp-0x80, struct.pack('<Q',OBJ)); uc.mem_write(rbp-0x90, struct.pack('<Q',OBJ))
    uc.mem_write(STACK+0x8000, struct.pack('<Q',STOP))
    uc.reg_write(UC_X86_REG_R14,OBJ); uc.reg_write(UC_X86_REG_R15,OBJ)
    uc.reg_write(UC_X86_REG_RBX,OBJ+0x1b52)
    uc.reg_write(UC_X86_REG_RBP,rbp); uc.reg_write(UC_X86_REG_RSP,STACK+0x9000)
    hit=[]; last=[]; calls=[]
    def hook(m,a,size,u):
        last.append(a)
        if a in (ASSIGNED,SKIP): hit.append(a); m.emu_stop()
        elif a == 0x49a8420: calls.append(a)
        elif not (0x400000 <= a < 0x8000000): raise RuntimeError(f'unexpected execution {a:#x}')
    uc.hook_add(UC_HOOK_CODE,hook)
    try: uc.emu_start(START,STOP,timeout=1_000_000,count=100_000)
    except Exception as exc:
        regs={n:hex(uc.reg_read(v)) for n,v in [('rdi',UC_X86_REG_RDI),('rsi',UC_X86_REG_RSI),('rdx',UC_X86_REG_RDX),('r8',UC_X86_REG_R8)]}
        raise RuntimeError(f'{exc}; last={list(map(hex,last[-12:]))}; regs={regs}') from exc
    rip=uc.reg_read(UC_X86_REG_RIP)
    weight=struct.unpack('<d',uc.mem_read(OBJ+0x18b8,8))[0]
    newmode=bytes(uc.mem_read(OBJ+0x1b52,30)).decode('ascii','replace')
    flag=struct.unpack('<I',uc.mem_read(control+4,4))[0]
    threshold=struct.unpack('<d', uc.mem_read(0x4a45e38, 8))[0]
    assigned=mode == 'CBD_PreRot' and curvature > threshold
    expected_mode='CBD_biasedRot' if assigned else mode
    passed=(bool(hit) and rip == (ASSIGNED if assigned else SKIP)
            and weight == (curvature if assigned else old_weight)
            and flag == (1 if assigned else rotstep)
            and newmode.rstrip() == expected_mode and len(calls) == 1)
    return dict(mode=mode,curvature=curvature,old_weight=old_weight,input_rotstep=rotstep,
                stop=hex(hit[-1] if hit else rip),weight=weight,new_mode=newmode,
                output_rotstep=flag,for_cpstr_calls=len(calls),passed=passed)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--elf',default=ELF_DEFAULT); ap.add_argument('--output',required=True); a=ap.parse_args()
    blob,segs=load_elf(a.elf); digest=hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256: raise ValueError('unsupported ELF digest')
    literal=None
    for va,ms,chunk in segs:
        offset=0x4a45e38-va
        if 0 <= offset <= len(chunk)-8: literal=chunk[offset:offset+8]
    if literal is None: raise ValueError('comparison literal not mapped')
    threshold=struct.unpack('<d',literal)[0]
    cases=[(m,c,3.25) for m in ('CBD_PreRot','OtherMode') for c in (2.5,0.,-5e-7,threshold,-1.000001e-6,-2.5)]
    results=[run(segs,m,c,w) for m,c,w in cases]
    report=dict(elf=a.elf,sha256=digest,entry=hex(START),assigned_stop=hex(ASSIGNED),skip_stop=hex(SKIP),
                comparison_literal_address=hex(0x4a45e38),comparison_literal=threshold,
                comparison_literal_bytes=literal.hex(),scope='instruction slice only; uploaded for_cpstr executed; no main/PES',cases=results,
                passed=all(x['passed'] for x in results))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)
if __name__=='__main__': main()
