"""Execute only the native VC stop-mask tail, no PES or LASP main."""
import hashlib
import itertools
import json
from pathlib import Path
import struct
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import (UC_X86_REG_RBX, UC_X86_REG_RBP,
    UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
    UC_X86_REG_RIP)
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF='/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
blob,segments=load_elf(ELF)
assert hashlib.sha256(blob).hexdigest()=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
# File-backed initialized int32 fields, before any synthetic writes. This
# establishes compiled values only, not absence of later runtime overrides.
budget_address=0x53ed7a0+0x2dd20
budget_segment=next((va,chunk) for va,_,chunk in segments
                    if va<=budget_address and budget_address+12<=va+len(chunk))
compiled_budgets=dict(zip(('ngaus_relax','ngaus_relax_ini','ngaus_relax_half'),
    struct.unpack_from('<iii',budget_segment[1],budget_address-budget_segment[0])))
u=Uc(UC_ARCH_X86,UC_MODE_64)
for va,memsz,chunk in segments:
    start=va & ~0xfff
    u.mem_map(start,((va+memsz+0xfff)&~0xfff)-start)
    if chunk:u.mem_write(va,chunk)
base=0x710000000000
u.mem_map(base,0x10000)
obj,para,control,bp=base,base+0x4000,base+0x6000,base+0x9000
u.mem_write(para+0xf4,struct.pack('<ii',4,5))
u.mem_write(bp-0x30,struct.pack('<Q',control))
rows=[]
for cell,short,allstop,ng in itertools.product((0,1),(0,1),(0,1),(2,4,5)):
    u.mem_write(obj+0x2260,bytes([cell]))
    u.mem_write(bp-0x110,struct.pack('<q',ng))
    u.mem_write(control+0x78,b'\x00'*8)
    for reg,value in ((UC_X86_REG_RBX,obj),(UC_X86_REG_R14,para),
        (UC_X86_REG_RBP,bp),(UC_X86_REG_R12,0xffffffff if short else 0),
        (UC_X86_REG_R13,0xffffffff if allstop else 0),(UC_X86_REG_R15,0)):
        u.reg_write(reg,value)
    u.emu_start(0x5f4a6b,0x5f4b6a,count=100)
    assert u.reg_read(UC_X86_REG_RIP)==0x5f4b6a
    actual_all,actual_short=struct.unpack('<II',u.mem_read(control+0x78,8))
    expected_all=bool(allstop or (short and ng==(5 if cell else 4)))
    expected_short=bool(short or expected_all)
    assert bool(actual_all&1)==expected_all
    assert bool(actual_short&1)==expected_short
    rows.append(dict(cell=cell,in_stage_stop=short,in_all_stop=allstop,index=ng,
        out_stage_stop=bool(actual_short&1),out_all_stop=bool(actual_all&1)))
# Independently execute the lower-energy all-stop predicate, including masks.
native_para=0x53ed7a0
pressure_cut=struct.unpack('<d',u.mem_read(0x4a46d80,8))[0]
margin=struct.unpack('<d',u.mem_read(0x4a46d48,8))[0]
lower_rows=[]
for delta,pressure,slab,fixed in itertools.product((-.2,-.1,0.),
        (pressure_cut-1.,pressure_cut+1.),(0,1),(0,1)):
    reference=-10.; candidate=reference+delta
    u.mem_write(bp-0x108,struct.pack('<d',reference))
    u.mem_write(bp-0x38,struct.pack('<d',candidate))
    u.mem_write(native_para+0x2db38,struct.pack('<d',pressure))
    u.mem_write(native_para+0x2dbe4,struct.pack('<i',slab))
    u.mem_write(native_para+0x2dde0,struct.pack('<iii',fixed,fixed,fixed))
    u.reg_write(UC_X86_REG_R14,native_para)
    u.reg_write(UC_X86_REG_RBP,bp)
    u.emu_start(0x5f462f,0x5f46cf,count=100)
    assert u.reg_read(UC_X86_REG_RIP)==0x5f46cf
    got=bool(u.reg_read(UC_X86_REG_R13)&1)
    expected=(candidate<reference-margin and pressure<pressure_cut
              and not (fixed==0 and slab))
    assert got==expected
    lower_rows.append(dict(candidate=candidate,reference=reference,
        external_pressure=pressure,slab=slab,min_fixlat=fixed,all_stop=got))
# Execute the budget/force merge through both N branches. Deliberately
# unequal synthetic budgets discriminate stage index from optimizer counter;
# these values are fixtures, not recovered native defaults.
stage_rows=[]
for ng,step,force,allstop in itertools.product((1,2),(1,2,3,4,5),(.25,.5,.75),(0,1)):
    ordinary_budget,initial_budget=2,4
    u.mem_write(native_para+0x2dd20,struct.pack('<ii',ordinary_budget,initial_budget))
    u.mem_write(bp-0x110,struct.pack('<q',ng))
    u.mem_write(bp-0x130,struct.pack('<d',force))
    u.mem_write(bp-0x180,struct.pack('<d',.5))
    u.mem_write(control+8,struct.pack('<i',step))
    u.mem_write(bp-0x30,struct.pack('<Q',control))
    for reg,value in ((UC_X86_REG_R14,native_para),(UC_X86_REG_RBP,bp),
                      (UC_X86_REG_R13,0xffffffff if allstop else 0),
                      (UC_X86_REG_R15,0)):
        u.reg_write(reg,value)
    u.emu_start(0x5f48b6,0x5f4970,count=100)
    assert u.reg_read(UC_X86_REG_RIP)==0x5f4970
    got=bool(u.reg_read(UC_X86_REG_R12)&1)
    budget=initial_budget if ng==1 else ordinary_budget
    expected=bool(step>budget or force<.5 or allstop)
    assert got==expected,(ng,step,force,allstop,got)
    stage_rows.append(dict(stage_index=ng,climbstep=step,force_measure=force,
        force_threshold=.5,initial_budget=initial_budget,
        ordinary_budget=ordinary_budget,in_all_stop=bool(allstop),
        out_stage_stop_before_energy=got))
out=Path('research/ga_ssw/evidence/native-vc-stop-levels.json')
out.write_text(json.dumps(dict(status='passed',cases=rows,lower_energy_cases=lower_rows,
    compiled_stage_budgets=compiled_budgets,compiled_stage_budget_address=hex(budget_address),stage_budget_cases=stage_rows,native_margin=margin,native_external_pressure_cut=pressure_cut,pes_requests=0,
    scope='Native tail, lower-energy predicate and budget/force merge only; force producer and trajectory energies not emulated'),indent=2)+'\n')
print(f'{len(rows)+len(lower_rows)+len(stage_rows)} isolated native cases passed; {out}')
