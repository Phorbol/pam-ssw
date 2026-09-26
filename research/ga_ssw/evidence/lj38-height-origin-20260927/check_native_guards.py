"""Static default provenance and conditional instruction checks; zero PES."""
import hashlib
import json
import struct
from pathlib import Path
from research.ga_ssw.probe_native_moveds_retry import run, ELF_DEFAULT, SHA256, ACCEPT, RETRY
from research.ga_ssw.probe_native_weight_emulated import load_elf

blob,segments=load_elf(ELF_DEFAULT)
assert hashlib.sha256(blob).hexdigest()==SHA256

def raw(address,n):
    for va,size,data in segments:
        if va<=address and address+n<=va+len(data):return data[address-va:address-va+n]
    raise ValueError(hex(address))

keys=[raw(0x4a4a538,16).decode().strip(),raw(0x4a4a54c,20).decode().strip()]
limits=[struct.unpack('<d',raw(a,8))[0] for a in (0x4a49950,0x4a49958)]
assert keys==['SSW.disp_perstep','SSW.bonddisp_perstep'] and limits==[2.,.25]
# Separate operands are deliberately different. The short-distance flag is stubbed false.
cases=[('actual_extreme_first',.39366449838960915,1.030004712874212*2.7,.8648961398089524*2.7,ACCEPT),
       ('relative_equality',.4,3.,2.25,ACCEPT),
       ('relative_reject',.4,3.,2.249,RETRY),
       ('max_atom_equality',2.,3.,3.,ACCEPT),
       ('max_atom_reject',2.001,3.,3.,RETRY)]
rows=[]
for name,maximum,before,after,expected in cases:
    row=run(segments,name,False,maximum,before,after,.6,1,
            disp_perstep=limits[0],bonddisp_perstep=limits[1])
    assert row['status']=='ok' and row['branch_target']==hex(expected),row
    rows.append(row)
out=Path(__file__).with_name('native-guard-instruction-check.json')
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(elf=ELF_DEFAULT,elf_sha256=SHA256,keys=keys,defaults=limits,
    scope='Conditional retry block only; present_tooshort flag is synthetic false, distances supplied, no full native move/PES',
    default_producer='get_real calls 0x68d2b7/0x68d32b, rcx rodata defaults copied at 0x60f390-0x60f3a5',rows=rows),indent=2)+'\n')
print('PASS: two literal defaults and five native conditional retry checks; zero PES')
