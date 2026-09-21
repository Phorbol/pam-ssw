"""Read CSSW type-bound tables from the ELF; executes no native instructions."""
import argparse, hashlib, json, struct, subprocess
from pathlib import Path


def inspect(path, table_bases=(0x53cd440,0x53cacc0)):
    blob=Path(path).read_bytes()
    if blob[:6] != b'\x7fELF\x02\x01':
        raise ValueError('little-endian ELF64 required')
    phoff=struct.unpack_from('<Q',blob,32)[0]
    size,count=struct.unpack_from('<HH',blob,54)
    segments=[]
    for i in range(count):
        kind,flags,offset,va,pa,fs,ms,align=struct.unpack_from('<IIQQQQQQ',blob,phoff+i*size)
        if kind==1: segments.append((va,blob[offset:offset+fs]))
    names={int(p[0],16):p[2] for line in subprocess.check_output(['nm','-n',str(path)],text=True).splitlines()
           if len(p:=line.split())>=3}
    def pointer(address):
        for va,data in segments:
            if va<=address and address+8<=va+len(data):
                return struct.unpack_from('<Q',data,address-va)[0]
        raise ValueError(hex(address))
    tables=[]
    for base in table_bases:
        slots={hex(o):dict(target=hex(pointer(base+o)),symbol=names.get(pointer(base+o)))
               for o in range(0,0x268,8)}
        tables.append(dict(table=hex(base),slots=slots))
    result=dict(elf_sha256=hashlib.sha256(blob).hexdigest(),tables=tables)
    assert result['elf_sha256']=='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('elf');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(json.dumps(inspect(a.elf),indent=2)+'\n')
