"""Bounded LS scalar instruction probes, not full LASP/paper parity.

No external runtime call or PES is executed. Entry register states are supplied
explicitly to isolated arithmetic blocks; scheduler guards remain static evidence.
"""
import argparse
import hashlib
import json
import struct
from research.ga_ssw.probe_native_ls_initialization import Oracle, PARA, DATA, STACK
from research.ga_ssw.probe_native_weight_emulated import load_elf
from unicorn.x86_const import *


def xmm(o, register, value):
    o.uc.reg_write(register, int.from_bytes(struct.pack('<d', value), 'little'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--elf',required=True);args=p.parse_args()
    blob,segments=load_elf(args.elf)
    digest=hashlib.sha256(blob).hexdigest()
    if digest!='bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':
        raise ValueError('wrong ELF')
    o=Oracle(segments)
    defaults={name:o.double(PARA+offset) for name,offset in
        [('bond_ener_scale',0x148),('steplenselfadapt',0x158),
         ('steplenselfadaptmax',0x160),('softratio',0x170),('biasperatomaim',0x180)]}
    defaults.update({name:o.integer(PARA+offset) for name,offset in
        [('freqselfadapt',0x150),('npreselfadapt',0x154),('softmodecycle',0x168)]})
    constants={hex(a):o.double(a) for a in [0x5520450,0x4a49c60,0x4a498d0,
        0x4a4c170,0x4a4c178,0x4a4c198,0x4a45e00]}
    input_names={hex(a):bytes(o.uc.mem_read(a,n)).decode('ascii') for a,n in
        [(0x4a49cc0,6),(0x4a49cc8,15),(0x4a49cd8,17)]}
    rows=[]
    # Actual 0x6c6130..0x6c6146 amplitude multiplications. Dummy bond descriptor
    # pointer is read at 0x6c613a but its consumers lie beyond the stop address.
    for b,fi,fj in [(0.06666666269302368,1.,1.),(.1111111044883728,1.,0.),(.1385542756031437,.5,.7)]:
        o=Oracle(segments);u=o.uc;rbp=STACK+0x80000
        u.reg_write(UC_X86_REG_RBP,rbp);o.putq(rbp-0x98,DATA+0x100)
        u.reg_write(UC_X86_REG_R9,DATA);u.reg_write(UC_X86_REG_RDX,0)
        u.reg_write(UC_X86_REG_RAX,DATA+8);u.reg_write(UC_X86_REG_R10,DATA+16)
        o.putd(DATA,b);o.putd(DATA+8,fi);o.putd(DATA+16,fj)
        amp=o.double(0x5520450);xmm(o,UC_X86_REG_XMM0,amp)
        u.emu_start(0x6c6130,0x6c6146,count=30,timeout=1000000)
        actual=struct.unpack('<d',u.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little')[:8])[0]
        expected=amp*b*fi*fj
        assert actual==expected
        rows.append(dict(block='effective_amplitude',B=b,amp_c=amp,filter_i=fi,filter_j=fj,
                         actual=actual,expected=expected))
    # Actual scalar matrix update block, Q supplied after static max-step analysis.
    for response,nb_new in [(0.,90),(20.,90),(40.,90),(10000.,80)]:
        b=.06666666269302368;n=60;nb_old=90
        eta=defaults['steplenselfadapt'];target=defaults['biasperatomaim']
        cap=defaults['steplenselfadaptmax']
        q=max(1.,abs(b*eta*n/nb_new*(response-target))/cap)
        o=Oracle(segments);u=o.uc;u.reg_write(UC_X86_REG_RBX,DATA)
        u.reg_write(UC_X86_REG_R12,0);o.putd(DATA,b)
        for reg,value in [(UC_X86_REG_XMM5,float(nb_old)),(UC_X86_REG_XMM8,eta*n),
                          (UC_X86_REG_XMM3,float(nb_new)),(UC_X86_REG_XMM9,response-target),
                          (UC_X86_REG_XMM7,q)]:xmm(o,reg,value)
        u.emu_start(0x6c8571,0x6c85a5,count=40,timeout=1000000)
        actual=o.double(DATA);expected=b*nb_old/nb_new-(b*eta*n/nb_new)*(response-target)/q
        assert abs(actual-expected)<1e-15
        rows.append(dict(block='matrix_update',B=b,N=n,nb_old=nb_old,nb_new=nb_new,
                         response=response,target=target,step=eta,Q_supplied=q,
                         actual=actual,expected=expected))
    # Response instructions fall outside the init Oracle whitelist: execute a
    # fresh mapped machine without its function-entry hook, no calls in block.
    rbp=STACK+0x80000
    # Change hook allowance only for this read-only arithmetic range.
    # Oracle's callback uses overridable method at construction; use a small
    # subclass so no unexpected control flow is silently accepted.
    class ResponseOracle(Oracle):
        def hook(self,u,address,size,user):
            if 0x5bf6b4<=address<0x5bf6f8:return
            return super().hook(u,address,size,user)
    o=ResponseOracle(segments);u=o.uc
    u.reg_write(UC_X86_REG_RBP,rbp);u.reg_write(UC_X86_REG_R14,DATA)
    o.puti(DATA,10);o.putd(0x78eae98,-100.);o.putd(rbp-0x348,-99.8)
    u.emu_start(0x5bf6b4,0x5bf6f8,count=40,timeout=1000000)
    actual=o.double(0x1d103350);expected=(-99.8+100.)/10*constants['0x4a45e00']
    assert actual==expected
    rows.append(dict(block='saved_response',energy_before=-100.,energy_after=-99.8,
                     N=10,scale=constants['0x4a45e00'],actual=actual,expected=expected))
    constants['0x4a4c170'] = '-Infinity'  # JSON-valid rendering of max-reduction seed.
    print(json.dumps(dict(elf_sha256=digest,static_defaults=defaults,constants=constants,
        input_names=input_names,rows=rows,limit='isolated arithmetic only; Q is statically derived and supplied, no scheduler/input execution'),indent=2))

if __name__=='__main__':main()
