"""Execute only native LS stop predicate; read parser default constants.

No main, input parser, optimizer, PES or protection code is executed.
"""
import argparse
import hashlib
import json
from research.ga_ssw.probe_native_ls_initialization import Oracle, DATA
from research.ga_ssw.probe_native_ls_response import xmm
from research.ga_ssw.probe_native_weight_emulated import load_elf
from unicorn.x86_const import UC_X86_REG_RBX, UC_X86_REG_R15, UC_X86_REG_XMM0


class StopOracle(Oracle):
    def hook(self, u, address, size, user):
        if address in (0x5bf6b4, 0x5be8aa):
            self.destination = address
            u.emu_stop()
        elif not 0x5bf699 <= address < 0x5bf6b4:
            raise RuntimeError(f'unexpected instruction {address:#x}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elf', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    o = StopOracle(segments)
    defaults = {
        'SSW.LSoptsoftmax': {'value': o.integer(0x4a499cc), 'constant_address': '0x4a499cc',
                           'parser_call': '0x68694a', 'field_offset': '0x2dad0'},
        'SSW.ftol': {'value': o.double(0x4a49900), 'constant_address': '0x4a49900',
                     'parser_call': '0x687d4e', 'field_offset': '0x2db28'},
    }
    names = [bytes(o.uc.mem_read(a, n)).decode('ascii').strip() for a, n in
             [(0x4a49d14, 16), (0x4a30898, 8)]]
    assert names == list(defaults)
    assert defaults['SSW.LSoptsoftmax']['value'] == 50
    assert defaults['SSW.ftol']['value'] == .1
    rows = []
    for force in (.099, .1, .101):
        for step in (49, 50, 51):
            o.uc.reg_write(UC_X86_REG_RBX, DATA)
            o.uc.reg_write(UC_X86_REG_R15, step)
            o.putd(DATA + 0x2db28, .1)
            o.puti(DATA + 0x2dad0, 50)
            xmm(o, UC_X86_REG_XMM0, force)
            o.destination = None
            o.uc.emu_start(0x5bf699, 0x5bf6b5, count=20, timeout=1000000)
            exit_branch = o.destination == 0x5bf6b4
            assert exit_branch == (force < .1 or step >= 50)
            rows.append(dict(force_measure=force, counter=step, exit_branch=exit_branch,
                             destination=hex(o.destination)))
    print(json.dumps(dict(elf_sha256=digest, parser_defaults=defaults, rows=rows,
        scope='Parser call argument constants plus isolated stop predicate; supplied finite force measure and counter. No optimizer failure flags, effective runtime overrides, iteration accounting or scientific convergence established.'), indent=2))


if __name__ == '__main__':
    main()
