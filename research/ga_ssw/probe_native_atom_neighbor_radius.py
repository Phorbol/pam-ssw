"""Isolated radius-mask write probe; optional actual distance geometry, no PES."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256

BASE, STACK, STOP = 0x720000010000, 0x720000030000, 0x720000050000
ENTRY, GET_DIST = 0x6e3590, 0x581240

def run(segments, n, center, far_candidate, far_distance, *, actual=False, positions=None):
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, size, data in segments:
        lo = va & ~4095
        uc.mem_map(lo, ((va + size + 4095) & ~4095) - lo)
        uc.mem_write(va, data)
    for address in (BASE, STACK, STOP):
        uc.mem_map(address, 0x10000)
    nptr, centerp, radiusp = BASE, BASE + 8, BASE + 16
    coords, cell, mask = BASE + 0x1000, BASE + 0x2000, BASE + 0x3000
    uc.mem_write(nptr, struct.pack('<i', n))
    uc.mem_write(centerp, struct.pack('<i', center))
    uc.mem_write(radiusp, struct.pack('<d', 12.0))
    uc.mem_write(coords, (np.asarray(positions,dtype='<f8').reshape(n,3) if positions is not None else np.zeros((n,3),dtype='<f8')).tobytes())
    uc.mem_write(cell, (40*np.eye(3,dtype='<f8')).tobytes() if actual else b'\0' * 72)
    uc.mem_write(mask, struct.pack('<' + 'i' * (3 * n), *([1] * (3 * n))))
    if actual:
        uc.mem_write(0x53e0f98, struct.pack('<Q', BASE + 0xf000))
    rsp = STACK + 0x8008
    uc.mem_write(rsp, struct.pack('<Q', STOP))
    # rdi/rsi/rdx/rcx/r8/r9 match the caller at 0x5d66ab.
    for reg, value in ((UC_X86_REG_RDI, centerp), (UC_X86_REG_RSI, radiusp),
                       (UC_X86_REG_RDX, nptr), (UC_X86_REG_RCX, coords),
                       (UC_X86_REG_R8, cell), (UC_X86_REG_R9, mask),
                       (UC_X86_REG_RSP, rsp)):
        uc.reg_write(reg, value)
    calls = []
    pending = {}
    distances = []
    def ret():
        sp=uc.reg_read(UC_X86_REG_RSP); target=struct.unpack('<Q',uc.mem_read(sp,8))[0]
        uc.reg_write(UC_X86_REG_RSP,sp+8); uc.reg_write(UC_X86_REG_RIP,target)
    def hook(m, address, size, user):
        if address == STOP:
            m.emu_stop(); return
        if address == BASE+0xf000 and actual:
            calls.append({'floor2':True})
            packed=uc.reg_read(UC_X86_REG_XMM0).to_bytes(16,'little'); vals=np.frombuffer(packed,dtype='<f8'); out=np.floor(vals).astype('<f8'); uc.reg_write(UC_X86_REG_XMM0,int.from_bytes(out.tobytes(),'little')); ret(); return
        if actual and address in pending:
            candidate, output = pending.pop(address)
            distances.append(dict(candidate=candidate,
                                  distance=struct.unpack('<d', m.mem_read(output,8))[0]))
        if actual and address == GET_DIST:
            candidate = struct.unpack('<i', m.mem_read(m.reg_read(UC_X86_REG_RCX),4))[0]
            target = struct.unpack('<Q',m.mem_read(m.reg_read(UC_X86_REG_RSP),8))[0]
            pending[target] = (candidate,m.reg_read(UC_X86_REG_RSI))
            return
        if address != GET_DIST or actual:
            return
        # get_dist's actual callee ABI: rdx=center pointer, rcx=candidate pointer.
        cptr = m.reg_read(UC_X86_REG_RCX)
        candidate = struct.unpack('<i', m.mem_read(cptr, 4))[0]
        calls.append(dict(rdi_nptr=hex(m.reg_read(UC_X86_REG_RDI)),
                          rdx_center_ptr=hex(m.reg_read(UC_X86_REG_RDX)),
                          rcx_candidate_ptr=hex(cptr), candidate=candidate,
                          distance=far_distance if candidate == far_candidate else 1.0))
        m.mem_write(m.reg_read(UC_X86_REG_RSI), struct.pack('<d', calls[-1]['distance']))
        sp = m.reg_read(UC_X86_REG_RSP)
        target = struct.unpack('<Q', m.mem_read(sp, 8))[0]
        m.reg_write(UC_X86_REG_RSP, sp + 8)
        m.reg_write(UC_X86_REG_RIP, target)
    uc.hook_add(UC_HOOK_CODE, hook)
    try:
        uc.emu_start(ENTRY, STOP, timeout=2_000_000, count=200_000)
    except Exception as exc:
        raise RuntimeError(f'actual geometry failed pc={uc.reg_read(UC_X86_REG_RIP):#x} calls={calls}') from exc
    if uc.reg_read(UC_X86_REG_RIP) != STOP:
        raise RuntimeError('atom_neighbor_radius did not return')
    got = list(struct.unpack('<' + 'i' * (3 * n), uc.mem_read(mask, 12 * n)))
    expected = [1] * (3 * n)
    # The observed branch addresses write the final triplet, regardless of candidate.
    if 1 <= far_candidate <= n and far_candidate != center and far_distance > 12.0:
        expected[-3:] = [0, 0, 0]
    if actual:
        assert len(distances) == n-1
        for observation in distances:
            expected_distance = np.linalg.norm(np.asarray(positions)[observation['candidate']-1]-np.asarray(positions)[center-1])
            observation['distance_error'] = abs(observation['distance']-expected_distance)
            # Native reciprocal/cell arithmetic differs by up to 1.82e-12 A here.
            # Check geometric identity at 1e-10 A, far below the 1 A gate margin.
            assert observation['distance_error'] < 1e-10, (distances, positions)
    return dict(n=n, center=center, far_candidate=far_candidate,
                far_distance=far_distance, positions=None if positions is None else np.asarray(positions).tolist(),
                calls=calls, measured_distances=distances, mask=got,
                expected=expected, passed=got == expected)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--actual-geometry', action='store_true')
    a = p.parse_args(); blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    if a.actual_geometry:
        pos=np.array([[15.,15.,15.],[28.,15.,15.],[15.,16.,15.],[15.,15.,16.]])
        cases=[run(segments,4,1,2,13.0,actual=True,positions=pos), run(segments,4,1,3,13.0,actual=True,positions=np.array([[15.,15.,15.],[16.,15.,15.],[28.,15.,15.],[15.,15.,16.]])), run(segments,4,1,0,1.0,actual=True,positions=np.array([[15.,15.,15.],[16.,15.,15.],[15.,16.,15.],[15.,15.,16.]]))]
    else:
        cases = []
    # One bounded four-atom geometry protocol: center atom 1, atom 4 remains near.
    # get_dist is replaced at its entry with the controlled PBC scalar, so this
    # isolates the wrapper's branch/write address rather than executing PES.
    if not a.actual_geometry:
        cases = [run(segments, 4, 1, 0, 1.0), run(segments, 4, 1, 2, 13.0), run(segments, 4, 1, 3, 13.0)]
    report = dict(elf=ELF_DEFAULT, sha256=ELF_SHA256, entry=hex(ENTRY),
                  hooked="floor2 only" if a.actual_geometry else hex(GET_DIST), cases=cases,
                  scope='native atom_neighbor_radius; actual geometry executes get_dist/reci and hooks only floor2 when --actual-geometry, otherwise controlled get_dist scalar; no main/PES/protection',
                  passed=all(c['passed'] for c in cases))
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], cases=len(cases))))
    if not report['passed']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
