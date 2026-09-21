"""Isolated native LBFGS oracle on frozen 246-D Fe7C3 VC objectives.

This harness executes only the uploaded ELF LBFGS reverse-communication
kernel. The caller supplies a chart E/F/stress oracle; it is a configuration
comparison against Safe-total, not a claim that native MCSRCH equals the
independent optimizer or that native flag 0 is a joint VC convergence test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
try:
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
    from unicorn.x86_const import *
    _UNICORN_ERROR = None
except ModuleNotFoundError as error:
    Uc = None
    _UNICORN_ERROR = error

from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening
from research.ga_ssw.compare_vc_arms import serial
from research.ga_ssw.compare_fe7c3_ls_frozen_quenches import _pick_failed, _softening, _frozen_bias, _atoms
try:
    from research.ga_ssw.probe_native_weight_emulated import load_elf, STOP, STACK, DATA
except ModuleNotFoundError:
    load_elf = None
    STOP, STACK, DATA = 0x700000000000, 0x710000000000, 0x720000000000

ELF = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
ENTRY = 0x6e87b0
ACCEPT_PC = 0x6e8dad
FLAG_ADDR = 0x791b938
PROFILE = dict(history=400, gtol=900., stpmin=.0001, maxstep=.5,
               ftol=.0001,
               eps=struct.unpack('<d', struct.pack('<Q', 0x3ee4f8b588e368f1))[0],
               xtol=struct.unpack('<d', struct.pack('<Q', 0x3c9cd2b297d889bc))[0],
               gradient_scale=1.)


def vc_norm(g, natoms):
    g = np.asarray(g, dtype=float)
    return max(float(np.linalg.norm(g[:-6].reshape(natoms, 3), axis=1).max()),
               float(np.linalg.norm(g[-6:])))


def native_workspace_bytes(dimension):
    """Bytes required by the dense profile workspace for one flat dimension."""
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError('dimension must be positive')
    return (dimension * 801 + 800) * 8


class FlatOracle:
    """Native LBFGS reverse-communication memory for any flat dimension."""
    def __init__(self, segments, x, *, accepted_step_cap, joint_gtol):
        if Uc is None:
            raise RuntimeError('unicorn is required for FlatOracle') from _UNICORN_ERROR
        self.uc = Uc(UC_ARCH_X86, UC_MODE_64)
        for va, size, chunk in segments:
            start = va & ~4095
            self.uc.mem_map(start, ((va + size + 4095) & ~4095) - start)
            self.uc.mem_write(va, chunk)
        # 246*801 doubles plus all pointer metadata requires >1 MiB.
        for address, size in ((STOP, 0x100000), (STACK, 0x100000), (DATA, 0x4000000)):
            self.uc.mem_map(address, size)
        self.cursor = DATA
        self.n = int(np.asarray(x).size)
        self.accepted_step_cap = int(accepted_step_cap)
        self.joint_gtol = float(joint_gtol)
        self.accepted = []
        self.calls = {}
        self.current = None
        self.stop_reason = None
        self.ptr = [self.alloc(struct.pack('<i', self.n)),
                    self.alloc(struct.pack('<i', PROFILE['history'])),
                    self.arr(x), self.arr([0.]), self.arr(np.zeros(self.n)),
                    self.alloc(struct.pack('<i', 0)), self.arr(np.ones(self.n)),
                    self.alloc(struct.pack('<ii', -1, 0)), self.arr([PROFILE['eps']]),
                    self.arr([PROFILE['xtol']]), self.arr(np.zeros(self.n * 801 + 800)),
                    self.alloc(struct.pack('<i', 0)), self.arr([PROFILE['maxstep']]),
                    self.arr([PROFILE['ftol']])]
        self.uc.hook_add(UC_HOOK_CODE, self.hook)

    def alloc(self, data):
        p = self.cursor
        self.cursor += (len(data) + 31) // 32 * 32
        self.uc.mem_write(p, data)
        return p

    def arr(self, value):
        return self.alloc(np.asarray(value, dtype='<f8').tobytes())

    def integer(self, address):
        return struct.unpack('<i', self.uc.mem_read(address, 4))[0]

    def vector(self):
        return np.frombuffer(self.uc.mem_read(self.ptr[2], self.n * 8), dtype='<f8').copy()

    def hook(self, machine, pc, size, data):
        if pc in (ENTRY, 0x6ea9d0, 0x6eb690):
            self.calls[hex(pc)] = self.calls.get(hex(pc), 0) + 1
        if pc == 0x4a102b0:  # memcpy PLT replacement, as in existing harness
            dst, src, count = [machine.reg_read(r) for r in
                               (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX)]
            machine.mem_write(dst, bytes(machine.mem_read(src, count)))
            sp = machine.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack('<Q', machine.mem_read(sp, 8))[0]
            machine.reg_write(UC_X86_REG_RSP, sp + 8)
            machine.reg_write(UC_X86_REG_RIP, ret)
            return
        if not ENTRY <= pc < 0x6ebce0:
            raise RuntimeError(f'unexpected native PC {pc:#x}')
        if pc == ACCEPT_PC and self.integer(FLAG_ADDR) == 1:
            accepted = dict(self.current)
            accepted['accepted_coordinate_error'] = float(
                np.max(np.abs(self.vector() - np.asarray(self.current['q']))))
            if accepted['accepted_coordinate_error'] > 1e-12:
                raise ValueError('native accepted iterate differs from requested q')
            self.accepted.append(accepted)
            if accepted['vc_norm'] <= self.joint_gtol:
                self.stop_reason = 'joint_gtol'
            elif len(self.accepted) >= self.accepted_step_cap:
                self.stop_reason = 'accepted_step_limit'
            if self.stop_reason:
                machine.emu_stop()

    def advance(self, objective, gradient, current):
        self.current = current
        u = self.uc
        u.mem_write(self.ptr[3], struct.pack('<d', float(objective)))
        # This harness receives the objective gradient directly. The old EMT
        # probe accepted a force and negated it; doing that here would make the
        # native optimizer climb the frozen objective.
        u.mem_write(self.ptr[4], np.asarray(gradient, dtype='<f8').tobytes())
        for register, pointer in zip((UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX,
                                      UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9), self.ptr):
            u.reg_write(register, pointer)
        sp = STACK + 0x80008
        u.mem_write(sp, struct.pack('<Q', STOP) +
                    b''.join(struct.pack('<Q', p) for p in self.ptr[6:]))
        u.reg_write(UC_X86_REG_RSP, sp)
        u.emu_start(ENTRY, STOP, timeout=10_000_000, count=10_000_000)
        if not self.stop_reason and u.reg_read(UC_X86_REG_RIP) != STOP:
            raise RuntimeError('native instruction/time cap')
        return self.integer(self.ptr[11])


def frozen_objective(chart, surface, softening, q, gaussians, pressure=0.):
    def combined(atoms):
        e, f, stress = surface.evaluate(atoms)
        le, lf, ls = softening.evaluate_stress(atoms)
        return e + le, f + lf, stress + ls
    ev = chart.evaluate(q, combined, pressure=pressure)
    objective, gradient = _frozen_bias(ev.objective, chart.project(ev.gradient), q, gaussians)
    return objective, gradient, ev


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run_case(source_path, model_path, calculator_factory, segments, *, request_cap, accepted_step_cap,
             joint_gtol, deadline, reference_path=None):
    source = json.loads(Path(source_path).read_text())
    record, climb = _pick_failed(source)
    chart = SymmetricLogStrainChart(_atoms(record['chart_reference']), strain_length=source['joint_config']['strain_length'])
    softening = _softening(record['frozen_softening'])
    gaussians = record['frozen_gaussians']
    last = gaussians[-1]
    q_start = np.asarray(last['center']) + float(last['width']) * np.asarray(last['direction'])
    if q_start.size != 246:
        raise ValueError(f'expected 246-D VC chart, got {q_start.size}')
    surface = ASEStressSurface(calculator_factory())
    oracle = FlatOracle(segments, q_start, accepted_step_cap=accepted_step_cap, joint_gtol=joint_gtol)
    reference = None if reference_path is None else json.loads(Path(reference_path).read_text())["safe_lbfgs"]["trace"][0]
    initial_agreement = None
    rows = []
    status = 'request_limit'
    error = None
    started = time.monotonic()
    for request in range(request_cap):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('600 second global deadline reached')
            q = oracle.vector()
            objective, gradient, ev = frozen_objective(chart, surface, softening, q, gaussians,
                                                       source['joint_config']['pressure'])
            row = dict(request=request + 1, q=q, objective=objective, gradient=gradient,
                       vc_norm=vc_norm(gradient, chart.natoms), atoms=ev.atoms)
            rows.append(row)
            if len(rows) == 1 and reference is not None:
                initial_agreement = dict(
                    q=float(np.max(np.abs(q-np.asarray(reference['q'])))),
                    energy=abs(objective-reference['energy']),
                    gradient=float(np.max(np.abs(gradient-np.asarray(reference['gradient'])))))
                if max(initial_agreement.values()) > 1e-8:
                    raise ValueError(f'frozen initial objective mismatch: {initial_agreement}')
            if len(rows) == 1 and row['vc_norm'] <= joint_gtol:
                status = 'initial_joint_gtol'
                break
            flag = oracle.advance(objective, gradient, row)
            if oracle.stop_reason:
                status = oracle.stop_reason
                break
            if flag != 1:
                status = 'native_flag_0_unqualified' if flag == 0 else 'native_failure'
                break
        except Exception as exc:
            status = 'time_limit' if isinstance(exc, TimeoutError) else 'oracle_error'
            error = repr(exc)
            break
    final = None
    fresh_surface = None
    try:
        if oracle.accepted:
            endpoint = oracle.accepted[-1]; endpoint_source = 'last_accepted'
        elif rows:
            endpoint = rows[0]; endpoint_source = 'evaluated_initial_no_accepted'
        else:
            endpoint = dict(q=q_start); endpoint_source = 'unevaluated_q_start_unknown'
        fresh_surface = ASEStressSurface(calculator_factory())
        objective, gradient, ev = frozen_objective(chart, fresh_surface, softening,
                                                   np.asarray(endpoint['q']), gaussians,
                                                   source['joint_config']['pressure'])
        final = dict(status='checked', objective=objective, gradient=gradient,
                     vc_norm=vc_norm(gradient, chart.natoms), atoms=ev.atoms,
                     source_q=np.asarray(endpoint['q']))
    except Exception as exc:
        final = dict(status='fresh_failed', error=repr(exc))
        endpoint_source = locals().get('endpoint_source', 'unknown')
    optimizer_requests = surface.requests
    fresh_requests = 0 if fresh_surface is None else fresh_surface.requests
    return dict(source_path=str(source_path), source_sha256=_sha256(source_path),
                model=str(model_path), model_sha256=_sha256(model_path), status=status,
                error=error, request_cap=request_cap, accepted_step_cap=accepted_step_cap,
                optimizer_requests=optimizer_requests, fresh_requests=fresh_requests,
                total_requests=optimizer_requests + fresh_requests,
                accepted_steps=len(oracle.accepted), native_flag=oracle.integer(oracle.ptr[11]), native_stop=oracle.stop_reason,
                q_start=q_start, q_failed=np.asarray(climb['q']),
                recorded_failure_status=climb['status'], recorded_failure_steps=climb['relaxation']['steps'],
                final_fresh=final, endpoint_source=endpoint_source, initial_agreement=initial_agreement,
                reference_path=None if reference_path is None else str(reference_path),
                calls=oracle.calls, seconds=time.monotonic()-started,
                evaluations=rows, accepted=oracle.accepted,
                profile=PROFILE, norm_definition='max(max atom-force norm, six-cell-block L2)',
                comparison_label='native LBFGS profile versus Safe-total; not an MCSRCH-only comparison')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='directory containing comparison/{ls_all,ls_filter}-seed{7,101}')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-cap', type=int, default=329)
    parser.add_argument('--accepted-step-cap', type=int, default=300)
    parser.add_argument('--joint-gtol', type=float, default=.001)
    args = parser.parse_args(argv)
    if args.request_cap < 1 or args.accepted_step_cap < 1 or args.joint_gtol <= 0:
        raise ValueError('caps and joint gtol must be positive')
    if load_elf is None:
        raise RuntimeError('unicorn is required to execute the isolated ELF harness') from _UNICORN_ERROR
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    sources = [args.root / 'comparison' / f'{arm}-seed{seed}' / 'result.json'
               for arm in ('ls_all', 'ls_filter') for seed in (7, 101)]
    plan = dict(elf=ELF, elf_sha256=hashlib.sha256(blob).hexdigest(), model=str(args.model),
                model_sha256=_sha256(args.model), request_cap=args.request_cap,
                accepted_step_cap=args.accepted_step_cap, joint_gtol=args.joint_gtol,
                profile=PROFILE, source_paths=[str(p) for p in sources],
                purpose='isolated flat 246-D native LBFGS configuration comparison')
    (output / 'plan.json').write_text(json.dumps(serial(plan), indent=2) + '\n')
    (output / 'script.py').write_text(Path(__file__).read_text())
    import torch
    from mace.calculators import MACECalculator
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    def calculator_factory():
        return MACECalculator(model_paths=str(args.model), device='cuda',
                               default_dtype='float64', enable_cueq=False)
    rows = []; global_deadline = time.monotonic() + 600.
    for source in sources:
        result = run_case(source, args.model, calculator_factory, segments,
                          request_cap=args.request_cap,
                          accepted_step_cap=args.accepted_step_cap,
                          joint_gtol=args.joint_gtol,
                          deadline=global_deadline,
                          reference_path=args.reference / f"seed{json.loads(source.read_text())['seed']}" / json.loads(source.read_text())["arm"] / "history10" / "result.json")
        path = output / f'{source.parent.name}.json'
        path.write_text(json.dumps(serial(result), indent=2, allow_nan=False) + '\n')
        rows.append({k: v for k, v in result.items() if k not in ('evaluations', 'accepted')})
        print(json.dumps({k: result[k] for k in ('source_path', 'status', 'optimizer_requests', 'fresh_requests', 'accepted_steps', 'error')}), flush=True)
    (output / 'summary.json').write_text(json.dumps(serial(dict(
        runs=rows, total_requests=sum(r['total_requests'] for r in rows),
        global_deadline_seconds=600)), indent=2) + '\n')


if __name__ == '__main__':
    main()
