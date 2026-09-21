"""Bounded BRZERO4 numerical oracle with explicitly replaced LAPACK only.

This extends the first-matrix Unicorn harness to keep the original BRZERO4
initial call and two history updates.  DGEGV is intercepted at its ABI entry
and evaluated with SciPy LAPACK on the matrices passed by the ELF; its inputs,
outputs and INFO are recorded.  No execution is performed unless
``--execute`` is supplied.
"""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_RSP, UC_X86_REG_RIP, UC_X86_REG_RBP
from research.ga_ssw.probe_native_broyden_first_matrix import PrefixOracle, DATA, STACK, STOP
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF_DEFAULT = '/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
ELF_SHA256 = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
BRZERO4 = 0x6f6c00
DGEGV = 0x149d0c0


class FullOracle(PrefixOracle):
    """PrefixOracle ABI with the BRZERO4 body and DGEGV handler exposed."""
    def __init__(self, segments):
        super().__init__(segments)
        self.dggev_calls = []
        self.spectral_checks = []

    def hook(self, uc, address, size, user):
        if address == STOP:
            uc.emu_stop(); return
        if address == 0x6fbfdc:
            bp=uc.reg_read(UC_X86_REG_RBP)
            self.spectral_checks.append(dict(trial_drop_count=self.i(bp-0x2d8),maximum=struct.unpack('<d',uc.mem_read(bp-0x378,8))[0],order=uc.reg_read(UC_X86_REG_RDX)))
        if address == DGEGV:
            self._dggev()
            return
        if 0x6f6c00 <= address < 0x7026c0 or 0x498a9c0 <= address < 0x498ac00:
            return
        self.runtime_calls[hex(address)] = self.runtime_calls.get(hex(address), 0) + 1
        rdi = uc.reg_read(UC_X86_REG_RDI)  # RDI
        rsi = uc.reg_read(UC_X86_REG_RSI)  # RSI
        rdx = uc.reg_read(UC_X86_REG_RDX)  # RDX
        if address == 0x498a070:
            uc.mem_write(rsi, struct.pack('<Q', self.heap)); self.heap += ((rdi + 63) // 64) * 64
            uc.reg_write(UC_X86_REG_RAX, 0)
        elif address == 0x4a102b0:
            uc.mem_write(rdi, bytes(uc.mem_read(rsi, rdx))); uc.reg_write(UC_X86_REG_RAX, rdi)
        elif address == 0x4a10430:
            uc.mem_write(rdi, bytes([rsi & 255]) * rdx); uc.reg_write(UC_X86_REG_RAX, rdi)
        elif address in (0x499e470, 0x49a01a0, 0x4998d70):
            uc.reg_write(UC_X86_REG_RAX, 0)
        else:
            raise RuntimeError(f'unexpected runtime call {address:#x}')
        self.ret()

    def _dggev(self):
        """Call SciPy's actual DGGEV on the pointers supplied by the ELF."""
        from scipy.linalg import lapack
        uc = self.uc
        regs = [uc.reg_read(reg) for reg in (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9)]
        rsp = uc.reg_read(UC_X86_REG_RSP)  # RSP
        stack = [struct.unpack('<Q', uc.mem_read(rsp + 8 * (i + 1), 8))[0] for i in range(11)]
        jobvl, jobvr, pn, pa, plda, pb = regs
        pldb, palphar, palphai, pbeta, pvl, pldvl, pvr, pldvr, pwork, plwork, pinfo = stack
        n = struct.unpack('<i', uc.mem_read(pn, 4))[0]
        lda = struct.unpack('<i', uc.mem_read(plda, 4))[0]
        ldb = struct.unpack('<i', uc.mem_read(pldb, 4))[0]
        a = np.frombuffer(uc.mem_read(pa, 8 * lda * n), dtype='<f8').reshape((lda, n), order='F').copy()
        b = np.frombuffer(uc.mem_read(pb, 8 * ldb * n), dtype='<f8').reshape((ldb, n), order='F').copy()
        left = bytes(uc.mem_read(jobvl, 1)).upper() == b'V'
        right = bytes(uc.mem_read(jobvr, 1)).upper() == b'V'
        if not 1 <= n <= 50 or left or right:
            raise ValueError('probe only supports actual N/N calls of order1..50')
        aa=np.array(a[:n,:n],order='F');bb=np.array(b[:n,:n],order='F')
        input_a=aa.tolist();input_b=bb.tolist()
        result = lapack.dggev(aa, bb, compute_vl=0, compute_vr=0, overwrite_a=1, overwrite_b=1)
        alphar, alphai, beta, vl, vr, work, info = result
        if info: raise RuntimeError(f'DGGEV info={info}')
        a[:n,:n]=aa;b[:n,:n]=bb
        uc.mem_write(pa,a.tobytes(order='F'));uc.mem_write(pb,b.tobytes(order='F'))
        def write_vec(ptr, values): uc.mem_write(ptr, np.asarray(values, dtype='<f8').tobytes())
        write_vec(pwork,work[:struct.unpack('<i',uc.mem_read(plwork,4))[0]])
        write_vec(palphar, alphar); write_vec(palphai, alphai); write_vec(pbeta, beta)
        if left: uc.mem_write(pvl, np.asarray(vl, dtype='<f8', order='F').tobytes(order='F'))
        if right: uc.mem_write(pvr, np.asarray(vr, dtype='<f8', order='F').tobytes(order='F'))
        uc.mem_write(pinfo, struct.pack('<i', int(info)))
        self.dggev_calls.append(dict(trial_drop_count=self.i(uc.reg_read(UC_X86_REG_RBP)-0x2d8),input_a=input_a,input_b=input_b,post_a=aa.tolist(),post_b=bb.tolist(),n=n, lda=lda, ldb=ldb, jobvl=left, jobvr=right,
                                     info=int(info), alpha_r=np.asarray(alphar).tolist(),
                                     alpha_i=np.asarray(alphai).tolist(), beta=np.asarray(beta).tolist()))
        self.ret()

    def full_call(self, x, f, g0, initial, *, iniangle=.5, langle=0, rotmode=0, iout=-1):
        # Same Fortran ABI packing as PrefixOracle.call, but stop only at the
        # synthetic return sentinel so BRZERO4 can reach all history updates.
        vals = [struct.pack('<i', len(x)), np.asarray(x, dtype='<f8').tobytes(),
                np.asarray(f, dtype='<f8').tobytes(), np.asarray(g0, dtype='<f8').tobytes(),
                struct.pack('<i', int(initial)), struct.pack('<i', -1), struct.pack('<i', iout),
                struct.pack('<d', iniangle), struct.pack('<i', 0), struct.pack('<i', langle), struct.pack('<i', rotmode)]
        ptrs=[]; cur=DATA
        for val in vals:
            ptrs.append(cur); self.uc.mem_write(cur, val); cur += (len(val)+31)//32*32
        for reg, ptr in zip((UC_X86_REG_RDI,UC_X86_REG_RSI,UC_X86_REG_RDX,UC_X86_REG_RCX,UC_X86_REG_R8,UC_X86_REG_R9), ptrs): self.uc.reg_write(reg, ptr)
        sp=STACK+0x80008; self.uc.mem_write(sp, struct.pack('<Q', STOP)+b''.join(struct.pack('<Q', p) for p in ptrs[6:])); self.uc.reg_write(UC_X86_REG_RSP,sp)
        self.uc.emu_start(BRZERO4, STOP, timeout=10_000_000,count=10_000_000)
        if self.uc.reg_read(UC_X86_REG_RIP) != STOP: raise RuntimeError(f'stopped at {self.uc.reg_read(UC_X86_REG_RIP):#x}')
        return np.frombuffer(self.uc.mem_read(ptrs[1], len(x)*8), dtype='<f8').copy(), ptrs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--elf',default=ELF_DEFAULT); ap.add_argument('--output',required=True); ap.add_argument('--execute',action='store_true'); ap.add_argument('--ndim',type=int,default=6);ap.add_argument('--updates',type=int,default=2);ap.add_argument('--seed',type=int,default=20260912);ap.add_argument('--graded-g0',action='store_true');ap.add_argument('--quartic',type=float,default=0.0);args=ap.parse_args()
    if args.ndim<=0 or args.ndim%3 or not 1<=args.updates<=8:raise ValueError('requires complete Cartesian blocks and 1..8 bounded updates')
    report=dict(elf=str(Path(args.elf).resolve()), entry=hex(BRZERO4), dggev=hex(DGEGV), history_updates=args.updates, n=args.ndim, seed=args.seed,graded_g0=args.graded_g0,quartic=args.quartic, hook_scope='runtime allocation/copy/zero/printing and DGEGV only; BRZERO4/LUDCM/LUBKS arithmetic executes ELF', status='prepared')
    blob,segments=load_elf(args.elf); report['sha256']=hashlib.sha256(blob).hexdigest()
    if report['sha256'] != ELF_SHA256: raise ValueError('wrong ELF')
    if args.execute:
        import scipy
        from research.ga_ssw.probe_native_broyden_first_matrix import DESCRIPTORS
        DESCRIPTORS.update(z=0x5522480,wi=0x5522780,t=0x5522540)
        oracle=FullOracle(segments);rng=np.random.default_rng(args.seed)
        x=rng.normal(size=args.ndim);h=np.diag(np.linspace(.5,2.,args.ndim));g0=np.linspace(.05,.2,args.ndim) if args.graded_g0 else np.full(args.ndim,.1)
        report.update(status='running',steps=[],scipy_version=scipy.__version__,oracle_boundary='BRZERO4 and inverse arithmetic native; DGEGV replaced by actual SciPy DGGEV with all input/output matrices recorded, not instruction parity')
        try:
            for step in range(args.updates+1):
                force=-h@x-args.quartic*x**3;before=x.copy();x,_=oracle.full_call(x,force,g0,step==0)
                arrays={k:oracle.array(k).tolist() for k in DESCRIPTORS}
                statics={name:np.frombuffer(oracle.uc.mem_read(addr,50*50*8),dtype='<f8').reshape((50,50),order='F')[:step,:step].tolist() for name,addr in dict(amat=0x7933660,beta=0x7938860,betaq=0x793dda0).items()} if step else {}
                report['steps'].append(dict(step=step,x_before=before.tolist(),force=force.tolist(),g0=g0.tolist(),x_after=x.tolist(),iteration=oracle.i(0x7942fa4),arrays=arrays,matrices=statics))
                assert np.isfinite(x).all()
            report['status']='completed'
        except Exception as error:
            report.update(status='blocked',error=repr(error),rip=hex(oracle.uc.reg_read(UC_X86_REG_RIP)))
        report.update(spectral_checks=oracle.spectral_checks,dggev_calls=oracle.dggev_calls,runtime_calls=oracle.runtime_calls,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hessian=h.tolist())

    Path(args.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k not in ('steps','dggev_calls')},indent=2))
    if report['status']=='blocked':raise SystemExit(1)


if __name__=='__main__': main()
