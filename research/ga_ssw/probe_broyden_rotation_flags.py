"""Replay archived raw inputs with recovered production rotation mode true."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from research.ga_ssw.probe_native_broyden_full import FullOracle, load_elf, ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.broyden_state_reconstruction import BroydenState


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute', action='store_true');ap.add_argument('--output',required=True);args=ap.parse_args()
    if not args.execute:raise SystemExit('requires --execute')
    blob,segments=load_elf(ELF_DEFAULT);assert hashlib.sha256(blob).hexdigest()==ELF_SHA256
    folder=Path('research/ga_ssw/evidence/native-broyden-full-probes-20260912'); results=[]
    for name in ('n6-seed11.json','n9-seed29-spectral.json','n6-seed29-quartic.json'):
        data=json.loads((folder/name).read_text());oracle=FullOracle(segments);rows=[]
        state=BroydenState(data['steps'][0]['g0'],weight=1000.,metric='native_block_sum',history_limit=50,spectral_limit=1e7)
        for row in data['steps']:
            # Scalar curvature deliberately varies to check whether saved
            # INIANGLE affects this body; this is not a full rotation caller.
            angle=(-1.)**row['step']*(row['step']+1.)
            x,p=oracle.full_call(row['x_before'],row['force'],row['g0'],row['step']==0,
                                iniangle=angle,langle=-1,rotmode=-1,iout=6)
            out=state.step(row['x_before'],row['force'])
            rows.append(dict(step=row['step'],iniangle=angle,iteration=oracle.i(0x7942fa4),
                             original_false_mode_error=float(np.max(abs(x-row['x_after']))),
                             reconstruction_error=float(np.max(abs(x-out.x))),
                             norder_after=oracle.i(p[8])))
        results.append(dict(source=name,rows=rows,spectral_checks=oracle.spectral_checks))
    output=dict(elf_sha256=ELF_SHA256,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                boundary='Isolated BRZERO4, SciPy DGGEV; true rotmode and output6, no full caller or PES',cases=results)
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
