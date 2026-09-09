"""Summarize recorded original LASP/SGN artifacts without certifying minima."""
import argparse
import json
from pathlib import Path
import re


def summarize(run):
    log=(run/'stdout.txt').read_text(errors='replace')
    tasks=[]
    for path in sorted((run/'output').rglob('lasp.in')):
        folder=path.parent
        output=(folder/'lasp.out').read_text(errors='replace') if (folder/'lasp.out').exists() else ''
        conf=path.read_text()
        steps=re.search(r'(?im)^ssw.sswsteps\s+(\d+)',conf)
        temp=re.search(r'(?im)^ssw.temp\s+(\S+)',conf)
        energies=[]
        if (folder/'all.arc').exists():
            for line in (folder/'all.arc').read_text().splitlines():
                parts=line.split()
                if parts and parts[0]=='Energy': energies.append(float(parts[3]))
        tasks.append(dict(path=str(folder.relative_to(run)),started=bool(output),
                          normal_end='SSW all done' in output,configured_steps=int(steps[1]) if steps else None,
                          temperature=float(temp[1]) if temp else None,
                          minimum_log_events=len(re.findall(r'^Minimum found',output,re.M)),
                          all_arc_frames=len(energies),lowest_recorded_energy=min(energies) if energies else None))
    return dict(status=json.loads((run/'status.json').read_text()),
                stages={name:token in log for name,token in {
                    'initial_classification':'Number of distinct generated initial structures:',
                    'ga_offspring':'Number of offspring structures:',
                    'quick_ssw':'The lowest energy of this generation SSW:',
                    'quick_completed':'Quick exploration completed',
                    'fine_started':'First execution of fine search'
                }.items()},
                java_exception_lines=[s for s in log.splitlines() if 'Exception' in s and not s.startswith('\tat')],
                lasp_tasks=tasks,energy_force_calls=None,
                qualification='Recorded energies and normal exits only; not independent force/chemical validation or efficiency evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);args=p.parse_args()
    result=summarize(args.run_dir)
    result['final_arc_outputs_exist']=any((args.run_dir/'output/final').rglob('*.arc'))
    if (args.run_dir/'finish-status.json').exists():
        result['staged_completion_status']=json.loads((args.run_dir/'finish-status.json').read_text())
    (args.run_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
