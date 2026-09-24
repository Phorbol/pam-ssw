"""Reproduce selected-stage cost attribution from frozen source records (no PES)."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    inputs = json.loads((HERE / 'inputs.json').read_text())
    rows = []
    for source in inputs['sources']:
        data = json.loads(Path(source['path']).read_text())
        for record in data['records']:
            if record['index'] not in source['selected']:
                continue
            for event in record['climb']:
                telemetry = event.get('optimizer_telemetry') or {}
                rows.append(dict(source=source['path'], record=record['index'],
                    stage=event['index'], accepted_steps=telemetry.get('accepted_steps'),
                    quench_requests=event.get('quench_requests'),
                    requests=event['requests'], status=event['status']))
    summary = {}
    for system in ('C4H6', 'C60'):
        sources = {r['source'] for r in rows
                   if ('c4h6' in r['source']) == (system == 'C4H6')}
        selected = [r for r in rows if r['source'] in sources]
        summary[system] = dict(stages=len(selected),
            status=sorted({r['status'] for r in selected}),
            all_stage_requests=sum(r['requests'] for r in selected),
            quench_requests_total=sum(r['quench_requests'] for r in selected))
        for field in ('accepted_steps', 'quench_requests'):
            values = [r[field] for r in selected]
            summary[system][field] = dict(median=statistics.median(values), max=max(values))
    result = dict(scope='Same frozen selected paths only; stages are correlated, no new PES',
                  summary=summary, rows=rows)
    (HERE / 'stage-cost-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
