"""Read-only completed-stage optimizer costs; no counterfactual speedup."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob('*/comparison/*/search-result.json')):
        data = json.loads(path.read_text())
        stages = [event for record in data['records'] if record.get('atomic')
                  for event in record['atomic']['climb'] if 'true_energy' in event]
        accepted = [event['optimizer_telemetry']['accepted_steps'] for event in stages]
        rows.append(dict(arm=path.parts[-4], seed=data['seed'],
            completed_gaussians=len(stages),
            accepted_min=min(accepted) if accepted else None,
            accepted_median=statistics.median(accepted) if accepted else None,
            accepted_max=max(accepted) if accepted else None,
            completed_biased_quench_requests=sum(e['quench_requests'] for e in stages),
            budget_stopped_stages=sum(e.get('stage_stop_reason') == 'iteration_budget' for e in stages),
            total_search_requests=data['search_requests'], source=str(path)))
    (args.root/'stage-cost-summary.json').write_text(json.dumps(dict(
        scope='Completed stages only; excludes incomplete-stage quench costs; no counterfactual speedup',
        rows=rows), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
