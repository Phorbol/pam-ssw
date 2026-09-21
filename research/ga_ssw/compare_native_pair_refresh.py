"""Compare independent Python geometry with saved canonical native executions."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.native_pair_selection import refresh_native_pair, native_pair_allowed


def compare(source):
    rows=[]
    for row in source['cases']:
        atoms=Atoms(numbers=row['numbers'],positions=row['positions'])
        draws=iter(row['draws'])
        initial=tuple(i-1 if i else None for i in row['pair_before'])
        got=refresh_native_pair(atoms,initial,draws,fixatom=row['fixatom'])
        wanted=tuple(i-1 if i else None for i in row['pair_after'])
        events=row['events']
        expected_counts=(events['distance_or_fixatom_rejections'],events['forbidden_rejections'],events['element_rejections'])
        counts=(got.distance_or_fixatom_rejections,got.forbidden_rejections,got.element_rejections)
        checks=all(native_pair_allowed(atoms,tuple(i-1 for i in check['pair']))==check['allowed'] for check in row['checks'])
        passed=(got.pair==wanted and got.draw_count==len(row['draws']) and
                got.geometry_accepted==bool(events['accepted_exit']) and counts==expected_counts and checks)
        rows.append(dict(name=row['name'],branch=row['branch'],passed=passed,
                         actual_pair=got.pair,expected_pair=wanted,actual_draws=got.draw_count,
                         expected_draws=len(row['draws']),counts=counts,expected_counts=expected_counts,
                         checks_match=checks,geometry_accepted=got.geometry_accepted))
    return rows


if __name__=='__main__':
    source=Path('research/ga_ssw/evidence/native-getpair-canonical-20260917.json')
    rows=compare(json.loads(source.read_text()))
    out=source.with_name('native-getpair-python-comparison-20260917.json')
    out.write_text(json.dumps(dict(source=str(source),cases=rows),indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),passed=sum(r['passed'] for r in rows),failures=[r for r in rows if not r['passed']])))
    assert all(r['passed'] for r in rows)
