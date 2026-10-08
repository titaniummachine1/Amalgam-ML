"""Dump-contract gate for extractor CSVs. Run after EVERY re-dump, before anything
consumes the files. Exits 1 (loud) on structural failure; prints a liveness
report otherwise.

Why this exists: this pipeline once trained for months on `eyey`, a column that
was a single frozen spawn value per entity (SourceTV never populates
m_angEyeAngles for remote players). Nothing checked, so nothing noticed. Any
column can go the same way silently - positions, mspd, flags. This script is the
gate that makes that impossible.

Checks (structural, fail loudly):
  1. header matches the dump contract exactly (no silent column drift)
  2. (ent,tick) unique
  3. ticks non-decreasing per entity (a reorder breaks every builder)
Liveness report (informational, flags STALE):
  4. per entity: rows, distinct ox/oy, distinct eyey, distinct mspd, distinct class
     An entity with >500 alive rows whose ox/oy never change is flagged STALE.
     An entity moving (distinct ox/oy large) whose eyey never changes is flagged
     EYE-FROZEN - expected in SourceTV dumps, but every consumer of eyey must see
     this flag and treat the column as a spawn constant, never a time series.

Usage: python ml/verify_csv.py <file.csv> [more...]
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mspd import PLAYER_COLS

STALE_MIN_ROWS = 500


def check(path):
    fails = []
    ents = defaultdict(lambda: {'n': 0, 'ox': set(), 'oy': set(), 'ey': set(),
                                'ms': set(), 'cls': set(), 'tk': [], 'life0': 0})
    seen = set()
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split(',')
        if header != PLAYER_COLS:
            fails.append('header mismatch in %s:\n got %s\nwant %s' % (path, header, PLAYER_COLS))
            return fails, []
        for ln, row in enumerate(csv.DictReader(fh, fieldnames=header), 2):
            try:
                e, t = int(row['ent']), int(row['tick'])
            except ValueError:
                fails.append('line %d: bad ent/tick' % ln)
                continue
            if (e, t) in seen:
                fails.append('line %d: duplicate (ent=%d, tick=%d)' % (ln, e, t))
            seen.add((e, t))
            s = ents[e]
            s['n'] += 1
            s['tk'].append(t)
            s['cls'].add(row['class'])
            if row['life'] != '0':
                continue
            s['life0'] += 1
            s['ox'].add(row['ox'])
            s['oy'].add(row['oy'])
            s['ey'].add(row['eyey'])
            s['ms'].add(row['mspd'])
    report = []
    for e in sorted(ents):
        s = ents[e]
        mono = all(b >= a for a, b in zip(s['tk'], s['tk'][1:]))
        if not mono:
            fails.append('ent %d: ticks not non-decreasing' % e)
        moving = len(s['ox']) > 1 or len(s['oy']) > 1
        stale = s['life0'] > STALE_MIN_ROWS and not moving
        eyef = moving and len(s['ey']) <= 1
        flags = []
        if stale:
            flags.append('STALE')
        if eyef:
            flags.append('EYE-FROZEN')
        if 'NaN' in s['ms'] or 'nan' in s['ms']:
            flags.append('MSPD-HAS-NAN')
        report.append('  ent %3d rows %7d alive %7d classes %s ox# %6d oy# %6d ey# %4d ms# %4d%s'
                      % (e, s['n'], s['life0'], '/'.join(sorted(s['cls'])),
                         len(s['ox']), len(s['oy']), len(s['ey']), len(s['ms']),
                         '  ' + ','.join(flags) if flags else ''))
    return fails, report


def main(paths):
    rc = 0
    for p in paths:
        print('== %s' % p)
        fails, report = check(p)
        print('\n'.join(report))
        if fails:
            rc = 1
            print('FAIL:')
            for f in fails[:20]:
                print('  ' + f)
        else:
            print('STRUCTURE OK')
    return rc


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: verify_csv.py <file.csv> [more...]')
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
