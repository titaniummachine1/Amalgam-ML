"""Ingest DemoLoop captures: %TEMP%\\demoloop_*.csv -> mge_csv + verify.
POISON RULE (user order): a capture is trusted ONLY with zero tick jumps:
per-ent tick sequences must be consecutive AND the .drops.log sidecar must
be empty. Files with markers are quarantined to mge_csv/_quarantine (never
deleted, never ingested) for inspection.
Usage: python ml/ingest_demoloop.py"""
import glob
import os
import shutil
import subprocess
import sys
from collections import defaultdict

TMP = os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Temp')
CSV = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
QUAR = os.path.join(CSV, '_quarantine')
os.makedirs(QUAR, exist_ok=True)
ML = os.path.dirname(os.path.abspath(__file__))


def check_continuity(path):
    """Returns (ok, info): per-ent ticks consecutive + no drop markers."""
    drops = path + '.drops.log'
    if os.path.exists(drops) and os.path.getsize(drops) > 0:
        return False, 'has drop markers'
    last = {}
    gaps = 0
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split(',')
        ie, it = header.index('ent'), header.index('tick')
        for line in fh:
            p = line.split(',')
            if len(p) != len(header):
                continue
            try:
                e, t = int(p[ie]), int(p[it])
            except ValueError:
                continue
            if e in last and t != last[e] + 1:
                gaps += 1
            last[e] = t
    if gaps:
        return False, '%d ent-tick gaps' % gaps
    return True, '%d ents' % len(last)


files = sorted(glob.glob(os.path.join(TMP, 'demoloop_*.csv')))
print('captures found: %d' % len(files), flush=True)
ingested, quarantined = 0, 0
for src in files:
    name = os.path.basename(src)
    dst = os.path.join(CSV, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        print('skip (exists) %s' % name, flush=True)
        continue
    ok, info = check_continuity(src)
    if not ok:
        shutil.copy(src, os.path.join(QUAR, name))
        print('QUARANTINE %s (%s)' % (name, info), flush=True)
        quarantined += 1
        continue
    r = subprocess.run([sys.executable, os.path.join(ML, 'verify_csv.py'), src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        shutil.copy(src, os.path.join(QUAR, name))
        print('QUARANTINE %s (verify failed)' % name, flush=True)
        print(r.stdout[-500:], flush=True)
        quarantined += 1
        continue
    shutil.copy(src, dst)
    print('OK %s (%s)' % (name, info), flush=True)
    ingested += 1
print('ingested=%d quarantined=%d' % (ingested, quarantined), flush=True)
