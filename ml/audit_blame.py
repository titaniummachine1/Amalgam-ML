"""Audit top-blame rows: join back to demos, classify noise vs modeling gap.
Input: TMP/blame_rows.jsonl {row (soldier order), h, blame, aneed}.
For each top row: fetch DB series window, print case file (did/ent/T, h,
grounded@T, speed, threat?, a_needed profile, horizon validity) + verdict:
  aneed unphysical (>4000 hu/s^2) -> NOISE (impulse/teleport residue)
  blame high + aneed normal + jump flight nearby -> GAP-JUMP (known, excluded)
  blame high + aneed normal, grounded, no jump -> GAP-BEHAVIOR (model blind spot)
Usage: python ml/audit_blame.py [topN=30]"""
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
con = sqlite3.connect(os.path.join(D, 'dataset_all.db'))
DT = 1 / 66.0


def main():
    topn = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    k = np.load(os.path.join(D, 'trainBk.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    rows = []
    with open(os.path.join(TMP, 'blame_rows.jsonl')) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print('no blame rows logged yet')
        return
    rows.sort(key=lambda r: -r['blame'])
    print('logged blame rows: %d, top blame: %.0f' % (len(rows), rows[0]['blame']), flush=True)
    by_type = defaultdict(int)
    for r in rows[:topn]:
        i, h = r['row'], r['h']
        if i >= len(Ks):
            continue
        did, ent, T = (int(Ks[i][0]), int(Ks[i][1]), int(Ks[i][2]))
        s = con.execute(
            'SELECT tick,ox,oy,oz,grounded,cls FROM ticks WHERE demo=? AND ent=? AND tick BETWEEN ? AND ? ORDER BY tick',
            (did, ent, T - 70, T + 140)).fetchall()
        tk = np.array([x[0] for x in s])
        jT = int(np.where(tk == T)[0][0]) if (tk == T).any() else -1
        grT = s[jT][4] if jT >= 0 else -1
        sp = 0.0
        if jT > 0:
            sp = float(np.hypot(s[jT][1] - s[jT - 1][1], s[jT][2] - s[jT - 1][2]) / DT)
        # jump flight nearby?
        oz = np.array([x[3] for x in s])
        gr = np.array([x[4] for x in s])
        v = np.diff(oz, prepend=oz[0]) / DT
        jump_near = bool((((gr[:-1] == 1) & (gr[1:] == 0) & (v[1:] > 150))).any())
        if r['aneed'] > 4000:
            verdict = 'NOISE'
        elif jump_near:
            verdict = 'GAP-JUMP'
        else:
            verdict = 'GAP-BEHAVIOR'
        by_type[verdict] += 1
        print('row=%d did=%d ent=%d T=%d h=%d blame=%.0f aneed=%.0f gr=%s sp=%.0f -> %s' % (
            i, did, ent, T, h, r['blame'], r['aneed'], grT, sp, verdict), flush=True)
    print('verdict histogram:', dict(by_type), flush=True)


if __name__ == '__main__':
    main()
