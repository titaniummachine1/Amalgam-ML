"""Inspect gap cases with DB context: what WAS the player doing where blame is
high but physics normal? Sample across horizons; print position/speed/ground
tracks + weapon + nearest-threat snapshot. Usage: python ml/audit_gaps.py [n=30]"""
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    k = np.load(os.path.join(D, 'evalBk.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    gaps = [json.loads(l) for l in open(os.path.join(TMP, 'gap_rows.jsonl'))]
    by_h = defaultdict(list)
    for g in gaps:
        by_h[min(g['h'] // 22, 5)].append(g)
    print('gap bins:', {kk: len(v) for kk, v in by_h.items()}, flush=True)
    did_count = defaultdict(int)
    for g in gaps:
        i = g['row']
        if i < len(Ks):
            did_count[int(Ks[i][0])] += 1
    print('top gap demos:', sorted(did_count.items(), key=lambda kv: -kv[1])[:8], flush=True)
    shown = 0
    for b in sorted(by_h):
        samp = np.random.RandomState(b).choice(by_h[b], min(n // 6 + 1, len(by_h[b])),
                                               replace=False)
        for g in samp:
            i, h = g['row'], g['h']
            if i >= len(Ks):
                continue
            did, ent, T = (int(Ks[i][0]), int(Ks[i][1]), int(Ks[i][2]))
            s = con.execute(
                'SELECT tick,ox,oy,oz,grounded,cls FROM ticks WHERE demo=? AND ent=? AND tick BETWEEN ? AND ? ORDER BY tick',
                (did, ent, T - 12, T + h + 12)).fetchall()
            if len(s) < 20:
                continue
            tk = np.array([x[0] for x in s])
            o = np.array([[x[1], x[2], x[3]] for x in s])
            gr = np.array([x[4] for x in s])
            jT = int(np.where(tk == T)[0][0]) if (tk == T).any() else -1
            v = np.diff(o, axis=0, prepend=o[0:1]) / DT
            sp = np.hypot(v[:, 0], v[:, 1])
            pre = sp[max(jT - 6, 0):jT + 1] if jT >= 0 else sp[:7]
            post = sp[jT:jT + h + 2] if jT >= 0 else sp[:h + 2]
            turns = ''
            if len(pre) > 2 and len(post) > 2:
                dpre = o[jT] - o[max(jT - 6, 0)] if jT >= 0 else o[6] - o[0]
                dpst = o[min(jT + h + 1, len(o) - 1)] - o[jT] if jT >= 0 else o[h + 1] - o[0]
                ang = abs(np.arctan2(dpre[0] * dpst[1] - dpre[1] * dpst[0],
                                     dpre.dot(dpst) + 1e-9) * 57.3)
                turns = ' turn=%.0fdeg' % ang
            air = ' AIR' if (gr[jT:jT + h + 1] == 0).any() and jT >= 0 else ''
            stop = ' STOP' if post.max() < 60 else ''
            wall = ''
            if jT >= 0 and len(post) > 3:
                decel = (sp[jT] - post.min()) / max(sp[jT], 1)
                if sp[jT] > 200 and decel > 0.85 and post.min() < 60:
                    wall = ' WALL?'
            print('did=%d ent=%d T=%d h=%d blame=%.0f aneed=%.0f gr=%s '
                  'spT=%.0f spmax=%.0f%s%s%s' % (
                      did, ent, T, h, g['blame'], g['aneed'],
                      int(gr[jT]) if jT >= 0 else -1,
                      sp[jT] if jT >= 0 else -1, post.max(), turns, air, stop + wall),
                  flush=True)
            shown += 1
            if shown >= n:
                return


if __name__ == '__main__':
    main()
