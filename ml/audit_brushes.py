"""audit_brushes.py — drop bogus model-space brushes (my own validation).
A kept brush whose planes swallow observed free-motion demo positions DEEP
(all nd < -5hu) cannot be world-space collision (demos move there freely);
it is an untransformed brush-entity model. Those brushes are dropped for the
BOX path only (tri soup unaffected — tris are real surfaces).
Usage: python ml/audit_brushes.py <brushes.npz> <csv> <out_drops.txt>
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import load_csv_live, FL_ONGROUND


def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d


def main():
    bpath, csvpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.load(bpath, allow_pickle=True)
    BP = d['bplanes'].astype(np.float64)
    BN = d['bnsides'].astype(np.int64)
    HALF = np.array([16., 16., 36.])
    S = BP.shape[1]
    OFFD = BP[:, :, 3] + np.abs(BP[:, :, :3]) @ HALF
    # csvpath = one csv OR a .txt list (audit across many demos)
    if csvpath.endswith('.txt'):
        csvs = [l.strip() for l in open(csvpath) if l.strip()]
    else:
        csvs = [csvpath]
    rows = {}
    P = []
    ndone = 0
    for csv in csvs:
        # stream: positions only, never hold all CSVs (107 files = GBs).
        # stride + per-CSV cap so ALL demos/areas are covered (phantoms are local).
        try:
            rone = load_csv_live(csv, classes=set(range(1, 10)))
        except Exception as ex:
            print('skip', csv, type(ex).__name__, flush=True)
            continue
        for e, rs in rone.items():
            for r in rs[::7]:
                try:
                    if not (int(f(r['flags'])) & FL_ONGROUND):
                        continue
                    P.append([f(r['ox']), f(r['oy']), f(r['oz']) + 36.0])
                    if len(P) >= 60000:
                        break
                except (ValueError, KeyError):
                    continue
            if len(P) >= 60000:
                break
        del rone
        ndone += 1
        if ndone % 20 == 0:
            print('csvs=%d positions=%d' % (ndone, len(P)), flush=True)
        if len(P) >= 60000:
            break
    print('csvs-used=%d sampled %d centers' % (ndone, len(P)), flush=True)
    P = np.array(P, dtype=np.float32)
    print('sampled %d centers' % len(P), flush=True)
    BPN = BP.astype(np.float32)
    OFF = OFFD.astype(np.float32)
    deep_count = np.zeros(BP.shape[0], dtype=np.int64)
    for s in range(0, len(P), 200):
        Pb = P[s:s + 200]
        nd = (BPN[:, :, :3] @ Pb.T).transpose(0, 2, 1) - OFF[:, None, :]
        valid = np.arange(S)[None, None, :] < BN[:, None, None]
        deep_count += np.all(~valid | (nd < -5.0), axis=2).sum(axis=1)
        print('batch %d/%d' % (s + len(Pb), len(P)), flush=True)
    drops = sorted(int(i) for i in np.nonzero(deep_count > 0)[0])
    print('bogus brushes (deep-swallow demo motion):', drops, flush=True)
    for i in drops:
        print('  brush %d ndeep=%d contents=%s' % (i, int(deep_count[i]), hex(int(d['bcontents'][i]))),
              flush=True)
    with open(out, 'w') as fh:
        fh.write('\n'.join(str(i) for i in drops))
    print('wrote', out, flush=True)


main()
