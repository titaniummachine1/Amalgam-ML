"""Per-sample max speed (DT_BasePlayer.m_flMaxspeed) for the input-space rollout.

Why: the wishspeed clamp in a ProcessMovement rollout must use the engine's own
per-tick MaxSpeed, not a per-class constant. TF2 raises MaxSpeed at runtime
(Demoman charge, speed boosts, ducking), and a charge moves the player FASTER than
the class run speed - which is exactly why the labels show radial accel at the
speed plateau that a 240-capped add-speed rule forbids.

Aligned 1:1 with the soldier rows of trainBk/evalBk (same order as the dense66 /
threatwin caches). Source: the re-dumped CSVs, which now carry an `mspd` column
(verified content-identical to the previous dumps on every shared column).

  MS  (N,)    float32  m_flMaxspeed at the label tick T (the runtime-knowable cap;
                      frozen through the rollout because the future cap is not
                      predictable). NaN where the tick is missing.
  MSd (N,66)  float32  per-tick max speed over [T, T+66] - DIAGNOSTIC ONLY. Using
                      the true future cap in the rollout would be label leakage:
                      the cap changes with unobserved weapon/boost events.
  CLS (N,)    int8     class implied by the modal MS over the window (mspd is a far
                      cleaner class signal than the DB `cls` column)
Output: trainB_mspd.npz / evalB_mspd.npz
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg as _dr

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
W = 66
SUF = os.environ.get('OUT_SUFFIX', 'B')
DID2CSV = _dr.reverse(_dr.load())
# TF2 run speeds as the engine reports them (measured, see STATUS.md 2026-09-30):
# these are the modal m_flMaxspeed per class id
CLS_OF_MSPD = {400.0: 1, 300.0: 2, 240.0: 3, 280.0: 4, 320.0: 8, 230.0: 6}


PLAYER_COLS = ("tick,ent,class,ox,oy,oz,vx,vy,vz,eyep,eyey,mspd,"
               "flags,health,life,team,fresh,wcls,w0,w1,w2").split(',')


def load_mspd(demo_base, ents):
    """tick -> {ent: mspd} for the requested entities only.

    Column is resolved BY NAME from the header, never by position: a positional
    read here silently returns the wrong quantity if the extractor ever adds or
    reorders a column, and nothing downstream would notice (that is how a whole
    pipeline once trained on a frozen eye-yaw column). The header is asserted
    exactly so a format drift fails LOUDLY instead of misreading.
    """
    path = os.path.join(D, demo_base)
    out = defaultdict(dict)
    want = {str(e) for e in ents}
    n_rows = n_alive = n_mspd = 0
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split(',')
        assert header == PLAYER_COLS, (
            'dump contract violated in %s:\n got %s\nwant %s' % (demo_base, header, PLAYER_COLS))
        ie, it, im = header.index('ent'), header.index('tick'), header.index('mspd')
        ilife = header.index('life')
        for line in fh:
            n_rows += 1
            p = line.split(',')
            if len(p) != len(header):
                continue
            if p[ie] not in want or p[ilife] != '0':
                continue
            n_alive += 1
            try:
                t, e, m = int(p[it]), int(p[ie]), p[im]
            except ValueError:
                continue
            if m.lower() in ('', 'nan', 'none'):
                continue
            try:
                out[t][e] = float(m)
            except ValueError:
                continue
            n_mspd += 1
    if n_alive and n_mspd < 0.5 * n_alive:
        raise SystemExit('mspd coverage %.1f%% in %s (alive rows %d) - extractor '
                         'regression, refusing to build' % (100.0 * n_mspd / n_alive,
                                                             demo_base, n_alive))
    return out


def build(tag):
    k = np.load(os.path.join(DS, tag + SUF + 'k.npz'))['K']
    sol = np.where(k[:, 3] == 3)[0]
    Ks = k[sol]
    N = len(Ks)
    MS = np.full(N, np.nan, dtype=np.float32)
    MSd = np.full((N, W), np.nan, dtype=np.float32)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(Ks[i][0]), int(Ks[i][1]))].append(i)
    by_demo = defaultdict(list)
    for key, idxs in groups.items():
        by_demo[key[0]].append((key[1], idxs))
    h = np.arange(W)
    for did, ent_groups in sorted(by_demo.items()):
        base = DID2CSV[did]
        mspd = load_mspd(base, [e for e, _ in ent_groups])
        for ent, idxs in ent_groups:
            tk = np.array(sorted(t for t, d in mspd.items() if ent in d))
            if len(tk) == 0:
                continue
            col = np.array([mspd[t][ent] for t in tk], dtype=np.float64)
            pos = {t: j for j, t in enumerate(tk)}
            for i in idxs:
                t = int(Ks[i][2])
                jT = pos.get(t)
                if jT is None:
                    continue
                MS[i] = col[jT]
                jj = jT + h
                if jj[-1] < len(tk) and np.array_equal(tk[jj], t + h):
                    MSd[i] = col[jj]
        print('  %s did %d done' % (tag, did), flush=True)
    # class implied by the modal MS over the window (falls back to the label tick)
    CLS = np.zeros(N, dtype=np.int8)
    src = np.where(np.isfinite(MSd), MSd, MS[:, None])
    with np.errstate(invalid='ignore'):
        for i in range(N):
            v = src[i]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            vals, cnts = np.unique(v, return_counts=True)
            CLS[i] = CLS_OF_MSPD.get(float(vals[np.argmax(cnts)]), 0)
    np.savez_compressed(os.path.join(DS, tag + SUF + '_mspd.npz'), MS=MS, MSd=MSd, CLS=CLS)
    ok = np.isfinite(MS)
    print('%s N=%d  MS finite %.4f' % (tag, N, ok.mean()), flush=True)
    if ok.any():
        v = MS[ok]
        vals, cnts = np.unique(np.round(v, 1), return_counts=True)
        top = sorted(zip(vals, cnts), key=lambda t: -t[1])[:10]
        print('%s MS top values: %s' % (tag, ', '.join('%.1f:%.1f%%' % (a, 100.0 * b / len(v))
                                                       for a, b in top)), flush=True)
    cnt = np.bincount(CLS[ok].astype(int), minlength=10)
    print('%s implied class histogram (by mspd): %s' % (tag, cnt.tolist()), flush=True)
    print('%s soldier rows whose implied class != 3: %.4f' % (tag, (CLS[ok] != 3).mean()),
          flush=True)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'eval')
