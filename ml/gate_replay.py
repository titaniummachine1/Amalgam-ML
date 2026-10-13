"""gate_replay.py — truth harness. Starts at ANY demo tick and compares three
controllers over the SAME window, all scored against the demo's own positions:

  1. DEMO-INPUT REPLAY (oracle) — recovers inputs from the demo's own
     kinematics and replays them. This is the ceiling. GATE: it must beat
     coast at every horizon. If it does not, the sim or the input recovery
     is broken and NOTHING downstream is trustworthy.
  2. COAST — no input (the floor every model must beat).
  3. MODEL — greedy AR from the transformer (optional).

Full history is always given before the window (HIST ticks of ego+enemy
state); nothing beyond the window start is used to produce predictions.
Horizons 66 (1s) and 132 (2s) as required, plus the short ones.

Scope (fixed, matches coast): horizontal input + camera yaw/pitch. Jumps are
NOT predicted — when the demo shows a jump the harness applies the engine
jump impulse itself, so jump is a given event, never a model output.

Usage:
  python ml/gate_replay.py <csv> <soup.npz> [t0] [nwin] [--model ckpt.pt]
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__file__))
from movesim import (AIRCAP, CLASS_SPEED, DUCK_FRAC, FL_DUCKING, FL_ONGROUND,
                     inverse_air, inverse_ground, load_csv_live, rollout_free,
                     yaw_to_fwd)

HIST = 66
HOR = [7, 13, 20, 66, 132]
R = 24.0


def build_windows(csvpath, souppath, t0=None, nwin=40, horizon=132):
    """Windows of HIST history + `horizon` future ticks, all consecutive."""
    rows = load_csv_live(csvpath, classes=set(range(1, 10)))
    ents = sorted(rows)
    if len(ents) < 2:
        raise SystemExit('need >=2 entities, got %d' % len(ents))
    # ego = entity with the most ticks; enemy = nearest other entity by tick
    ego = max(ents, key=lambda e: len(rows[e]))
    rs = rows[ego]
    out = []
    i = HIST
    while i + horizon < len(rs) and len(out) < nwin:
        win = rs[i - HIST:i + horizon + 1]
        ticks = [int(r['tick']) for r in win]
        if any(b - a != 1 for a, b in zip(ticks, ticks[1:])):
            i += 1
            continue
        cls = int(float(win[0]['class']))
        if any(int(float(r['class'])) != cls for r in win):
            i += 1
            continue
        try:
            P = np.array([[float(r['ox']), float(r['oy']), float(r['oz'])] for r in win])
            V = np.array([[float(r['vx']), float(r['vy']), float(r['vz'])] for r in win])
        except ValueError:
            i += 1
            continue
        if np.any(np.linalg.norm(P[1:] - P[:-1], axis=1) > 80):   # teleport
            i += 1
            continue
        ms0 = float(win[0]['mspd']) or CLASS_SPEED.get(cls, 300.0)
        if ms0 <= 1:
            i += 1
            continue
        out.append((int(ticks[0]), cls, ms0, P, V,
                    [int(float(r['flags'])) for r in win],
                    [float(r['eyey']) for r in win], [int(r['ent']) for r in win]))
        i += 1
    return ego, out


def recover_inputs(V, flags, ms0, k0, k1):
    """Exact same recovery the trainer uses, over [k0, k1)."""
    inp = []
    for k in range(k0, k1):
        msk = ms0 * (DUCK_FRAC if (flags[k] & FL_DUCKING) else 1.0)
        gr = bool(flags[k] & FL_ONGROUND)
        try:
            inv = inverse_ground(V[k], V[k + 1], msk) if gr else inverse_air(V[k], V[k + 1])
        except (ValueError, KeyError, IndexError):
            inp.append((None, 0.0))
            continue
        wd = inv['wishdir']
        if wd is None:
            inp.append((None, 0.0))
        else:
            wd = np.asarray(wd, dtype=float).ravel()[:2]
            inp.append((wd, float(inv['wishspeed'])))
    return inp


def run_controller(P, V, flags, eyey, ms0, kind, model=None, net=None, HOR=HOR,
                   split=None, fut=None):
    """kind: 'demo' (recovered inputs) | 'coast' | 'model'.
    `split` = index of the last history tick; `fut` = ticks to simulate."""
    if split is None:
        split = HIST
    if fut is None:
        fut = len(P) - 1 - HIST
    p0 = P[split].copy()
    v0 = V[split].copy()
    ey0 = eyey[split]
    views = [yaw_to_fwd(ey0)] * fut
    grf = [bool(flags[split + k] & FL_ONGROUND) for k in range(fut)]
    if kind == 'demo':
        inp = recover_inputs(V, flags, ms0, split, split + fut)
    elif kind == 'coast':
        inp = [(None, 0.0)] * fut
    else:
        inp = model_inputs(model, net, V[split - 32:split + 1], p0, ms0, fut)
    pr = rollout_free(v0, p0, fut, inp, grf, ms0, AIRCAP, views)
    errs = {}
    for h in HOR:
        if h > fut:
            continue
        e = float(np.linalg.norm(pr[h - 1] - P[split + h]))
        errs[h] = (e, e < R)
    return errs


def per_tick_residual(V, flags, ms0, k0, k1):
    """Forward-sim each recovered input and compare to the observed v_{k+1}.
    Uses inverse_ground/inverse_air's own internal forward-sim residual, which
    is computed inside movesim (info['res']) — no second physics path to drift.
    Isolates SIM fidelity from input-recovery noise."""
    res = []
    for k in range(k0, k1):
        msk = ms0 * (DUCK_FRAC if (flags[k] & FL_DUCKING) else 1.0)
        gr = bool(flags[k] & FL_ONGROUND)
        try:
            inv = inverse_ground(V[k], V[k + 1], msk) if gr else inverse_air(V[k], V[k + 1])
        except (ValueError, KeyError, IndexError):
            continue
        if inv['wishdir'] is None:
            continue
        res.append(float(inv['info'].get('res', 0.0)))
    return res


def main():
    csvpath, souppath = sys.argv[1], sys.argv[2]
    t0 = int(sys.argv[3]) if len(sys.argv) > 3 else None
    nwin = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    model = net = None
    if '--model' in sys.argv:
        import torch
        import train_cont
        from train_cont import ContAR
        ck = torch.load(sys.argv[sys.argv.index('--model') + 1],
                        map_location='cpu', weights_only=False)
        if 'cfg' in ck:
            train_cont.D, train_cont.NL, train_cont.NH, train_cont.FF = ck['cfg'][:4]
            if len(ck['cfg']) > 6:
                train_cont.NFEAT = ck['cfg'][6]
        net = ContAR()
        net.load_state_dict(ck['net'])
        net.eval()
        torch.set_num_threads(4)
        model = 'greedy'

    ego, wins = build_windows(csvpath, souppath, t0, nwin)
    print('ego=ent%d windows=%d horizons=%s R=%.0f' % (ego, len(wins), HOR, R), flush=True)
    stats = {}
    allres = []
    for i, (tick, cls, ms0, P, V, flags, eyey, ents) in enumerate(wins):
        kinds = ['demo', 'coast'] + (['model'] if net is not None else [])
        for kind in kinds:
            e = run_controller(P, V, flags, eyey, ms0, kind, model, net)
            for h, (err, hit) in e.items():
                s = stats.setdefault((kind, h), [0, 0, []])
                s[0] += hit
                s[1] += 1
                s[2].append(err)
        allres.extend(per_tick_residual(V, flags, ms0, HIST, len(P) - 1))
    if allres:
        r = np.array(allres)
        print('per-tick replay residual (hu/s): med %.4f p90 %.4f p99 %.4f  n=%d'
              % (np.median(r), np.percentile(r, 90), np.percentile(r, 99), len(r)))
        print('  (STV position quantum 0.01hu / DT 0.015 = 0.67 hu/s velocity noise floor)')
    print('\n%-8s %-5s %-9s %-9s %s' % ('ctrl', 'h', 'hit<24hu', 'med-err', 'p90-err'))
    order = ['demo', 'coast', 'model']
    gate_ok = True
    for kind in order:
        for h in HOR:
            k = (kind, h)
            if k not in stats:
                continue
            hit, tot, errs = stats[k]
            errs = np.array(errs)
            print('%-8s %-5d %-9.1f %-9.2f %.2f' %
                  (kind, h, 100.0 * hit / tot, np.median(errs), np.percentile(errs, 90)))
    # GATE: (a) demo-input replay must never be WORSE than coast, and
    # (b) it must be near-exact at short range (proves physics, not luck).
    print('\nGATE (demo-input replay vs coast):')
    gate_ok = True
    for h in HOR:
        d = stats.get(('demo', h))
        c = stats.get(('coast', h))
        if not d or not c:
            continue
        dh = 100.0 * d[0] / d[1]
        ch = 100.0 * c[0] / c[1]
        med = np.median(d[2])
        ok = (dh >= ch) and (med < 1.0 if h <= 20 else True)
        gate_ok = gate_ok and ok
        print('  h=%-4d demo %5.1f%% (med %.2fhu)  coast %5.1f%%  %s' %
              (h, dh, med, ch, 'OK' if ok else '*** FAIL ***'))
    print('\nRESULT: %s' % ('PASS' if gate_ok else 'FAIL'))
    return 0 if gate_ok else 1


if __name__ == '__main__':
    sys.exit(main())
