"""Dataset vB: ALL classes + per-state IMPULSE SPLIT (knockback is external force,
not input: ground |A|>14*MAXV, air |A|>1500 -> drop sample). Uses dataset_all.db
(built by build_db_all.py). Series split on class change. Per-class maxspeed
norm + one-hot class features (9). Pure-input accel labels. trainB/evalB.npz."""
import csv, glob, os, sqlite3
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg
import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
os.makedirs(OUT, exist_ok=True)
# Generation switch: default reads dataset_all.db and writes the B files (frozen
# baseline - do not change). EYE generation sets ML_DB=dataset_eye.db,
# OUT_SUFFIX=C, EYE_FRAME=1 for the true view-relative frame.
DB = os.path.join(OUT, os.environ.get('ML_DB', 'dataset_all.db'))
SUF = os.environ.get('OUT_SUFFIX', 'B')
EYE_FRAME = os.environ.get('EYE_FRAME', '0') == '1'
# WEAPON_DIMS=1 (generation E): append 7 weapon dims AFTER the 14 base dims
# (warm-start contract: first 14 dims EXACTLY match B, new cols zero-init):
#   active slot one-hot (3: primary/secondary/melee, all-zero = unknown) +
#   active class id scalar (1) + loadout slot class ids (3, one per slot).
# Identity is the server class (defindex is NOT networked in STV demos -
# verified 2026-10-08; model-index props are polluted: arms parenting +
# stale values). Active slot+class come from m_hActiveWeapon resolution;
# loadout slots come from owner inversion (m_hOwner on weapon entities -
# verified: player simultaneously owns rocket AND shovel). Slot comes from
# the class->slot table below (no m_hMyWeapons in STV either). MG vs stock
# shovel share CTFShovel: accepted blur. C++ BuildWarmup mirrors this
# EXACTLY (same table, live entity classes; defid ignored for parity).
WEAPON_DIMS = os.environ.get('WEAPON_DIMS', '0') == '1'
# (slot, class_id). Soldier table v1: classes OBSERVED on cls=3 rows in
# dataset_eye.db (2026-10-08 census). Unknown classes -> (None, 0):
# all-zero slot + zero scalar (safe default, identical to pre-weapon
# behavior). MG/stock/Disciplinary/Equalizer share CTFShovel (accepted blur).
WCLASS = {
    'CTFRocketLauncher': (0, 1),
    'CTFRocketLauncher_DirectHit': (0, 2),
    'CTFRocketLauncher_AirStrike': (0, 3),
    'CTFParticleCannon': (0, 4),
    'CTFShotgun_Soldier': (1, 5),
    'CTFBuffItem': (1, 6),
    'CTFShovel': (2, 7),
    'CTFKatana': (2, 8),
    'CTFSword': (2, 9),
}
WCLS_SCALE = 16.0

# TF2 base run speeds hu/s by class id
MAXV_CLS = {1: 400.0, 2: 300.0, 3: 240.0, 4: 280.0, 5: 320.0,
            6: 230.0, 7: 300.0, 8: 300.0, 9: 300.0}

W = 66; DT = 1 / 66.0
AFK_WIN = 132          # +-2s window for AFK test
APPROACH = 66          # keep 1s approach ramp into movement (dodge onsets)
TELEPORT = 300.0       # hu single-step split
HOR = [7, 13, 20, 33, 66]  # 100/200/300/500/1000ms horizons
EVAL_NAMES = {'tfmge_2026-09-28__21-42', 'tfmge_2026-09-28__22-02',
              'tfmge_2026-09-28__23-02', 'tfmge_2026-09-29__00-02',
              'tfmge_2026-09-28__20-55'}

con = sqlite3.connect(DB)
con.execute('PRAGMA journal_mode=WAL')

files = sorted(f for f in glob.glob(os.path.join(D, '*.csv'))
               if not f.endswith('.proj.csv') and not f.endswith('.hurt.csv'))
demo_id = didreg.register([os.path.basename(f) for f in files])  # registry, never positional

# per (demo,ent) series -> AFK mask, segments, windows
def series(did, ent):
    if WEAPON_DIMS:
        q = con.execute('SELECT tick,ox,oy,oz,grounded,eyey,cls,wcls,w0,w1,w2 FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                        (did, ent)).fetchall()
    else:
        q = con.execute('SELECT tick,ox,oy,oz,grounded,eyey,cls FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                        (did, ent)).fetchall()
    return q

Xtr, Ytr, Wtr, Htr, Ktr, Xev, Yev, Wev, Hev, Kev = [], [], [], [], [], [], [], [], [], []
n_impulse = [0]
eye_var_total = eye_var_moved = 0
from collections import Counter, defaultdict
cls_counts = Counter()
# EVAL_ONLY=1: rebuild just the eval split (recovery path; train arrays stay
# empty and are NOT saved, so existing train files are untouched).
EVAL_ONLY = os.environ.get('EVAL_ONLY', '0') == '1'
# FILTER AUDIT (per class): ticks at each stage, so over-pruning is visible.
# stages: raw series ticks -> in ok-runs -> window candidates -> kept windows.
# A stage that eats a class disproportionately is a bug, not a filter.
audit_raw = Counter()    # ticks seen per class (known classes only)
audit_ok = Counter()     # ticks inside ok-runs
audit_cand = Counter()   # window candidates (label ticks reaching impulse check)
audit_imp = Counter()    # dropped by impulse cap
audit_keep = Counter()   # final windows
for name, did in sorted(demo_id.items(), key=lambda kv: kv[1]):
    is_eval = any(k in name for k in EVAL_NAMES)
    if EVAL_ONLY and not is_eval:
        continue
    ents = [r[0] for r in con.execute('SELECT DISTINCT ent FROM ticks WHERE demo=?', (did,))]
    for ent in ents:
        s = series(did, ent)
        # SHORT SERIES STAY: a window needs 66 warmup + 1 label tick (67). Future
        # horizons just go NaN where unavailable (masked in loss). The old 133
        # minimum threw away every short engagement (spawns, brief duels).
        if len(s) < W + 1:
            continue
        t = np.array([r[0] for r in s])
        o = np.array([[r[1], r[2], r[3]] for r in s])
        gr = np.array([r[4] for r in s], dtype=np.float64)
        ey = np.array([r[5] for r in s])
        cl = np.array([r[6] for r in s], dtype=np.int64)
        wc = np.array([r[7] if len(r) > 7 and r[7] else '' for r in s]) if WEAPON_DIMS else None
        wl = ([np.array([r[8 + k] if len(r) > 8 + k and r[8 + k] else '' for r in s]) for k in range(3)]
              if WEAPON_DIMS else None)
        for c in cl[(cl >= 1) & (cl <= 9)]:
            audit_raw[int(c)] += 1
        # consecutive + teleport + CLASS-CHANGE split (history invalid across classes)
        # + known-class only (cls 0 = unclassified entity, not trainable)
        consec = np.ediff1d(t, to_begin=2) == 1
        samecls = np.ediff1d(cl, to_begin=0) == 0
        knowncls = (cl >= 1) & (cl <= 9)
        step = np.linalg.norm(np.diff(o, axis=0, prepend=o[0:1]), axis=1)
        tele = step > TELEPORT
        # AFK: max displacement in +-AFK_WIN window
        keep = np.zeros(len(s), dtype=bool)
        # AFK: any real step (>0.5hu) within +-AFK_WIN ticks? O(n) via cumsum dilation.
        raw = step > 0.5
        cs = np.concatenate([[0], np.cumsum(raw.astype(np.int64))])
        idx_all = np.arange(len(s))
        a = np.clip(idx_all - AFK_WIN, 0, len(s) - 1)
        b = np.clip(idx_all + AFK_WIN + 1, 0, len(s))
        # count of moving ticks in [i-WIN, i+WIN]: careful with edges; use searchsorted-free approx:
        moving = (cs[b] - cs[a]) > 0
        keep = np.zeros(len(s), dtype=bool)
        # keep moving + APPROACH-tick ramp before each moving stretch (dodge onsets)
        idx = np.where(moving)[0]
        keep[idx] = True
        if len(idx):
            # ramp: for each moving tick, keep APPROACH ticks before it (vectorized via cumsum)
            mk = np.zeros(len(s) + 1, dtype=np.int64)
            mk[idx] += 1
            cov = np.cumsum(mk)
            # keep[i] if any moving tick in (i, i+APPROACH]
            b2 = np.clip(idx_all + APPROACH + 1, 0, len(s))
            keep |= (cov[b2] - cov[idx_all]) > 0
        # eye freshness check (global stats)
        eye_var_total += 1
        if np.nanstd(ey) > 1.0:
            eye_var_moved += 1
        # segments of kept & consecutive & no-teleport & same-class & known-class
        ok = keep & consec & (~tele) & samecls & knowncls
        # derived velocity (backward diff, hu/s) — networked velocity is stale in demos
        v = np.diff(o, axis=0, prepend=o[0:1])[:, :2] / DT
        j = 0
        while j < len(s):
            if not ok[j]:
                j += 1
                continue
            k = j
            while k < len(s) and ok[k]:
                k += 1
            run = np.arange(j, k)  # kept consecutive run
            rcls = Counter(int(c) for c in cl[run] if 1 <= int(c) <= 9)
            for c, n in rcls.items():
                audit_ok[c] += n
            if len(run) >= W + 1:
                for st in range(0, len(run) - W, 4):
                    w = run[st:st + W]
                    lt = run[st + W]
                    cls = int(cl[lt])
                    MAXV = MAXV_CLS.get(cls, 300.0)
                    cls_counts[cls] += 1
                    audit_cand[cls] += 1
                    # ego frame: velocity heading when moving, else eye yaw.
                    # (No eye-std threshold: it was knife-edge unstable (0.5) and
                    # made old demos (frozen eye->velocity) inconsistent with new
                    # demos (live eye->view). Motion direction is the frame when
                    # moving; view when still. Deterministic, window-free.)
                    yaw = ey[lt]
                    if not np.isfinite(yaw):
                        yaw = 0.0
                    if EYE_FRAME:
                        # true view-relative frame (live eye data): use it unless
                        # the eye itself is missing, then velocity heading
                        if not np.isfinite(ey[lt]):
                            yaw = (np.rad2deg(np.arctan2(v[lt, 1], v[lt, 0]))
                                   if np.linalg.norm(v[lt]) > 1.0 else 0.0)
                    elif np.linalg.norm(v[lt]) > 50:
                        yaw = np.rad2deg(np.arctan2(v[lt, 1], v[lt, 0]))
                    a = np.deg2rad(yaw)
                    c, sn = np.cos(-a), np.sin(-a)
                    R = np.array([[c, -sn], [sn, c]])
                    ve = (v[w] @ R.T) / MAXV
                    sp = np.linalg.norm(v[w], axis=1) / MAXV
                    dy = np.deg2rad(np.ediff1d(np.nan_to_num(ey[w]), to_begin=0.0))
                    dy = (dy + np.pi) % (2 * np.pi) - np.pi
                    oh = np.zeros(9, dtype=np.float32)
                    if 1 <= cls <= 9:
                        oh[cls - 1] = 1.0
                    Xb = np.stack([ve[:, 0], ve[:, 1], sp, gr[w], dy], axis=1).astype(np.float32)
                    X = np.concatenate([Xb, np.repeat(oh[None, :], W, axis=0)], axis=1)
                    if WEAPON_DIMS:
                        # per-tick window weapon feats (history, like everything
                        # else): active slot one-hot + active class scalar +
                        # loadout slot class scalars (owner inversion).
                        wsl = np.zeros((W, 3), dtype=np.float32)
                        wci = np.zeros((W, 1), dtype=np.float32)
                        wld = np.zeros((W, 3), dtype=np.float32)
                        for ti, cname in enumerate(wc[w]):
                            se = WCLASS.get(cname)
                            if se is not None and se[0] is not None:
                                wsl[ti, se[0]] = 1.0
                                wci[ti, 0] = se[1] / WCLS_SCALE
                        for k in range(3):
                            for ti, cname in enumerate(wl[k][w]):
                                se = WCLASS.get(cname)
                                if se is not None:
                                    wld[ti, k] = se[1] / WCLS_SCALE
                        X = np.concatenate([X, wsl, wci, wld], axis=1)
                    # PURE INPUT accel (behavior, not physics): cancel TF2 ground friction
                    # (drop = max(speed,100)*sv_friction(4)*dt). Air: dV/dt is already pure.
                    v0r = v[lt]
                    v1r = v[lt + 1] if (lt + 1 < len(s) and ok[lt + 1]) else v0r
                    A = (v1r - v0r) / DT
                    s0 = float(np.linalg.norm(v0r))
                    if gr[lt] > 0.5 and s0 > 1.0:
                        A = A + (v0r / s0) * max(s0, 100.0) * 4.0
                    # IMPULSE SPLIT: explosion knockback is external force, not input.
                    # Ground input caps at ~accel(10)*wishspeed + friction residual
                    # ~= 14*MAXV; air input is small ONLY in the sense of no
                    # friction - the accel ceiling is the same 10*MAXV, so the
                    # air cap is 10*MAXV (NOT 1500: that ate legit air-strafes,
                    # which must be learned). Drop, don't learn.
                    cap = (14.0 * MAXV if gr[lt] > 0.5 else 10.0 * MAXV)
                    if float(np.linalg.norm(A)) > cap:
                        n_impulse[0] += 1
                        audit_imp[cls] += 1
                        continue
                    Y = (R @ (A / MAXV)).astype(np.float32)
                    wt = float(np.clip(sp.mean() / 0.5, 0.1, 1.0))
                    H = np.full((len(HOR), 2), np.nan, dtype=np.float32)
                    for hi, h in enumerate(HOR):
                        if lt + h < len(s) and ok[lt + 1:lt + h + 1].all():
                            H[hi] = R @ (o[lt + h, :2] - o[lt, :2])
                    if is_eval:
                        Xev.append(X); Yev.append(Y); Wev.append(wt); Hev.append(H); Kev.append((did, ent, int(t[lt]), cls))
                    else:
                        Xtr.append(X); Ytr.append(Y); Wtr.append(wt); Htr.append(H); Ktr.append((did, ent, int(t[lt]), cls))
                    audit_keep[cls] += 1
            j = k
print('class window counts:', dict(sorted(cls_counts.items())), flush=True)
print('series with varying eye yaw: %d/%d' % (eye_var_moved, eye_var_total), flush=True)

def save(tag, X, Y, W8, H, K):
    X = np.array(X); Y = np.array(Y); W8 = np.array(W8); H = np.array(H); K = np.array(K)
    np.savez_compressed(os.path.join(OUT, tag + SUF + '.npz'), X=X, Y=Y, W=W8, H=H)
    np.savez_compressed(os.path.join(OUT, tag + SUF + 'k.npz'), K=K)
    print(tag + SUF, X.shape, 'meanW=%.3f' % W8.mean(), flush=True)

save('train', Xtr, Ytr, Wtr, Htr, Ktr) if not EVAL_ONLY else None
save('eval', Xev, Yev, Wev, Hev, Kev)
print('impulse-split dropped samples:', n_impulse[0])
print('FILTER AUDIT per class: raw -> ok-runs -> candidates -> impulse-dropped -> kept')
print(' cls |     raw |      ok |    cand |     imp |    kept | kept/raw')
for c in range(1, 10):
    r, o, d, im, kp = (audit_raw[c], audit_ok[c], audit_cand[c],
                       audit_imp[c], audit_keep[c])
    print(' %3d | %7d | %7d | %7d | %7d | %7d | %.3f' % (
        c, r, o, d, im, kp, kp / max(r, 1)))
print('done, no VACUUM (shared DB)')
