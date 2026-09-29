"""Dataset v2: origin-derived velocity, bidirectional AFK slice, movement weights,
multi-horizon labels. Stores kept ticks in SQLite (Turso-compatible file), exports
train/eval windows to npz."""
import csv, glob, os, sqlite3
import numpy as np

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
os.makedirs(OUT, exist_ok=True)
DB = os.path.join(OUT, 'dataset.db')
if os.path.exists(DB):
    os.remove(DB)

W = 66; DT = 1 / 66.0; MAXV = 240.0
AFK_WIN = 132          # +-2s window for AFK test
AFK_DIST = 16.0        # hu displacement threshold
APPROACH = 66          # keep 1s approach ramp into movement (dodge onsets)
TELEPORT = 300.0       # hu single-step split
HOR = [7, 13, 20, 33, 66]  # 100/200/300/500/1000ms horizons
EVAL_NAMES = {'tfmge_2026-09-28__21-42', 'tfmge_2026-09-28__22-02',
              'tfmge_2026-09-28__23-02', 'tfmge_2026-09-29__00-02',
              'tfmge_2026-09-28__20-55'}

con = sqlite3.connect(DB)
con.execute('PRAGMA journal_mode=WAL')
con.execute('''CREATE TABLE ticks(demo SMALLINT, ent TINYINT, tick INT,
  ox REAL, oy REAL, oz REAL, grounded TINYINT, eyey REAL,
  PRIMARY KEY(demo, ent, tick))''')

files = sorted(f for f in glob.glob(os.path.join(D, '*.csv')) if not f.endswith('.proj.csv'))
eye_var_total = eye_var_moved = 0
demo_id = {}
for f in files:
    name = os.path.basename(f)
    did = demo_id.setdefault(name, len(demo_id))
    batch = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0' or row['class'] != '3':
                continue
            try:
                batch.append((did, int(row['ent']), int(row['tick']),
                              float(row['ox']), float(row['oy']), float(row['oz']),
                              1 if (int(row['flags']) & 1) else 0, float(row['eyey'])))
            except ValueError:
                continue
    con.executemany('INSERT OR IGNORE INTO ticks VALUES (?,?,?,?,?,?,?,?)', batch)
    con.commit()
    print('loaded', name, len(batch), flush=True)
con.execute('CREATE INDEX ix_ticks ON ticks(demo, ent, tick)')

# per (demo,ent) series -> AFK mask, segments, windows
def series(did, ent):
    q = con.execute('SELECT tick,ox,oy,oz,grounded,eyey FROM ticks WHERE demo=? AND ent=? ORDER BY tick',
                    (did, ent)).fetchall()
    return q

Xtr, Ytr, Wtr, Htr, Xev, Yev, Wev, Hev = [], [], [], [], [], [], [], []
for name, did in sorted(demo_id.items(), key=lambda kv: kv[1]):
    is_eval = any(k in name for k in EVAL_NAMES)
    ents = [r[0] for r in con.execute('SELECT DISTINCT ent FROM ticks WHERE demo=?', (did,))]
    for ent in ents:
        s = series(did, ent)
        if len(s) < W + 67:
            continue
        t = np.array([r[0] for r in s])
        o = np.array([[r[1], r[2], r[3]] for r in s])
        gr = np.array([r[4] for r in s], dtype=np.float64)
        ey = np.array([r[5] for r in s])
        # consecutive + teleport split
        consec = np.ediff1d(t, to_begin=2) == 1
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
        # segments of kept & consecutive & no-teleport
        ok = keep & consec & (~tele)
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
            if len(run) >= W + 67:
                for st in range(0, len(run) - W - 66, 4):
                    w = run[st:st + W]
                    lt = run[st + W]
                    # ego frame: eye yaw at label tick, fallback velocity heading
                    yaw = ey[lt]
                    if not np.isfinite(yaw):
                        yaw = 0.0
                    if np.nanstd(ey[w]) < 0.5 and np.linalg.norm(v[lt]) > 50:
                        yaw = np.rad2deg(np.arctan2(v[lt, 1], v[lt, 0]))
                    a = np.deg2rad(yaw)
                    c, sn = np.cos(-a), np.sin(-a)
                    R = np.array([[c, -sn], [sn, c]])
                    ve = (v[w] @ R.T) / MAXV
                    sp = np.linalg.norm(v[w], axis=1) / MAXV
                    dy = np.deg2rad(np.ediff1d(np.nan_to_num(ey[w]), to_begin=0.0))
                    dy = (dy + np.pi) % (2 * np.pi) - np.pi
                    X = np.stack([ve[:, 0], ve[:, 1], sp, gr[w], dy], axis=1).astype(np.float32)
                    acc = ((v[lt + 1] - v[lt]) / DT / MAXV) if (lt + 1 < len(s) and ok[lt + 1]) else np.zeros(2)
                    Y = (R @ acc).astype(np.float32)
                    wt = float(np.clip(sp.mean() / 0.5, 0.1, 1.0))
                    H = np.full((len(HOR), 2), np.nan, dtype=np.float32)
                    for hi, h in enumerate(HOR):
                        if lt + h < len(s) and ok[lt + 1:lt + h + 1].all():
                            H[hi] = R @ (o[lt + h, :2] - o[lt, :2])
                    if is_eval:
                        Xev.append(X); Yev.append(Y); Wev.append(wt); Hev.append(H)
                    else:
                        Xtr.append(X); Ytr.append(Y); Wtr.append(wt); Htr.append(H)
            j = k
print('series with varying eye yaw: %d/%d' % (eye_var_moved, eye_var_total), flush=True)

def save(tag, X, Y, W8, H):
    X = np.array(X); Y = np.array(Y); W8 = np.array(W8); H = np.array(H)
    np.savez_compressed(os.path.join(OUT, tag + '2.npz'), X=X, Y=Y, W=W8, H=H)
    print(tag, X.shape, 'meanW=%.3f' % W8.mean(), flush=True)

save('train', Xtr, Ytr, Wtr, Htr)
save('eval', Xev, Yev, Wev, Hev)
con.execute('VACUUM')
print('db MB:', os.path.getsize(DB) // 1048576)
