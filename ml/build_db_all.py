"""DB build for all classes: ticks(demo,ent,tick,ox,oy,oz,grounded,eyey,cls).
Any class, alive only. Excludes .proj.csv/.hurt.csv sidecars.
dids come from did_registry.json (APPEND-ONLY, stable forever).
Incremental: skips demos already present (INSERT OR IGNORE also guards)."""
import csv, glob, os, sqlite3
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import didreg

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DB = os.path.join(DS, 'dataset_all.db')
# NOTE: never delete — incremental appends only (registry keeps dids stable)

con = sqlite3.connect(DB)
con.execute('PRAGMA journal_mode=WAL')
con.execute('PRAGMA synchronous=NORMAL')
con.execute('''CREATE TABLE IF NOT EXISTS ticks(demo SMALLINT, ent SMALLINT, tick INT,
  ox REAL, oy REAL, oz REAL, grounded TINYINT, eyey REAL, cls TINYINT,
  PRIMARY KEY(demo, ent, tick))''')

reg = didreg.load()
have = {r[0] for r in con.execute('SELECT DISTINCT demo FROM ticks').fetchall()}
files = sorted(f for f in glob.glob(os.path.join(D, '*.csv'))
               if not f.endswith('.proj.csv') and not f.endswith('.hurt.csv'))
# register any unseen files (append-only dids)
didreg.register([os.path.basename(f) for f in files])
reg = didreg.load()
for f in files:
    name = os.path.basename(f)
    did = reg[name]
    if did in have:
        print('skip (in DB)', name, flush=True)
        continue
    batch = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row['life'] != '0':
                continue
            try:
                batch.append((did, int(row['ent']), int(row['tick']),
                              float(row['ox']), float(row['oy']), float(row['oz']),
                              1 if (int(row['flags']) & 1) else 0,
                              float(row['eyey']), int(row['class'])))
            except ValueError:
                continue
    con.executemany('INSERT OR IGNORE INTO ticks VALUES (?,?,?,?,?,?,?,?,?)', batch)
    con.commit()
    print('loaded', name, len(batch), flush=True)
con.execute('CREATE INDEX IF NOT EXISTS ix_ticks ON ticks(demo, ent, tick)')
con.execute('VACUUM')
print('db MB:', os.path.getsize(DB) // 1048576, flush=True)
