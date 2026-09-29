"""DB rebuild for all classes: ticks(demo,ent,tick,ox,oy,oz,grounded,eyey,cls).
Any class, alive only. Excludes .proj.csv sidecars."""
import csv, glob, os, sqlite3

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
DB = os.path.join(DS, 'dataset_all.db')
if os.path.exists(DB):
    os.remove(DB)

con = sqlite3.connect(DB)
con.execute('PRAGMA journal_mode=WAL')
con.execute('PRAGMA synchronous=NORMAL')
con.execute('''CREATE TABLE ticks(demo SMALLINT, ent SMALLINT, tick INT,
  ox REAL, oy REAL, oz REAL, grounded TINYINT, eyey REAL, cls TINYINT,
  PRIMARY KEY(demo, ent, tick))''')

files = sorted(f for f in glob.glob(os.path.join(D, '*.csv')) if not f.endswith('.proj.csv'))
for f in files:
    name = os.path.basename(f)
    did = files.index(f)
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
con.execute('CREATE INDEX ix_ticks ON ticks(demo, ent, tick)')
con.execute('VACUUM')
print('db MB:', os.path.getsize(DB) // 1048576, flush=True)
