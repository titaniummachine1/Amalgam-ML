import os
import sqlite3

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(D, 'dataset_all.db'))
rows = con.execute(
    'SELECT tick,ox,oy,oz,grounded FROM ticks WHERE demo=39 AND ent=2 AND tick BETWEEN 166050 AND 166075 ORDER BY tick').fetchall()
px, py = None, None
for t, x, y, z, g in rows:
    d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 if px is not None else 0
    print(t, 'd=%.1f' % d, 'pos=(%.1f,%.1f,%.1f)' % (x, y, z), 'gr=%s' % g)
    px, py = x, y
print('db rows:', len(rows))
