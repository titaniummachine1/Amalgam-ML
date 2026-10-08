"""Verify H0: did == index in sorted list of the 25 ORIGINAL player CSVs.
Reconstruct the original list (mge_csv player CSVs minus any that postdate the
DB build), then spot-check DB values vs CSV for did 0, 12, 24."""
import csv
import glob
import os
import sqlite3

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_csv')
DS = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
con = sqlite3.connect(os.path.join(DS, 'dataset_all.db'))

files = sorted(f for f in glob.glob(os.path.join(D, '*.csv'))
               if not f.endswith('.proj.csv') and not f.endswith('.hurt.csv'))
print('current player CSVs:', len(files))
# original 25 = tfmge_ only (orange came later); print them
orig = [f for f in files if os.path.basename(f).startswith('tfmge_')]
print('tfmge_ CSVs:', len(orig))
for did in (0, 12, 24):
    if did >= len(orig):
        print(f'did {did}: out of range')
        continue
    f = orig[did]
    print(f'--- did {did} ?= {os.path.basename(f)[:40]}')
    # first DB row for this did
    row = con.execute('SELECT ent,tick,ox,oy FROM ticks WHERE demo=? ORDER BY tick LIMIT 1',
                      (did,)).fetchone()
    if row is None:
        print('  NO DB ROWS for did', did)
        continue
    ent, tick, ox, oy = row
    # find matching row in CSV
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r['life'] != '0':
                continue
            if int(r['ent']) == ent and int(r['tick']) == tick:
                match = (abs(float(r['ox']) - ox) < 0.01 and abs(float(r['oy']) - oy) < 0.01)
                print(f'  ent={ent} tick={tick} DB=({ox},{oy}) CSV=({r["ox"]},{r["oy"]}) MATCH={match}')
                break
        else:
            print('  row not found in CSV')
# distinct dids in DB
print('distinct dids in DB:', con.execute('SELECT COUNT(DISTINCT demo) FROM ticks').fetchone())
print('max did in DB:', con.execute('SELECT MAX(demo) FROM ticks').fetchone())
