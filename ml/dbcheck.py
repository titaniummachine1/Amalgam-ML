import sqlite3
con = sqlite3.connect(r'C:\Users\Terminatort8000\Downloads\mge_ds\dataset_all.db')
print(con.execute("SELECT sql FROM sqlite_master WHERE name='ticks'").fetchone())
print('rows:', con.execute('SELECT COUNT(*) FROM ticks').fetchone())
print('demos:', con.execute('SELECT COUNT(DISTINCT demo) FROM ticks').fetchone())
