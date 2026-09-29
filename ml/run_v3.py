import subprocess, sys
T = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
LOG = open(T + r'\run_v3.log', 'a')
for s in ['train3.py', 'finetune3.py', 'tail_eval3.py']:
    LOG.write(f'===== {s} =====\n')
    LOG.flush()
    r = subprocess.run([sys.executable, '-u', T + '\\' + s],
                       stdout=LOG, stderr=subprocess.STDOUT)
    LOG.write(f'===== {s} exit={r.returncode} =====\n')
    LOG.flush()
    if r.returncode != 0:
        break
