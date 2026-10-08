import os, urllib.request, http.cookiejar, re

OUT_MGE = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
OUT_OR = os.path.join(os.path.expanduser('~'), 'Downloads', 'orange_demos')
os.makedirs(OUT_MGE, exist_ok=True)
os.makedirs(OUT_OR, exist_ok=True)

# (drive_id, filename, outdir) — 21 new MGE + 3 orange
DEMOS = [
 ('11NJji_HtIgw04Nvi8UzHdUpeznChz6yv', 'tfmge_2026-09-30__19-40_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1ZUgmLBUjplERvADW3bOGK1Xrg46CRfHE', 'tfmge_2026-09-30__19-04_mge_training_v8_beta4b.dem', OUT_MGE),
 ('103IsDgJIlZ5vMSBglmEbDyp3gn7LXMph', 'tfmge_2026-09-30__19-03_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1i74GgQCPdCdSIVF3ULoJ95Zw1LHBsOwL', 'tfmge_2026-09-30__18-49_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1tyhqH0QC7HVV7euFyF3IHYe7e9_Zd2ML', 'tfmge_2026-09-30__18-39_mge_training_v8_beta4b.dem', OUT_MGE),
 ('11EhmBbdMSL-a_8mRwVYI5qM64ycTe3Tk', 'tfmge_2026-09-30__18-14_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1q4tkVHF3k38yYuVcmOxdhmVUsle0L_y1', 'tfmge_2026-09-30__17-14_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1jddc_ZDWJXQuBjkQTUgdnDR6rsIuB_5P', 'tfmge_2026-09-30__16-08_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1KeXK9zAMHf-snt9da_LJkmsw7GGRiRMr', 'tfmge_2026-09-30__16-06_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1NU8RXDoqoompaWVUpBKF3YSZ1hOiCc8k', 'tfmge_2026-09-30__14-25_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1rXE10DnT4K48ETnp6agztQTZ6XjFEzhh', 'tfmge_2026-09-29__21-59_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1fnU04dbSKbYc0XrCTGpN83_Yh7g8QOI5', 'tfmge_2026-09-29__21-58_mge_training_v8_beta4b.dem', OUT_MGE),
 ('112V3biUqjeRF8dKpmKknnHa5RSIxSUVa', 'tfmge_2026-09-29__21-09_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1aqofifm1sK_s7QyfhCNBpoCuTGgV_gNG', 'tfmge_2026-09-29__20-09_mge_training_v8_beta4b.dem', OUT_MGE),
 ('14ehU9UwRqEKxNW7cYib-bPT7CAwB6ZiW', 'tfmge_2026-09-29__19-09_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1WGtLQVg508mcldf1NAZupow5EP4eVAwA', 'tfmge_2026-09-29__19-03_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1_GHcZc-CsF8BFghNLaoWm6u3mPHSPYwt', 'tfmge_2026-09-29__17-40_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1h9GOkr2bBQb8bYQ2xngx6mnweSu8Gp7B', 'tfmge_2026-09-29__17-24_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1OIBMJxF1gqo9RtXS8qKt4ZUm7trYFK-v', 'tfmge_2026-09-29__17-19_mge_training_v8_beta4b.dem', OUT_MGE),
 ('12ak1Hqbbcbqy5cgMjE1HocFbeQjtTkzF', 'tfmge_2026-09-29__16-48_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1RdlVSu_z-wzzKHYxpyDnE2FYknZx0qDQ', 'tfmge_2026-09-29__15-17_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1bIVUU3AVuK8Pafucy4Tk43XBQnFwNfFx', 'tforange_2026-09-30__19-55_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1m5_gF_NuWYygyfMg2-i_k5-BSmIMxOvZ', 'tforange_2026-09-30__16-58_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1tVzwYrvNJXNxbVc4tqG-4XZTKDj_2qus', 'tforange_2026-09-29__22-48_cp_orange_x5_bg_v34.dem', OUT_OR),
]

def download(fid, dest):
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [('User-Agent', 'Mozilla/5.0')]
    url = 'https://drive.google.com/uc?export=download&id=' + fid
    r = op.open(url)
    data = r.read()
    ctype = r.headers.get_content_type()
    if b'confirm' in data[:200000] and ('text/html' in ctype):
        m = re.search(rb'confirm=([0-9A-Za-z-_]+)', data)
        if m:
            url = 'https://drive.google.com/uc?export=download&confirm=%s&id=%s' % (m.group(1).decode(), fid)
            r = op.open(url)
            data = r.read()
        else:
            m2 = re.search(rb'name="uuid" value="([^"]+)"', data)
            if m2:
                q = urllib.parse.urlencode({'id': fid, 'export': 'download', 'confirm': 't', 'uuid': m2.group(1).decode()})
                r = op.open('https://drive.google.com/uc?' + q)
                data = r.read()
    ok = data[:8] == b'HL2DEMO\x00'
    if ok:
        with open(dest, 'wb') as f:
            f.write(data)
    return ok, len(data)

fails = []
for i, (fid, name, outdir) in enumerate(DEMOS, 1):
    dest = os.path.join(outdir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        print(f'[{i}/{len(DEMOS)}] skip (exists) {name}')
        continue
    try:
        ok, n = download(fid, dest)
        print(f'[{i}/{len(DEMOS)}] {"OK " if ok else "FAIL"} {name} bytes={n}')
        if not ok:
            fails.append(name)
            if os.path.exists(dest):
                os.remove(dest)
    except Exception as e:
        print(f'[{i}/{len(DEMOS)}] ERR {name}: {e}')
        fails.append(name)
print('FAILS:', fails if fails else 'none')
