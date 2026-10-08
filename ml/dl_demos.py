import os, sys, urllib.request, urllib.parse, http.cookiejar

OUT = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
os.makedirs(DEMOS_OUT := OUT, exist_ok=True)

# (drive_id, filename) in page order; #1 (00-02) already in Downloads, skipped
DEMOS = [
 ('1tY8nuEBUS6NwOd8fbuUtew1m8ixOCpYw', 'tfmge_2026-09-28__23-02_mge_training_v8_beta4b.dem'),
 ('1-RmG1F7QuJDi9SWPsSO20tT72P_jtWCF', 'tfmge_2026-09-28__22-02_mge_training_v8_beta4b.dem'),
 ('1PIJg37vcnsZKSEZCL82w-F5k-1VuZqO5', 'tfmge_2026-09-28__21-42_mge_training_v8_beta4b.dem'),
 ('1tMz0OrvXdXDDrZtVZOToYP2uaIBZZo3P', 'tfmge_2026-09-28__20-55_mge_training_v8_beta4b.dem'),
 ('1CliufvFz78KCQ6coYXzVvdU_-UUxOPfk', 'tfmge_2026-09-28__19-29_mge_training_v8_beta4b.dem'),
 ('18J59eKQVQU5KV5lC5OxC_Q_R1_LFU4vv', 'tfmge_2026-09-28__19-28_mge_training_v8_beta4b.dem'),
 ('10ZASNVlEQmhSYp7mxN4M7ySzQMxLnYbs', 'tfmge_2026-09-28__18-20_mge_training_v8_beta4b.dem'),
 ('1uBjWB-ezs9y8_RtdQXYDb1gr8GR24PJv', 'tfmge_2026-09-28__17-20_mge_training_v8_beta4b.dem'),
 ('1ZqHhze1Yy43Tj7qEUCNQvUYQ3h8Y3dZI', 'tfmge_2026-09-28__17-18_mge_training_v8_beta4b.dem'),
 ('1JCcN8gdSGrsqKtGbDv8STGHfk9KS1a9z', 'tfmge_2026-09-28__16-58_mge_training_v8_beta4b.dem'),
 ('1x3CVOow4fm_w4m516ssXWCmjPC6Mo5Bb', 'tfmge_2026-09-28__16-56_mge_training_v8_beta4b.dem'),
 ('1ATgtCqCelCVwVVskIWuR6Mz3sFWBvKmD', 'tfmge_2026-09-28__16-39_mge_training_v8_beta4b.dem'),
 ('1vk_m-4kbbkpSGntWiEUMlwZBSxHI7Now', 'tfmge_2026-09-28__15-39_mge_training_v8_beta4b.dem'),
 ('1t63QvVCvLg0Nms9deaN1A6HmPjPr4ZLm', 'tfmge_2026-09-28__00-43_mge_training_v8_beta4b.dem'),
 ('1TCnssezIOH3gCn7T8cZIU1VpO-kYM-di', 'tfmge_2026-09-27__23-43_mge_training_v8_beta4b.dem'),
 ('157W5XeyHpc88yLKAlntHVXiN6Z0eqOXk', 'tfmge_2026-09-27__22-43_mge_training_v8_beta4b.dem'),
 ('1RjpD6XNXxQ8D6Ig_4szIjn3TyiB5RqQ7', 'tfmge_2026-09-27__21-03_mge_training_v8_beta4b.dem'),
 ('14ZayBEDrd3llDFC4qlUH2PoDJptZia2y', 'tfmge_2026-09-27__20-03_mge_training_v8_beta4b.dem'),
 ('112wkX0phmRjKK8xGvg3L72CnINI9QsNd', 'tfmge_2026-09-27__19-34_mge_training_v8_beta4b.dem'),
 ('1kIB65sUvgybVcFheuAMWlqmUzAyw64iR', 'tfmge_2026-09-27__18-10_mge_training_v8_beta4b.dem'),
 ('1bX2pV42z4zJcROWFwptyyXyqZYoh2vna', 'tfmge_2026-09-27__16-51_mge_training_v8_beta4b.dem'),
 ('1AwgTp3VgFT1r5iKB1vshuBZbI3weDDQT', 'tfmge_2026-09-27__15-51_mge_training_v8_beta4b.dem'),
 ('12dgZMJyaeDGIri3ARpPWOmKDi8KlzrFu', 'tfmge_2026-09-27__14-35_mge_training_v8_beta4b.dem'),
 ('1am_CdP17QwoWQLaVn5mweqXFi5F0Lkw0', 'tfmge_2026-09-27__13-35_mge_training_v8_beta4b.dem'),
]

def download(fid, dest):
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [('User-Agent', 'Mozilla/5.0')]
    url = 'https://drive.google.com/uc?export=download&id=' + fid
    # first pass: may return confirm page for virus scan
    r = op.open(url)
    data = r.read()
    ctype = r.headers.get_content_type()
    if b'confirm' in data[:200000] and ('text/html' in ctype):
        import re
        m = re.search(rb'confirm=([0-9A-Za-z-_]+)', data)
        if m:
            url = 'https://drive.google.com/uc?export=download&confirm=%s&id=%s' % (m.group(1).decode(), fid)
            r = op.open(url)
            data = r.read()
        else:
            # form-based confirm (uuid)
            m2 = re.search(rb'name="uuid" value="([^"]+)"', data)
            if m2:
                q = urllib.parse.urlencode({'id': fid, 'export': 'download', 'confirm': 't', 'uuid': m2.group(1).decode()})
                r = op.open('https://drive.google.com/uc?' + q)
                data = r.read()
    if data[:8] == b'HL2DEMO\x00':
        ok = True
    else:
        ok = False
    if ok:
        with open(dest, 'wb') as f:
            f.write(data)
    return ok, len(data), data[:8]

fails = []
for i, (fid, name) in enumerate(DEMOS, 1):
    dest = os.path.join(OUT, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        print(f'[{i}/{len(DEMOS)}] skip (exists) {name}')
        continue
    try:
        ok, n, magic = download(fid, dest)
        print(f'[{i}/{len(DEMOS)}] {"OK " if ok else "FAIL"} {name} bytes={n} magic={magic}')
        if not ok:
            fails.append(name)
            if os.path.exists(dest):
                os.remove(dest)
    except Exception as e:
        print(f'[{i}/{len(DEMOS)}] ERR {name}: {e}')
        fails.append(name)
print('FAILS:', fails if fails else 'none')
