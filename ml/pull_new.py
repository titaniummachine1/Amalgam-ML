"""Fourth pull: everything on brygadagraczy.pl/sourcetv (tfmge + tforange)
not already on disk. Self-contained (dl_new has import side effects).
PAGE env: 0=tfmge only, 1=tforange only, else both. Resumable: skips existing."""
import http.cookiejar
import os
import re
import sys
import urllib.parse
import urllib.request

OUT_MGE = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_demos')
OUT_OR = os.path.join(os.path.expanduser('~'), 'Downloads', 'orange_demos')
os.makedirs(OUT_MGE, exist_ok=True)
os.makedirs(OUT_OR, exist_ok=True)


def download(fid, dest):
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [('User-Agent', 'Mozilla/5.0')]
    url = 'https://drive.google.com/uc?export=download&id=' + fid
    r = op.open(url, timeout=120)
    data = r.read()
    ctype = r.headers.get_content_type()
    if b'confirm' in data[:200000] and ('text/html' in ctype):
        m = re.search(rb'confirm=([0-9A-Za-z-_]+)', data)
        if m:
            url = 'https://drive.google.com/uc?export=download&confirm=%s&id=%s' % (
                m.group(1).decode(), fid)
            r = op.open(url, timeout=120)
            data = r.read()
        else:
            m2 = re.search(rb'name="uuid" value="([^"]+)"', data)
            if m2:
                q = urllib.parse.urlencode({'id': fid, 'export': 'download',
                                            'confirm': 't', 'uuid': m2.group(1).decode()})
                r = op.open('https://drive.google.com/uc?' + q, timeout=120)
                data = r.read()
    ok = data[:8] == b'HL2DEMO\x00'
    if ok:
        with open(dest, 'wb') as f:
            f.write(data)
    return ok, len(data)


PAGES = [
    ('https://brygadagraczy.pl/sourcetv/sv.php?id=tfmge', OUT_MGE),
    ('https://brygadagraczy.pl/sourcetv/sv.php?id=tforange', OUT_OR),
]
page_sel = os.environ.get('PAGE', '')
if page_sel in ('0', '1'):
    PAGES = [PAGES[int(page_sel)]]

total_ok, total_skip, fails = 0, 0, []
for url, outdir in PAGES:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'replace')
    pairs = re.findall(r"DownloadDemo\('([^']+)'\)\"><div class=\"display\">([^<]+\.dem)</div>", html)
    print('PAGE %s -> %d entries' % (url, len(pairs)), flush=True)
    for fid, name in pairs:
        dest = os.path.join(outdir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100000:
            total_skip += 1
            continue
        try:
            ok, n = download(fid, dest)
            print('%s %s bytes=%d' % ('OK  ' if ok else 'FAIL', name, n), flush=True)
            if ok:
                total_ok += 1
            else:
                fails.append(name)
                if os.path.exists(dest):
                    os.remove(dest)
        except Exception as e:
            print('EXC %s %s' % (name, e), flush=True)
            fails.append(name)
print('DONE ok=%d skip=%d fails=%s' % (total_ok, total_skip, fails if fails else 'none'), flush=True)
