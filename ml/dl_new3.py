"""Third expansion pull (2026-10-06): 12 MGE + 3 orange from brygadagraczy.pl/sourcetv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_new import download, OUT_MGE, OUT_OR

DEMOS = [
 ('1jw3ISRMuUo1Jpk14SS3mdbRZJa7gHfw9', 'tfmge_2026-10-06__03-09_mge_training_v8_beta4b.dem', OUT_MGE),
 ('11zKdymZ_vAuoIUezYTR1aKpldMvG8YsC', 'tfmge_2026-10-06__03-06_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1eVqymg1Uiw5_GriweBuouHrLqzFkMIyo', 'tfmge_2026-10-06__03-05_mge_training_v8_beta4b.dem', OUT_MGE),
 ('19cU73rkpCtN3TM7nQy_4Q9IIjTf_xZbS', 'tfmge_2026-10-05__22-53_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1dUXXv_aeYvt-QH3tF0Fw4Oo6Qdl44AI4', 'tfmge_2026-10-05__22-08_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1WmpbzPJVfBHHKPOQyaBLxgFi_Vy6UGzX', 'tfmge_2026-10-05__22-05_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1eurdKGLQ8VfeK7JtTKQkXtAh2tkhvnkc', 'tfmge_2026-10-05__22-01_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1H3hMXhUf6KRmDjPnFa8KNTXMzntlFEcy', 'tfmge_2026-10-05__21-35_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1Bt66Ai2chbABmIYQZKxFE9xW5WpwHfi_', 'tfmge_2026-10-05__20-35_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1fMeBg1rglLeIVOwGaXLzFNv4LDvRdRsd', 'tfmge_2026-10-05__19-31_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1ldBL3Y56FZGFJkan49b6oUeQFnY_C0NC', 'tfmge_2026-10-05__17-55_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1kEEfvoYUlAJHn_c4P-BVzZvi8LOY7h2k', 'tfmge_2026-10-05__16-54_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1u69ROUNRN_IjlQL3zUVpaNE-eZBcXY9P', 'tforange_2026-10-05__19-52_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1tSkoPh6bO2ePgLV-BCJhzu3Hr9zDdFwR', 'tforange_2026-10-05__18-12_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1rSwPW4ClJggBJLL2wTv4wljXQ6RdIAdt', 'tforange_2026-10-05__16-12_cp_orange_x5_bg_v34.dem', OUT_OR),
]

if __name__ == '__main__':
    fails = []
    for i, (fid, name, outdir) in enumerate(DEMOS, 1):
        dest = os.path.join(outdir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100000:
            print('[%d/%d] skip (exists) %s' % (i, len(DEMOS), name), flush=True)
            continue
        try:
            ok, n = download(fid, dest)
            print('[%d/%d] %s %s bytes=%d' % (i, len(DEMOS), 'OK ' if ok else 'FAIL', name, n),
                  flush=True)
            if not ok:
                fails.append(name)
        except Exception as e:
            print('[%d/%d] EXC %s %s' % (i, len(DEMOS), name, e), flush=True)
            fails.append(name)
    print('FAILS: %s' % (fails if fails else 'none'), flush=True)
