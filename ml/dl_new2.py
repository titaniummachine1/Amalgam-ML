"""Second expansion pull (2026-10-01): 12 MGE + 4 orange from brygadagraczy.pl/sourcetv
(sv.php?id=tfmge / tforange), everything newer than the 09-30 19:40 pull."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_new import download, OUT_MGE, OUT_OR

DEMOS = [
 ('1CnJIgO3CR5OPsXz7zgpLIB5ekqfcSehd', 'tfmge_2026-10-01__17-56_mge_training_v8_beta4b.dem', OUT_MGE),
 ('12BJCaKLX0NWaT-N5t70-lg1NfxOs3NKg', 'tfmge_2026-10-01__16-56_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1c1HGVnr61qIJIOq2ZKcdX7TVT0UihpuI', 'tfmge_2026-10-01__16-36_mge_training_v8_beta4b.dem', OUT_MGE),
 ('195SR52-pdSkfhdh-MT82Wg_tJBspvgww', 'tfmge_2026-10-01__16-20_mge_training_v8_beta4b.dem', OUT_MGE),
 ('160TdPtGz3m5yBcP-DLzRmZ1tpPisbKBt', 'tfmge_2026-10-01__14-48_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1IbEwO-vaeBoB0uj92qS5K6djn1quiSEc', 'tfmge_2026-10-01__07-51_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1RTxNKBAX90xOxz8xa_P_IFYHMlyNqAK7', 'tfmge_2026-10-01__07-50_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1Mm55DP385Ql_VunqweyOQqbF8HKa4Ykb', 'tfmge_2026-10-01__07-49_mge_training_v8_beta4b.dem', OUT_MGE),
 ('13z7wi9n2NegzPxdx0vLFypc3hGm_qp_k', 'tfmge_2026-10-01__07-46_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1ObG7xVySYP3lbtXDlfg6FrBhFjnQg9gZ', 'tfmge_2026-09-30__23-33_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1-pPRstDluHbJ3aIxm_ufKZsjAFpdS6xH', 'tfmge_2026-09-30__22-33_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1FShZuz78pidF03saA3tUppqboG0kyYt_', 'tfmge_2026-09-30__19-50_mge_training_v8_beta4b.dem', OUT_MGE),
 ('1rAcIhsWb1GjM0ZHfn93eNXNUtWUTh9ns', 'tforange_2026-10-01__17-49_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1AbrLp0aBSj3LJygO7LjghkOSWwfoGROo', 'tforange_2026-10-01__16-43_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('1yRqzJpCeT72sOWUZ6I7p26SGJj0ZQhuD', 'tforange_2026-09-30__23-10_cp_orange_x5_bg_v34.dem', OUT_OR),
 ('15_RwDjK50oqd0-qQYDqv3JwV9eDz-NSG', 'tforange_2026-09-30__20-48_cp_orange_x5_bg_v34.dem', OUT_OR),
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
