"""Variant A: champion GRU(20,96) + single-head attention over the 66 warmup hidden
states ("does simple attention help?"). Per rollout step: query=hh, K/V=warmup H,
ctx=softmax(qK/sqrt(d))V, policy head sees h + Wc*ctx (Wc zero-init). GRU input
extended 20->22 with coast-displacement suggestion (new cols zero-init), so the
net starts EXACTLY as the champion and can only diverge by learning.
Threat-sim recompute + friction-first + dense hitbox loss, identical to champion.
Resume-safe: 1 epoch per invocation, best-eval-only save.
OUT: Downloads/mge_ds/gru_attn66.pt ; log: TMP/attn66.log.jsonl"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
TAG = 'attn66'
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, 'gru_threatsimB.pt')
OUT = os.path.join(D, 'gru_attn66.pt')
DT = 1 / 66.0
HB = 24.0
W = np.maximum((np.arange(1, 67) / 66.0) ** 1.3718, 0.02).astype(np.float32)
WT = torch.from_numpy(W)
_LN_EPS = 1e-12


def slen(v, dim=-1):
    return torch.sqrt((v ** 2).sum(dim=dim, keepdim=True) + _LN_EPS)


MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)


class NetA(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(22, h, batch_first=True)
        self.q = nn.Linear(h, 32)
        self.k = nn.Linear(h, 32)
        self.wc = nn.Linear(h, h, bias=False)
        nn.init.zeros_(self.wc.weight)
        self.head = nn.Linear(h, 2)

    def attend(self, hh, H):
        q = self.q(hh[-1]).unsqueeze(1)
        k = self.k(H)
        w = torch.softmax((q @ k.transpose(1, 2)) / np.sqrt(32), dim=-1)
        return (w @ H).squeeze(1)


def load(tag):
    # Repacked contiguous files (repack20.py) when present: sequential block
    # reads instead of scattered memmap page faults. Returns
    # (mode, X-or-X20, spos-or-None, TW-or-None, S0, H, V).
    P = os.path.join(D, tag + 'B_X20sol')
    if os.path.exists(P + '.raw'):
        import json as _j
        try:
            _done = _j.load(open(P + '.done.json'))['done']
        except Exception:
            _done = -1
        k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
        if _done == int((k[:, 3] == 3).sum()):
            n = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
            X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(n, 66, 20))
            H = np.memmap(P.replace('X20sol', 'HD66') + '.raw', dtype=np.float32, mode='r',
                          shape=(n, 66, 2))
            V = np.load(P.replace('X20sol', 'VD66') + '.npy', mmap_mode='r')
            tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'))
            S0 = np.array(tw['TS'])[:, -1, :]
            tw.close()
            print('repack %s rows=%d' % (tag, n), flush=True)
            return ('repack', X, None, None, S0, H, V)
    d = np.load(os.path.join(D, tag + 'B.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'))
    sol = k[:, 3] == 3
    TW = np.array(tw['TW'])
    assert TW.shape[0] == sol.sum()
    S0 = np.array(tw['TS'])[:, -1, :]
    tw.close()
    dn = np.load(os.path.join(D, tag + 'B_dense66.npz'), mmap_mode='r')
    return ('legacy', d['X'], np.where(sol)[0], TW, S0, dn['Hd'], dn['Vd'])


def load_dense(tag):
    k = np.load(os.path.join(D, tag + 'B' + 'k.npz'))['K']
    dn = np.load(os.path.join(D, tag + 'B' + '_dense66.npz'), mmap_mode='r')
    return dn['Hd'], dn['Vd']


def get_batch(Xb, spos, TW, S0, jdx):
    rows = spos[jdx]
    xb = np.concatenate([np.array(Xb[rows]), np.array(TW[jdx])], axis=2)
    return torch.from_numpy(xb), torch.from_numpy(np.array(S0[jdx]))


mode_tr, Xb_tr, spos_tr, TW_tr, S0_tr, HtrD, VtrD = load('train')
mode_ev, Xb_ev, spos_ev, TW_ev, S0_ev, HevD, VevD = load('eval')
REPACK = (mode_tr == 'repack')
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    HtrD, VtrD, S0_tr = HtrD[:256], VtrD[:256], S0_tr[:256]
    HevD, VevD, S0_ev = HevD[:512], VevD[:512], S0_ev[:512]
    if not REPACK:
        spos_tr, TW_tr = spos_tr[:256], TW_tr[:256]
        spos_ev, TW_ev = spos_ev[:512], TW_ev[:512]
N_TR = (256 if SMOKE else len(S0_tr))
N_EV = (512 if SMOKE else len(S0_ev))
print('train', N_TR, 'eval', N_EV, 'repack' if REPACK else 'legacy', flush=True)


def apply_friction(v, gr):
    sp = slen(v)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def unroll(net, xb, s0):
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
    zero2 = torch.zeros(xb.size(0), 66, 2)
    _, h = net.gru(torch.cat([xb, zero2], dim=-1))
    # warmup hidden sequence = attention keys/values (coast cols zero: no "p" yet)
    Hw, _ = net.gru(torch.cat([xb, zero2], dim=-1))
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    traj = []
    for step in range(66):
        ctx = net.attend(hh, Hw)
        a = net.head(hh[-1] + net.wc(ctx)) * mv.squeeze(1)
        v = apply_friction(v, gr) + a * DT
        p = p + v * DT
        q = q + vth * DT
        rel = q - p
        dist = slen(rel, dim=1).clamp_min(1e-6)
        ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
        ang = torch.atan2(rel[:, 1:2], rel[:, 0:1])
        infov = (ang.abs() < (np.pi / 4)).float()
        appr = -((vth - v) * rel).sum(-1, keepdim=True) / dist
        tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                          torch.ones_like(dist))
        live = torch.cat([ndist, nspd, infov, cnt, mem, tti], dim=-1).unsqueeze(1)
        thr = present.unsqueeze(1) * live + (1.0 - present.unsqueeze(1)) * thr_frozen
        sp = slen(v) / mv.squeeze(1)
        dyn = torch.cat([(v / mv.squeeze(1)).unsqueeze(1), sp.unsqueeze(1),
                         gr.unsqueeze(1), torch.zeros_like(sp).unsqueeze(1),
                         thr], dim=-1)
        # coast suggestion: where frictionless coast from current v would be next tick
        xin22 = torch.cat([dyn, static, (v * DT / 1000.0).unsqueeze(1)], dim=-1)
        _, hh = net.gru(xin22, hh)
        traj.append(p)
    return torch.stack(traj, dim=1)


def dense_loss(net, xb, s0, Hb, Vb):
    traj = unroll(net, xb, s0)
    # NOTE: Hd holds NaN on invalid ticks; NaN*mask(0) = NaN (not 0) and would
    # poison the batch. Neutralize BEFORE the error so masked ticks score exactly 0.
    tgt = torch.nan_to_num(torch.from_numpy(np.array(Hb)), nan=0.0)
    m = torch.from_numpy(np.array(Vb)).unsqueeze(-1).float()
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    per = ((lin + lout) * WT.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def main():
    net = NetA()
    champ = torch.load(WARM, map_location='cpu')
    sd = net.state_dict()
    sd['gru.weight_ih_l0'][:, :20] = champ['gru.weight_ih_l0']
    sd['gru.weight_ih_l0'][:, 20:] = 0
    sd['gru.weight_hh_l0'] = champ['gru.weight_hh_l0']
    sd['gru.bias_ih_l0'] = champ['gru.bias_ih_l0']
    sd['gru.bias_hh_l0'] = champ['gru.bias_hh_l0']
    sd['head.weight'] = champ['head.weight']
    sd['head.bias'] = champ['head.bias']
    net.load_state_dict(sd)
    print('warm-started from champion (exact at init)', flush=True)
    if os.path.exists(OUT):
        net.load_state_dict(torch.load(OUT, map_location='cpu'))
        print('resumed', OUT, flush=True)
    # mid-epoch resume: completed block ids persist in PROG so kills/OOMs only
    # lose the current block, never the epoch.
    done_blocks = set()
    try:
        with open(PROG) as f:
            done_blocks = set(json.load(f).get('done_blocks', []))
    except Exception:
        pass
    if done_blocks:
        print('resuming epoch, skipping %d done blocks' % len(done_blocks), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    B = 128
    N = N_TR
    S = 2048
    nblocks = (N + S - 1) // S
    try:
        with open(PROG) as f:
            _pg = json.load(f)
        if len(_pg.get('border', [])) == nblocks and not SMOKE:
            border = np.array(_pg['border'])
            print('reusing stored block order', flush=True)
        else:
            raise ValueError('new epoch')
    except Exception:
        border = np.random.permutation(nblocks)
        with open(PROG, 'w') as f:
            json.dump({'border': [int(b) for b in border], 'done_blocks': [],
                       'best': None}, f)
    bl = []
    t0 = time.time()
    net.train()
    max_b = 2 if SMOKE else 10 ** 9
    nb = 0
    done = False
    nblk = 0
    for bi in border:
        if done:
            break
        if int(bi) in done_blocks:
            continue
        a, bb = int(bi * S), int(min((bi + 1) * S, N))
        if REPACK:
            Xblk = np.array(Xb_tr[a:bb])
        else:
            Xblk = np.concatenate([np.array(Xb_tr[spos_tr[a:bb]]),
                                   np.array(TW_tr[a:bb])], axis=2)
        Sblk = np.array(S0_tr[a:bb])
        Hblk = np.array(HtrD[a:bb])
        Vblk = np.array(VtrD[a:bb])
        for i in range(0, bb - a, B):
            if nb >= max_b:
                done = True
                break
            nb += 1
            xb = torch.from_numpy(Xblk[i:i + B])
            sb = torch.from_numpy(Sblk[i:i + B])
            opt.zero_grad()
            loss = dense_loss(net, xb, sb, Hblk[i:i + B], Vblk[i:i + B])
        if not np.isfinite(loss.item()):
            log({'type': 'SKIP-NONFINITE', 'batch': i})
            continue
        loss.backward()
        if not all(torch.isfinite(p.grad).all().item()
                   for p in net.parameters() if p.grad is not None):
            log({'type': 'SKIP-NONFINITE-GRAD', 'batch': i})
            continue
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        bl.append(loss.item())
        if nb % 200 == 0:
            print('batch %d loss %.3f' % (nb, loss.item()), flush=True)
        done_blocks.add(int(bi))
        nblk += 1
        if nblk % 50 == 0:
            # mid-epoch checkpoint: weights + completed blocks (kill-proof)
            torch.save(net.state_dict(), OUT)
            with open(PROG, 'w') as f:
                json.dump({'border': [int(b) for b in border],
                           'done_blocks': sorted(done_blocks), 'best': None}, f)
            print('ckpt @block %d/%d' % (nblk, nblocks), flush=True)
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, N_EV, 512):
            j0, j1 = i, min(i + 512, N_EV)
            if REPACK:
                xb = torch.from_numpy(np.array(Xb_ev[j0:j1]))
                sb = torch.from_numpy(np.array(S0_ev[j0:j1]))
            else:
                xb, sb = get_batch(Xb_ev, spos_ev, TW_ev, S0_ev, np.arange(j0, j1))
            el.append(dense_loss(net, xb, sb, HevD[j0:j1], VevD[j0:j1]).item())
            ew.append(j1 - j0)
        em = float(np.average(el, weights=ew))
        es = float(np.sqrt(np.average((np.array(el) - em) ** 2, weights=ew)))
    best = None
    try:
        with open(PROG) as f:
            best = json.load(f).get('best', None)
    except Exception:
        best = None
    improved = (best is None or em < best) and not SMOKE
    if improved:
        best = em
        torch.save(net.state_dict(), OUT)
    b = np.array(bl)
    rec = {'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'best': best, 'saved': improved}
    log(rec)
    if not SMOKE:
        with open(PROG, 'w') as f:
            json.dump({'best': best, 'done': False}, f)


if __name__ == '__main__':
    main()
