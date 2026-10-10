"""Corrector-transformer (1-layer bidir, d32, 2 heads): ONE parallel pass over all
132 future ticks, adjusting the kinematic chain (user's "attention is all you need"
bracket, variant C). Base = coast rollout AS INPUT (suggestion, not answer):
tokens carry coast displacement + live threat picture recomputed vs the coast
ego position (threat moves linearly, independent of ego -> no recurrence needed).
Head outputs per-tick displacement corrections; loss = dense hitbox loss with
LINEAR far-weighted W132 (slightly elevated far reward) on Vd132 mask.
Resume-safe: EPOCHS=1 per invocation, best-eval-only save, ckpt reload.
Usage: python ml/train_correct132.py   ( повторяйте for more epochs )
OUT: Downloads/mge_ds/correct132.pt ; log: TMP/correct132.log.jsonl"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
TAG = 'correct132'
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
OUT = os.path.join(D, 'correct132.pt')
DT = 1 / 66.0
HB = 24.0
HOR = 132
MAXV_CLS = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)
# champion's power weight, extended to 132: far emphasis WITHOUT the quadratic
# far-gradient monopoly (the friction base keeps corrections small everywhere,
# so absolute scales stay comparable across ticks).
W132 = np.maximum((np.arange(1, HOR + 1, dtype=np.float32) / 66.0) ** 1.3718, 0.02).astype(np.float32)
WT = torch.from_numpy(W132)
_LN_EPS = 1e-12


def slen(v, dim=-1):
    return torch.sqrt((v ** 2).sum(dim=dim, keepdim=True) + _LN_EPS)


class Corrector(nn.Module):
    def __init__(self, d=32, heads=2):
        super().__init__()
        self.prefix = nn.Linear(8 * 20 + 14, d)
        self.embed = nn.Linear(11, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        # zero-init: net starts EXACTLY as the coast base (92/74/52/30/11/7/5),
        # learns refinements only. Random init buries the good base in noise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pref, tok):
        p = self.prefix(pref).unsqueeze(1)
        x = torch.cat([p, self.embed(tok)], dim=1)
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.n1(x + a)
        x = self.n2(x + self.ff(x))
        return self.head(x[:, 1:])


def load(tag):
    # Repacked contiguous files when complete, else memmap slices (never full
    # np.array copies: X alone is 12GB). Returns (X, S0, H, V, n).
    P = os.path.join(D, tag + 'B_X20sol')
    use_repack = False
    if os.path.exists(P + '.raw'):
        import json as _j
        try:
            _done = _j.load(open(P + '.done.json'))['done']
        except Exception:
            _done = -1
        k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
        use_repack = (_done == int((k[:, 3] == 3).sum()))
    if use_repack:
        n = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
        X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(n, 66, 20))
        H = np.memmap(P.replace('X20sol', 'HD132') + '.raw', dtype=np.float32, mode='r',
                      shape=(n, 132, 2))
        V = np.load(P.replace('X20sol', 'VD132') + '.npy', mmap_mode='r')
        # S0 via threatwin npz (small file, sequential read; the S0.npy sidecar
        # hits a flaky Errno 22 on this machine, so don't depend on it).
        tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'))
        S0 = np.array(tw['TS'])[:, -1, :]
        tw.close()
        print('repack %s rows=%d' % (tag, n), flush=True)
        return X, None, None, S0, H, V, n
    d = np.load(os.path.join(D, tag + 'B.npz'), mmap_mode='r')
    k = np.load(os.path.join(D, tag + 'Bk.npz'))['K']
    tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'), mmap_mode='r')
    dn = np.load(os.path.join(D, tag + 'B_dense132.npz'), mmap_mode='r')
    sol = np.where(k[:, 3] == 3)[0]
    n = len(sol)
    print('legacy %s rows=%d' % (tag, n), flush=True)
    return (d['X'], sol, tw['TW'], np.array(tw['TS'])[:, -1, :],
            dn['Hd'], dn['Vd'], n)


def get_batch(Xb, sol, TW, S0, H, V, j0, j1):
    if sol is None:  # repack: already joined, sequential slice
        return (torch.from_numpy(np.array(Xb[j0:j1])),
                torch.from_numpy(np.array(S0[j0:j1])),
                torch.from_numpy(np.array(H[j0:j1])),
                torch.from_numpy(np.array(V[j0:j1])))
    rows = sol[j0:j1]
    return (torch.from_numpy(np.concatenate([np.array(Xb[rows]), np.array(TW[j0:j1])], axis=2)),
            torch.from_numpy(np.array(S0[j0:j1])),
            torch.from_numpy(np.array(H[j0:j1])),
            torch.from_numpy(np.array(V[j0:j1])))


Xtr, sol_tr, TW_tr, Sotr, Htr, Vtr, N_TR = load('train')
Xev, sol_ev, TW_ev, Soev, Hev, Vev, N_EV = load('eval')
REPACK = sol_tr is None
print('train', N_TR, 'eval', N_EV, 'repack' if REPACK else 'legacy', flush=True)


def apply_friction_base(v, gr):
    sp = slen(v)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def build_tokens(xb, s0):
    """xb (B,66,20) warmup, s0 (B,5) threat state. Returns pref, tok, coast, mask-src."""
    B = xb.shape[0]
    mv = MAXV_CLS[3]
    v0 = xb[:, -1, :2] * mv
    with torch.no_grad():
        v = xb[:, -1, :2] * mv
        gr = xb[:, -1, 3:4]
        # friction-coast base (much closer to truth than frictionless: the
        # corrections stay small everywhere, so one linear head can learn all
        # horizons jointly instead of drowning the near field in far error).
        p = torch.zeros_like(v)
        coast_steps = []
        for _ in range(HOR):
            v = apply_friction_base(v, gr)
            p = p + v * DT
            coast_steps.append(p.clone())
        coast = torch.stack(coast_steps, dim=1)
    h = torch.arange(1, HOR + 1).float().view(1, -1, 1) * DT
    csp = slen(coast / (h * mv + 1e-9)).squeeze(-1)
    # threat linear sim (independent of ego): q(h) = q0 + vth*h
    q0 = s0[:, 0:2].unsqueeze(1)
    vth = s0[:, 2:4].unsqueeze(1)
    present = s0[:, 4:5].unsqueeze(1)
    q = q0 + vth * h
    rel = q - coast
    dist = slen(rel, dim=2).clamp_min(1e-6)
    ndist = torch.clamp(dist / 2000.0, 0.0, 1.0)
    ang = torch.atan2(rel[:, :, 1:2], rel[:, :, 0:1])
    infov = (ang.abs() < (np.pi / 4)).float()
    nspd = torch.clamp(slen(vth, dim=2) / 3000.0, 0.0, 1.0).expand(-1, HOR, -1)
    appr = -((vth - v0.unsqueeze(1)) * rel).sum(-1, keepdim=True) / dist
    tti = torch.where(appr > 50.0, torch.clamp(dist / appr.clamp_min(1e-9) / 5.0, 0.0, 1.0),
                      torch.ones_like(dist))
    tti = tti.squeeze(-1) if tti.shape[-1] == 1 else tti
    cnt = xb[:, -1, 17:18].unsqueeze(1).expand(-1, HOR, -1)
    mem = xb[:, -1, 18:19].unsqueeze(1).expand(-1, HOR, -1)
    live = torch.cat([ndist.unsqueeze(-1) if ndist.dim() == 2 else ndist,
                      nspd, infov, cnt, mem,
                      tti.unsqueeze(-1) if tti.dim() == 2 else tti], dim=-1)
    thr = present.expand(-1, HOR, -1) * live + (1.0 - present.expand(-1, HOR, -1)) * \
        xb[:, -1:, 14:20].expand(-1, HOR, -1)
    tt = (torch.arange(HOR).float().view(1, -1, 1) / HOR).expand(B, -1, -1)
    tok = torch.cat([coast / 1000.0, csp.unsqueeze(-1), gr.unsqueeze(1).expand(-1, HOR, -1),
                     tt, thr], dim=-1)
    pref = torch.cat([xb[:, -8:, :].reshape(B, -1), xb[:, -1, 5:19]], dim=-1)
    return pref, tok, coast


# Uncertainty ramp: corrections grow with horizon (near-field coast is already
# 92% hits; far field needs room). Fixed ramp, not learned: h=7 -> 30hu max,
# h=66 -> 200hu, h=132 -> 300hu. Keeps one head from drowning near precision.
RAMP = torch.from_numpy(
    (np.minimum(np.arange(1, HOR + 1, dtype=np.float32) / 66.0, 1.5) * 200.0).astype(np.float32))


def corr_scaled(net, pref, tok):
    return net(pref, tok) * RAMP.view(1, -1, 1)


def tick_loss(err):
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    return lin + lout


def loss_fn(net, xb, s0, Hb, Vb):
    pref, tok, coast = build_tokens(xb, s0)
    traj = coast + corr_scaled(net, pref, tok)
    # Hd carries NaN on invalid ticks; neutralize before the error (NaN*0 = NaN).
    tgt = torch.nan_to_num(Hb, nan=0.0)
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    # Absolute champion-formula loss (relative form gamed via sign flips).
    # The friction-coast base keeps corrections small, so absolute scales stay
    # learnable across all 132 ticks; power weight gives the far emphasis.
    m = Vb.unsqueeze(-1).float()
    per = (tick_loss(err) * WT.view(1, -1, 1) * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def main():
    net = Corrector()
    if os.path.exists(OUT):
        net.load_state_dict(torch.load(OUT, map_location='cpu'))
        print('resumed', OUT, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    B = 256
    S = 2048
    nblocks = (N_TR + S - 1) // S
    border = np.random.permutation(nblocks)
    bl = []
    t0 = time.time()
    net.train()
    nb = 0
    for bi in border:
        a, bb = int(bi * S), int(min((bi + 1) * S, N_TR))
        for i in range(a, bb, B):
            j0, j1 = i, min(i + B, bb)
            xb, sb, Hb, Vb = get_batch(Xtr, sol_tr, TW_tr, Sotr, Htr, Vtr, j0, j1)
            opt.zero_grad()
            loss = loss_fn(net, xb, sb, Hb, Vb)
            if not np.isfinite(loss.item()):
                log({'type': 'SKIP-NONFINITE', 'batch': j0})
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            bl.append(loss.item())
            nb += 1
            if nb % 200 == 0:
                print('batch %d loss %.3f' % (nb, loss.item()), flush=True)
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        for i in range(0, N_EV, 512):
            j0, j1 = i, min(i + 512, N_EV)
            xb, sb, Hb, Vb = get_batch(Xev, sol_ev, TW_ev, Soev, Hev, Vev, j0, j1)
            el.append(loss_fn(net, xb, sb, Hb, Vb).item())
            ew.append(j1 - j0)
        em = float(np.average(el, weights=ew))
    best = None
    try:
        with open(PROG) as f:
            best = json.load(f).get('best', None)
    except Exception:
        best = None
    improved = best is None or em < best
    if improved:
        best = em
        torch.save(net.state_dict(), OUT)
    b = np.array(bl)
    rec = {'train_mean': float(b.mean()), 'eval_mean': em, 'ep_sec': round(time.time() - t0, 1),
           'best': best, 'saved': improved}
    log(rec)
    with open(PROG, 'w') as f:
        json.dump({'best': best, 'done': False}, f)


if __name__ == '__main__':
    main()
