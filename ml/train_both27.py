"""Combined threat+weapon model (27-dim = 14 base + 6 threat + 7 weapon).
The champion (threatsimB, 20-dim, no weapon) and denseE (21-dim, no threat)
are comparable (97.7/87.2/61.2 vs 92.0/81.0/58.7 on matched subsets) and their
naive average LOSES (blur). This trains ONE sharp policy on BOTH signals:
warm-start trunk 20 cols from champion + 7 weapon cols zero-init (epoch 0 ==
champion exactly). Threat-sim rollout + weapon frozen @T. Champion dense
hitbox loss. Resume-safe: 1 epoch/invocation, best-eval-only + mid-epoch
ckpts with block resume. OUT: gru_both27.pt ; log: TMP/both27.log.jsonl"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
TMP = r'C:\Users\Terminatort8000\AppData\Local\Temp\opencode'
TAG = 'both27'
LOG = os.path.join(TMP, TAG + '.log.jsonl')
PROG = os.path.join(TMP, TAG + '_progress.json')
WARM = os.path.join(D, os.environ.get('WARM_CKPT', 'gru_threatsimB.pt'))
OUT = os.path.join(D, 'gru_both27.pt')
BESTOUT = os.path.join(D, 'gru_both27_best.pt')
LR = float(os.environ.get('LR', '1e-4'))
DT = 1 / 66.0
HB = 24.0
HOR = 132
# Far weight STEEPENED +20% (1.3718 -> 1.646, 2026-10-11): the net was
# overfitting the short field (@100-200 saturated) while @500 wobbled.
W132 = np.maximum((np.arange(1, HOR + 1) / 66.0) ** 1.646, 0.02).astype(np.float32)
WT = torch.from_numpy(W132)
_LN_EPS = 1e-12
# Adaptive difficulty (2026-10-11): per-horizon EMA of the tick loss drives the
# weights (gradient-magnitude-tuned, no hand curve to babysit). w = structural
# power curve x blame x adapt, each normalized ~mean 1. Updated per batch.
EMA132 = torch.ones(HOR, 1)
EMA_MOM = 0.99


def slen(v, dim=-1):
    return torch.sqrt((v ** 2).sum(dim=dim, keepdim=True) + _LN_EPS)


MAXV1 = np.array([400, 300, 240, 280, 320, 230, 300, 300, 300], dtype=np.float32)


class Net27(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.gru = nn.GRU(27, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


def load(tag):
    P = os.path.join(D, tag + 'B_X20sol')
    n = int(os.path.getsize(P + '.raw') // (66 * 20 * 4))
    X = np.memmap(P + '.raw', dtype=np.float32, mode='r', shape=(n, 66, 20))
    # W7 is savez_compressed: preload once (2.5GB train). Per-block member
    # access would full-decompress every block.
    W7 = np.array(np.load(os.path.join(D, tag + 'B_w7.npz'))['W7'])
    assert W7.shape[0] == n, (W7.shape, n)
    H = np.memmap(P.replace('X20sol', 'HD132') + '.raw', dtype=np.float32, mode='r',
                  shape=(n, HOR, 2))
    V = np.load(P.replace('X20sol', 'VD132') + '.npy', mmap_mode='r')
    tw = np.load(os.path.join(D, tag + 'B_threatwin.npz'))
    S0 = np.array(tw['TS'])[:, -1, :]
    tw.close()
    JM = JH = None
    if tag == 'train':
        jm = np.load(os.path.join(D, 'trainB_jumpmask.npz'))
        JM = np.array(jm['JM'])
        JH = np.array(jm['JH132'])
        assert JM.shape[0] == n, (JM.shape, n)
        print('jumpmask: rows-kept=%.4f tick-keep=%.4f' % (
            JM.mean(), JH.mean()), flush=True)
    print('%s rows=%d' % (tag, n), flush=True)
    return X, W7, S0, H, V, JM, JH, n


Xtr, Wtr, S0tr, Htr, Vtr, JMtr, JHtr, N_TR = load('train')
Xev, Wev, S0ev, Hev, Vev, _, _, N_EV = load('eval')
SMOKE = os.environ.get('SMOKE', '0') == '1'
BLAME_LOG_THR = float(os.environ.get('BLAME_LOG_THR', '4000.0'))
BLAME_LOG = os.path.join(TMP, 'blame_rows.jsonl')
print('train', N_TR, 'eval', N_EV, flush=True)


def apply_friction(v, gr):
    sp = slen(v)
    drop = torch.clamp(sp, min=100.0) * 4.0 * DT
    newsp = torch.clamp(sp - drop, min=0.0)
    scale = torch.where(sp > 0, newsp / sp.clamp_min(1e-6), torch.ones_like(sp))
    return v * (gr * scale + (1.0 - gr))


def unroll(net, xb, s0, nsteps=HOR, ret_acc=False):
    mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
    _, h = net.gru(xb)
    v = xb[:, -1, :2] * mv.squeeze(1)
    p = torch.zeros_like(v)
    gr = xb[:, -1, 3:4]
    static = xb[:, -1:, 5:14]
    wstat = xb[:, -1:, 20:27]
    q = s0[:, 0:2].clone()
    vth = s0[:, 2:4].clone()
    present = s0[:, 4:5]
    thr_frozen = xb[:, -1:, 14:20]
    nspd = torch.clamp(slen(vth) / 3000.0, 0.0, 1.0)
    cnt = thr_frozen[:, 0, 3:4]
    mem = thr_frozen[:, 0, 4:5]
    hh = h
    traj = []
    acc = [] if ret_acc else None
    for _ in range(nsteps):
        a = net.head(hh[-1]) * mv.squeeze(1)
        if ret_acc:
            acc.append(a.clone())
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
        _, hh = net.gru(torch.cat([dyn, static, wstat], dim=-1), hh)
        traj.append(p)
    traj = torch.stack(traj, dim=1)
    if ret_acc:
        return traj, torch.stack(acc, dim=1)
    return traj


def dense_loss(net, xb, s0, Hb, Vb, Jm=None, rows=None):
    """Counterfactual-blame + adaptive-difficulty + jump-masked dense loss.
    blame_h = |a_pred,h - a_needed,h| (inverse dynamics from Hd, detached):
    the tick whose wrong intent caused the miss gets the gradient. Jm = jump
    flight mask (train only). rows = global soldier-row idx (blame audit log).
    Returns loss (mean)."""
    global EMA132
    traj, acc = unroll(net, xb, s0, nsteps=HOR, ret_acc=True)
    tgt = torch.nan_to_num(torch.from_numpy(np.array(Hb)), nan=0.0)
    m = torch.from_numpy(np.array(Vb)).unsqueeze(-1).float()
    if Jm is not None:
        m = m * torch.from_numpy(np.array(Jm)).unsqueeze(-1).float()
    err = torch.abs(traj - tgt).amax(dim=-1, keepdim=True)
    inside = (err <= HB).float()
    lin = torch.log((err + 1.0) / (HB + 1.0)) * inside
    lout = 4.0 * ((err - HB) / HB).clamp_min(0.0) ** 2 * (1.0 - inside)
    base = lin + lout
    # --- counterfactual blame from inverse dynamics (detached) ---
    with torch.no_grad():
        mv = (xb[:, -1:, 5:14] * torch.from_numpy(MAXV1)).sum(-1, keepdim=True)
        v0 = xb[:, -1, :2] * mv.squeeze(1)
        gr = xb[:, -1, 3:4]
        Hd0 = torch.cat([torch.zeros_like(tgt[:, :1]), tgt], dim=1)
        Vt = (Hd0[:, 1:] - Hd0[:, :-1]) / DT
        Vf = apply_friction(torch.cat([v0.unsqueeze(1), Vt[:, :-1]], dim=1),
                            gr.unsqueeze(1))
        Aneed = (Vt - Vf) / DT
        # --- unnatural-tick gate (physics plausibility, 2026-10-11) ---
        # |a_needed| beyond any human input = external impulse (blast/
        # knockback/teleport residue), NOT behavior. Caps: ground 4000
        # (C++ clamp 14*MAXV=3360 + margin), air 3000 (strafe max 2400 +
        # margin). Distinct from C++ output clamping (bounding predictions)
        # vs this (excluding supervision). Audit: blame log top is all
        # teleports (aneed ~1M, sp=0); blast flights 5-50k.
        cap = torch.where(gr.unsqueeze(1) > 0.5, 14.0 * mv * (4000.0 / 3360.0),
                          torch.full_like(mv, 3000.0)).expand(-1, HOR, -1)
        unnat = Aneed.abs().amax(dim=-1, keepdim=True) > cap
        m = m * (~unnat).float()
        global UNNAT_FRAC
        try:
            UNNAT_FRAC.append(float(unnat.float().mean().item()))
        except NameError:
            UNNAT_FRAC = [float(unnat.float().mean().item())]
        bl = (acc.detach() - Aneed).abs().mean(-1, keepdim=True)
        # neutral (1.0 pre-norm) where targets invalid
        Vm = m.squeeze(-1).bool()
        Vm2 = Vm & torch.cat([Vm[:, :1], Vm[:, :-1]], dim=1)
        bl = torch.where(Vm2.unsqueeze(-1), bl, torch.zeros_like(bl))
        bmean = bl.mean(dim=1, keepdim=True).clamp_min(1e-6)
        blame_n = bl / bmean
        w_blame = 0.25 + 0.75 * blame_n
        # --- adaptive difficulty: EMA of per-horizon loss tunes weights ---
        # NOTE: base goes NEGATIVE inside the hitbox (lin = log((err+1)/25)).
        # EMA must track magnitude (clamp_min(0)): signed values drive the EMA
        # negative and sqrt(negative) NaNs the whole run (killed ep7 @47% skip).
        tick = ((base.clamp_min(0) * m).sum(0) / m.sum(0).clamp_min(1)).squeeze(-1).detach()
        tick = torch.nan_to_num(tick, nan=0.0)
        EMA132.mul_(EMA_MOM).add_(tick.unsqueeze(-1), alpha=1 - EMA_MOM)
        adapt = (EMA132 / EMA132.mean().clamp_min(1e-9)).sqrt().detach()
        # --- noise/gap audit: log outlier rows with keys ---
        if net.training and rows is not None:
            mx, mh = bl.squeeze(-1).max(dim=1)
            hot = torch.where(mx > BLAME_LOG_THR)[0]
            if len(hot):
                ana = Aneed[hot].abs().amax(dim=-1)[torch.arange(len(hot)), mh[hot]]
                with open(BLAME_LOG, 'a') as f:
                    for k, rr in enumerate(hot.tolist()):
                        f.write(json.dumps({'row': int(rows[rr]),
                                            'h': int(mh[rr].item()),
                                            'blame': round(float(mx[rr].item()), 1),
                                            'aneed': round(float(ana[k].item()), 1)}) + '\n')
    tick_w = WT.view(1, -1, 1) * w_blame * adapt.view(1, -1, 1)
    per = ((base) * tick_w * m).sum(dim=1) / (WT.view(1, -1, 1) * m).sum(dim=1).clamp_min(1)
    return per.mean()


def log(rec):
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def main():
    net = Net27()
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
    done_blocks = set()
    try:
        with open(PROG) as f:
            _pg = json.load(f)
            if isinstance(_pg.get('done_blocks'), list):
                done_blocks = set(_pg['done_blocks'])
    except Exception:
        pass
    if done_blocks:
        print('resuming epoch, skipping %d done blocks' % len(done_blocks), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    B = 128
    N = N_TR
    S = 2048
    nblocks = (N + S - 1) // S
    try:
        with open(PROG) as f:
            _pg = json.load(f)
        if len(_pg.get('border', [])) == nblocks:
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
    nb = 0
    nblk = 0
    max_b = 2 if SMOKE else 10 ** 9
    for bi in border:
        if nb >= max_b:
            break
        a, bb = int(bi * S), int(min((bi + 1) * S, N))
        if int(bi) in done_blocks:
            continue
        Xblk = np.concatenate([np.array(Xtr[a:bb]), np.array(Wtr[a:bb])], axis=2)
        Sblk = np.array(S0tr[a:bb])
        Hblk = np.array(Htr[a:bb])
        Vblk = np.array(Vtr[a:bb])
        Jblk = np.array(JHtr[a:bb])
        rws = np.arange(a, bb)
        for i in range(0, bb - a, B):
            xb = torch.from_numpy(Xblk[i:i + B])
            sb = torch.from_numpy(Sblk[i:i + B])
            opt.zero_grad()
            loss = dense_loss(net, xb, sb, Hblk[i:i + B], Vblk[i:i + B],
                              Jm=Jblk[i:i + B], rows=rws[i:i + B])
            if not np.isfinite(loss.item()):
                log({'type': 'SKIP-NONFINITE', 'batch': a + i})
                continue
            loss.backward()
            if not all(torch.isfinite(p.grad).all().item()
                       for p in net.parameters() if p.grad is not None):
                log({'type': 'SKIP-NONFINITE-GRAD', 'batch': a + i})
                continue
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            bl.append(loss.item())
            nb += 1
            if nb % 200 == 0:
                print('batch %d loss %.3f' % (nb, loss.item()), flush=True)
        done_blocks.add(int(bi))
        nblk += 1
        if not SMOKE and nblk % 50 == 0:
            torch.save(net.state_dict(), OUT)
            with open(PROG, 'w') as f:
                json.dump({'border': [int(b) for b in border],
                           'done_blocks': sorted(done_blocks), 'best': None}, f)
            print('ckpt @block %d/%d' % (nblk, nblocks), flush=True)
    net.eval()
    with torch.no_grad():
        el, ew = [], []
        ev_n = 512 if SMOKE else N_EV
        for i in range(0, ev_n, 512):
            j0, j1 = i, min(i + 512, N_EV)
            xb = torch.from_numpy(np.concatenate([np.array(Xev[j0:j1]),
                                                  np.array(Wev[j0:j1])], axis=2))
            sb = torch.from_numpy(np.array(S0ev[j0:j1]))
            el.append(dense_loss(net, xb, sb, Hev[j0:j1], Vev[j0:j1]).item())
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
        torch.save(net.state_dict(), BESTOUT)
        print('new best-ever -> gru_both27_best.pt', flush=True)
    b = np.array(bl)
    try:
        unnat_mean = float(np.mean(UNNAT_FRAC)) if 'UNNAT_FRAC' in globals() else -1.0
    except Exception:
        unnat_mean = -1.0
    rec = {'train_mean': float(b.mean()), 'train_std': float(b.std()),
           'eval_mean': em, 'eval_std': es, 'ep_sec': round(time.time() - t0, 1),
           'best': best, 'saved': improved, 'unnat_frac': unnat_mean}
    log(rec)
    if not SMOKE:
        with open(PROG, 'w') as f:
            json.dump({'best': best, 'done': True, 'done_blocks': []}, f)


if __name__ == '__main__':
    main()
