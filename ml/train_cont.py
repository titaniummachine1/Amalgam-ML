"""train_cont.py — continuous-input AR transformer (kills the 25-bin ceiling).
Same TinyAR backbone/history as train_tx.py (d64/2L/4H winner arch) but the
future is continuous: wishdir xy (world frame, unit-or-zero) + wishspeed/400
+ jump bit. Regression heads (tanh/sigmoid) + jump/conf heads.
Teacher forcing on WC sidecars (build_cont.py); loss = masked MSE(wish,speed)
+ 0.3*BCE(jump) + 0.3*BCE(conf vs ERR<24). Mask weights = same classes
(clean 1.0, capped 0.5, flight/blast 0.0). No class rebalance (regression).
World-frame targets sidestep frozen-eye entirely (eye only enters eval via
the back-clamp view, same as physics).
Usage: python ml/train_cont.py <train_all.txt> [epochs] [out.pt]
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn

D = 64
NL = 2
NH = 4
FF = 128
HIST = 33
FUT = 20
NFEAT = 22
NCONF = 6
MASKW = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}


class ContAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_proj = nn.Linear(NFEAT, D)
        self.fut_proj = nn.Linear(3, D)
        self.jmp_emb = nn.Embedding(2, D)
        layer = nn.TransformerDecoderLayer(d_model=D, nhead=NH, dim_feedforward=FF,
                                           batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=NL)
        self.mem_proj = nn.Linear(D, D)
        self.whead = nn.Linear(D, 2)
        self.shead = nn.Linear(D, 1)
        self.jhead = nn.Linear(D, 1)
        self.chead = nn.Linear(D, NCONF)
        pe = torch.zeros(HIST + FUT, D)
        pos = torch.arange(HIST + FUT, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, X, fin, jin):
        B = X.size(0)
        h = self.hist_proj(X) + self.pe[:HIST].unsqueeze(0)
        mem = self.mem_proj(h)
        f = self.fut_proj(fin) + self.jmp_emb(jin) + self.pe[HIST:HIST + FUT].unsqueeze(0)
        causal = torch.triu(torch.ones(FUT, FUT, dtype=torch.bool), 1).to(X.device)
        y = self.dec(f, mem, tgt_mask=causal)
        return (torch.tanh(self.whead(y)), torch.sigmoid(self.shead(y)).squeeze(-1),
                self.jhead(y).squeeze(-1), self.chead(y.mean(dim=1)))


def main():
    src = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = sys.argv[3] if len(sys.argv) > 3 else 'tx_cont.pt'
    files = [l.strip() for l in open(src) if l.strip()] if src.endswith('.txt') else [src]
    per = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if len(d['X']) == 0:
            continue
        try:
            wc = np.load(fp + '.cont.npz', allow_pickle=True)['WC']
        except OSError:
            print('NO CONTCAR, skip', fp, flush=True)
            continue
        if len(wc) != len(d['X']):
            print('LEN MISMATCH, skip', fp, flush=True)
            continue
        per.append((d['X'], wc, d['JMP'], d['MSK'], d['ERR']))
    print('files=%d/%d' % (len(per), len(files)), flush=True)
    # FILE-split val (never random windows: stride-1 neighbours are
    # near-duplicates, random split leaks train into val).
    order = np.arange(len(per))
    rng = np.random.RandomState(7)
    rng.shuffle(order)
    nvf = max(1, len(per) // 10)
    vset, tset = set(order[:nvf].tolist()), set(order[nvf:].tolist())
    if not tset:
        tset = set(order.tolist())
    Tr = [per[j] for j in sorted(tset)]
    Va = [per[j] for j in sorted(vset)]
    X = np.concatenate([a[0] for a in Tr])
    W = np.concatenate([a[1] for a in Tr])
    JMP = np.concatenate([a[2] for a in Tr])
    MSK = np.concatenate([a[3] for a in Tr])
    ERR = np.concatenate([a[4] for a in Tr])
    Xv = np.concatenate([a[0] for a in Va])
    Wv = np.concatenate([a[1] for a in Va])
    JMPv = np.concatenate([a[2] for a in Va])
    MSKv = np.concatenate([a[3] for a in Va])
    ERRv = np.concatenate([a[4] for a in Va])
    N = len(X)
    print('rows=%d train=%d val=%d valfiles=%d' % (N + len(Xv), N, len(Xv), len(Va)), flush=True)
    dev = 'cpu'
    torch.set_num_threads(4)
    net = ContAR().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    conf_lab_all = (ERR < 24.0).astype(np.float32)

    def batch(idxs, Ds, ns=0.0):
        XX, WW, JJ, MM, EE = Ds
        Xt = torch.tensor(XX[idxs], dtype=torch.float32, device=dev)
        Wt_ = torch.tensor(WW[idxs], dtype=torch.float32, device=dev)
        Jt = torch.tensor(JJ[idxs], dtype=torch.float32, device=dev)
        Mt = torch.tensor(MM[idxs], dtype=torch.long, device=dev)
        Wt = torch.tensor(np.vectorize(MASKW.get)(MM[idxs]), dtype=torch.float32, device=dev)
        Ct = torch.tensor((EE[idxs] < 24.0).astype(np.float32), dtype=torch.float32, device=dev)
        fin = torch.zeros_like(Wt_)
        fin[:, 1:] = Wt_[:, :-1]
        jin = torch.zeros_like(Jt)
        jin[:, 1:] = Jt[:, :-1]
        if ns > 0.0:
            # AR-robustness: eval feeds the net's OWN noisy wishes back for
            # 20 steps; pure teacher forcing never shows it that regime.
            # Noise on the teacher context simulates AR drift (cheap
            # scheduled sampling; ramped in by epoch).
            fin = fin + ns * torch.randn_like(fin)
        w, s, jlog, conf = net(Xt, fin, jin.long())
        se = ((w - Wt_[:, :, :2]) ** 2).sum(-1) + (s - Wt_[:, :, 2]) ** 2
        je = bce(jlog, Jt)
        wsum = Wt.sum(-1).clamp_min(1.0)
        row = ((se + 0.3 * je) * Wt).sum(-1) / wsum
        has = (Wt.sum(-1) > 1e-6).float()
        loss_main = (row * has).sum() / has.sum().clamp_min(1.0)
        lc = bce(conf, Ct).mean(dim=1).mean()
        loss = loss_main + 0.3 * lc
        with torch.no_grad():
            werr = torch.sqrt((((w - Wt_[:, :, :2]) ** 2).sum(-1) * (Wt > 0)).sum()
                              / (Wt > 0).float().sum().clamp_min(1.0))
        return loss, werr

    best = 1e18
    B = 128
    Dtr = (X, W, JMP, MSK, ERR)
    Dva = (Xv, Wv, JMPv, MSKv, ERRv)
    ti = np.arange(N)
    vi = np.arange(len(Xv))
    for ep in range(epochs):
        net.train()
        rng.shuffle(ti)
        ns = 0.3 * min(1.0, ep / 20.0)
        tot, na = 0.0, 0
        for s in range(0, len(ti), B):
            bi = ti[s:s + B]
            opt.zero_grad()
            loss, _ = batch(bi, Dtr, ns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(bi)
            na += len(bi)
        net.eval()
        with torch.no_grad():
            vl, vw = 0.0, 0.0
            for s in range(0, len(vi), B):
                bi = vi[s:s + B]
                loss, werr = batch(bi, Dva)
                vl += loss.item() * len(bi)
                vw += werr.item() * len(bi)
            vl /= max(1, len(vi))
            vw /= max(1, len(vi))
        print('ep=%d train=%.4f val=%.4f werr=%.4f' % (ep, tot / na, vl, vw), flush=True)
        if vl < best:
            best = vl
            torch.save({'net': net.state_dict(), 'cfg': (D, NL, NH, FF, HIST, FUT, NFEAT),
                        'cont': True}, out)
    print('best=%.4f saved %s' % (best, out), flush=True)


if __name__ == '__main__':
    main()
