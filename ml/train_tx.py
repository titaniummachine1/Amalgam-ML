"""train_tx.py — tiny early-exit AR transformer (Maia-style, tick-tokens).
Input: build_rollout.py npz (X history 33x14, TOK/JMP/MSK future 20,
  ERR @6 horizons, POS). Predicts NEXT INPUT TOKEN autoregressively
  (teacher forcing) + jump flag + confidence head (P(match<24hu) @ horizons,
  = future AI hitchance + early-exit signal).
Model: decoder-only, d=64/2L/4H/ff128 (~120k params), sincos positions,
  causal mask, history projected from 14-dim, future from token emb.
Loss: CE(tokens)*maskw + 0.3*BCE(jump)*maskw + 0.3*BCE(conf vs ERR<24).
Mask weights: clean 1.0, capped 0.5, flight/blast 0.0 (never reward/penalty
  on unpredictable ticks; context preserved in history).
Positions are NOT predicted (sim integrates inputs — physics stays exact).
Export + int8 + C++ port at ship time (pattern: export_int8.py).
Usage: python ml/train_tx.py <train.npz> [epochs] [out.pt]
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
NVOC = 25
HIST = 33
FUT = 20
NFEAT = 14
NCONF = 6
MASKW = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}


class TinyAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_proj = nn.Linear(NFEAT, D)
        self.tok_emb = nn.Embedding(NVOC, D)
        self.jmp_emb = nn.Embedding(2, D)
        layer = nn.TransformerDecoderLayer(d_model=D, nhead=NH, dim_feedforward=FF,
                                           batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=NL)
        self.mem_proj = nn.Linear(D, D)
        self.lm = nn.Linear(D, NVOC)
        self.jhead = nn.Linear(D, 1)
        self.chead = nn.Linear(D, NCONF)
        # fixed sincos positions (no extra params)
        pe = torch.zeros(HIST + FUT, D)
        pos = torch.arange(HIST + FUT, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, X, tok_in, jmp_in):
        B = X.size(0)
        h = self.hist_proj(X) + self.pe[:HIST].unsqueeze(0)
        mem = self.mem_proj(h)
        f = self.tok_emb(tok_in) + self.jmp_emb(jmp_in) + self.pe[HIST:HIST + FUT].unsqueeze(0)
        causal = torch.triu(torch.ones(FUT, FUT, dtype=torch.bool), 1).to(X.device)
        y = self.dec(f, mem, tgt_mask=causal)
        return self.lm(y), self.jhead(y).squeeze(-1), self.chead(y.mean(dim=1))


def main():
    npz = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = sys.argv[3] if len(sys.argv) > 3 else 'tx_tiny.pt'
    d = np.load(npz, allow_pickle=True)
    X, TOK, JMP, MSK, ERR = d['X'], d['TOK'], d['JMP'], d['MSK'], d['ERR']
    N = len(X)
    idx = np.arange(N)
    rng = np.random.RandomState(7)
    rng.shuffle(idx)
    nv = max(100, N // 10)
    vi, ti = idx[:nv], idx[nv:]
    print('rows=%d train=%d val=%d' % (N, len(ti), len(vi)), flush=True)
    dev = 'cpu'
    torch.set_num_threads(4)
    net = TinyAR().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    ce = nn.CrossEntropyLoss(reduction='none')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    conf_lab_all = (ERR < 24.0).astype(np.float32)

    def batch(idxs, train):
        Xt = torch.tensor(X[idxs], dtype=torch.float32, device=dev)
        Tt = torch.tensor(TOK[idxs], dtype=torch.long, device=dev)
        Jt = torch.tensor(JMP[idxs], dtype=torch.float32, device=dev)
        Mt = torch.tensor(MSK[idxs], dtype=torch.long, device=dev)
        Wt = torch.tensor(np.vectorize(MASKW.get)(MSK[idxs]), dtype=torch.float32, device=dev)
        Ct = torch.tensor(conf_lab_all[idxs], dtype=torch.float32, device=dev)
        # teacher forcing: input = <start=none> + TOK[:-1]
        tin = torch.zeros_like(Tt)
        tin[:, 1:] = Tt[:, :-1]
        jin = torch.zeros_like(Jt)
        jin[:, 1:] = Jt[:, :-1]
        logits, jlog, conf = net(Xt, tin, jin.long())
        lce = (ce(logits.reshape(-1, NVOC), Tt.reshape(-1)).reshape_as(Tt) * Wt)
        lj = (bce(jlog, Jt) * Wt)
        lc = bce(conf, Ct).mean(dim=1)
        denom = Wt.sum().clamp_min(1.0)
        loss = lce.sum() / denom + 0.3 * lj.sum() / denom + 0.3 * lc.mean()
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = ((pred == Tt).float() * (Wt > 0)).sum() / (Wt > 0).float().sum().clamp_min(1.0)
        return loss, acc

    best = 1e18
    B = 128
    for ep in range(epochs):
        net.train()
        rng.shuffle(ti)
        tot, na = 0.0, 0
        for s in range(0, len(ti), B):
            bi = ti[s:s + B]
            opt.zero_grad()
            loss, _ = batch(bi, True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
            na += len(bi)
        net.eval()
        with torch.no_grad():
            vl, vacc = 0.0, 0.0
            for s in range(0, len(vi), B):
                bi = vi[s:s + B]
                loss, acc = batch(bi, False)
                vl += loss.item() * len(bi)
                vacc += acc.item() * len(bi)
            vl /= len(vi)
            vacc /= len(vi)
        print('ep=%d train=%.4f val=%.4f vacc=%.3f' % (ep, tot / na, vl, vacc), flush=True)
        if vl < best:
            best = vl
            torch.save({'net': net.state_dict(), 'cfg': (D, NL, NH, FF, HIST, FUT)}, out)
    print('best=%.4f saved %s' % (best, out), flush=True)


main()
