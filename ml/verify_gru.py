"""Verify the C++ GRU port math == torch: numpy reimplementation following
BehaviorPrediction.cpp gate order/row-major layout, dequantized int8 weights,
vs torch forward on a real eval warmup. Must match to ~1e-5."""
import os
import numpy as np
import torch
import torch.nn as nn

D = os.path.join(os.path.expanduser('~'), 'Downloads', 'mge_ds')
H, NIN = 96, 14


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NIN, H, batch_first=True)
        self.head = nn.Linear(H, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


# int8 roundtrip exactly like export_int8.py (ALL six arrays incl. biases)
sd = torch.load(os.path.join(D, 'gru_soldier_dense66.pt'), map_location='cpu')
q = {}
for n, p in sd.items():
    s = p.float().abs().max().item() / 127.0
    q[n] = (torch.round(p.float() / s).clamp(-128, 127).to(torch.int8), s)


def dq(n):
    qq, s = q[n]
    return (qq.float() * s).numpy()


Wih, Whh = dq('gru.weight_ih_l0'), dq('gru.weight_hh_l0')
Bih, Bhh = dq('gru.bias_ih_l0'), dq('gru.bias_hh_l0')
Hw, Hb = dq('head.weight'), dq('head.bias')

ev = np.load(os.path.join(D, 'evalA.npz'), mmap_mode='r')
k = np.load(os.path.join(D, 'evalAk.npz'))['K']
X = np.array(ev['X'])
m = (k[:, 3] == 3)
X = X[m][:8]


def cpp_gru(xb):
    h = np.zeros(H, dtype=np.float64)
    for t in range(xb.shape[0]):
        x = xb[t].astype(np.float64)
        hn = np.zeros(H)
        for j in range(H):
            ir = Bih[j] + x @ Wih[j] + h @ Whh[j]
            iz = Bih[H + j] + x @ Wih[H + j] + h @ Whh[H + j]
            inn = Bih[2 * H + j] + x @ Wih[2 * H + j] + h @ Whh[2 * H + j]
            r = 1 / (1 + np.exp(-ir))
            z = 1 / (1 + np.exp(-iz))
            # torch: n = tanh(b_in + W_in.x + r*(b_hn + W_hn.h)) — mirrors C++ exactly
            nn_ = np.tanh((Bih[2 * H + j] + x @ Wih[2 * H + j]) +
                          r * (Bhh[2 * H + j] + h @ Whh[2 * H + j]))
            hn[j] = (1 - z) * nn_ + z * h[j]
        h = hn
    return Hw @ h + Hb


net = Net()
dsd = {n: torch.from_numpy(dq(n).astype(np.float32)) for n in sd}
net.load_state_dict(dsd)
net.eval()
with torch.no_grad():
    ref = net(torch.from_numpy(X)).numpy()
got = np.stack([cpp_gru(xb) for xb in X])
print('max abs diff vs torch:', np.abs(ref - got).max())
