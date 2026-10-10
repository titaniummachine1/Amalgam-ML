"""Init-exactness check: variant-A net (champion warm start, zero-init deltas)
must reproduce the champion trajectory bit-near-exactly on eval rows."""
import os
import sys

print('dbg_init start', flush=True)
import numpy as np
print('numpy ok', flush=True)
import torch

print('torch ok', flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_attn66 as T
from hit_eval_final import Net20, rollout20

D = T.D
champ = Net20()
champ.load_state_dict(torch.load(os.path.join(D, 'gru_threatsimB.pt'), map_location='cpu'))
champ.eval()
net = T.NetA()
cs = torch.load(T.WARM, map_location='cpu')
sd = net.state_dict()
for k in ['gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0',
          'head.weight', 'head.bias']:
    sd[k] = cs[k]
sd['gru.weight_ih_l0'][:, :20] = cs['gru.weight_ih_l0']
sd['gru.weight_ih_l0'][:, 20:] = 0
net.load_state_dict(sd)
net.eval()
xb, sb = T.get_batch(T.Xb_ev, T.spos_ev, T.TW_ev, T.S0_ev, np.arange(256))
with torch.no_grad():
    a = T.unroll(net, xb, sb).numpy()
b = rollout20(champ, xb.numpy(), sb.numpy(), True)
print('max|mine - champ| =', float(np.abs(a - b).max()))
print('mean|mine - champ| =', float(np.abs(a - b).mean()))
