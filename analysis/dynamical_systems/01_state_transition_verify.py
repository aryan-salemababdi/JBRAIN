import torch, numpy as np, time
from torch.func import jvp
from common import *

set_seed()
m = load_model(torch.float64)
ids, chash = corpus_ids()
res = {"env": env_info(), "corpus_sha256": chash, "corpus_tokens": int(len(ids))}
toks = torch.tensor(windows(ids, 1, 600)[0])
emb = m.embeddings(toks)[None]  # [1,T,D]

# hand-written transition H_t = g e^{i dphi} H_{t-1} + val vs layer code
st = zero_state(1)
maxerr = 0.0
with torch.no_grad():
    for t in range(64):
        x = emb[:, t:t + 1]
        for i, layer in enumerate(m.layers):
            u = layer.ln1(x)
            g, p, vr, vi, _ = gates(layer.core, u, st[i][0])
            Hp = torch.complex(st[i][1], st[i][2])
            Hn = (g.unsqueeze(-1) * torch.exp(1j * p.unsqueeze(-1))) * Hp + torch.complex(vr, vi)
            x, st[i] = layer(x, st[i])
            Hc = torch.complex(st[i][1], st[i][2])
            maxerr = max(maxerr, (Hn - Hc).abs().max().item())
res["hand_equation_vs_code_max_abs_err"] = maxerr

with torch.no_grad():
    l1, s1 = m(toks[None, :1])
    l2, s2 = m(toks[None, :1], zero_state(1))
res["zero_state_vs_None_logit_maxdiff"] = (l1 - l2).abs().max().item()

with torch.no_grad():
    lp, sp = m(toks[None, :200])
    s = None
    for t in range(200):
        lr, s = m(toks[None, t:t + 1], s)
res["parallel_vs_recurrent_last_logit_maxdiff_fp64"] = (lp[:, -1] - lr[:, -1]).abs().max().item()
res["parallel_vs_recurrent_h_maxdiff_fp64"] = max(
    max((a[1] - b[1]).abs().max().item(), (a[2] - b[2]).abs().max().item()) for a, b in zip(sp, s))

tr = trajectory_gates(m, emb[0], record_h=True)
H = tr["H"][0]  # [T, 27648]
mod_max = 0.0
for l in range(NL):
    o = l * (D + 2 * D)
    hr, hi = H[:, o + D:o + 2 * D], H[:, o + 2 * D:o + 3 * D]
    mod_max = max(mod_max, torch.sqrt(hr ** 2 + hi ** 2).max().item())
res["max_channel_modulus_natural_600"] = mod_max
res["gamma_min"] = tr["gamma"].min().item(); res["gamma_max"] = tr["gamma"].max().item()
res["gamma_mean"] = tr["gamma"].mean().item()
res["dphi_abs_mean"] = tr["dphi"].abs().mean().item()

t0 = 300
with torch.no_grad():
    st = zero_state(1)
    for t in range(t0):
        _, st = step(m, emb[:, t:t + 1], st)
s0 = flat_state(st)[0].clone()
e_next = emb[:, t0:t0 + 1]


def f(v):
    _, nx = step(m, e_next, unflat_state(v[None]))
    return flat_state(nx)[0]

blk = D * 3  # per-layer block: cache(768) + hr(768) + hi(768)
tri = []
diag_err = []
gate_t = trajectory_gates(m, emb[0, :t0 + 1])  # gamma/dphi at step t0
for lp in range(NL):          # perturb layer lp's h only
    tang = torch.zeros_like(s0)
    o = lp * blk
    tang[o + D:o + 3 * D] = torch.randn(2 * D, dtype=torch.float64)
    _, out = jvp(f, (s0,), (tang,))
    for l in range(NL):
        n = out[l * blk:(l + 1) * blk].norm().item()
        if l < lp:
            tri.append(n)  # must be exactly 0 (earlier layer cannot see later layer's state)
        if l == lp:
            hr, hi = out[l * blk + D:l * blk + 2 * D], out[l * blk + 2 * D:l * blk + 3 * D]
            th = tang[o + D:o + 2 * D], tang[o + 2 * D:o + 3 * D]
            g = gate_t["gamma"][t0, l, 0].repeat_interleave(HD); p = gate_t["dphi"][t0, l, 0].repeat_interleave(HD)
            dh = g * torch.exp(1j * p) * torch.complex(*th)
            diag_err.append((torch.complex(hr, hi) - dh).abs().max().item())
res["upper_triangle_block_max_abs"] = max(tri)
res["diag_block_vs_gamma_exp_i_dphi_max_err"] = max(diag_err)
# diagonal block cache->cache must be zero (cache_t = ln1(x_t) independent of layer-l state)
tang = torch.zeros_like(s0); tang[:D] = torch.randn(D, dtype=torch.float64)
_, out = jvp(f, (s0,), (tang,))
res["cache0_to_cache0_block_norm"] = out[:D].norm().item()
save_json("01_state_transition_verify.json", res)
print(json.dumps({k: v for k, v in res.items() if k not in ("env",)}, indent=1))
