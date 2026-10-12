import json, time, numpy as np, torch
from torch.func import jvp, vmap
from common import *

set_seed()
m = load_model(torch.float64, fast_conv=True)
ids, _ = corpus_ids()
T = 1000
PRE = 128
win = windows(ids, 3, PRE + T)
EPS = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
res = {"env": env_info(), "T": T, "eps": EPS}


def h_vec(st):  # all h (real 18432) per batch row, layer-major
    return torch.cat([torch.cat([hr.reshape(hr.shape[0], -1), hi.reshape(hi.shape[0], -1)], 1) for _, hr, hi in st], 1)


def per_layer_dist(a, b):
    out = []
    for (_, ar, ai), (_, br, bi) in zip(a, b):
        out.append(torch.sqrt(((ar - br) ** 2 + (ai - bi) ** 2).reshape(ar.shape[0], -1).sum(1)))
    return torch.stack(out, 1)  # [B,NL]


fd = {}
for bname, seqi in [("zero_init", 0), ("after_128tok_prefix", 1), ("after_128tok_prefix_seq2", 2)]:
    toks = torch.tensor(win[seqi])
    emb_all = m.embeddings(toks)
    st0 = zero_state(1)
    t_start = 0
    if bname != "zero_init":
        with torch.no_grad():
            for t in range(PRE):
                _, st0 = step(m, emb_all[None, t:t + 1], st0)
        t_start = PRE
    nE = len(EPS)
    # batch: row0 = reference, rows 1..nE perturbed (direction random, h-only, d_0 = eps exactly)
    g = torch.Generator().manual_seed(SEED + seqi)
    base = [tuple(x.expand(nE + 1, *x.shape[1:]).clone() for x in s) for s in st0]
    dirs = torch.randn(nE, NL, 2, D, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.reshape(nE, -1).norm(dim=1).view(nE, 1, 1, 1)
    for i, e in enumerate(EPS):
        for l in range(NL):
            base[l][1][i + 1] += (e * dirs[i, l, 0]).view(NH, HD)
            base[l][2][i + 1] += (e * dirs[i, l, 1]).view(NH, HD)
    st = base
    dist = np.zeros((T, nE)); dist_l = np.zeros((T, nE, NL))
    with torch.no_grad():
        for t in range(T):
            e_t = emb_all[None, t_start + t:t_start + t + 1].expand(nE + 1, 1, D)
            _, st = step(m, e_t, st)
            ref = [(s[0][:1].expand(nE, -1, -1), s[1][:1].expand(nE, -1, -1), s[2][:1].expand(nE, -1, -1)) for s in st]
            pert = [(s[0][1:], s[1][1:], s[2][1:]) for s in st]
            dl = per_layer_dist(ref, pert)
            dist_l[t] = dl.numpy(); dist[t] = dl.pow(2).sum(1).sqrt().numpy()
    fd[bname] = {"dist": dist, "dist_per_layer": dist_l[::10]}
    d0 = np.array(EPS)
    tt = np.arange(1, T + 1)[:, None]
    fd[bname]["finite_time_lyapunov"] = np.log(dist / d0) / tt
    print(bname, "d_T/d_0 per eps:", (dist[-1] / d0).round(4), flush=True)
res["finite_difference"] = fd

# Benettin tangent propagation, top-k exponents, along natural trajectory
K = 4
toks = torch.tensor(win[0]); emb_all = m.embeddings(toks)
N = NL * 3 * D
st = zero_state(1)
g = torch.Generator().manual_seed(SEED)
Q = torch.zeros(N, K, dtype=torch.float64)
for l in range(NL):
    Q[l * 3 * D + D:(l + 1) * 3 * D] = torch.randn(2 * D, K, generator=g, dtype=torch.float64)
Q, _ = torch.linalg.qr(Q)
logsum = torch.zeros(K, dtype=torch.float64)
hist = []
t0 = time.time()
for t in range(T):
    e_t = emb_all[None, t:t + 1]
    x0 = flat_state(st)[0]

    def f(v, e_t=e_t):
        _, nx = step(m, e_t, unflat_state(v[None]))
        return flat_state(nx)[0]
    with torch.no_grad():
        Z = vmap(lambda tv: jvp(f, (x0,), (tv,))[1], in_dims=1, out_dims=1)(Q)
        _, st = step(m, e_t, st)
    Q, R = torch.linalg.qr(Z)
    logsum += torch.log(torch.abs(torch.diagonal(R)))
    if (t + 1) % 10 == 0:
        hist.append((logsum / (t + 1)).numpy().copy())
    if (t + 1) % 100 == 0:
        print("benettin", t + 1, (logsum / (t + 1)).numpy().round(5), f"{time.time() - t0:.0f}s", flush=True)
res["benettin"] = {"K": K, "T": T, "exponents_per_step": (logsum / T).numpy(), "history_every10": np.array(hist)}

# theory: bound from exact block-triangular structure => max over (l,h) of time-mean log gamma
tr = trajectory_gates(m, emb_all[None, :T])
lg = torch.log(tr["gamma"][:, :, 0]).mean(0).numpy()  # [NL,NH]
res["theory"] = {"max_over_channels_mean_log_gamma": float(lg.max()), "min_over_channels_mean_log_gamma": float(lg.min()),
                 "sorted_top8_mean_log_gamma": np.sort(lg.ravel())[::-1][:8], "log_0.999": float(np.log(0.999))}
save_json("03_lyapunov.json", res)
