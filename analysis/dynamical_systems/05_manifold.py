import json, os, time, numpy as np, torch
from common import *

set_seed()
NSEQ, T, SKIP, STRIDE = int(os.environ.get("NSEQ", 96)), 512, 64, 8
m = load_model(torch.float64, fast_conv=True)
ids, _ = corpus_ids()
win = windows(ids, NSEQ, T)
st = zero_state(NSEQ)
samples = []  # [S, NSEQ, 12, 2, 768]
t0 = time.time()
with torch.no_grad():
    for t in range(T):
        e = m.embeddings(torch.tensor(win[:, t])).unsqueeze(1)
        _, st = step(m, e, st)
        if t >= SKIP and (t - SKIP) % STRIDE == 0:
            samples.append(torch.stack([torch.stack([hr.reshape(NSEQ, -1), hi.reshape(NSEQ, -1)], 1) for _, hr, hi in st], 1).float().numpy())
        if (t + 1) % 64 == 0: print(t + 1, f"{time.time() - t0:.0f}s", flush=True)
X = np.stack(samples, 0)           # [S, NSEQ, NL, 2, D]
S = X.shape[0]
seq_id = np.tile(np.arange(NSEQ), S)
X = X.reshape(S * NSEQ, NL, 2, D)
Nsamp = X.shape[0]
res = {"env": env_info(), "n_samples": int(Nsamp), "n_sequences": NSEQ, "positions_per_seq": int(S), "complex_dim_per_layer": D,
       "real_dim_per_layer": 2 * D, "complex_dim_full": NL * D, "real_dim_full": 2 * NL * D}
THR = [0.5, 0.8, 0.9, 0.95, 0.99]


def spectrum_stats(ev):
    ev = np.clip(ev, 0, None); tot = ev.sum(); c = np.cumsum(ev) / tot
    out = {f"n_for_{int(q*100)}pct": int(np.searchsorted(c, q) + 1) for q in THR}
    out["participation_ratio"] = float(tot ** 2 / (ev ** 2).sum()); out["rank_nonzero(>1e-10*max)"] = int((ev > 1e-10 * ev.max()).sum())
    out["top1_frac"] = float(ev[0] / tot); out["top10_frac"] = float(ev[:10].sum() / tot)
    return out, ev / tot


def pca_real(Y):
    Y = Y - Y.mean(0)
    if Y.shape[0] >= Y.shape[1]:
        ev = np.linalg.eigvalsh(Y.T.astype(np.float64) @ Y / (len(Y) - 1))[::-1]
    else:
        ev = np.linalg.eigvalsh(Y.astype(np.float64) @ Y.T / (len(Y) - 1))[::-1]
    return ev


per_layer = {}
for l in range(NL):
    Z = X[:, l, 0] + 1j * X[:, l, 1]                    # complex [N, 768]
    Zc = Z - Z.mean(0)
    evc = np.linalg.eigvalsh(Zc.conj().T.astype(np.complex128) @ Zc / (len(Z) - 1))[::-1]   # Hermitian PCA
    evr = pca_real(np.concatenate([X[:, l, 0], X[:, l, 1]], 1))
    sc, cc = spectrum_stats(evc); sr, cr = spectrum_stats(evr)
    per_layer[l] = {"complex_pca": sc, "real_equiv_pca": sr, "mean_modulus": float(np.abs(Z).mean())}
    per_layer[l]["cum_var_complex_first64"] = np.cumsum(cc)[:64]; per_layer[l]["cum_var_real_first128"] = np.cumsum(cr)[:128]
    per_layer[l]["real_spectrum_normalized_first256"] = cr[:256]
    print("layer", l, sc, flush=True)
res["per_layer"] = per_layer
# full concatenated state (real-equivalent 18432); sample-Gram trick
full = np.concatenate([X[:, :, 0].reshape(Nsamp, -1), X[:, :, 1].reshape(Nsamp, -1)], 1)
evf = pca_real(full); sf, cf = spectrum_stats(evf)
res["full_state_real_equiv_pca"] = sf; res["full_state_cum_var_first512"] = np.cumsum(cf)[:512]
Zf = X[:, :, 0].reshape(Nsamp, -1) + 1j * X[:, :, 1].reshape(Nsamp, -1); Zf = Zf - Zf.mean(0)
evfc = np.linalg.eigvalsh(Zf.astype(np.complex128) @ Zf.conj().T / (len(Zf) - 1))[::-1]
sfc, _ = spectrum_stats(evfc); res["full_state_complex_pca"] = sfc
res["pca_caveat"] = "N_samples < real_dim_full => rank <= N-1; samples within a sequence are highly autocorrelated."

# intrinsic dimension (real-equivalent vectors), neighbours from other sequences only
def knn_other(Y, k=20):
    Y = torch.tensor(Y, dtype=torch.float32)
    d = torch.cdist(Y, Y)
    sid = torch.tensor(seq_id)
    d[sid[:, None] == sid[None, :]] = float("inf")
    dk, _ = torch.topk(d, k, largest=False)
    return dk.numpy().astype(np.float64)

def twonn(dk):
    mu = dk[:, 1] / dk[:, 0]; mu = mu[np.isfinite(mu) & (mu > 1)]
    mu = np.sort(mu); n = len(mu); F = np.arange(1, n + 1) / n
    sel = slice(0, int(0.9 * n))
    x, y = np.log(mu[sel]), -np.log(1 - F[sel]); return float((x * y).sum() / (x * x).sum())

def mle(dk, k=10):
    d = dk[:, :k]; return float(1.0 / np.mean(np.log(d[:, k - 1:k] / d[:, :k - 1]).mean(1)))

sub = np.random.RandomState(0).choice(Nsamp, min(Nsamp, 4000), replace=False)
seq_id_full = seq_id; seq_id = seq_id_full[sub]
idd = {}
for l in range(NL):
    Y = np.concatenate([X[sub, l, 0], X[sub, l, 1]], 1)
    dk = knn_other(Y)
    idd[l] = {"twonn": twonn(dk), "mle_k10": mle(dk, 10), "mle_k20": mle(dk, 20)}
    print("ID layer", l, idd[l], flush=True)
Y = np.concatenate([X[sub][:, :, 0].reshape(len(sub), -1), X[sub][:, :, 1].reshape(len(sub), -1)], 1)
dk = knn_other(Y); idd["full"] = {"twonn": twonn(dk), "mle_k10": mle(dk, 10), "mle_k20": mle(dk, 20)}
res["intrinsic_dimension"] = idd
res["id_caveat"] = "k-NN ID estimators are biased low for N << exp(d) and by non-uniform density; treat as upper-/lower-bound-ish indicators."

# local PCA dimension (layer-wise, k=50 neighbours from other sequences, 90% local variance), 300 query points
loc = {}
for l in [0, 5, 11]:
    Y = np.concatenate([X[sub, l, 0], X[sub, l, 1]], 1).astype(np.float32)
    Yt = torch.tensor(Y); d = torch.cdist(Yt, Yt); sid = torch.tensor(seq_id)
    d[sid[:, None] == sid[None, :]] = float("inf")
    nn = torch.topk(d, 50, largest=False)[1].numpy()
    dims = []
    for q in range(0, len(sub), len(sub) // 300):
        P = Y[nn[q]] - Y[nn[q]].mean(0)
        ev = np.linalg.svd(P, compute_uv=False) ** 2; c = np.cumsum(ev) / ev.sum()
        dims.append(int(np.searchsorted(c, 0.9) + 1))
    loc[l] = {"local_pca_dim_90pct_k50_mean": float(np.mean(dims)), "median": float(np.median(dims)), "max_possible": 49}
res["local_pca"] = loc
save_json("05_manifold.json", res)
