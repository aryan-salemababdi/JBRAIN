import json, time, numpy as np, torch
from scipy.sparse.linalg import LinearOperator, svds
from torch.func import jvp, vjp, jacfwd
from common import *

set_seed()
m = load_model(torch.float64)
ids, _ = corpus_ids()
T = 600
nat = windows(ids, 2, T)
rng = np.random.RandomState(SEED)
seqs = {"natural_A": nat[0], "natural_B": nat[1],
        "repeat_token": np.full(T, ids[1000]), "uniform_random_tokens": rng.randint(0, VOCAB, T)}
res = {"env": env_info(), "T": T}

# analytic spectra along trajectories
G = {}
for name, s in seqs.items():
    tr = trajectory_gates(m, m.embeddings(torch.tensor(s)))
    G[name] = (tr["gamma"][:, :, 0].numpy(), tr["dphi"][:, :, 0].numpy())  # [T,NL,NH]
spec = {}
for name, (g, p) in G.items():
    rho_t = g.reshape(T, -1).max(1)
    spec[name] = {
        "spectral_radius_t": rho_t, "mean_modulus_t": g.reshape(T, -1).mean(1),
        "min_modulus_t": g.reshape(T, -1).min(1),
        "frac_modulus_gt_0.99": float((g > 0.99).mean()), "frac_modulus_gt_0.98": float((g > 0.98).mean()),
        "frac_modulus_gt_0.95": float((g > 0.95).mean()), "frac_modulus_lt_0.92": float((g < 0.92).mean()),
        "rho_max_over_t": float(rho_t.max()), "rho_mean_over_t": float(rho_t.mean()),
        "gamma_hist_counts": np.histogram(g.ravel(), bins=np.linspace(0.9, 1.0, 51))[0],
        "abs_dphi_mean": float(np.abs(p).mean()), "abs_dphi_quantiles": np.quantile(np.abs(p), [.1, .5, .9]),
        "mean_gamma_per_layer_head": g.mean(0),            # [NL,NH]
        "mean_log_gamma_per_layer_head": np.log(g).mean(0),
        "mean_dphi_per_layer_head": p.mean(0),
    }
res["analytic_spectrum"] = spec
res["eigenvalue_count"] = {"real_state_dim": NL * 3 * D, "zero_modes(cache)": NL * D, "nonzero_modes(h, real dims)": NL * 2 * D,
                           "distinct_complex_eigs_per_step": NL * NH, "multiplicity_each_conj_pair": HD}

# full-Jacobian numerical validation on 2-layer sub-stack at t0
t0 = 300
s = seqs["natural_A"]; emb = m.embeddings(torch.tensor(s))[None]
with torch.no_grad():
    st = zero_state(1)
    for t in range(t0):
        _, st = step(m, emb[:, t:t + 1], st, upto=2)
x0 = flat_state(st)[0]
e_next = emb[:, t0:t0 + 1]


def f2(v):
    _, nx = step(m, e_next, unflat_state(v[None], 2), upto=2)
    return flat_state(nx)[0]

t1 = time.time()
J = jacfwd(f2)(x0).numpy()
ev = np.linalg.eigvals(J)
print("full J 2-layer", J.shape, "eig time", time.time() - t1)
tr2 = trajectory_gates(m, emb[0, :t0 + 1])
g2 = tr2["gamma"][t0, :2, 0].numpy(); p2 = tr2["dphi"][t0, :2, 0].numpy()
an = np.concatenate([np.repeat((g2 * np.exp(1j * p2)).ravel(), HD), np.repeat((g2 * np.exp(-1j * p2)).ravel(), HD)])
nz = ev[np.abs(ev) > 1e-6]
a_sorted = np.sort(np.abs(an)); n_sorted = np.sort(np.abs(nz))
res["full_jacobian_check_2layers"] = {
    "dim": list(J.shape), "n_eig_nonzero(|l|>1e-6)": int(len(nz)), "n_analytic": int(len(an)),
    "max_abs_diff_sorted_moduli": float(np.abs(a_sorted - n_sorted).max()) if len(nz) == len(an) else None,
    "numeric_spectral_radius": float(np.abs(ev).max()), "analytic_spectral_radius": float(np.abs(an).max()),
    "numeric_sigma_max": float(np.linalg.svd(J, compute_uv=False)[0]),
    "n_eig_abs<1e-6": int((np.abs(ev) <= 1e-6).sum()),
    "max_abs_eig_among_expected_zero_modes": float(np.sort(np.abs(ev))[:2 * D].max()),
}
print(res["full_jacobian_check_2layers"])

# non-normal amplification: top singular values of the full 12-layer J_t
N = NL * 3 * D
sig = {}
for name in ["natural_A", "natural_B", "repeat_token", "uniform_random_tokens"]:
    emb = m.embeddings(torch.tensor(seqs[name]))[None]
    out = {}
    with torch.no_grad():
        st = zero_state(1)
        t = 0
        for tp in [1, 5, 10, 25, 50, 100, 200, 400, 599]:
            while t < tp:
                _, st = step(m, emb[:, t:t + 1], st); t += 1
            x0 = flat_state(st)[0].clone()
            e_n = emb[:, tp:tp + 1]

            def f(v, e_n=e_n):
                _, nx = step(m, e_n, unflat_state(v[None]))
                return flat_state(nx)[0]
            _, vjp_fn = vjp(f, x0)
            op = LinearOperator((N, N), dtype=np.float64,
                                matvec=lambda v: jvp(f, (x0,), (torch.from_numpy(np.asarray(v).ravel().copy()),))[1].numpy(),
                                rmatvec=lambda v: vjp_fn(torch.from_numpy(np.asarray(v).ravel().copy()))[0].numpy())
            sv = svds(op, k=6, which="LM", return_singular_vectors=False, tol=1e-6, maxiter=200)
            out[tp] = sorted(sv.tolist(), reverse=True)
            print(name, tp, [round(x, 4) for x in out[tp]], flush=True)
            st = [tuple(x.clone() for x in s_) for s_ in st]
    sig[name] = out
res["sigma_top6_full_jacobian"] = sig
save_json("02_jacobian.json", res)
