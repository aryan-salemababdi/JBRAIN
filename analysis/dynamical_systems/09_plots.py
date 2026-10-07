import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); P = os.path.join(os.path.dirname(R), "plots")
os.makedirs(P, exist_ok=True)
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A7A"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130,
                     "axes.prop_cycle": plt.cycler(color=C), "font.size": 9})

def J(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None

def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(P, n)); plt.close(fig); print("plot", n)

j2 = J("02_jacobian.json")
if j2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (n, s) in j2["analytic_spectrum"].items():
        ax[0].plot(s["spectral_radius_t"], label=n, lw=1)
    ax[0].axhline(1, c="k", ls="--", lw=.8); ax[0].axhline(.999, c="k", ls=":", lw=.8)
    ax[0].set_ylim(.95, 1.005); ax[0].set_xlabel("timestep t"); ax[0].set_ylabel("spectral radius  max|λ|"); ax[0].legend(fontsize=7); ax[0].set_title("Exact one-step spectral radius (=max γ)")
    for i, (n, d) in enumerate(j2["sigma_top6_full_jacobian"].items()):
        ts = sorted(d, key=int); ax[1].plot([int(t) for t in ts], [d[t][0] for t in ts], "o-", c=C[i], label=n, ms=3)
    ax[1].axhline(1, c="k", ls="--", lw=.8); ax[1].set_xscale("log"); ax[1].set_xlabel("timestep t"); ax[1].set_ylabel("σ_max(J_t)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Largest singular value of full J_t (non-normal gain)")
    save(fig, "1_jacobian_spectral_radius_vs_t.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); e = np.linspace(.9, 1.0, 51)
    for i, (n, s) in enumerate(j2["analytic_spectrum"].items()):
        h = np.array(s["gamma_hist_counts"], float); ax[0].step(e[:-1], h / h.sum(), where="post", label=n, c=C[i])
    ax[0].set_yscale("log"); ax[0].set_xlabel("|λ| = γ"); ax[0].set_ylabel("fraction of modes"); ax[0].legend(fontsize=7); ax[0].set_title("Eigenvalue modulus distribution")
    mg = np.array(j2["analytic_spectrum"]["natural_A"]["mean_gamma_per_layer_head"]); im = ax[1].imshow(mg, aspect="auto", cmap="viridis")
    ax[1].set_xlabel("head"); ax[1].set_ylabel("layer"); ax[1].set_title("mean γ per (layer, head), natural_A"); plt.colorbar(im, ax=ax[1]); ax[1].grid(False)
    save(fig, "2_eigenvalue_modulus_distribution.png")

j3 = J("03_lyapunov.json")
if j3:
    fd = j3["finite_difference"]; fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for bi, (bn, v) in enumerate(list(fd.items())[:2]):
        d = np.array(v["dist"]); 
        for i, e in enumerate(j3["eps"]): ax[bi].plot(d[:, i], c=C[i], label=f"ε={e:g}")
        ax[bi].set_yscale("log"); ax[bi].set_xlabel("t"); ax[bi].set_ylabel("d_t = ||H_t−H'_t||"); ax[bi].set_title(bn); ax[bi].legend(fontsize=7)
    b = np.array(j3["benettin"]["history_every10"]); xs = np.arange(1, len(b) + 1) * 10
    for k in range(b.shape[1]): ax[2].plot(xs, b[:, k], label=f"λ{k+1}")
    ax[2].axhline(j3["theory"]["max_over_channels_mean_log_gamma"], c="k", ls="--", lw=.8, label="max mean log γ (theory)")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("Lyapunov exponent (per step)"); ax[2].legend(fontsize=7); ax[2].set_title("Benettin QR exponents (top-4)")
    save(fig, "3_lyapunov_divergence.png")
    fig, ax = plt.subplots(figsize=(5, 3.6)); v = fd["zero_init"]
    for i, e in enumerate(j3["eps"]): ax.plot(np.arange(1, len(v["finite_time_lyapunov"]) + 1), np.array(v["finite_time_lyapunov"])[:, i], c=C[i], label=f"ε={e:g}")
    ax.set_xscale("log"); ax.set_xlabel("t"); ax.set_ylabel("λ(t) = (1/t) ln(d_t/d_0)"); ax.legend(fontsize=7); save(fig, "3b_finite_time_lyapunov.png")

j4 = J("04_attractor.json")
if j4:
    n = j4["stream_names"]; fig, ax = plt.subplots(1, 3, figsize=(14, 3.6)); t = np.arange(len(j4["norm_h"])) * 10
    for i, nm in enumerate(n):
        ax[0].plot(t, np.array(j4["norm_h"])[:, i], label=nm, c=C[i % 7], ls="-" if i < 7 else "--", lw=1)
        ax[1].plot(t[1:], np.array(j4["step_change_norm"])[1:, i], c=C[i % 7], ls="-" if i < 7 else "--", lw=1)
    ax[0].set_ylabel("||H_t|| (all layers)"); ax[1].set_yscale("log"); ax[1].set_ylabel("||H_t − H_{t−1}||"); ax[0].legend(fontsize=6)
    for k, v in j4["ic_spread_max_pairwise"].items(): ax[2].plot(t, v, label=k)
    ax[2].set_yscale("log"); ax[2].set_ylabel("max pairwise distance across 7 initial conditions"); ax[2].legend(fontsize=7)
    for a in ax: a.set_xlabel("t")
    save(fig, "4_attractor_dynamics.png")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, nm in zip(ax, ["const_the", "period7", "repeat_sentence40", "natural_text"]):
        im = a.imshow(np.array(j4["recurrence_matrices"][nm]), cmap="magma"); a.set_title(nm, fontsize=8); a.grid(False)
    save(fig, "4b_recurrence_matrices.png")

j5 = J("05_manifold.json")
if j5:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for l in [0, 3, 6, 9, 11]:
        ax[0].plot(np.arange(1, 129), j5["per_layer"][str(l)]["cum_var_real_first128"], label=f"layer {l}")
    ax[0].plot(np.arange(1, 129), np.array(j5["full_state_cum_var_first512"])[:128], "k--", label="full state")
    ax[0].set_xlabel("# PCs (real-equivalent)"); ax[0].set_ylabel("cumulative explained variance"); ax[0].legend(fontsize=7); ax[0].set_title("PCA explained variance")
    L = list(range(12)); 
    for q, c in zip(["n_for_50pct", "n_for_90pct", "n_for_99pct"], C): ax[1].plot(L, [j5["per_layer"][str(l)]["real_equiv_pca"][q] for l in L], "o-", label=q, c=c, ms=3)
    ax[1].plot(L, [j5["per_layer"][str(l)]["real_equiv_pca"]["participation_ratio"] for l in L], "s--", label="participation ratio", c=C[3], ms=3)
    ax[1].set_yscale("log"); ax[1].set_xlabel("layer"); ax[1].set_ylabel("dimensions"); ax[1].legend(fontsize=7); ax[1].set_title("Linear effective dimension (of R^1536)")
    for k, c in zip(["twonn", "mle_k10", "mle_k20"], C): ax[2].plot(L, [j5["intrinsic_dimension"][str(l)][k] for l in L], "o-", label=k, c=c, ms=3)
    ax[2].set_xlabel("layer"); ax[2].set_ylabel("intrinsic dimension estimate"); ax[2].legend(fontsize=7); ax[2].set_title("Nonlinear intrinsic dimension (kNN)")
    save(fig, "5_pca_intrinsic_dimension.png")

j6 = J("06_memory.json")
if j6:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ks = np.arange(0, 701, 5); allM = np.stack([np.array(v["M_total"])[::5] for v in j6["token_substitution"].values()])
    for v in list(j6["token_substitution"].values()): ax[0].plot(ks, np.array(v["M_total"])[::5], c="#999", lw=.5, alpha=.5)
    ax[0].plot(ks, allM.mean(0), c=C[0], lw=2, label="mean of 18 perturbations"); ax[0].set_yscale("log"); ax[0].set_xlabel("k"); ax[0].set_ylabel("M(k)=||δH_{t+k}||/||δH_t||"); ax[0].legend(fontsize=7); ax[0].set_title("Token-substitution memory")
    for i, (kd, v) in enumerate(j6["state_perturbation"].items()): ax[1].plot(np.arange(701), v["M_total"], label=kd, c=C[i])
    ax[1].set_yscale("log"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=7); ax[1].set_title("State perturbation: radial vs phase vs random")
    lg = np.array(j6["mean_log_gamma_per_head_continuation"]).ravel(); rt = np.array(j6["per_head_decay_rate_token_pert"]).ravel()
    ax[2].scatter(lg, rt, s=8, c=C[0]); ax[2].plot([lg.min(), lg.max()], [lg.min(), lg.max()], "k--", lw=.8)
    ax[2].set_xlabel("mean log γ (Jacobian mode)"); ax[2].set_ylabel("measured decay rate (token pert, k=20..400)"); ax[2].set_title("Decay rate vs Jacobian eigenmode")
    save(fig, "6_memory_retention.png")

j8 = J("08_latency.json")
if j8:
    res = {k: v for k, v in j8["results"].items() if "unsupported" not in v}; fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for i, (k, r) in enumerate(res.items()):
        a = np.concatenate([x["forward_ms_raw"] for x in r["runs"]]); ax[0].hist(a, bins=np.linspace(0, np.percentile(a, 99.9) * 1.1, 80), alpha=.6, label=k, color=C[i])
    ax[0].set_xlabel("ms / token (forward only)"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=7); ax[0].set_title("Decode latency distribution")
    for i, (k, r) in enumerate(res.items()):
        cs = r["context_scaling"]; L = sorted(cs, key=int)
        ax[1].plot([int(l) for l in L], [cs[l]["decode_ms_after_prefill"]["mean"] for l in L], "o-", label=k, c=C[i], ms=3)
        ax[2].plot([int(l) for l in L], [cs[l]["prefill_tok_per_s"] for l in L], "o-", label=k, c=C[i], ms=3)
    ax[1].set_xscale("log"); ax[1].set_ylim(bottom=0); ax[1].set_xlabel("context length (tokens prefilled)"); ax[1].set_ylabel("decode ms/token"); ax[1].set_title("Decode latency vs context"); ax[1].legend(fontsize=7)
    ax[2].set_xscale("log"); ax[2].set_xlabel("prompt length"); ax[2].set_ylabel("prefill tokens/s"); ax[2].set_title("Prefill throughput vs length")
    save(fig, "7_8_latency.png")

j7 = J("07_flops.json")
if j7:
    pf = j7["prefill_analytic_per_token"]; L = sorted(pf, key=int); fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([int(l) for l in L], [pf[l]["flops_per_token_wT=1"] / 1e6 for l in L], "o-", label="prefill, analytic (chunk≤512)", c=C[0], ms=3)
    ax.axhline(j7["decode_L1_analytic"]["total_flops_wT=1"] / 1e6, c=C[1], label="decode (L=1) – independent of context")
    fcp = j7["flopcounter_per_token_prefill"]; ax.plot([int(l) for l in fcp], [v / 1e6 for v in fcp.values()], "x", c="k", label="FlopCounter (matmul/conv only)")
    ax.set_xscale("log"); ax.set_xlabel("sequence/chunk length"); ax.set_ylabel("MFLOPs per token"); ax.legend(fontsize=7); save(fig, "9_flops_per_token.png")
 