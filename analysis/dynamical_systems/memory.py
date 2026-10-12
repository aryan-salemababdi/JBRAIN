import json, time, numpy as np, torch
from common import *

set_seed()
m = load_model(torch.float64, fast_conv=True)
ids, _ = corpus_ids()
K = 700; PRE = 200
KS = [1, 5, 10, 20, 50, 100, 200, 500]
win = windows(ids, 6, PRE + K + 1)
res = {"env": env_info(), "K": K, "ks": KS}


def hdist(a, b, per="layer"):
    """a,b: states with batch dim: returns [B,NL,NH] distance per layer/head"""
    out = []
    for (_, ar, ai), (_, br, bi) in zip(a, b):
        out.append(torch.sqrt(((ar - br) ** 2 + (ai - bi) ** 2).sum(-1)))
    return torch.stack(out, 1)

def logits_of(x):
    return m.lm_head(m.ln_f(x))[:, 0]

# token substitution, 6 contexts x 3 replacement types
ctx_res = []
rep_types = {"random_token": None, "same_class_common": None, "rare_token": None}
rng = np.random.RandomState(SEED)
tok_cnt = np.bincount(ids, minlength=VOCAB)
common_ids = np.argsort(-tok_cnt)[:200]; rare_ids = np.where((tok_cnt > 0) & (tok_cnt <= 2))[0]
curves = {}
with torch.no_grad():
    for ci in range(6):
        seq = torch.tensor(win[ci])
        st = zero_state(1); e = m.embeddings(seq)
        for t in range(PRE):
            _, st = step(m, e[None, t:t + 1], st)
        orig = int(seq[PRE])
        reps = {"random_token": int(rng.randint(0, VOCAB)), "common_token": int(rng.choice(common_ids)), "rare_token": int(rng.choice(rare_ids))}
        for rn, rt in reps.items():
            B = 2
            s2 = [tuple(x.expand(B, *x.shape[1:]).clone() for x in s) for s in st]
            tk = torch.tensor([orig, rt])
            x_, s2 = step(m, m.embeddings(tk).unsqueeze(1), s2)
            d = np.zeros((K + 1, NL, NH)); dl = np.zeros(K + 1)
            d[0] = hdist([(c[:1], r[:1], i[:1]) for c, r, i in s2], [(c[1:], r[1:], i[1:]) for c, r, i in s2])[0].numpy()
            for k in range(1, K + 1):
                tk = seq[PRE + k].repeat(B)
                x_, s2 = step(m, m.embeddings(tk).unsqueeze(1), s2)
                d[k] = hdist([(c[:1], r[:1], i[:1]) for c, r, i in s2], [(c[1:], r[1:], i[1:]) for c, r, i in s2])[0].numpy()
                if k % 5 == 0:
                    lg = logits_of(x_); dl[k] = (lg[0] - lg[1]).norm().item()
            curves[(ci, rn)] = {"d": d, "dlogit": dl}
            tot = np.sqrt((d ** 2).sum((1, 2)))
            print("tok-subst ctx", ci, rn, "M(k):", [round(tot[k] / tot[0], 4) for k in KS], flush=True)
res["token_substitution"] = {f"ctx{c}_{r}": {"M_total": (np.sqrt((v["d"] ** 2).sum((1, 2))) / np.sqrt((v["d"][0] ** 2).sum())),
                                              "M_per_layer": (np.sqrt((v["d"] ** 2).sum(2)) / np.sqrt((v["d"][0] ** 2).sum()))[::5],
                                              "dlogit": v["dlogit"][::5]} for (c, r), v in curves.items()}
tot_all = np.stack([np.sqrt((v["d"] ** 2).sum((1, 2))) / np.sqrt((v["d"][0] ** 2).sum()) for v in curves.values()])
res["token_substitution_M_mean_over_18"] = {str(k): float(tot_all[:, k].mean()) for k in [0] + KS}
res["token_substitution_M_std_over_18"] = {str(k): float(tot_all[:, k].std()) for k in [0] + KS}
Hs = np.stack([v["d"] for v in curves.values()])          # [18,K+1,NL,NH] absolute distance
# per-head decay-rate fit from k=20..400 using mean over the 18 perturbations
rate = np.zeros((NL, NH)); r2 = np.zeros((NL, NH))
ks = np.arange(20, 401)
for l in range(NL):
    for h in range(NH):
        y = np.log(Hs[:, ks, l, h].mean(0) + 1e-300)
        A = np.polyfit(ks, y, 1); rate[l, h] = A[0]
res["per_head_decay_rate_token_pert"] = rate

# state perturbation: radial vs tangential, per (layer, head)
seq = torch.tensor(win[0]); e = m.embeddings(seq)
with torch.no_grad():
    st = zero_state(1)
    for t in range(PRE):
        _, st = step(m, e[None, t:t + 1], st)
    eps = 1e-3
    out_state = {}
    for kind in ["radial", "tangential", "random_complex"]:
        # perturb ALL layers/heads simultaneously with a fixed per-element-RMS perturbation (norm eps overall)
        B = 2
        s2 = [tuple(x.expand(B, *x.shape[1:]).clone() for x in s) for s in st]
        g = torch.Generator().manual_seed(1)
        tot2 = 0.0; pert = []
        for l in range(NL):
            hr, hi = st[l][1][0], st[l][2][0]
            ph = torch.complex(hr, hi); mod = ph.abs() + 1e-12
            if kind == "radial": dv = ph / mod                       # along h
            elif kind == "tangential": dv = 1j * ph / mod            # perpendicular (phase rotation)
            else:
                dv = torch.complex(torch.randn(NH, HD, generator=g, dtype=torch.float64), torch.randn(NH, HD, generator=g, dtype=torch.float64))
            pert.append(dv); tot2 += (dv.abs() ** 2).sum().item()
        sc = eps / np.sqrt(tot2)
        for l in range(NL):
            s2[l][1][1] += sc * pert[l].real; s2[l][2][1] += sc * pert[l].imag
        d = np.zeros((K + 1, NL, NH))
        d[0] = hdist([(c[:1], r[:1], i[:1]) for c, r, i in s2], [(c[1:], r[1:], i[1:]) for c, r, i in s2])[0].numpy()
        for k in range(1, K + 1):
            tk = seq[PRE + k - 1].repeat(B)
            _, s2 = step(m, m.embeddings(tk).unsqueeze(1), s2)
            d[k] = hdist([(c[:1], r[:1], i[:1]) for c, r, i in s2], [(c[1:], r[1:], i[1:]) for c, r, i in s2])[0].numpy()
        tot = np.sqrt((d ** 2).sum((1, 2)))
        out_state[kind] = {"M_total": tot / tot[0], "M_per_layer": (np.sqrt((d ** 2).sum(2)) / tot[0])[::5], "d_head": d[[0, 1, 5, 10, 20, 50, 100, 200, 400, 700]]}
        print("state-pert", kind, [round(tot[k] / tot[0], 4) for k in KS], flush=True)
res["state_perturbation"] = out_state

# correlation with Jacobian eigenmodes: mean log gamma / decay timescale on the same trajectory
tr = trajectory_gates(m, e[None, :PRE + K])
lg = torch.log(tr["gamma"][PRE:, :, 0]).mean(0).numpy()  # [NL,NH]
res["mean_log_gamma_per_head_continuation"] = lg
from scipy.stats import spearmanr, pearsonr
kk = np.array([20, 50, 100, 200, 400]); idx = [4, 5, 6, 7, 8]
srate = {}
for kind, v in out_state.items():
    dh_ = v["d_head"][idx]                       # [5,NL,NH]
    srate[kind] = np.array([[np.polyfit(kk, np.log(dh_[:, l, h] + 1e-300), 1)[0] for h in range(NH)] for l in range(NL)])
res["per_head_decay_rate_state_pert"] = srate
res["corr_rate_vs_mean_log_gamma"] = {"spearman_token_pert": float(spearmanr(lg.ravel(), rate.ravel())[0]),
    "pearson_token_pert": float(pearsonr(lg.ravel(), rate.ravel())[0])}
for kind, r_ in srate.items():
    res["corr_rate_vs_mean_log_gamma"]["spearman_state_" + kind] = float(spearmanr(lg.ravel(), r_.ravel())[0])
    res["corr_rate_vs_mean_log_gamma"]["pearson_state_" + kind] = float(pearsonr(lg.ravel(), r_.ravel())[0])
# per-head decay rate from random state perturbation (single trajectory, direct)
save_json("06_memory.json", res)
