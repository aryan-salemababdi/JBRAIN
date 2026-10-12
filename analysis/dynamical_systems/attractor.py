import time
import json 
import numpy as np
import torch
import os
from pathlib import Path
from common import *

set_seed()
T = int(os.environ.get("ATT_T", 4000)); PRE = 128; SNAP = 10
m = load_model(torch.float64, fast_conv=True)
ids, _ = corpus_ids()
the = tokenizer().encode(" the")[0]; rare = int(ids[5000])
win = windows(ids, 8, PRE)
nat_long = windows(ids, 1, T, seed=7)[0]
sent = windows(ids, 1, 40, seed=11)[0]
rng = np.random.RandomState(SEED)

streams = {  # name -> token array [T], zero-input flagged separately
    "zero_input": None,
    "const_the": np.full(T, the), "const_rare": np.full(T, rare),
    "period2": np.tile(ids[2000:2002], T // 2), "period7": np.tile(ids[3000:3007], T // 7 + 1)[:T],
    "period31": np.tile(ids[4000:4031], T // 31 + 1)[:T],
    "repeat_sentence40": np.tile(sent, T // 40 + 1)[:T],
    "natural_text": nat_long, "uniform_random": rng.randint(0, VOCAB, T),
}
names = list(streams)
R0 = len(names)

# initial conditions: 6 natural prefixes + zero
ic_states = [zero_state(1)]
with torch.no_grad():
    for w in win[:6]:
        st = zero_state(1); e = m.embeddings(torch.tensor(w))
        for t in range(PRE):
            _, st = step(m, e[None, t:t + 1], st)
        ic_states.append(st)
NIC = len(ic_states)  # 7
B = R0 + 2 * NIC
st = zero_state(B)
for k in range(NIC):
    for l in range(NL):
        for j in range(3):
            st[l][j][R0 + k] = ic_states[k][l][j][0]
            st[l][j][R0 + NIC + k] = ic_states[k][l][j][0]
tok = np.zeros((T, B), dtype=np.int64)
zero_mask = np.zeros(B, dtype=bool)
for i, n in enumerate(names):
    if streams[n] is None:
        zero_mask[i] = True
    else:
        tok[:, i] = streams[n]
tok[:, R0:R0 + NIC] = the                 # const-token from 7 ICs
zero_mask[R0 + NIC:] = True               # zero-input from 7 ICs
zm = torch.tensor(~zero_mask, dtype=torch.float64).view(B, 1, 1)
W = m.embeddings.weight


def hvec(st):
    return torch.cat([torch.cat([hr.reshape(B, -1), hi.reshape(B, -1)], 1) for _, hr, hi in st], 1)

hn = np.zeros((T, B)); dh = np.zeros((T, B)); hmax = np.zeros((T, B))
ic_max_pair = np.zeros((T, 2)); snaps = []
prev = hvec(st); t0 = time.time()
with torch.no_grad():
    for t in range(T):
        e = W[torch.tensor(tok[t])].unsqueeze(1) * zm
        _, st = step(m, e, st)
        h = hvec(st)
        hn[t] = h.norm(dim=1).numpy(); dh[t] = (h - prev).norm(dim=1).numpy()
        hr, hi = h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]
        for gi, sl in enumerate([slice(R0, R0 + NIC), slice(R0 + NIC, R0 + 2 * NIC)]):
            c = h[sl]
            ic_max_pair[t, gi] = torch.cdist(c, c).max().item()
        if t % SNAP == 0:
            snaps.append(h[:R0].to(torch.float32).numpy())
        prev = h
        if (t + 1) % 250 == 0:
            print(t + 1, f"{time.time() - t0:.0f}s", "ICspread const/zero:", ic_max_pair[t].round(6), flush=True)
snaps = np.stack(snaps, 1)  # [R0, T/SNAP, DIM]
# note: all-layer h; channel moduli max
res = {"env": env_info(), "T": T, "stream_names": names, "snapshot_stride": SNAP}
res["norm_h"] = hn[::10]; res["step_change_norm"] = dh[::10]
res["ic_spread_max_pairwise"] = {"const_the": ic_max_pair[::10, 0], "zero_input": ic_max_pair[::10, 1]}
summ = {}
for i, n in enumerate(names):
    S = snaps[i]; fin = S[-1]
    dist_fin = np.linalg.norm(S - fin, axis=1)
    last = slice(-40, None)                   # last 400 steps
    rec = np.linalg.norm(S[last, None] - S[None, last], axis=2)   # recurrence distance matrix
    # exponential rate of convergence to final state over t in [T/10, T/2]
    a, b = int(len(S) * .1), int(len(S) * .5)
    tt = np.arange(len(S)) * SNAP
    ok = dist_fin[a:b] > 1e-12
    slope = float(np.polyfit(tt[a:b][ok], np.log(dist_fin[a:b][ok]), 1)[0]) if ok.sum() > 5 else None
    # period test: relative distance ||h_t - h_{t-p}||, p in {1,2,7,31,40}
    per = {p: float(np.linalg.norm(snaps[i, -1] - snaps[i, -1 - max(1, p // SNAP)]) / np.linalg.norm(fin)) for p in [10, 20, 70, 310, 400]}
    summ[n] = {"norm_h_final": float(hn[-1, i]), "norm_h_max": float(hn[:, i].max()),
               "step_change_final": float(dh[-1, i]), "step_change_mean_last500": float(dh[-500:, i].mean()),
               "step_change_t100": float(dh[100, i]),
               "rel_step_change_final": float(dh[-1, i] / hn[-1, i]),
               "dist_to_final_at_T/4": float(dist_fin[len(S) // 4]), "dist_to_final_at_T/2": float(dist_fin[len(S) // 2]),
               "log_dist_slope_per_step(T/10..T/2)": slope,
               "recurrence_dist_mean_last400": float(rec[np.triu_indices(40, 1)].mean()),
               "rel_recurrence_dist_mean_last400": float(rec[np.triu_indices(40, 1)].mean() / np.linalg.norm(fin)),
               "rel_dist_h_t_vs_h_t-lag(steps=10,20,70,310,400)": per}
    res.setdefault("recurrence_matrices", {})[n] = rec
res["summary"] = summ
res["ic_convergence"] = {"const_the_final_maxpair": float(ic_max_pair[-1, 0]), "const_the_initial_maxpair": float(ic_max_pair[0, 0]),
                          "zero_final_maxpair": float(ic_max_pair[-1, 1]), "zero_initial_maxpair": float(ic_max_pair[0, 1])}
# channel-modulus saturation fraction (|h|>0.9) at final step per stream
hh = snaps[:, -1].reshape(R0, NL, 2, D)
mod = np.sqrt(hh[:, :, 0] ** 2 + hh[:, :, 1] ** 2)
res["max_channel_modulus_final"] = mod.reshape(R0, -1).max(1); res["mean_channel_modulus_final"] = mod.reshape(R0, -1).mean(1)
save_json("04_attractor.json", res)
if os.environ.get("SAVE_SNAPSHOTS", "0") == "1":
    output_dir = Path(__file__).resolve().parent / "results" / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "04_snapshots_float32.npy", snaps)