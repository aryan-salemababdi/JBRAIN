import json, numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from common import *

D_, H_, HD_, FF, V = D, NH, HD, 4 * D, VOCAB


def layer_decode(D=D_, H=H_, FF=FF):
    mm = {"linear_Wtheta_Wu_Wkey_out (4 x DxD)": 4 * 2 * D * D, "gate_heads W_gamma,W_dphi (2 x DxH)": 2 * 2 * D * H,
          "ffn w1 (DxFF)": 2 * D * FF, "ffn w2 (FFxD)": 2 * FF * D, "depthwise_conv k=2": 4 * D}
    ew, tr = {}, {}
    ew["rmsnorm1"] = 4 * D + 1; tr["rmsnorm1"] = 1
    ew["conv_bias"] = D
    ew["silu_conv_out"] = D; tr["silu_conv_out"] = D
    ew["linear_biases(3 DxD + 2 heads)"] = 3 * D + 2 * H
    ew["theta,theta_q scaling 2pi"] = 2 * D
    tr["tanh_u"] = D
    ew["gamma = .9+.099*sigmoid"] = 2 * H; tr["sigmoid_gamma"] = H
    ew["dphi = pi*tanh"] = H; tr["tanh_dphi"] = H
    ew["bound_factor 1-gamma"] = H
    tr["cos/sin(theta) for value"] = 2 * D; ew["value u*cos*bf, u*sin*bf"] = 4 * D
    tr["cos/sin(dphi)"] = 2 * H
    ew["recurrence H_t = g e^{i dphi} H + V (10 flops/complex elem)"] = 10 * D
    tr["cos/sin(theta_q) readout"] = 2 * D; ew["readout Re(H e^{-i th_q}) * scale"] = 4 * D
    ew["out_proj bias"] = D
    ew["residual1"] = D
    ew["rmsnorm2"] = 4 * D + 1; tr["rmsnorm2"] = 1
    ew["ffn biases"] = FF + D; ew["ffn silu mul"] = FF; tr["ffn silu sigmoid"] = FF
    ew["residual2"] = D
    return mm, ew, tr


def totals(mm, ew, tr, wT=1):
    return sum(mm.values()) + sum(ew.values()) + wT * sum(tr.values())


mm, ew, tr = layer_decode()
head_mm = 2 * D_ * V
fin_ew = 4 * D_ + 1; fin_tr = 1
res = {"env": env_info()}
dec = {"per_layer_matmul_conv": sum(mm.values()), "per_layer_elementwise": sum(ew.values()), "per_layer_transcendental_events": sum(tr.values()),
       "layers": NL, "lm_head_matmul": head_mm, "final_rmsnorm_elementwise": fin_ew, "final_rmsnorm_transc": fin_tr}
dec["total_matmul_conv"] = NL * sum(mm.values()) + head_mm
dec["total_elementwise"] = NL * sum(ew.values()) + fin_ew
dec["total_transc_events"] = NL * sum(tr.values()) + fin_tr
for w in [1, 10, 20]:
    dec[f"total_flops_wT={w}"] = dec["total_matmul_conv"] + dec["total_elementwise"] + w * dec["total_transc_events"]
dec["state_update_only_flops_wT=1 (no lm_head/ln_f)"] = NL * totals(mm, ew, tr, 1)
dec["softmax_sampling_extra_flops (3V + V temp, wT=1)"] = 5 * V
dec["breakdown_per_layer"] = {"matmul_conv": mm, "elementwise": ew, "transcendental": tr}
dec["lm_head_share_of_total_wT=1"] = head_mm / dec["total_flops_wT=1"]
res["decode_L1_analytic"] = dec

# cross-check matmul/conv portion with FlopCounterMode on real forward
m = load_model(torch.float32)
fc = {}
with torch.no_grad():
    for L in [1, 16, 64, 256, 512]:
        ids = torch.randint(0, VOCAB, (1, L))
        st = zero_state(1, torch.float32) if L == 1 else None
        with FlopCounterMode(display=False) as f:
            m(ids, st)
        fc[L] = f.get_total_flops()
res["flopcounter_total_matmul_conv_flops"] = fc
res["analytic_decode_matmul_conv"] = dec["total_matmul_conv"]
res["decode_matmul_rel_diff_vs_flopcounter"] = (fc[1] - dec["total_matmul_conv"]) / fc[1]


# ---- prefill (parallel scan, past_state=None, chunk length L): analytic per-token vs L
def prefill_per_token(L, with_head=True):
    mm_, ew_, tr_ = layer_decode()
    mm_ = dict(mm_); ew_ = dict(ew_); tr_ = dict(tr_)
    # replace the recurrence terms (10D basic, 2H transc) with parallel scan terms, per token (divide by L)
    ew_.pop("recurrence H_t = g e^{i dphi} H + V (10 flops/complex elem)"); tr_.pop("cos/sin(dphi)")
    scan_mm = 8 * L * D_                               # 4 matmuls [LxL]@[Lxd] per head: 8*L^2*d*H / L
    scan_ew = (6 * L * H_) + 4 * D_ + 4                # Lx L: sub x2, mul x2, mask x2 per head /L ; + adds in B_r,B_i (2 per elem) + cumsums
    scan_tr = 3 * L * H_ + 2 * H_                      # exp, cos, sin of LxL per head /L ; + log per token
    mm_["scan matmuls"] = scan_mm; ew_["scan elementwise"] = scan_ew; tr_["scan transc"] = scan_tr
    per = totals(mm_, ew_, tr_, 1) * NL + (head_mm if with_head else 0) + fin_ew + fin_tr
    return per, {"matmul": NL * sum(mm_.values()) + (head_mm if with_head else 0), "scan_matmul": NL * scan_mm}

pf = {}
for L in [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    # L>512 is processed in chunks of 512 by encode_prompt_chunked (with past_state => extra decay terms); use chunk=min(L,512)
    Lc = min(L, 512)
    per, parts = prefill_per_token(Lc)
    pf[L] = {"chunk": Lc, "flops_per_token_wT=1": per, **parts}
res["prefill_analytic_per_token"] = pf
# FlopCounter per-token for prefill
res["flopcounter_per_token_prefill"] = {L: fc[L] / L for L in [16, 64, 256, 512]}
res["decode_flops_vs_context_length"] = "constant: the L=1 recurrent step touches only the fixed-size state (O(1) in context)."
res["state_size"] = {"floats_per_layer": D_ + 2 * D_, "floats_total": NL * 3 * D_, "bytes_fp32": NL * 3 * D_ * 4,
                     "bytes_bf16": NL * 3 * D_ * 2}
save_json("07_flops.json", res)
print(json.dumps({k: v for k, v in res.items() if k not in ("env", "prefill_analytic_per_token")}, indent=1)[:3500])
