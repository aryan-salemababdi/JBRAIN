import hashlib, json, os, platform, subprocess, sys, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "analysis", "dynamical_systems")
RES = os.path.join(OUT, "results")
PLOTS = os.path.join(OUT, "plots")
sys.path.insert(0, ROOT)
import architecture_v2 as av2  # noqa: E402

CKPT = os.path.join(ROOT, "model", "jbr_v2_ce_weights.pt")
VOCAB, D, NH, NL = 100277, 768, 12, 12
HD = D // NH
SEED = 1234


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def _fast_conv_forward(self, x, past_x=None, _orig=av2.CausalConv1d.forward):
    """Analysis-only speed-up for the L=1 recurrent path: the depthwise k=2 conv equals
    w[:,0]*x_{t-1} + w[:,1]*x_t + b (elementwise). PyTorch's CPU grouped-conv kernel loops per channel (~10x slower).
    Equivalence to the original forward is verified in 00_patch_equivalence.json. NEVER used for latency benchmarks."""
    if past_x is not None and x.shape[1] == 1:
        w = self.conv.weight[:, 0, :]
        return past_x[:, -1:, :] * w[:, 0] + x * w[:, 1] + self.conv.bias, x
    return _orig(self, x, past_x)


def load_model(dtype=torch.float64, device="cpu", fast_conv=False):
    m = av2.JBRLanguageModel(VOCAB, D, NH, NL)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    m = m.to(device=device, dtype=dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if fast_conv:
        import types
        for layer in m.layers:
            layer.core.conv.forward = types.MethodType(_fast_conv_forward, layer.core.conv)
    return m


def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def corpus_ids():
    import pydoc_data.topics as t
    txt = "\n\n".join(t.topics[k] for k in sorted(t.topics))
    ids = tokenizer().encode(txt, disallowed_special=())
    return np.array(ids, dtype=np.int64), hashlib.sha256(txt.encode()).hexdigest()


def zero_state(B, dtype=torch.float64, device="cpu"):
    """The true initial condition: conv cache = 0 (== left zero padding), h = 0."""
    return [(torch.zeros(B, 1, D, dtype=dtype, device=device),
             torch.zeros(B, NH, HD, dtype=dtype, device=device),
             torch.zeros(B, NH, HD, dtype=dtype, device=device)) for _ in range(NL)]


def step(model, emb, state, upto=NL):
    """One recurrent step for the stack of real `JBR_Block`s (L=1). `emb` [B,1,D].
    Identical to JBRLanguageModel.forward minus ln_f/lm_head (which do not feed back
    into the state). Returns (final residual stream x, next_state)."""
    x = emb
    nxt = []
    for i in range(upto):
        x, st = model.layers[i](x, state[i])
        nxt.append(st)
    return x, nxt


def flat_state(state):
    return torch.cat([torch.cat([c.reshape(c.shape[0], -1), hr.reshape(hr.shape[0], -1),
                                 hi.reshape(hi.shape[0], -1)], 1) for c, hr, hi in state], 1)


def unflat_state(v, upto=NL):
    B = v.shape[0]
    out, o = [], 0
    for _ in range(upto):
        c = v[:, o:o + D].reshape(B, 1, D); o += D
        hr = v[:, o:o + D].reshape(B, NH, HD); o += D
        hi = v[:, o:o + D].reshape(B, NH, HD); o += D
        out.append((c, hr, hi))
    return out


def h_complex(state):
    """list over layers of complex h [B, 768] (head-major flatten)."""
    return [torch.complex(hr.reshape(hr.shape[0], -1), hi.reshape(hi.shape[0], -1))
            for _, hr, hi in state]


def env_info():
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty_files": sh("git status --short"),
        "checkpoint": CKPT,
        "checkpoint_sha256_16": hashlib.sha256(open(CKPT, "rb").read(1 << 26)).hexdigest()[:16] + " (first 64MiB)",
        "checkpoint_bytes": os.path.getsize(CKPT),
        "model_config": dict(vocab_size=VOCAB, d_model=D, n_heads=NH, head_dim=HD, n_layers=NL, ffn_expand=4),
        "tokenizer": "tiktoken cl100k_base (n_vocab=100277); NOT bundled in repo, matched by vocab size",
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "os": platform.platform(), "machine": platform.machine(),
        "cpu": sh("sysctl -n machdep.cpu.brand_string"), "ncpu": os.cpu_count(),
        "ram_bytes": sh("sysctl -n hw.memsize"),
        "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
        "seed": SEED,
    }


def save_json(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved", name)


def windows(ids, n, L, seed=SEED):
    """n disjoint L-token windows of the corpus (deterministic)."""
    starts = np.arange(0, len(ids) - L, L)
    rng = np.random.RandomState(seed)
    rng.shuffle(starts)
    return np.stack([ids[s:s + L] for s in starts[:n]])


def gates(core, u, past_x):
    """Re-derivation (analysis side) of the per-step coefficients inside
    JBRHolographicRecurrentCore.forward for L=1. u = ln1(x) [B,1,D]. Returns
    gamma, dphi [B,H], val_r, val_i [B,H,hd], theta_q [B,H,hd]."""
    import math
    import torch.nn.functional as F
    xc, _ = core.conv(u, past_x)
    xm = F.silu(xc)
    B = u.shape[0]
    theta = core.W_theta(xm).view(B, NH, HD) * (2 * math.pi)
    uu = torch.tanh(core.W_u(xm)).view(B, NH, HD)
    thq = core.W_key_theta(xm).view(B, NH, HD) * (2 * math.pi)
    gamma = 0.9 + 0.099 * torch.sigmoid(core.W_gamma(xm))[:, 0]
    dphi = torch.tanh(core.W_delta_phi(xm))[:, 0] * math.pi
    bf = (1.0 - gamma).unsqueeze(-1)
    return gamma, dphi, uu * torch.cos(theta) * bf, uu * torch.sin(theta) * bf, thq


def trajectory_gates(model, emb_seq, record_h=False):
    """Run a single stream (emb_seq [T,D] or [B,T,D]) step by step, recording per-step
    per-layer (gamma, dphi) and optionally h. Returns dict of tensors."""
    if emb_seq.dim() == 2:
        emb_seq = emb_seq[None]
    B, T, _ = emb_seq.shape
    st = zero_state(B, emb_seq.dtype, emb_seq.device)
    G = torch.zeros(T, NL, B, NH, dtype=emb_seq.dtype)
    P = torch.zeros(T, NL, B, NH, dtype=emb_seq.dtype)
    Hs = [] if record_h else None
    with torch.no_grad():
        for t in range(T):
            x = emb_seq[:, t:t + 1]
            for i, layer in enumerate(model.layers):
                u = layer.ln1(x)
                g, p, _, _, _ = gates(layer.core, u, st[i][0])
                G[t, i], P[t, i] = g, p
                x, st[i] = layer(x, st[i])
            if record_h:
                Hs.append(flat_state(st))
    return {"gamma": G, "dphi": P, "H": torch.stack(Hs, 1) if record_h else None, "final_state": st}
