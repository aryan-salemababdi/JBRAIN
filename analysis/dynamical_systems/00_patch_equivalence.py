import torch
from common import *
set_seed(); ids, _ = corpus_ids(); tk = torch.tensor(ids[:80])
out = {}
for dt in [torch.float64, torch.float32]:
    a, b = load_model(dt), load_model(dt, fast_conv=True)
    sa, sb = zero_state(1, dt), zero_state(1, dt); worst = 0.0; wl = 0.0
    with torch.no_grad():
        for t in range(80):
            e = a.embeddings(tk[t:t + 1])[None]
            xa, sa = step(a, e, sa); xb, sb = step(b, e, sb)
            worst = max(worst, (flat_state(sa) - flat_state(sb)).abs().max().item()); wl = max(wl, (xa - xb).abs().max().item())
    out[str(dt)] = {"max_abs_state_diff_80_steps": worst, "max_abs_residual_stream_diff": wl}
save_json("00_patch_equivalence.json", out); print(out)
