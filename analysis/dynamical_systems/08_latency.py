"""Part 2.B/C: steady-state next-token latency & throughput of the UNMODIFIED inference path
(model(next_token, past_states) exactly as in JBRLanguageModel.generate_o1_memory), batch=1.
Must run alone on the machine (no other CPU load)."""
import json, os, time, numpy as np, torch
import torch.nn.functional as F
from common import *

WARM = 100
NRUNS = {"cpu": (300, 3), "mps": (1000, 5)}   # CPU path is ~140 ms/token (slow grouped-conv kernel) => fewer samples
PROMPT = 64
configs = [("cpu", torch.float32), ("mps", torch.float32), ("mps", torch.float16)]
ids, _ = corpus_ids()
res = {"env": env_info(), "warmup_steps": WARM, "measured_steps_per_run_runs": NRUNS, "batch_size": 1,
       "prompt_len": PROMPT, "torch_threads": torch.get_num_threads(), "temperature": 0.7,
       "note": "fp16 on MPS is a precision VARIANT of the same weights; accuracy not evaluated here."}


def sync(dev):
    if dev == "mps": torch.mps.synchronize()


def stats(a):
    a = np.asarray(a)
    return {"mean": a.mean(), "median": np.median(a), "std": a.std(ddof=1), "min": a.min(), "max": a.max(),
            "p50": np.percentile(a, 50), "p90": np.percentile(a, 90), "p95": np.percentile(a, 95), "p99": np.percentile(a, 99)}


out = {}
for dev, dt in configs:
    key = f"{dev}_{str(dt).split('.')[-1]}"
    if dev == "mps" and not torch.backends.mps.is_available(): continue
    try:
        N, RUNS = NRUNS[dev]
        t0 = time.perf_counter()
        mdl = av2.JBRLanguageModel(VOCAB, D, NH, NL)
        sd = torch.load(CKPT, map_location="cpu"); mdl.load_state_dict(sd, strict=True); del sd
        mdl = mdl.to(device=dev, dtype=dt).eval(); sync(dev)
        t_load = time.perf_counter() - t0
        r = {"model_load_s": t_load, "runs": []}
        # warm-up (kernel compile / allocator) on a prompt + decode steps
        tw = time.perf_counter()
        with torch.inference_mode():
            p = torch.tensor(ids[:PROMPT])[None].to(dev)
            lg, st = mdl.encode_prompt_chunked(p, 512)
            nt = lg[:, -1].argmax(-1, keepdim=True)
            for _ in range(WARM):
                lg, st = mdl(nt, past_states=st); nt = lg[:, -1].argmax(-1, keepdim=True)
            sync(dev)
        r["warmup_s"] = time.perf_counter() - tw
        for run in range(RUNS):
            g = torch.Generator(device="cpu").manual_seed(SEED + run)
            off = 1000 + 5000 * run
            with torch.inference_mode():
                p = torch.tensor(ids[off:off + PROMPT])[None].to(dev)
                sync(dev); tp = time.perf_counter()
                lg, st0 = mdl.encode_prompt_chunked(p, 512); sync(dev)
                prompt_s = time.perf_counter() - tp
                # A) forward-only (token ids fed from corpus; sync every step)
                st = st0; fwd = []
                for i in range(N):
                    tk = torch.tensor([[int(ids[off + PROMPT + i])]], device=dev)
                    sync(dev); a = time.perf_counter()
                    lg, st = mdl(tk, past_states=st); sync(dev)
                    fwd.append((time.perf_counter() - a) * 1e3)
                # B) end-to-end predict: forward + temperature softmax + multinomial + .item()
                st = st0; nt = lg[:, -1].argmax(-1, keepdim=True); e2e = []
                for i in range(N):
                    a = time.perf_counter()
                    lg, st = mdl(nt, past_states=st)
                    pr = F.softmax(lg[:, -1].float() / 0.7, dim=-1)
                    nt = torch.multinomial(pr, 1, generator=None); _ = nt.item()
                    e2e.append((time.perf_counter() - a) * 1e3)
            r["runs"].append({"prompt_prefill_s": prompt_s, "forward_ms": stats(fwd), "e2e_ms": stats(e2e),
                              "forward_ms_raw": np.array(fwd, dtype=np.float32), "e2e_ms_raw": np.array(e2e, dtype=np.float32)})
            print(key, "run", run, "fwd mean %.3f ms  e2e mean %.3f ms" % (np.mean(fwd), np.mean(e2e)), flush=True)
        allf = np.concatenate([x["forward_ms_raw"] for x in r["runs"]]); alle = np.concatenate([x["e2e_ms_raw"] for x in r["runs"]])
        r["pooled_forward_ms"] = stats(allf); r["pooled_e2e_ms"] = stats(alle)
        r["tokens_per_s_forward(1/mean)"] = 1e3 / allf.mean(); r["tokens_per_s_e2e(1/mean)"] = 1e3 / alle.mean()
        r["run_means_forward_ms"] = [x["forward_ms"]["mean"] for x in r["runs"]]
        # context scaling: prefill throughput & decode latency after prefill of Lc tokens (chunk 512 as shipped)
        sc = {}
        with torch.inference_mode():
            for Lc in [128, 512, 1024, 2048, 4096, 8192]:
                ptk = torch.tensor(ids[:Lc])[None].to(dev)
                ts = []
                for rep in range(3):
                    sync(dev); a = time.perf_counter(); lg, stc = mdl.encode_prompt_chunked(ptk, 512); sync(dev)
                    ts.append(time.perf_counter() - a)
                nt = lg[:, -1].argmax(-1, keepdim=True); d = []
                for i in range(200):
                    sync(dev); a = time.perf_counter(); lg, stc = mdl(nt, past_states=stc); sync(dev)
                    d.append((time.perf_counter() - a) * 1e3); nt = lg[:, -1].argmax(-1, keepdim=True)
                sc[Lc] = {"prefill_s_median": float(np.median(ts)), "prefill_s_all": ts, "prefill_tok_per_s": Lc / float(np.median(ts)),
                          "decode_ms_after_prefill": stats(d)}
                print(key, "ctx", Lc, "prefill tok/s %.0f" % sc[Lc]["prefill_tok_per_s"], "decode mean %.3f ms" % np.mean(d), flush=True)
        r["context_scaling"] = sc
    except Exception as ex:
        r = {"unsupported": f"{type(ex).__name__}: {ex}"}
        print(key, "FAILED:", r["unsupported"], flush=True)
    out[key] = r
    res["results"] = out; save_json("08_latency.json", res)
    del mdl
    if dev == "mps": torch.mps.empty_cache()
res["results"] = out
save_json("08_latency.json", res)
