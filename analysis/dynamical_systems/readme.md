# J-BRAIN v2 — Dynamical-Systems and Inference-Cost Analysis

Scope: analysis only. `architecture_v2.py`, the checkpoint `model/jbr_v2_ce_weights.pt`, the tokenizer choice and all hyperparameters are unchanged. Nothing was retrained.
All numbers below come from `results/*.json`; plots are in `plots/`; every script is in this directory (`run_all.sh` reproduces everything).

Evidence labels used throughout: **[PROVEN]** = follows from the code by derivation (and was also checked numerically), **[EMPIRICAL]** = measured on the trajectories we ran, **[HYPOTHESIS]** = plausible mechanism, not tested, **[INCONCLUSIVE]** = data do not settle it.

---

## 1. Executive summary

* **[PROVEN]** The recurrent state update is a gated complex linear recurrence: `H_t = γ_t e^{iΔφ_t} ⊙ H_{t-1} + (1-γ_t) u_t e^{iθ_t}`, with `γ_t ∈ (0.9, 0.999)`. Every state channel stays inside the unit disk. The coefficients depend on the input, so the system is non-autonomous, and the nonlinearity lives in how the coefficients are computed, not in the state update itself.
* **[PROVEN + verified]** The one-step Jacobian of the full state (all 12 layers, conv caches included) is block lower-triangular in layer order. Its eigenvalues are exactly `{0} ∪ {γ e^{±iΔφ}}`, so its spectral radius is `max γ ≤ 0.999 < 1` for every input. We checked this against a full numerical Jacobian and eigendecomposition on a 2-layer sub-stack (moduli agree to 1.3e-13).
* **[EMPIRICAL]** The spectral radius sits at 0.9989–0.9990 at every timestep, essentially at the upper clamp, and about 60% of all modes have modulus > 0.99. Local stability is therefore "contractive, but with a large band of nearly neutral (very slow) modes", not "strongly contractive".
* **[EMPIRICAL]** The spectral radius hides strong non-normality: the largest singular value of the full one-step Jacobian is about 10–50. Short-term gain exists for some perturbation directions (token substitution gives M(1) ≈ 1.4), but random state perturbations never grew (max d_t/d_0 = 0.97).
* **[EMPIRICAL]** State perturbations contract at every ε from 1e-8 to 1 (d_T/d_0 ≈ 0.17 at T=1000, finite-time exponent ≈ −0.0018/step). The Benettin top-4 exponents are about −0.0017 to −0.0018 and are *not* asymptotically converged. We found **no evidence of chaos or exponential divergence**. Open-loop only: the token sequence is fixed; generation feedback was not studied.
* **[EMPIRICAL]** Under constant/zero input the state moves toward a fixed point (spread across 7 initial conditions shrinks 31 → 0.26–0.30 in 4000 steps, not yet fully converged). Under periodic or repeated input it locks onto a forced periodic orbit (relative recurrence error ≲ 2e-4 at the input period versus 0.15–0.47 at other lags). Natural text and random tokens never recur.
* **[EMPIRICAL]** Memory is multi-timescale. A substituted token's effect falls to 50% after ~16 steps and 10% after ~58, then has a slow tail (2.2% at k=500, 1.6% at k=700). A uniform state perturbation has half-life ≈ 245 steps. Long memory lives mostly in layers 0, 10 and 11 (all heads mean γ ≈ 0.9985–0.9989); layers 1 and 4 are fast (mean γ ≈ 0.91–0.94).
* **[EMPIRICAL]** Hidden states are low-dimensional in the linear sense (participation ratio ≈ 71 out of 18,432 real dimensions) and have kNN intrinsic-dimension estimates ≈ 24–28. This does **not** establish an invariant manifold.
* **Cost:** 324.9 MFLOP per decoded token (47% in the LM head), constant in context length. Measured on an Apple M3 Pro: **188 ms/token on CPU** (5.3 tok/s, dominated by a slow PyTorch CPU grouped-convolution kernel) and **13.9 ms/token on MPS fp32** (71.7 tok/s). Decode latency is flat from 128 to 8192 tokens of context.

---

## 2. Exact implemented state-transition equation

Derived from `architecture_v2.py` (`JBRHolographicRecurrentCore.forward`, `L==1` branch, lines 93–171), not from the paper's continuous-time form.

For layer ℓ, with `x_t` the layer input (residual stream), `u_t = RMSNorm(x_t)`:

```
c_t      = u_{t-1}                                   (conv cache, previous normalised input)
z_t      = SiLU( w0 ⊙ c_t + w1 ⊙ u_t + b )           (depthwise conv, kernel 2)
γ_t,h    = 0.9 + 0.099·σ(W_γ z_t)_h                  ∈ (0.9, 0.999)   per head h
Δφ_t,h   = π·tanh(W_Δφ z_t)_h                        ∈ (−π, π)        per head
a_t      = tanh(W_u z_t)            ∈ (−1,1)^768
θ_t      = 2π·W_θ z_t ,   θq_t = 2π·W_key z_t
V_t      = (1−γ_t,h)·a_t·e^{iθ_t}                    (input injection)
H_t      = γ_t,h·e^{iΔφ_t,h} ⊙ H_{t−1} + V_t         (complex, head-wise scalar coefficient)
R_t      = Re( H_t · e^{−iθq_t} ) / √64              (readout)
x'_t     = x_t + W_out R_t ;   x_{t+1}^{(ℓ+1)} = x'_t + FFN(RMSNorm(x'_t))
```

The code splits `H` into real and imaginary parts (`prev_h_r`, `prev_h_i`), so the recurrence runs on real tensors. Checked against the code: our hand-written equation matches the layer output to 3.1e-16 over 64 steps × 12 layers (fp64).

Where each ingredient enters:

| Ingredient | Where |
|---|---|
| γ (decay) | multiplicative coefficient on `H_{t-1}`; also scales the input via `(1−γ)` |
| Phase rotation Δφ | complex rotation of `H_{t-1}` by `e^{iΔφ}` (same angle for all 64 channels of a head) |
| Normalisation | RMSNorm before the core (`ln1`) and before the FFN (`ln2`), plus `ln_f` before the head; none inside the recurrence |
| Convolution | k=2 depthwise causal conv on the normalised input; its cache is part of the state |
| Nonlinearities | SiLU, sigmoid, tanh, cos/sin, RMSNorm and FFN SiLU all act on the *input path* that produces coefficients and values |
| Input injection | `V_t`, bounded by `(1−γ)` |

**Linearity.** The update is affine in `H_{t-1}` for fixed input, so the system is linear in `H` but non-autonomous (coefficients vary with input), and the full multi-layer map is nonlinear because the coefficients of layer ℓ+1 depend on `H` of layer ℓ.

**Mismatch with the README/paper.** The README's continuous-time equation `Ḣ = (−λ+iω)H + λV` is only loosely related to the discrete update above. In the code the input drive is `(1−γ)·a·e^{iθ}` with `a` from `tanh` (not `λV` with a free `V`). Also, parallel (training) and recurrent (inference) paths are *not bit-identical*: the scan uses `log(γ + 1e-8)`, and in fp64 the last-logit difference over 200 tokens is 2.7e-6 (state difference 7.2e-7). That is tiny, but it is not "100% numerical equivalence". We attribute it to the 1e-8 term but did not isolate it.

**Boundedness [PROVEN].** For each complex channel, `|H_t| ≤ γ|H_{t−1}| + (1−γ)|a_t| < γ|H_{t−1}| + (1−γ)`, so `|H_t| < 1` if it starts inside the disk. Observed maximum channel modulus: 0.9806 (natural text, 600 steps) and 0.999 over all 4000-step regimes.

---

## 3. State-space definition

| Quantity | Value |
|---|---|
| Recurrent complex state `H` per layer | 12 heads × 64 = **768 complex** (1,536 real) |
| Complex state, all 12 layers | **9,216 complex** = **18,432 real** |
| Conv cache per layer | 768 real (previous normalised input) |
| Full state (caches + `H`) | **27,648 real** (110,592 B in fp32) |
| Time-dependent quantities | `γ_t, Δφ_t, a_t, θ_t, θq_t`, caches, `H_t` |
| Non-autonomous? | Yes: coefficients depend on the token stream |
| Nonlinear? | Yes in the full stack (coefficient computation + cross-layer coupling); affine in `H` within one layer |

Model: 12 layers, d_model 768, 12 heads, head_dim 64, FFN ×4, vocab 100,277 (tied embedding/LM head). **162,298,656 unique parameters** (239.3 M if the tied matrix is counted twice, as in the checkpoint file).

---

## 4. Stability analysis (local)

**Structure [PROVEN].** Layer ℓ's input depends only on lower layers, so `∂S^ℓ_t/∂S^{ℓ'}_{t-1} = 0` for `ℓ' > ℓ`. Each layer's cache at time t is a function of lower layers only, so its diagonal block is `[[0, 0],[A, D_t]]` with `D_t = diag(γ e^{iΔφ})`. Verified numerically (fp64, t=300): upper-triangle blocks = 0 exactly, cache→cache block = 0 exactly, diagonal block equals `γ e^{iΔφ}` to 6.3e-16.

Consequence: the spectrum of the one-step Jacobian is `{0}` (9,216 cache modes) `∪ {γ_{ℓ,h} e^{±iΔφ_{ℓ,h}}}` (each with multiplicity 64). Full-Jacobian check on a 2-layer sub-stack (4,608 real dims): 3,072 non-zero eigenvalues, moduli agree with the analytic ones to 1.3e-13; 1,536 zero modes (≤ 2.4e-15).

**Spectral radius [EMPIRICAL on 4 streams × 600 steps; value follows from the derivation].** `ρ(J_t) = max γ`: 0.9989–0.9990 at every t (plot `1_jacobian_spectral_radius_vs_t.png`). Max over the run is 0.999 (= the upper clamp), so the dominant modes are pinned at the edge of what the parametrisation allows.

**Mode distribution [EMPIRICAL].** On natural text: 61% of modes have |λ| > 0.99, 63% > 0.98, 66% > 0.95, and 31% < 0.92. The distribution is strongly bimodal and saturated at both clamps: roughly a quarter of modes sit in the lowest bin (γ ≈ 0.90, half-life ≈ 7 steps) and roughly 45% in the top bin (γ ≥ 0.998), with a thin tail between (`2_eigenvalue_modulus_distribution.png`, left; the repeated-token stream is the exception, with far fewer modes at 0.90). Most (layer, head) pairs are either "fast" or "slow" (right panel). Mean |Δφ| ≈ 1.4 rad, so the slow modes also rotate substantially.

**Non-normality [EMPIRICAL].** Largest singular value of the full 12-layer `J_t` (ARPACK on jvp/vjp): natural_A 9.6–21 (median 17), natural_B 18–49 (median 22), repeated token 20–27, random tokens 19–48. So `σ_max ≫ ρ`; the off-diagonal blocks (lower-layer state → higher-layer state) give large one-step gain even though every eigenvalue is inside the unit disk. We did not compute asymptotic growth from this; the Lyapunov section does.

**Verdict.** Locally **contractive in the eigenvalue sense (ρ ≤ 0.999) with a large fraction of nearly neutral modes and strong non-normal transient gain** — i.e. "mixed", not "strongly contractive". This is a statement about the one-step map along the trajectories we ran, **not** global stability, and not closed-loop (generation) stability.

---

## 5. Lyapunov / sensitivity analysis

Method: two copies of the system from `H_0` and `H_0 + ε·(random direction in the h-subspace, norm exactly ε)`, identical token stream, ε ∈ {1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1}, T=1000 steps, three base conditions (zero state, and two states after a 128-token prefix). fp64. Plus Benettin/QR propagation of 4 tangent vectors via forward-mode autodiff along one trajectory (T=1000).

**Finite-difference results [EMPIRICAL].**

* `d_T/d_0` ≈ 0.16–0.18 for all ε and all base states; the curves for different ε are parallel on a log scale (`3_lyapunov_divergence.png`), so the response is in the linear regime even at ε=1.
* Max over t of `d_t/d_0` is 0.96–0.97 (no transient growth for random state perturbations).
* Finite-time exponent `λ(T) = (1/T)ln(d_T/d_0)` = −0.0017 to −0.0018 per step, identical across ε.
* No saturation behaviour was seen (it would show as curves converging for large ε; they do not).

**Benettin exponents [EMPIRICAL, finite-time].** Top-4: −0.00175, −0.00178, −0.00174, −0.00177. The four are nearly equal, consistent with many modes of similar slow decay; the running estimates still oscillate by ±0.0003, so they are not converged to an asymptotic value.

**Theory comparison.** For a fixed token stream, a product of block-triangular Jacobians is block-triangular, so the Lyapunov exponents are those of the diagonal-block products: at most `max_{ℓ,h} mean_t log γ`. On this trajectory that is −0.00109 (and ≤ log 0.999 = −0.00100 for any input). The measured −0.0017/−0.0018 is more negative than this bound, which is allowed at finite T (a likely reason, not tested here, is that the random initial tangent puts most of its mass in faster-decaying modes, so it takes longer than T=1000 for the slowest modes to dominate). The bound itself rests on the triangular-cocycle argument with bounded off-diagonal blocks; we observed finite blocks (σ_max ≤ 49) but did not prove a uniform bound, so we call the bound **derived and numerically consistent**, not rigorously proven.

**Verdict.** Exponential **contraction** (all measured exponents negative; none near zero or positive). Evidence of sensitivity (exponential divergence) is absent. We cannot rule out chaos for other regimes (very different inputs, closed-loop sampling, other ε directions in the cache subspace); we simply did not observe any. The word "chaotic" is not supported.

---

## 6. Attractor / long-term dynamics

Nine drives, T=4000, from the zero state: zero input (embedding forced to 0), constant common token, constant rare token, periodic (periods 2, 7, 31), repeated 40-token passage, natural text, uniform random tokens. Plus two multi-initial-condition tests (7 initial conditions each) under constant-token and zero input.

| Drive | final ‖H‖ | ‖H_t−H_{t−1}‖ (mean, last 500) | Behaviour |
|---|---|---|---|
| zero input | 37.3 | 1.1e-3 | slow convergence toward a fixed point |
| constant common token | 39.1 | 1.7e-3 | same |
| constant rare token | 36.2 | 3.4e-4 | same |
| period 2 | 39.2 | 7.9 | forced 2-cycle (recurrence error 1e-4 at multiples of 2) |
| period 7 | 36.6 | 6.4 | forced 7-cycle (recurrence error 1e-4 at lag 70) |
| period 31 | 33.9 | 5.6 | forced 31-cycle (error ≈ 0 at lag 310) |
| repeat 40-token passage | 33.7 | 5.7 | forced 40-cycle (error 2e-4 at lag 400) |
| natural text | 33.6 | 5.7 | bounded, no recurrence (error 0.37–0.46 at all lags) |
| uniform random tokens | 35.2 | 4.3 | bounded, no recurrence |

Max channel modulus at the end of every run: 0.965–0.999 (< 1, consistent with the bound).

**Initial-condition convergence [EMPIRICAL].** Max pairwise distance among 7 initial conditions falls from 31.2 to 0.26 (constant token) and from 31.1 to 0.30 (zero input) over 4000 steps (≈ −0.0012/step, in line with the slowest mode rates). Not converged to numerical zero because the slowest modes have time constants near 1000 steps.

**Interpretation.** The system is driven, so "attractor" here means a driven attractor: for constant drive, a fixed point; for periodic drive, a forced periodic orbit with the drive's period (entrainment). These are what contraction predicts. We did **not** find intrinsic autonomous attractors, multiple coexisting attractors (all 7 initial conditions drift together), or quasi-periodic or chaotic regimes. Natural and random streams show bounded, non-recurrent motion with no convergence (log-distance slope ≈ 0). Clustering of states in later PCA/ID analyses is finite-sample and should not be read as attractor structure. Multi-attractor behaviour is **not excluded**: 7 initial conditions, 2 drives and a finite horizon are a weak test.

---

## 7. Low-dimensional structure

Data: 96 natural-text sequences × positions 64..512 (stride 8) → 5,376 states per layer (`05_manifold.json`). Complex handling: Hermitian PCA on `C^768`, and PCA on the real-equivalent `R^1536` (Re,Im stacked). Full state: `C^9216` / `R^18432`. Caveat: for the full state, N=5,376 < 18,432 so rank ≤ 5,375, and samples within a sequence are autocorrelated.

Dimensions needed for explained variance (complex PCA, per layer, out of 768):

| layer | 50% | 80% | 90% | 95% | 99% | participation ratio |
|---|---|---|---|---|---|---|
| 0 | 7 | 39 | 90 | 150 | 304 | 17.2 |
| 3 | 11 | 58 | 100 | 134 | 177 | 23.2 |
| 6 | 4 | 34 | 72 | 109 | 166 | 7.8 |
| 9 | 10 | 34 | 56 | 77 | 112 | 26.5 |
| 11 | 4 | 24 | 56 | 103 | 239 | 10.5 |

Full state: complex PCA needs 40 / 319 / 769 / 1,299 / 2,418 complex dimensions for 50/80/90/95/99% (real-equivalent: 42 / 373 / 963 / 1,687 / 3,242 real dimensions); participation ratio ≈ 69 (complex) / 71 (real); top PC 7.2%, top-10 30%.

Intrinsic dimension (kNN, neighbours taken only from *other* sequences to avoid trivial temporal-neighbour bias, 4,000 sampled points): full state TwoNN 23.9, MLE 25.4 (k=10) / 27.6 (k=20); per layer ≈ 18–48 (layers 0 and 11 TwoNN 41 and 48, others 21–24). Local PCA (50 neighbours, 90% variance): 28 (layer 0), 35 (layer 5), 18 (layer 11) of a maximum 49.

**What this does and does not show.**

* **[EMPIRICAL] Low-dimensional representation:** a few dozen directions carry half the variance of an 18,432-dimensional state.
* **[EMPIRICAL, estimator-dependent] Low intrinsic dimension:** ≈ 20–30 per the kNN estimators (biased low for finite samples; they disagree with the linear n90 of several hundred, as expected for a curved, spread-out set).
* **[NOT SHOWN] Invariant manifold:** nothing here tests invariance (that trajectories started on the set stay on it). We make no such claim. The state's low dimensionality is also partly explained by the fact that `H` is a filtered version of a lower-dimensional input stream.

---

## 8. Jacobian / eigenvalue analysis

* Eigenvalue moduli, real and imaginary parts: moduli are `γ ∈ [0.9, 0.999]`; the eigenvalues are `γ cos Δφ ± iγ sin Δφ`, with |Δφ| averaging 1.4 rad (a fast rotation for most modes, quantiles in `02_jacobian.json`).
* Mode counts: ≈ 61% of 18,432 non-zero eigenvalues have |λ| > 0.99; ≈ 31% have |λ| < 0.92; **none** exceed 1; 9,216 are exactly zero (caches).
* Strongly contracting modes: |λ| < 0.92, ≈ 31%. Potentially expanding modes: none by eigenvalue; **non-normal** gain is possible (σ_max ≈ 10–50).
* Complex/non-Hermitian: the map is not Hermitian, so singular values and eigenvalues differ widely (σ_max ≈ 20 vs ρ ≈ 1). The full Jacobian was treated as a real-linear map on `R^{27648}`, whose eigenvalues come in conjugate pairs `γe^{±iΔφ}`.
* Layer structure of the slow modes (mean γ per layer, natural_A): layer 0: 0.9989, layers 10–11: 0.9984; layers 1, 4 ≈ 0.91, 0.94; the rest 0.95–0.98.
* Correlation with retention: see next section.

---

## 9. Memory-retention analysis

Token substitution: at t0 = 200, replace one token (random / one of the 200 most common / a rare one) in 6 contexts (18 perturbations), same continuation for 700 steps; `M(k) = ‖δH_{t0+k}‖ / ‖δH_{t0}‖` over all layers.

| k | 1 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 700 |
|---|---|---|---|---|---|---|---|---|---|
| mean M(k), 18 runs | 1.41 | 1.09 | 0.74 | 0.37 | 0.116 | 0.060 | 0.038 | 0.022 | 0.016 |
| std | 0.21 | 0.22 | 0.15 | 0.09 | 0.044 | 0.022 | 0.018 | 0.014 | — |

* Half-life (first k with M < 0.5): ≈ 16 steps; M < 0.1 at ≈ 58 steps. The tail is slow and heavy: about 2% survives to k=500. That shape is what a sum of exponentials with rates spread over `1−γ ∈ [0.001, 0.1]` would produce **[HYPOTHESIS, not fitted]**.
* The initial M(1)=1.41 > 1 is transient growth through the layer stack (non-normal gain), a case where the large σ_max shows up in a real perturbation.
* Token type: random / common / rare replacements give similar curves (M(50) = 0.121 / 0.101 / 0.126; half-life 16 / 15 / 17). Context matters more: M(50) ranges from 0.064 to 0.184 across the six contexts.
* The effect reaches the output: the L2 change in logits is ≈ 106 at k=5, 38 at k=20, ≈ 26 at k=35.

State perturbation (all layers at once, total norm 1e-3, same continuation): radial (along `H`, "magnitude"), tangential (perpendicular, "phase") and random-complex perturbations are indistinguishable: M(100) = 0.656 / 0.658 / 0.660, M(500) = 0.361 / 0.355 / 0.351, half-life ≈ 245 / 243 / 245 steps. This is expected from the structure: a complex scalar multiplication treats magnitude and phase symmetrically **[PROVEN for the diagonal blocks]**, so the model's memory has no magnitude/phase preference.

**Link to Jacobian eigenmodes.**

* Token-substitution per-head decay rate (fit over k=20–400) vs the head's mean `log γ`: Spearman 0.78, Pearson 0.76 (`6_memory_retention.png`, right). Heads with `γ` near 0.999 retain longest. Heads with fast γ decay slower than their own `γ` predicts (rate floor ≈ −0.01): consistent with their state being re-driven by slower lower layers **[HYPOTHESIS]**.
* The same comparison for the simultaneous all-layer state perturbation gives a *negative* correlation (Spearman −0.14 to −0.25). That per-head rate is confounded by re-injection from lower layers, so this analysis is **[INCONCLUSIVE]**; it does not contradict the token-based result.

**Mechanism [EMPIRICAL + derived].** Persistent information pathways exist and come from near-unit-modulus diagonal modes (γ ≈ 0.999 gives a half-life of ≈ 693 steps for an isolated mode), concentrated in layers 0, 10, 11, plus a spread of faster modes. Rotation (Δφ) does not itself preserve information longer; it sets the phase at which injected values accumulate.

---

## 10. FLOPs per prediction

One next-token prediction = one recurrent step with L=1 through 12 layers + final norm + LM head. Counted op-by-op from the code (`07_flops.py`).

Conventions: matmul/linear/conv = 2·MAC; depthwise k=2 conv = 4 FLOP/channel; add/sub/mul/div = 1 FLOP; transcendental (exp, log, sin, cos, tanh, sigmoid, rsqrt) counted as events with weight 1 (a lower bound; 10 and 20 given as sensitivity); SiLU = sigmoid + 1 mul. Complex arithmetic: the code never uses complex dtypes, so we count the real operations executed: `γ e^{iΔφ} H + V` costs 10 FLOP per complex element (4 mul + 2 add for the rotation, 2 mul for γ, 2 add for injection). A generic complex multiply is 6 FLOP and a complex add 2. Memory ops (cat, view, embedding lookup) are free.

| Component | FLOPs |
|---|---|
| per layer, matmul/conv | 14,195,712 |
| per layer, elementwise | 34,634 |
| per layer, transcendental events | 7,730 |
| 12 layers + final norm | 170,856,912 (+ ln_f) |
| LM head (768 × 100,277) | 154,025,472 (47.4%) |
| **Total per decoded token (weight 1)** | **324,885,458 ≈ 324.9 MFLOP** |
| weight 10 / 20 for transcendentals | 325.7 / 326.6 MFLOP |
| softmax + sampling (outside the model) | ≈ 0.5 MFLOP extra |

Cross-check: PyTorch `FlopCounterMode` on the real forward gives 324,374,016 matmul/conv FLOPs, exactly equal to our matmul/conv count (relative difference 0.0).

**Scaling with context.** Decode cost per token is **constant** in context length (fixed-size state). For parallel prefill (chunks ≤ 512, as shipped) the scan adds `O(L)` per token: 325.9 MFLOP/token at L=16, 344 at 256, 363 at 512, constant beyond because of chunking (analytic; FlopCounter matmul-only gives 325.6 / 343.2 / 362.1, matching). The shipped prefill also computes LM-head logits for every prompt token, which is 47% of its cost; we did not change this.

---

## 11. Prediction latency

Hardware: Apple M3 Pro (11 cores, 18 GB), macOS 26.3, PyTorch 2.13.0, Python 3.13.3, torch threads 5. Batch size 1, fp32 weights, prompt 64 tokens, unmodified `model(next_token, past_states)`. Load time excludes warm-up; warm-up is 100 decode steps after the prompt. Results in `08_latency.json` (raw per-step arrays included).

| | CPU fp32 | MPS fp32 |
|---|---|---|
| model load | 3.5 s | 4.0 s |
| warm-up (prompt + 100 steps) | 19.0 s | 1.0 s |
| measured | 3 runs × 300 steps | 5 runs × 1000 steps |
| forward-only mean / median / std | 187.95 / 186.38 / 4.20 ms | 13.95 / 14.17 / 0.69 ms |
| p90 / p95 / p99 | 192.5 / 194.6 / 203.6 ms | 14.6 / 14.7 / 15.5 ms |
| end-to-end (forward + temp. softmax + multinomial + `.item()`) mean | 188.35 ms | 14.51 ms |
| end-to-end p50 / p90 / p95 / p99 | 187.6 / 191.8 / 194.1 / 196.2 | 14.78 / 15.14 / 15.24 / 15.79 |
| run-to-run spread of mean (forward) | 186.8–190.0 ms | 13.93–13.97 ms |

**fp16 on MPS is not supported by the shipped code**: the parallel-scan prefill builds `causal_mask` with `torch.ones` (fp32) and matmuls it against fp16 tensors (`RuntimeError: Expected arguments of same type but got Float and Half`). We did not alter the code, so no fp16 or bf16 numbers are reported.

**Why CPU is slow (profiled).** One CPU step spends 70–95% of its time in PyTorch's `_slow_conv2d_forward`, called once per channel (768 × 12 layers per step) for the depthwise `Conv1d(groups=768, kernel=2)`; the matmuls are about 3%. So the CPU number is an artefact of this implementation on this PyTorch CPU backend, not of the FLOP count (≈ 1.7 GFLOP/s effective). An elementwise form of the same op (bit-identical outputs, tested) ran a step in 14.7 ms vs 189 ms; we used that only inside the analysis scripts 03–06 and **not** for any latency measurement.

---

## 12. Throughput

| | CPU fp32 | MPS fp32 |
|---|---|---|
| decode tokens/s (1 / mean forward) | 5.3 | 71.7 |
| decode tokens/s end-to-end | 5.3 | 68.9 |
| prefill tokens/s (chunk 512), 128 / 512 / 1024 / 2048 / 4096 / 8192 | 480 / 937 / 943 / 964 / 968 / 957 | 4341 / 4565 / 4503 / 4418 / 4414 / 4419 |
| decode ms/token after those prefills | 186.8 – 193.4 | 14.1 – 14.3 |

Decode latency does not grow with context (128 → 8192 tokens): this empirically confirms constant-time decoding. Prefill throughput is flat for ≥ 512 tokens.

The README's "> 18,700 tokens/second" is not reproduced here and is a different measurement (hardware not stated for it; the bundled `benchmark_results.json` describes a different 110M model on an RTX 4060 at ≈ 84 tok/s decode). Our M3 Pro numbers are not comparable to those.

---

## 13. Limitations

1. The checkpoint's vocabulary (100,277) matches `tiktoken` `cl100k_base`, which is not in the repo. We **assumed** it is the right tokenizer. This affects only the "natural text" token streams (Python reference docs, 116,356 tokens, SHA-256 prefix `8d5b3cbd…`); all conclusions about the dynamics use the real weights. Natural-text results depend on that assumption and on the text being English technical prose. No model quality (perplexity, generation) was evaluated.
2. All stability/Lyapunov results are **open-loop** with a fixed token sequence, local, finite-time, from the zero state or short prefixes; no statement about closed-loop generation (where tokens depend on the state) or global behaviour.
3. Lyapunov exponents: finite T=1000, one trajectory for Benettin; not converged. Perturbations were applied to `H` only (the conv-cache subspace, where the largest non-normal gains may sit, was not perturbed in the finite-difference runs).
4. Dynamics were computed in fp64 (exact upcast of the fp32 weights); latency is fp32. fp32 vs fp64 dynamics were not compared.
5. The analytic spectrum is verified exactly on a 2-layer sub-stack and by structural tests on the full 12 layers (triangularity and diagonal blocks), not by a full 27,648-dimensional eigendecomposition.
6. PCA/ID: N=5,376 samples (full-state PCA rank-limited), autocorrelated samples, kNN estimators biased; 96 sequences from one corpus.
7. Attractor tests use 7 initial conditions and two drive types; periodicity checks were at stride-10 snapshots.
8. Memory: 6 contexts, 3 token types, one trajectory for the state-perturbation test; per-head state-perturbation rates are confounded.
9. Latency measured on one machine; CPU used 300 × 3 samples (reduced from 1000 × 5 because of the slow conv kernel); MPS 1000 × 5. Machine was otherwise idle but not isolated.
10. The `FLOP` totals are an analytic count; transcendental weighting is a convention.
11. `analysis/` and `model/` are untracked in git (commit `4877549`).

---

## 14. Reproducibility

* Git commit (repo): `4877549f51f6bb233badaa9fc669a47e7c2d0534`; `model/` and `analysis/` untracked.
* Checkpoint: `model/jbr_v2_ce_weights.pt` (649,280,810 bytes; sha256 of first 64 MiB = `09366f35d69c1ebf…`); loads with `strict=True` into `architecture_v2.JBRLanguageModel(100277, 768, 12, 12)`; 162,298,656 unique parameters.
* Tokenizer: `tiktoken` 0.14.0, `cl100k_base`. Corpus: `pydoc_data.topics` (stdlib), concatenated in key order.
* Seed 1234 (NumPy/PyTorch; windows selected with fixed RNG).
* Software: Python 3.13.3, PyTorch 2.13.0, NumPy 2.3.4, SciPy 1.16.2, scikit-learn 1.7.2; Apple M3 Pro, 19,327,352,832 B RAM, macOS 26.3 (Darwin 25.3.0); no CUDA.
* dtype: fp64 for dynamics, fp32 for FLOP checks and latency. Batch size 1 for latency; warm-up 100 steps; measured 3×300 (CPU), 5×1000 (MPS); sequence/context lengths 128–8192.
* Run: `./run_all.sh` (≈ 1 hour; scripts 07–08 must run on an idle machine). Each JSON stores its own `env` block (commit, hardware, versions, seed).
* Analysis-only conv shortcut (`common.py`, `fast_conv=True`) is bit-identical to `CausalConv1d` for L=1 (`00_patch_equivalence.json`: max diff 0.0 in fp64 and fp32); scripts 01, 02 and 08 use the original code.
* Results: `results/00…08_*.json` and `.log`; plots: `plots/1…9_*.png`. Large state snapshots from script 04 are not stored (written to a scratch directory).

---

## 15. Conclusions

1. **Proven from the code:** a gated complex linear recurrence with `|H| < 1`, exactly block-triangular one-step Jacobian, spectrum `{0} ∪ {γe^{±iΔφ}}`, `ρ ≤ 0.999`.
2. **Observed:** locally contractive state dynamics (no divergence for ε from 1e-8 to 1), many near-neutral slow modes, non-normal transient gain (σ_max ≈ 10–50), forced fixed points/periodic orbits under regular input, and no recurrence under natural text.
3. **Memory:** real, multi-timescale, with a heavy tail and long-lived modes concentrated in layers 0, 10 and 11; magnitude and phase perturbations are retained equally.
4. **Not supported:** "chaotic", "has an invariant manifold", "is globally stable". The data support "locally contractive along the studied open-loop trajectories", "low linear and kNN-estimated dimension", and "bounded states by construction".
5. **Cost:** 324.9 MFLOP/token constant in context length; 13.9 ms (71.7 tok/s) on MPS fp32 and 188 ms (5.3 tok/s) on CPU fp32 for this implementation, the latter dominated by the grouped-conv CPU kernel.

### Summary table

| Metric | Result | Interpretation | Confidence |
|---|---|---|---|
| Parameter count | 162,298,656 unique (tied head) | 239.3 M if the tied matrix is counted twice | High |
| State dimensionality | 9,216 complex `H` (18,432 real) + 9,216 real cache = 27,648 real | constant per-token memory (110.6 KB fp32) | High |
| Spectral radius (one-step Jacobian) | 0.9989–0.9990 at every step | at the γ upper clamp; local eigen-stability but near-neutral | High (derived, verified) |
| Dominant eigenvalue magnitude | max γ = 0.999 (61% of modes > 0.99; none > 1) | slow modes dominate retention | High |
| Largest singular value of full J_t | ≈ 10–50 (median ≈ 20) | strong non-normal transient gain despite ρ < 1 | Medium (4 streams, 36 points) |
| Lyapunov estimate | −0.0017 to −0.0018 /step (T=1000, finite-time); analytic upper bound −0.0011 | contraction; not a converged asymptotic value; no sign of chaos | Medium (open-loop, one trajectory) |
| Effective / intrinsic dimension | PR ≈ 71 (of 18,432 real); n90 = 963 real; kNN ID ≈ 24–28 | low-dimensional representation; no invariant-manifold claim | Medium-low (estimator/sample dependent) |
| Memory retention | token substitution: M=0.5 at k≈16, 0.1 at k≈58, 0.022 at k=500; uniform state perturbation half-life ≈ 245 | multi-timescale memory with a long tail | Medium-high |
| FLOPs/token | 324.9 MFLOP (47% LM head); constant in context; 325–363 MFLOP for chunked prefill | matmul/conv count matches PyTorch FlopCounter exactly; transcendental weighting is a convention | High |
| ms/token | CPU fp32 188.0 (p99 203.6); MPS fp32 13.95 (p99 15.5); e2e 188.4 / 14.5 | CPU dominated by slow grouped-conv kernel; flat in context | High for this machine |
| Tokens/s | CPU 5.3; MPS 71.7 (68.9 e2e); prefill 960 (CPU) / 4,400 (MPS) tok/s | one Apple M3 Pro, batch 1; not comparable to README's 18.7k figure | High for this machine |
