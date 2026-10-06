# J-BRAIN v2: Joint-Behavioral Resonant Artificial Intelligence Network

> **Constant Inference Memory Language Modeling via Selective Non-Hermitian Wave Dynamics and Holographic Uncertainty Bounds**  
> **Author:** Aryan Salemabadi & Soroush Tanzadeh Mojarad

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime%20INT8-005CED.svg?style=flat&logo=onnx)](https://onnxruntime.ai/)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-WearOS-black.svg)](https://pytorch.org/executorch/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Executive Overview

Modern Large Language Models (LLMs) heavily rely on the Transformer architecture, which suffers from an $O(N)$ memory bottleneck during inference due to Key-Value (KV) cache scaling linearly with sequence length. While recent State Space Models (SSMs) and RNNs achieve constant inference memory, effectively parameterizing data-dependent phase dynamics while retaining strict boundedness remains a fundamental challenge.

**J-BRAIN v2** advances discrete dissipative representations into **selective non-Hermitian wave mechanics**. By encoding token information into complex-valued amplitudes and phases via Euler's identity, and governing state propagation through input-dependent selective dissipation ($\gamma_t$) and unitary phase rotation ($\Delta\phi_t$), J-BRAIN v2 maintains a strictly bounded, **constant-memory footprint ($O(1)$)** while achieving massive training parallelization.

$$V_t = u_t \odot e^{i \theta_t} \odot (1 - \gamma_t)$$

---

## Key Architectural Innovations

### 1. Selective Non-Hermitian Recurrent Core
The hidden state evolves within a multi-head complex phase space governed by driven non-Hermitian wave dynamics. In the continuous-time limit, the state vector $H(t)$ evolves according to:
$$\dot{H}(t) = (-\lambda(t) + i\omega(t))H(t) + \lambda(t)V(t)$$
This couples input-dependent phase decay ($\gamma_t \in (0.90, 0.999)$) with instantaneous angular phase rotations $\Delta\phi_t \in (-\pi,\pi)$

### 2. Analytical Unit-Disk State Boundedness
By modulating the incoming complex wave packet with an input-convex bound factor $(1-\gamma_t)$, the recurrent state trajectory is analytically confined within the complex unit disk ($\Vert{}H_t\Vert{}_\infty \le 1$) across infinite sequence horizons, completely preventing numerical divergence without explicit gradient clipping.

### 3. Exact Parallel GPU Training (`complex_phase_parallel_scan`)
Non-Hermitian recurrence is unrolled into parallel matrix operations via cumulative logarithmic decay and cumulative phase matrices. This achieves 100% numerical equivalence with sequential edge inference, eliminating the $O(L)$ recurrent training bottleneck on GPU Tensor Cores.

### 4. Holographic Wave Interference Readout
Context retrieval is executed via variance-scaled holographic wave interference readout stabilized through head-wise RMSNorm, dynamically routing information through constructive resonance and destructive noise cancellation.

---

## Fundamental Theoretical Bounds

J-BRAIN v2 mathematically guarantees long-term memory coherence and structural identifiability through several fundamental invariants natively derived from its architecture:

| Fundamental Bound / Invariant | Symbol | Analytical Value | Architectural Implication |
| :--- | :---: | :---: | :--- |
| **Wave Uncertainty Invariant** | $\mathcal{I}_S$ | $2$ | Scale-invariant saturation of the time–frequency bound. |
| **Resonance Quality Factor** | $\mathcal{Q}_S$ | $\approx 3140\text{ rad}$ | Maximum coherent phase accumulation prior to amplitude dissipation. |
| **Incoherent Noise Floor** | $\sigma^2_{\text{phase}}$ | $1/2$ | Expected projection variance of aliased modes inducing interference degradation. |
| **Holographic Orthogonality Bound** | $\mathcal{K}_{\text{holo}}$ | $2d_h = 128$ | Absolute capacity limit of superimposed independent modes per head. |
| **Phase-Space Arc Length** | $\mathcal{S}_{\text{arc}}$ | $\approx 3140$ | Total geometric distance a localized wave packet travels in Hilbert space. |

---

## Empirical Benchmarks: Unprecedented Sample Efficiency

A 163M-parameter J-BRAIN v2 model trained on merely **~4 billion tokens** demonstrates exceptional sample efficiency, outperforming significantly larger-budget baselines trained on 300B tokens (e.g., Pythia, Mamba) on standard zero-shot reasoning benchmarks.

| Model | Architecture | Parameters | Pretraining Tokens | ARC-Easy (acc ↑) | PIQA (acc ↑) | HellaSwag (acc_norm ↑) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **J-BRAIN v2** | **Non-Hermitian Wave** | **163M** | **4B** | **50.76%** | **61.53%** | **30.02%** |
| GPT-2 | Transformer | 124M | ~100B | 43.50% | 62.90% | 31.10% |
| Pythia | Transformer | 160M | 300B | 43.20% | 61.40% | 30.20% |
| Mamba | State Space | 130M | 300B | 48.00% | 64.50% | 35.30% |
| RWKV-4 | Linear RNN | 169M | 332B | 47.47% | 65.07% | 32.26% |

### Hardware Execution & Edge Deployment
Hardware profiling confirms true constant-memory execution. A 163M J-BRAIN v2 instance operates with a minimal recurrent state footprint of **just 36 KB (BF16)**, zero KV-cache allocation, and achieves inference throughput exceeding **18,700 tokens/second**, establishing immediate viability for extreme edge computing and low-resource devices (e.g., Wear OS).

---

## Project Structure

```text
JBRAIN/
├── architecture.py              # Core J-BRAIN v2 PyTorch layer & complex_phase_parallel_scan
├── test_equivalence.py          # Numerical equivalence verification (Recurrent vs Parallel Scan)
├── export_onnx.py               # ONNX computational graph export pipeline
├── quantize_jbrain.py           # Static INT8/BF16 quantization scripts
├── test_smartwatch.py           # Constant memory Edge/Wear OS simulation environment
├── benchmark_results.md         # Detailed hardware execution and zero-shot metrics
└── Data/
    └── jbrain_tokenizer.json    # Calibrated Tokenizer schema