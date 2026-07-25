# J-BRAIN: Joint-Behavioral Resonant Artificial Intelligence Network

> **An $O(1)$ Holographic Language Model Inspired by Discrete Dissipative Flow and Wave Mechanics**  
> **Author:** Aryan Salemabadi ([aryansab80@gmail.com](mailto:aryansab80@gmail.com))

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime%20INT8-005CED.svg?style=flat&logo=onnx)](https://onnxruntime.ai/)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-WearOS-black.svg)](https://pytorch.org/executorch/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Executive Overview

Modern Large Language Models (LLMs) based on the Transformer architecture suffer from an $O(N)$ memory bottleneck during inference due to linear scaling of the Key-Value (KV) cache. While State Space Models (SSMs) and RNNs achieve $O(1)$ inference memory, they often suffer from catastrophic forgetting or gradient instability when forcing sequence histories into real-valued vectors.

**J-BRAIN** bridges deep wave mechanics and sequence modeling by representing token information through complex-valued amplitudes and phases via Euler's identity:

$$V_t = u_t \odot e^{i \theta_t} = u_t \cos(\theta_t) + i u_t \sin(\theta_t)$$

By governing hidden state propagation through parametric discrete dissipation ($\gamma \in (0.90, 0.999)$), J-BRAIN retains context within a **strictly fixed-size state memory footprint ($O(1)$ space complexity)** while retaining full training parallelism on modern GPUs via the `parallel_decay_scan` algorithm.

---

## Key Architectural Innovations

### 1. Complex Wave Encoding & Hilbert Space Projection
Rather than static spatial position embeddings, J-BRAIN projects contextual features $X_{\text{mixed}}$ into a complex Hilbert space:
* **Semantic Amplitude ($u_t \in (-1, 1)$):** Encodes semantic energy/density.
* **Spatial Phase Coordinate ($\theta_t \in [0, 2\pi)$):** Encodes positional and topological routing as continuous complex rotations.
* **Query Reference Phase Beam ($\theta_{q,t}$):** Acts as a directional illumination beam during context retrieval.

### 2. Holographic Wave Interference Readout
Information decoding is executed via constructive/destructive phase interference, analogous to optical holography:

$$R_k = \frac{1}{\sqrt{d_h}} \left( H_r \odot \cos(\theta_q) + H_i \odot \sin(\theta_q) \right)$$

This allows sharp associative retrieval over extended context windows without accumulating an expanding KV-cache.

### 3. Exact Parallel GPU Training (`parallel_decay_scan`)
Linear dissipative recurrence ($H_t = \gamma H_{t-1} + V_t$) is reformulated into an exact lower-triangular matrix multiplication scan ($M_{\text{decay}}$). This bypasses sequential GPU execution bottlenecks during pre-training while matching $O(1)$ step-by-step updates during edge inference.

### 4. Segmented Prompt Processing (`encode_prompt_chunked`)
Long prompts ($L \gg 512$) are ingested in fixed chunks (e.g., $C=512$) using state passing. This protocol eliminates Out-Of-Memory (OOM) spikes and achieves sub-linear scaling during the prompt prefill phase on resource-constrained devices.

---

## Empirical Benchmarks & Hardware Performance

Evaluating a 110M-parameter J-BRAIN model quantized to INT8 ONNX on wearable processors (Wear OS) demonstrates deterministic low-resource execution:

| Metric | Transformer (Standard Attention) | J-BRAIN (110M INT8 ONNX) |
| :--- | :--- | :--- |
| **Inference Memory Footprint** | $O(N)$ (Linear KV Cache growth) | **$O(1)$ Deterministic (~20 MB RAM)** |
| **Prefill Time Complexity** | $O(N^2)$ Quadratic Overhead | **Sub-Linear Chunked Prefill** |
| **Generation Speed** | Slows down over extended contexts | **Sustained ~85 tokens/sec** |
| **Edge Deployment Hardware** | Requires high-end mobile NPUs | **Wear OS Smartwatches (ARM CPUs)** |

---

## Project Structure

```text
JBRAIN/
├── architecture.py              # Core J-BRAIN PyTorch layer & parallel_decay_scan
├── test_equivalence.py          # Numerical equivalence verification (RNN loop vs Parallel Scan)
├── export_onnx.py               # ONNX computational graph export pipeline
├── quantize_jbrain.py           # Static INT8 quantization script
├── test_onnx.py                 # ONNX Runtime evaluation and correctness test
├── test_smartwatch.py           # Wear OS / Low-resource simulation environment
├── benchmark_results.md         # Detailed hardware execution metrics
├── benchmark_prefill.png        # Sub-linear prefill scaling benchmark chart
├── benchmark_results.png        # Long-context memory stress-test visualization
└── Data/
    └── jbrain_persian_tokenizer.json  # Calibrated BPE Tokenizer schema