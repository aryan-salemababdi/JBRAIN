#!/bin/bash
# Reproduces every measurement. Run from this directory. Part 2 (07,08) must run with no other load.
cd "$(dirname "$0")"
for s in 00_patch_equivalence 01_state_transition_verify 02_jacobian 03_lyapunov 04_attractor 05_manifold 06_memory 07_flops 08_latency; do
  echo "=== $s $(date)"; python3 $s.py > results/$s.log 2>&1 || echo "FAILED $s"
done
python3 09_plots.py > results/09_plots.log 2>&1 || echo "FAILED plots"
echo "=== ALL DONE $(date)"
