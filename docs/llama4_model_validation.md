# Llama 4 Scout Model-Level Validation Report

**Date:** 2026-05-13
**Device:** Intel Arc B580 (BMG)
**PyTorch:** Built from source with XPU (USE_SYCLTLA=ON)

## Model Architecture

| Parameter | Value |
|---|---|
| Model | Llama 4 Scout (109B/17B-16E) |
| hidden_size (K) | 5120 |
| intermediate_size (N) | 8192 |
| num_local_experts (E) | 16 |
| num_experts_per_tok | 1 (top-1 routing) |
| MoE layers | All 48 decoder layers |

## Test Methodology

The torchao MoE FP8 training path calls `torch._scaled_grouped_mm` with the **2Dx3D input mode** (ragged A with offsets along M):

- **Gate/Up projection:** `A=(M, 5120) FP8` × `B=(G, 5120, 8192) FP8` → `(M, 8192) BF16`
- **Down projection:** `A=(M, 8192) FP8` × `B=(G, 8192, 5120) FP8` → `(M, 5120) BF16`

Where G = experts per device (varies by Expert Parallelism degree).

The full model (M=32768, G=16) is too large for Arc B580 (~12GB). We use scaled-down M (512–2048) while keeping K, N, G at real model dimensions.

### Cross-Device Comparison

Each test:
1. Creates FP8 tensors and float32 scales on **CPU**
2. Computes reference on **CPU** via float32 dequantize + matmul + cast to BF16
3. Copies tensors to **XPU**, runs the SYCL kernel
4. Compares XPU output against CPU reference

This validates cross-device numerical equivalence, not just XPU self-consistency.

## Results — Local SYCL Extension

All 9 tests **PASSED**. Total time: 120s.

| Test | M | G | K | N | CPU (ms) | XPU (ms) | Max Error | Mean Error |
|---|---|---|---|---|---|---|---|---|
| gate_up G1 | 512 | 1 | 5120 | 8192 | 6069.6 | 14.4 | 2.0000 | 0.0504 |
| gate_up G2 | 1024 | 2 | 5120 | 8192 | 12102.4 | 31.0 | 2.0000 | 0.0466 |
| gate_up G4 | 2048 | 4 | 5120 | 8192 | 24099.8 | 41.8 | 2.0000 | 0.0481 |
| gate_up G16 | 512 | 16 | 5120 | 8192 | 6583.1 | 108.2 | 2.0000 | 0.0469 |
| down G1 | 512 | 1 | 8192 | 5120 | 6234.5 | 12.2 | 2.0000 | 0.0634 |
| down G2 | 1024 | 2 | 8192 | 5120 | 12439.7 | 28.3 | 2.0000 | 0.0604 |
| down G4 | 2048 | 4 | 8192 | 5120 | 24698.4 | 39.0 | 2.0000 | 0.0590 |
| down G16 | 512 | 16 | 8192 | 5120 | 6590.5 | 205.9 | 2.0000 | 0.0578 |
| gate_up unbalanced | 1024 | 4 | 5120 | 8192 | — | — | — | PASS |

## Results — Full PyTorch Dispatch Chain (`torch._scaled_grouped_mm`)

All 9 tests **PASSED**. Total time: 121s.

| Test | M | G | K | N | CPU (ms) | XPU (ms) | Max Error | Mean Error |
|---|---|---|---|---|---|---|---|---|
| gate_up G1 | 512 | 1 | 5120 | 8192 | 6211.1 | 7.8 | 2.0000 | 0.0475 |
| gate_up G2 | 1024 | 2 | 5120 | 8192 | 12046.2 | 18.4 | 2.0000 | 0.0472 |
| gate_up G4 | 2048 | 4 | 5120 | 8192 | 24045.0 | 31.4 | 2.0000 | 0.0474 |
| gate_up G16 | 512 | 16 | 5120 | 8192 | 6572.9 | 82.1 | 2.0000 | 0.0471 |
| down G1 | 512 | 1 | 8192 | 5120 | 6138.6 | 6.3 | 2.0000 | 0.0617 |
| down G2 | 1024 | 2 | 8192 | 5120 | 12463.0 | 15.5 | 2.0000 | 0.0608 |
| down G4 | 2048 | 4 | 8192 | 5120 | 25253.7 | 28.4 | 2.0000 | 0.0603 |
| down G16 | 512 | 16 | 8192 | 5120 | 6580.6 | 180.7 | 2.0000 | 0.0602 |
| dispatch unbalanced | 1024 | 4 | 5120 | 8192 | — | — | — | PASS |

## Analysis

### Accuracy

- **Max absolute error:** 2.0 across all tests (consistent, BF16 quantization boundary)
- **Mean absolute error:** 0.047–0.063 (well within acceptable range)
- **Error source:** XPU kernel dequantizes FP8→BF16 then accumulates in BF16, while CPU reference accumulates in float32. With K=5120–8192, ~5K–8K additions each lose BF16 precision. The max error of 2.0 corresponds to a single BF16 ULP at typical output magnitudes (~300).

### Performance

- **XPU vs CPU speedup:** 150–1000× (CPU reference is unoptimized float32 dequant+matmul)
- **XPU scaling:** Time scales roughly linearly with total compute (M×G×K×N)
  - G1 M=512: ~7–14ms
  - G4 M=2048: ~28–42ms
  - G16 M=512: ~82–206ms (16× more weight data to process)
- **Dispatch overhead:** Negligible — dispatch chain matches extension performance

### Coverage

| Scenario | G | M | Description |
|---|---|---|---|
| EP=16 | 1 | 512 | Expert Parallelism across 16 devices |
| EP=8 | 2 | 1024 | Expert Parallelism across 8 devices |
| EP=4 | 4 | 2048 | Expert Parallelism across 4 devices |
| No EP | 16 | 512 | All experts on single device, small batch |
| Unbalanced | 4 | 1024 | Non-uniform token distribution (128/256/512/128) |

## Conclusion

The `_scaled_grouped_mm` XPU kernel correctly handles all Llama 4 Scout MoE shapes across both the local SYCL extension and the full PyTorch dispatch chain. Cross-device comparison confirms numerical equivalence with CPU reference within expected BF16 accumulation tolerances.

## Test Script

`test/test_llama4_model_shapes.py` — not included in upstream PR per design.
