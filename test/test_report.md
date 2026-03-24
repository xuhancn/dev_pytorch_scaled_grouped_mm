# Local Kernel Test Report: `scaled_grouped_mm`

**Date**: 2026-03-25
**Device**: Intel Arc B580 (BMG)
**Environment**: Intel oneAPI 2025.x, PyTorch with XPU support, conda env `xu_pytorch`

## Summary

All 10 tests pass. The SYCL `scaled_grouped_mm` extension correctly implements FP8×FP8→BF16 grouped GEMM with rowwise float32 scaling across all 4 input modes.

## Approach

Since sycl-tla lacks a combined grouped+FP8+rowwise-scaling dispatch policy, the kernel:
1. **Dequantizes** FP8 inputs to BF16 with rowwise scale application
2. **Dispatches** to the existing BF16 grouped GEMM sycl-tla kernel

This introduces slightly more numerical error than a fused kernel (scales are applied in BF16 precision before the GEMM, rather than in float32 within the epilogue).

## Test Results

```
test_3d_3d .................... ok    (batched grouped GEMM, no offsets)
test_2d_3d .................... ok    (ragged A / MoE pattern with M offsets)
test_3d_2d .................... ok    (ragged B with N offsets)
test_2d_2d .................... ok    (ragged K with K offsets)
test_single_group_3d_3d ....... ok    (edge case: G=1)
test_min_k_size ............... ok    (edge case: K=16)
test_larger_shapes ............ ok    (G=8, M=64, N=128, K=128)
test_wrong_a_dtype ............ ok    (validation: rejects non-FP8 input)
test_missing_offs_for_2d ...... ok    (validation: requires offsets for 2D input)
test_bias_not_supported ....... ok    (validation: rejects bias argument)

Ran 10 tests in 0.725s — OK
```

## Accuracy

Reference: per-group `(A_fp8.float() * scale_a) @ (B_fp8.float() * scale_b).T → bf16`

Test shapes: M=16, N=32, K=64, G=4 (plus edge cases).

Tolerances: `atol=0.2, rtol=0.05`. These are wider than the CUDA kernel's tolerances (`atol=5e-2, rtol=5e-4`) because the dequantize-then-GEMM approach truncates intermediate values to BF16 before the matrix multiply. A future fused FP8+scale sycl-tla kernel would allow tighter tolerances.

## Benchmark

| Mode | G | M | N | K | Avg (ms) | TFLOPS |
|------|---|---|---|---|----------|--------|
| 3D×3D | 4 | 16 | 32 | 64 | 0.181 | 0.001 |
| 3D×3D | 4 | 64 | 128 | 128 | 0.183 | 0.046 |
| 3D×3D | 8 | 64 | 128 | 128 | 0.192 | 0.087 |
| 3D×3D | 8 | 128 | 256 | 256 | 0.177 | 0.757 |
| 3D×3D | 16 | 128 | 256 | 256 | 0.232 | 1.158 |
| 2D×3D | 4 | 16 | 32 | 64 | 0.153 | 0.002 |
| 2D×3D | 4 | 64 | 128 | 128 | 0.163 | 0.052 |
| 2D×3D | 8 | 64 | 128 | 128 | 0.178 | 0.094 |
| 2D×3D | 8 | 128 | 256 | 256 | 0.249 | 0.540 |
| 2D×3D | 16 | 128 | 256 | 256 | 0.328 | 0.819 |

Settings: 10 warmup iterations, 50 timed iterations, `torch.xpu.synchronize()` after each batch.

## Key Issues Found and Fixed

1. **dtype promotion (bf16 × f32 → f32)**: Multiplying a BF16 tensor by a float32 scale silently promotes the result to float32. The sycl-tla kernel interprets the data pointer as `bfloat16_t`, causing garbage output. Fixed by adding `.to(torch::kBFloat16)` after every scale multiplication.

2. **B matrix layout**: The sycl-tla grouped GEMM kernel expects B as (K,N) row-major contiguous. The input `mat_b` is transposed (logical (K,N), physical (N,K)). Fixed by calling `mat_b.contiguous()` to materialize the logical layout.

3. **2D×2D dimension indexing**: After making B contiguous as (total_K, N), dimension indices for N and K-slice were swapped. Fixed `N = b_bf16.size(1)` and slice along dim 0.
