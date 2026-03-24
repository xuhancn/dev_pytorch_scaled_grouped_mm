# CUDA `_scaled_grouped_mm` Analysis

## Operator Schema

From `native_functions.yaml:7344`:
```yaml
- func: _scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b,
                           Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None,
                           ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
  variants: function
  dispatch:
    CUDA: _scaled_grouped_mm_cuda
  tags: needs_exact_strides
```

No XPU dispatch yet. There is also a `_scaled_grouped_mm_v2` with recipe-based scales (out of scope for initial port).

## Dispatch Chain

```
torch._scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, ...)
  → aten dispatcher (CUDA key)
  → _scaled_grouped_mm_cuda()           [GroupedBlas.cpp:418]
    ├─ Validation (lines 428-478)
    ├─ create_grouped_gemm_output_tensor()  [GroupedMMUtils.h:38]
    ├─ MXFP8 path (if float8_e8m0fnu scales + USE_MSLK)
    │   → _mx8_mx8_bf16_grouped_mm_mslk()   [CUDA-only, skip for XPU]
    └─ Rowwise FP8 path (float32 scales)
        → _f8_f8_bf16_rowwise_grouped_mm()   [GroupedBlas.cpp:210]
          → _f8_f8_bf16_rowwise_grouped_mm_cuda()  [ScaledGroupMM.cu]
            → CUTLASS f8f8bf16_grouped_gemm_impl_sm90 template
```

## Validation Rules (GroupedBlas.cpp:428-478)

1. **Device**: SM 9.0+, SM 10.0+, or ROCm MI300+
2. **Strides**: `mat_a` NOT transposed, `mat_b` transposed (via `check_valid_strides_and_return_transposed`)
3. **Dimensions**: Both 2D or 3D
4. **Contraction dim**: `mat_a.size(-1) == mat_b.size(-2)` when not both-2D
5. **Divisibility**: `mat_a.size(-1) % 16 == 0`, `mat_b.size(-2) % 16 == 0`, `mat_b.size(-1) % 16 == 0`
6. **Unsupported**: `bias` and `scale_result` must be nullopt
7. **Offsets**: Required iff either input is 2D; must be 1D int32
8. **Scales**: Both float32 (rowwise) or both float8_e8m0fnu (MXFP8)
9. **Output dtype**: BF16 only

### Scale Validation (rowwise float32)

`_check_scales_fp8_rowwise()` at GroupedBlas.cpp:301:
- **2D mat**: scale is 1D, contiguous, size = `mat.size(dim) * scale_multiplier`
  - `scale_multiplier` = `offs->size(0)` for 2D×2D case, else 1
- **3D mat**: scale is 2D, stride(1)==1, size(0)==mat.size(0), size(1)==mat.size(1+dim)

For scale_a: dim=0 (rows), for scale_b: dim=1 (columns).

## Input Modes

| Mode | mat_a | mat_b (transposed) | offs | output |
|------|-------|---------------------|------|--------|
| 2D×2D | (M, total_K) | (total_K, N) transposed | (G,) cum K offsets | (G, M, N) |
| 2D×3D | (total_M, K) | (G, K, N) transposed | (G,) cum M offsets | (total_M, N) |
| 3D×2D | (G, M, K) | (total_N, K) transposed | (G,) cum N offsets | (M, total_N) |
| 3D×3D | (G, M, K) | (G, K, N) transposed | none | (G, M, N) |

**Note**: "transposed" means B is stored column-major (stride[-2]==1, stride[-1]>=size[-2]).

## CUTLASS Kernel Details

From `ScaledGroupMM.cu`:
- ElementA/B: `float_e4m3_t` (FP8)
- ElementAccumulator: `float`
- ElementOutput: `bfloat16_t`
- DtypeScale: `float`
- Layout A: RowMajor, Layout B: ColumnMajor
- Alignment: 16 bytes
- Tile shapes dispatched based on problem size
- Per-row scale_a broadcast via `Sm90ColBroadcast`, per-col scale_b via `Sm90RowBroadcast`
- Max 1024 groups

## Key Helpers to Reuse

From `GroupedMMUtils.h`:
- `check_valid_strides_and_return_transposed()` — stride validation
- `create_grouped_gemm_output_tensor()` — output allocation with TMA-aligned strides

## XPU Porting Strategy

sycl-tla lacks a combined grouped+FP8+rowwise-scaling dispatch policy. Approach:

1. **Dequantize**: Cast FP8→BF16, apply rowwise scales per group
2. **GEMM**: Reuse existing BF16 grouped GEMM kernel from `_grouped_mm` port
3. **Scope**: Rowwise FP8 only (float32 scales), skip MXFP8
