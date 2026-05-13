# PR Submit Plan: `_grouped_mm` + `_scaled_grouped_mm` XPU Support

## Split Rationale

The original combined PR (#3172, +1927 lines, 11 files) bundled both `_grouped_mm` and `_scaled_grouped_mm` into a single PR. We split it into two sequential PRs for better reviewability:

- **Size reduction**: 1928 lines → 724 + 1203 lines per PR
- **Focused review**: Each PR covers one operator with its own tests, reducing cognitive load
- **Independent landing**: `_grouped_mm` (BF16 GEMM) is simpler and can land first; `_scaled_grouped_mm` (FP8 dequant + scaled GEMM) depends on it and lands second
- **Review history**: 46 automated review comments across 6 rounds on the combined PR, mixing concerns from both operators
- **Existing PR pair**: PR #3122 / #178242 already exist for standalone `_grouped_mm`

### Dependency Chain

```
PR #3122 (torch-xpu-ops: _grouped_mm kernel)
  └→ PR #178242 (pytorch: _grouped_mm dispatch)
       └→ PR #3172 (torch-xpu-ops: _scaled_grouped_mm kernel)
            └→ PR #178354 (pytorch: _scaled_grouped_mm dispatch)
```

## PR Inventory

| # | Repo | Operator | Branch | Status | Lines |
|---|------|----------|--------|--------|-------|
| #3122 | torch-xpu-ops | `_grouped_mm` | `xpu-grouped-mm` | Open, needs update | ~724 |
| #178242 | pytorch | `_grouped_mm` | `xpu-grouped-mm` | Open, draft, needs update | ~40 |
| #3172 | torch-xpu-ops | `_scaled_grouped_mm` | `xpu-scaled-grouped-mm` | Open, needs trimming | ~1203 |
| #178354 | pytorch | `_scaled_grouped_mm` | `xpu-scaled-grouped-mm` | Open, draft, needs trimming | ~165 |

---

## Phase 1: `_grouped_mm` — BF16 Grouped GEMM

### PR #3122 — torch-xpu-ops (submit FIRST)

**Repository**: `intel/torch-xpu-ops`
**Branch**: `xpu-grouped-mm` at `xuhancn/torch-xpu-ops`
**PR**: https://github.com/intel/torch-xpu-ops/pull/3122

#### Files

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `src/ATen/native/xpu/sycltla/GroupedMM.cpp` | NEW | 462 | sycl-tla BF16 grouped GEMM kernel |
| `src/ATen/native/xpu/sycltla/GroupedMM.h` | NEW | 24 | kernel header (`at::xpu::detail` namespace) |
| `src/ATen/native/xpu/GroupedMM.cpp` | NEW | 42 | `#ifdef USE_SYCLTLA` guarded wrapper |
| `src/ATen/native/xpu/GroupedMM.h` | NEW | 26 | wrapper header: `is_grouped_mm_available()` + `bf16bf16_grouped_mm()` |
| `src/ATen/CMakeLists.txt` | EDIT | 2 | add `native/xpu/sycltla/*.cpp` glob + header install |
| `test/xpu/test_grouped_mm_xpu.py` | NEW | 168 | unit tests for all 4 input modes |

#### Checklist

- [ ] Sync #3122 branch with GroupedMM files from #3172 HEAD (`ee36cf93`)
- [ ] Verify all GroupedMM files are included and up-to-date
- [ ] Push to `xuhancn/torch-xpu-ops` branch `xpu-grouped-mm` (new commits, no force push)
- [ ] Update PR #3122 description if needed
- [ ] Wait for CI to pass
- [ ] Request human review
- [ ] Merge #3122

### PR #178242 — PyTorch (submit after #3122 merges)

**Repository**: `pytorch/pytorch`
**Branch**: `xpu-grouped-mm` at `xuhancn/pytorch`
**PR**: https://github.com/pytorch/pytorch/pull/178242

#### Files

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `aten/src/ATen/native/native_functions.yaml` | EDIT | 1 | Add `XPU: _grouped_mm_xpu` dispatch key |
| `aten/src/ATen/native/xpu/GroupedBlas.cpp` | NEW | 32 | XPU dispatch function for `_grouped_mm` |
| `aten/src/ATen/CMakeLists.txt` | EDIT | 4 | Add `native/xpu/*.cpp` and `native/xpu/*.h` glob patterns |
| `third_party/xpu.txt` | EDIT | 1 | Update to #3122 merge commit hash |

#### Checklist

- [ ] Update #178242 branch with GroupedBlas.cpp from #178354 HEAD
- [ ] Update `xpu.txt` to point to merged #3122 commit on `intel/torch-xpu-ops`
- [ ] Update `caffe2/CMakeLists.txt` repo URL back to `intel/torch-xpu-ops` (if needed)
- [ ] Push to `xuhancn/pytorch` branch `xpu-grouped-mm` (new commits, no force push)
- [ ] Un-draft PR #178242
- [ ] Wait for CI to pass
- [ ] Request human review
- [ ] Merge #178242

---

## Phase 2: `_scaled_grouped_mm` — FP8 Dequant + Scaled Grouped GEMM

### PR #3172 — torch-xpu-ops (submit after #3122 merges)

**Repository**: `intel/torch-xpu-ops`
**Branch**: `xpu-scaled-grouped-mm` at `xuhancn/torch-xpu-ops`
**PR**: https://github.com/intel/torch-xpu-ops/pull/3172

#### Files (after trimming GroupedMM)

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `src/ATen/native/xpu/sycltla/ScaledGroupedMM.cpp` | NEW | 551 | FP8 dequant + rowwise scale + grouped GEMM |
| `src/ATen/native/xpu/sycltla/ScaledGroupedMM.h` | NEW | 25 | kernel header (`at::xpu::detail` namespace) |
| `src/ATen/native/xpu/ScaledGroupedMM.cpp` | NEW | 44 | `#ifdef USE_SYCLTLA` guarded wrapper |
| `src/ATen/native/xpu/ScaledGroupedMM.h` | NEW | 27 | wrapper header: `is_scaled_grouped_mm_available()` + `f8f8bf16_scaled_grouped_mm()` |
| `test/xpu/test_scaled_grouped_mm_xpu.py` | NEW | 556 | unit tests for all 4 modes + FP8 + negative tests |

#### Checklist

- [ ] Rebase #3172 branch on new `main` (after #3122 merged)
- [ ] Remove GroupedMM files from #3172 (they landed via #3122)
- [ ] Remove CMakeLists.txt changes from #3172 (they landed via #3122)
- [ ] Verify ScaledGroupedMM kernel still compiles and links against merged GroupedMM
- [ ] Push to `xuhancn/torch-xpu-ops` branch `xpu-scaled-grouped-mm` (new commits, no force push)
- [ ] Update PR #3172 description to reference #3122 as dependency
- [ ] Wait for CI to pass
- [ ] Request human review
- [ ] Merge #3172

### PR #178354 — PyTorch (submit after #3172 merges)

**Repository**: `pytorch/pytorch`
**Branch**: `xpu-scaled-grouped-mm` at `xuhancn/pytorch`
**PR**: https://github.com/pytorch/pytorch/pull/178354

#### Files (after trimming GroupedBlas)

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `aten/src/ATen/native/native_functions.yaml` | EDIT | 1 | Add `XPU: _scaled_grouped_mm_xpu` dispatch key |
| `aten/src/ATen/native/xpu/ScaledGroupedBlas.cpp` | NEW | 158 | XPU dispatch function with FP8 validation |
| `torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h` | EDIT | 3 | AOTInductor C shim (with `TORCH_FEATURE_VERSION` guard) |
| `third_party/xpu.txt` | EDIT | 1 | Update to #3172 merge commit hash |

#### Checklist

- [ ] Rebase #178354 branch on new `main` (after #178242 merged)
- [ ] Remove GroupedBlas.cpp from #178354 (landed via #178242)
- [ ] Remove GroupedBlas CMake changes from #178354 (landed via #178242)
- [ ] Update `xpu.txt` to point to merged #3172 commit on `intel/torch-xpu-ops`
- [ ] Update `caffe2/CMakeLists.txt` repo URL back to `intel/torch-xpu-ops` (if needed)
- [ ] Push to `xuhancn/pytorch` branch `xpu-scaled-grouped-mm` (new commits, no force push)
- [ ] Un-draft PR #178354
- [ ] Wait for CI to pass
- [ ] Request human review
- [ ] Merge #178354

---

## Model-Level Validation (Llama 4 Scout)

Validated `_scaled_grouped_mm` with real tensor shapes from Llama 4 Scout (109B/17B-16E) MoE architecture on Intel Arc B580. Full model too large for device (~12GB VRAM); tested with scaled-down sequence lengths (M=512–2048) while keeping real model dimensions for K, N, G.

### Model Architecture

| Parameter | Value |
|---|---|
| Model | Llama 4 Scout (109B/17B-16E) |
| hidden_size (K) | 5120 |
| intermediate_size (N) | 8192 |
| num_local_experts (E) | 16 |
| MoE layers | All 48 decoder layers |

### Test Methodology

Cross-device comparison (CPU float32 reference vs XPU SYCL kernel):
1. Create FP8 tensors and float32 scales on CPU
2. Compute reference on CPU via float32 dequantize + matmul + cast to BF16
3. Copy tensors to XPU, run the SYCL kernel
4. Compare XPU output against CPU reference

### Results Summary

**18 of 18 tests PASSED** (9 extension + 9 dispatch), covering:

| Test Shape | M | G | K | N | Max Error | Mean Error | Verdict |
|---|---|---|---|---|---|---|---|
| Gate/Up G=1 | 512 | 1 | 5120 | 8192 | 2.0 | 0.050 | 1 ULP ✅ |
| Gate/Up G=2 | 1024 | 2 | 5120 | 8192 | 2.0 | 0.047 | 1 ULP ✅ |
| Gate/Up G=4 | 2048 | 4 | 5120 | 8192 | 2.0 | 0.048 | 1 ULP ✅ |
| Gate/Up G=16 | 512 | 16 | 5120 | 8192 | 2.0 | 0.047 | 1 ULP ✅ |
| Down G=1 | 512 | 1 | 8192 | 5120 | 2.0 | 0.063 | 1 ULP ✅ |
| Down G=2 | 1024 | 2 | 8192 | 5120 | 2.0 | 0.060 | 1 ULP ✅ |
| Down G=4 | 2048 | 4 | 8192 | 5120 | 2.0 | 0.059 | 1 ULP ✅ |
| Down G=16 | 512 | 16 | 8192 | 5120 | 2.0 | 0.058 | 1 ULP ✅ |
| Unbalanced routing | 1024 | 4 | 5120 | 8192 | — | — | ✅ |

### Error Analysis

- **Max error = exactly 1 BF16 ULP** at every worst-case element
- **~50% of elements match exactly** between CPU and XPU
- **99.9% of elements** have error ≤ 0.5
- Error scales with √K (central limit theorem for BF16 accumulation)
- Worst case: `XPU=-258.0, CPU_ref=-260.0, error=2.0` → `|260| in [256,512)`, so `1 ULP = 2.0`

### Conclusion

All errors are within 1 BF16 ULP — the theoretical minimum for BF16 vs float32 accumulation. This is identical behavior to CUDA FP8 grouped GEMM kernels.

---

## Scope Decisions

| Item | Decision | Reason |
|------|----------|--------|
| MXFP8 (float8_e8m0fnu scales) | Skip | CUDA-only (USE_MSLK); not needed for Intel BMG |
| `_scaled_grouped_mm_v2` | Skip | Recipe-based API; port v1 first |
| Fused FP8+scale kernel | Skip (future) | sycl-tla lacks combined grouped+FP8+scaling policy |
| Fallback when no SYCLTLA | `TORCH_CHECK(false)` | Matches CUDA pattern (no fallback for scaled) |
| Output dtype | BF16 only | Matches CUDA constraint |

## Key Design Decisions

1. **`_check_scales_fp8_rowwise` copied into XPU dispatch**: This validation function lives in CUDA-only `cuda/GroupedBlas.cpp`, not in the shared `GroupedMMUtils.h`. We copied it into `xpu/ScaledGroupedBlas.cpp` for minimal diff.

2. **CUTLASS kernel types duplicated**: The sycl-tla kernel type definitions (TileShape, TiledMma, etc.) are identical to `GroupedMM.cpp`. Since both `.cpp` files are compiled separately, the types are duplicated in `ScaledGroupedMM.cpp`.

3. **Wider tolerances**: `atol=0.2, rtol=0.05` vs CUDA's `atol=5e-2, rtol=5e-4`. The dequantize-then-GEMM approach truncates to BF16 before the matrix multiply, losing precision compared to a fused kernel.

## Reference

| Kernel | Dev repo | torch-xpu-ops PR | PyTorch PR |
|--------|----------|-------------------|------------|
| `_grouped_mm` | `dev_pytorch_group_mm` | [#3122](https://github.com/intel/torch-xpu-ops/pull/3122) | [#178242](https://github.com/pytorch/pytorch/pull/178242) |
| `_scaled_grouped_mm` | `dev_pytorch_scaled_grouped_mm` | [#3172](https://github.com/intel/torch-xpu-ops/pull/3172) | [#178354](https://github.com/pytorch/pytorch/pull/178354) |
