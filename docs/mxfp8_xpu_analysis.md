# MXFP8 Support for Intel XPU — Feasibility Analysis

## Background

### What is MXFP8?

MXFP8 (Microscaling FP8) uses FP8 data tensors (`float8_e4m3fn` or `float8_e5m2`) paired with
**block-shared scale factors** stored in `float8_e8m0fnu` (8-bit exponent-only, unsigned). Each
block of 32 consecutive FP8 elements shares one `e8m0` scale. The effective value is:

```
value_i = fp8_data_i × 2^(e8m0_exponent - 127)
```

This differs from **rowwise FP8** (the current XPU implementation), where each row of the matrix
has a single `float32` scale applied uniformly across all its elements.

### How CUDA handles MXFP8 in `_scaled_grouped_mm`

From `docs/cuda_analysis.md` and `aten/src/ATen/native/cuda/GroupedBlas.cpp`:

```
torch._scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, ...)
  └─ scale_a.dtype == float8_e8m0fnu  →  MXFP8 path (requires USE_MSLK build flag)
       → _mx8_mx8_bf16_grouped_mm_mslk()
```

**CUDA MXFP8 constraints:**
- Only supported with `USE_MSLK` build flag (MSLK = Microscaling Library for Kepler/Hopper/Blackwell)
- Scale layout must be `SWIZZLE_32_4_4` (a specific tiled memory layout for hardware consumption)
- Only `2D×2D` and `2D×3D` input modes supported (no batched A)
- `offs` tensor is required (group offsets into 2D matrices)
- Uses NVIDIA Blackwell SM100 native `mxf8f6f4.block_scale` tensor core instruction

---

## sycl-tla Dispatch Policy Survey

The following Xe dispatch policies exist in `third_party/sycl-tla/include/cutlass/gemm/dispatch_policy.hpp`:

| Policy | Grouped (ptr-array) | FP8 inputs | Scale factors |
|--------|--------------------|-----------:|---------------|
| `MainloopIntelXeXMX16` | ❌ | ❌ | ❌ |
| `MainloopXeL1StagedGroup` | ✅ | via pre-convert | ❌ |
| `MainloopIntelXeXMX16Group` | ✅ | via pre-convert | ❌ |
| `MainloopIntelXeXMX16GroupFP8` | ✅ | ✅ (convert in mainloop) | ❌ |
| `MainloopIntelXeXMX16FP8Scaling` | ❌ (dense only) | ✅ | ✅ float16 per group |
| `MainloopIntelXeXMX16MixedPrecision` | ❌ | N/A | N/A |
| `MainloopIntelXeXMX16GroupMixedPrecision` | ✅ | N/A | N/A |

**Gap: There is no policy that combines grouped GEMM + FP8 inputs + scale factors.**

The closest existing policy for grouped FP8 is `MainloopIntelXeXMX16GroupFP8`, which inherits from
`MainloopIntelW8A8` — it converts FP8 to BF16 in the mainloop but applies **no scaling at all**.

The closest policy for FP8 with scaling is `MainloopIntelXeXMX16FP8Scaling` (used in
`examples/08_bmg_gemm_f8/08_bmg_gemm_f8_scaling.cpp`), but it only supports a single dense GEMM
(not pointer-array grouped). Its scale type in the example is `half_t` (FP16 rowwise scales), not
`float_ue8m0_t`.

---

## `float_ue8m0_t` Support in sycl-tla

### Type availability

`float_ue8m0_t` is defined in sycl-tla and re-exported via
`cute/numeric/numeric_types.hpp` (line 106):
```cpp
using cutlass::float_ue8m0_t;
```

### Conversion support

`cutlass/numeric_conversion.h` provides two conversion paths for `float_ue8m0_t ↔ float`:

```cpp
// Fast path — NVIDIA CUDA PTX only (NOT available on Intel SYCL targets):
#if defined(CUDA_PTX_UE8M0_CVT_ENABLED)
  "cvt.rn.bf16x2.ue8m0x2 %0, lo;\n"   // ue8m0 → bf16
  "cvt.rp.satfinite.ue8m0x2.f32 ..."  // float → ue8m0

// Software fallback (C++, works everywhere including Intel SYCL):
float val = ldexpf(1.0f, static_cast<int>(bits) - 127);
```

**On Intel Xe targets, `CUDA_PTX_UE8M0_CVT_ENABLED` is never defined.** The software fallback
(`ldexpf`) compiles and runs correctly under SYCL/DPC++.

### Implication

`static_cast<bfloat16_t>(float_ue8m0_t_value)` works on Intel Xe via the software path.
The `transform_A` function inside `xe_mma_fp8_scaling.hpp` applies scales via:
```cpp
auto scale = static_cast<DstType>(shfl_sync(0xFFFFFFFF, tCrS_input(m), i));
out(_, _, k)[m * 16 + i] *= scale;
```
This would work if `ElementScaleA = float_ue8m0_t` and `DstType = bfloat16_t`, but only for the
**dense** GEMM path (no grouped/pointer-array variant exists).

---

## Intel Xe Hardware Capabilities

### FP8 tensor core support

Intel Battlemage (BMG / Xe2 architecture) supports XMX (Xe Matrix eXtensions) instructions that
accept FP8 (`float_e4m3_t`, `float_e5m2_t`) inputs and accumulate in BF16/FP32. These are exposed
through sycl-tla as the `XE_DPAS_TT<8, float, bfloat16_t, bfloat16_t, float>` MMA atom.

### No hardware MXFP8 blockscaled instruction

Intel Xe (BMG/Xe2) has **no native hardware MXFP8 blockscaled XMX instruction.** The
`mxf8f6f4.block_scale` tensor core instruction is specific to NVIDIA Blackwell SM100 (introduced
with the GB200 architecture). On Intel Xe, all e8m0 scale decoding must be done in software
(register-level computation before the XMX instruction fires).

This means even a fused "grouped FP8 scaling" kernel on Intel Xe would still be doing:
1. Load FP8 data tiles
2. Load e8m0 scale values, decode to float via software (`ldexpf`)
3. Multiply FP8-converted BF16 values by decoded float scales
4. Issue XMX DPAS with scaled BF16 inputs

There is no hardware shortcut that eliminates steps 2–3.

---

## Three Implementation Options

### Option A — Software dequant pre-pass (feasible now, ~1–2 days)

**Approach:** Add a standalone SYCL dequantization kernel that decodes MXFP8 to BF16 before
dispatching to the existing `MainloopXeL1StagedGroup` grouped GEMM.

**Steps:**
1. Extend `validate_inputs()` in `scaled_grouped_mm_ops.sycl` to accept `float8_e8m0fnu` scale
   dtype (alongside the existing `float32` check)
2. Restrict MXFP8 mode to 2D×2D and 2D×3D inputs only (matching CUDA limitations)
3. Require `offs` tensor for MXFP8 (group boundary offsets into 2D matrices)
4. Add a new `dequantize_mxfp8()` SYCL kernel:
   - Load FP8 data in blocks of 32
   - Load one `e8m0` scale per block, decode via `ldexpf(1.0f, (int)e8m0 - 127)`
   - Multiply and write BF16 output
5. Feed the BF16 tensors into the existing grouped GEMM dispatch path

**Pros:** No sycl-tla changes; reuses the proven BF16 grouped GEMM kernel.

**Cons:** Two passes over the FP8 data (dequant then GEMM); higher peak memory (BF16 intermediates
are 2× larger than FP8 inputs); no memory-bandwidth benefit from FP8 format.

**Scale layout note:** The CUDA path uses `SWIZZLE_32_4_4` layout. For XPU, callers can pass
canonical (non-swizzled) row-major layout since we are not using hardware MXFP8 tensor cores that
require the swizzled layout. The dequant kernel should document the expected layout.

---

### Option B — New `MainloopXeL1StagedGroupFP8Scaling` in sycl-tla (~2–4 weeks)

**Approach:** Extend sycl-tla with a new grouped mainloop dispatch policy that fuses FP8
loading + e8m0 scale decoding + BF16 conversion in a single kernel pass.

**Steps:**
1. Add `struct MainloopXeL1StagedGroupFP8Scaling` dispatch policy in `dispatch_policy.hpp`
   (inherits from `MainloopIntelXeXMX16<Stages_, KernelXePtrArrayCooperative>`)
2. Add a `CollectiveMma` specialization in `collective_mma.hpp` that:
   - Takes pointer arrays for A, B, ScaleA, ScaleB (one pointer per group)
   - Loads data using the existing `MainloopXeL1StagedGroup` tile copy logic
   - Applies scale via the `transform_A/B` pattern from `xe_mma_fp8_scaling.hpp`,
     using `float_ue8m0_t → bfloat16_t` software conversion
3. Set `group_size=32` for MXFP8 alignment
4. Validate on Intel BMG via `examples/09_bmg_grouped_gemm_f8/` as a reference test

**Subgroup shuffle consideration:** `transform_A` in `xe_mma_fp8_scaling.hpp` uses `shfl_sync`
to broadcast scale values across subgroup lanes. This maps to SYCL subgroup shuffle (`sycl::ext::oneapi::experimental::cuda::masked_shuffles`). This needs testing on Intel Xe, since the
existing dense FP8Scaling example (08) exercises this path on BMG.

**Pros:** Single-pass fused kernel; lower memory bandwidth; FP8 tensors never materialize as BF16
in global memory.

**Cons:** Non-trivial sycl-tla kernel engineering; requires upstream contribution to sycl-tla;
scale broadcast via subgroup shuffle must be validated on Xe. e8m0 decode is still software —
no true hardware speedup over Option A on current Intel Xe hardware.

---

### Option C — True hardware MXFP8 (not feasible today)

**Approach:** Use native hardware blockscaled FP8 XMX instructions on Intel Xe.

**Status:** **Not possible.** Intel Battlemage (BMG/Xe2) has no `mxf8f6f4.block_scale`-equivalent
tensor core instruction. This would require new Intel GPU silicon, new ISA extensions, and
corresponding sycl-tla MMA atom definitions — none of which exist at the time of this analysis.

---

## Recommendation

| Option | Effort | Correctness risk | Perf vs Option B |
|--------|--------|-----------------|-----------------|
| A — Software dequant pre-pass | Low (~1–2 days) | Low | ~50% slower (2 passes) |
| B — New sycl-tla grouped FP8 Scaling mainloop | Medium (~2–4 weeks) | Medium | Baseline |
| C — Hardware MXFP8 | Not feasible | N/A | N/A |

**Short term:** Implement **Option A**. The dequantize-then-GEMM pattern is already established
in this codebase for the rowwise FP8 path. Adding MXFP8 dequantization follows the same structure
and can be validated against the CUDA reference immediately.

**Medium term:** File an issue or PR against `intel/sycl-tla` proposing **Option B** as a
`MainloopXeL1StagedGroupFP8Scaling` contribution. This benefits not only `_scaled_grouped_mm`
but any grouped FP8 GEMM workload (e.g., MoE layers) on Intel Xe.

---

## Key Files for Implementation

| File | Relevance |
|------|-----------|
| `sycl_kernel/scaled_grouped_mm_ops.sycl:99–103` | Scale dtype validation — add `float8_e8m0fnu` |
| `sycl_kernel/scaled_grouped_mm_ops.sycl:220–260` | Dequant dispatch — add MXFP8 branch |
| `sycl_kernel/grouped_mm_kernel.hpp` | Existing grouped GEMM entry point (unchanged for Option A) |
| `third_party/sycl-tla/include/cutlass/gemm/collective/xe_mma_fp8_scaling.hpp` | Reference for scale transform logic (Option B) |
| `third_party/sycl-tla/include/cutlass/gemm/dispatch_policy.hpp:1267–1290` | Where to add new policy (Option B) |
| `third_party/sycl-tla/include/cutlass/numeric_conversion.h:2335–2499` | `float_ue8m0_t` software conversion |
| `docs/cuda_analysis.md` | CUDA MXFP8 path, scale layout, input mode constraints |
