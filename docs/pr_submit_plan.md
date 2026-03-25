# PR Submit Plan: `_scaled_grouped_mm` XPU Support

## Overview

Upstream `_scaled_grouped_mm` (FP8×FP8→BF16 grouped GEMM with rowwise float32 scaling) to Intel XPU via two PRs, following the pattern established by the `_grouped_mm` port (torch-xpu-ops PR #3122 + PyTorch PR #178242).

**Kernel approach**: Dequantize FP8→BF16 with rowwise scale application, then dispatch to the existing sycl-tla BF16 grouped GEMM kernel. Supports all 4 input modes: 3D×3D, 2D×3D, 3D×2D, 2D×2D.

## PR 1: torch-xpu-ops (submit FIRST)

**Repository**: `intel/torch-xpu-ops`
**Branch**: `xpu-scaled-grouped-mm` (based on `xpu-grouped-mm`)
**Commit**: `2d6ca38` — "Add scaled_grouped_mm kernel for XPU (FP8 x FP8 -> BF16)"
**Push URL**: https://github.com/xuhancn/torch-xpu-ops/pull/new/xpu-scaled-grouped-mm

### Files

| File | Action | Description |
|------|--------|-------------|
| `src/ATen/native/xpu/ScaledGroupedMM.h` | NEW | Wrapper header (`at::native::xpu` namespace): `is_scaled_grouped_mm_available()` + `f8f8bf16_scaled_grouped_mm()` |
| `src/ATen/native/xpu/ScaledGroupedMM.cpp` | NEW | `#ifdef USE_SYCLTLA` guarded wrapper forwarding to `at::xpu::detail::` |
| `src/ATen/native/xpu/sycltla/ScaledGroupedMM.h` | NEW | Kernel header (`at::xpu::detail` namespace) |
| `src/ATen/native/xpu/sycltla/ScaledGroupedMM.cpp` | NEW | sycl-tla kernel: dequantize FP8→BF16 + grouped GEMM (531 lines) |
| `test/xpu/test_scaled_grouped_mm_xpu.py` | NEW | 5 accuracy tests across all 4 input modes |

### No CMake changes needed

The existing glob `file(GLOB xpu_sycltla ... "native/xpu/sycltla/*.cpp")` and `install_xpu_headers("native/xpu/sycltla")` already pick up new files.

### Commit Summary

**`2d6ca38` — Add scaled_grouped_mm kernel for XPU (FP8 x FP8 -> BF16)**

Add `_scaled_grouped_mm` support for Intel XPU using sycl-tla. The kernel
dequantizes FP8 inputs to BF16 with rowwise float32 scale application,
then dispatches to the existing BF16 grouped GEMM sycl-tla kernel.

Supports all 4 input modes: 3D×3D (batched), 2D×3D (ragged A / MoE),
3D×2D (ragged B), and 2D×2D (ragged K).

Files:
- `ScaledGroupedMM.{h,cpp}`: USE_SYCLTLA-guarded wrapper
- `sycltla/ScaledGroupedMM.{h,cpp}`: sycl-tla kernel implementation
- `test_scaled_grouped_mm_xpu.py`: accuracy tests for all 4 modes

```
 src/ATen/native/xpu/ScaledGroupedMM.cpp         |  44 ++
 src/ATen/native/xpu/ScaledGroupedMM.h            |  27 ++
 src/ATen/native/xpu/sycltla/ScaledGroupedMM.cpp  | 531 +++++++++++++++++
 src/ATen/native/xpu/sycltla/ScaledGroupedMM.h    |  25 ++
 test/xpu/test_scaled_grouped_mm_xpu.py           | 220 +++++++
 5 files changed, 847 insertions(+)
```

## PR 2: PyTorch (submit SECOND)

**Repository**: `pytorch/pytorch`
**Branch**: `xpu-scaled-grouped-mm` (based on `xpu-grouped-mm`)
**Commits**:
- `073a25b` — "Add XPU dispatch for _scaled_grouped_mm"
- `b27627f` — "Update AOTInductor C shim for _scaled_grouped_mm XPU dispatch"

**Push URL**: https://github.com/xuhancn/pytorch/pull/new/xpu-scaled-grouped-mm

### Files

| File | Action | Description |
|------|--------|-------------|
| `aten/src/ATen/native/native_functions.yaml` | EDIT | Add `XPU: _scaled_grouped_mm_xpu` dispatch key |
| `aten/src/ATen/native/mkldnn/xpu/ScaledGroupedBlas.cpp` | NEW | XPU dispatch function with full validation (158 lines) |
| `torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h` | EDIT | Auto-generated AOTInductor C shim entry |
| `third_party/xpu.txt` | EDIT | Update to torch-xpu-ops commit `2d6ca382598655ccb32775947eae816973326c4c` |

### Build Note

After editing `native_functions.yaml` with a new XPU dispatch key, you **must** regenerate the AOTInductor C shim headers before building:

```bash
PYTHONPATH=/path/to/pytorch python torchgen/gen.py --update-aoti-c-shim
```

This adds the `aoti_torch_xpu__scaled_grouped_mm` entry to `c_shim_xpu.h`. Without this step, the build fails with an "AOTInductor C shim header files have unexpectedly changed" error.

### Commit Summary

**`073a25b` — Add XPU dispatch for _scaled_grouped_mm**

Register XPU dispatch key for `_scaled_grouped_mm` in `native_functions.yaml`
and add the dispatch function in `ScaledGroupedBlas.cpp`. The kernel
validates inputs (FP8 dtypes, transposition, divisibility, rowwise float32
scales), creates the output tensor, and forwards to the sycl-tla kernel
via the USE_SYCLTLA-guarded wrapper in torch-xpu-ops.

Update `third_party/xpu.txt` to include the scaled_grouped_mm kernel commit.

```
 aten/src/ATen/native/mkldnn/xpu/ScaledGroupedBlas.cpp | 158 +++++++++++++++++
 aten/src/ATen/native/native_functions.yaml            |   1 +
 third_party/xpu.txt                                   |   2 +-
 3 files changed, 160 insertions(+), 1 deletion(-)
```

**`b27627f` — Update AOTInductor C shim for _scaled_grouped_mm XPU dispatch**

Regenerated via: `python torchgen/gen.py --update-aoti-c-shim`

```
 torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h | 1 +
 1 file changed, 1 insertion(+)
```

## Validation

### After PR 1 (torch-xpu-ops)

```bash
# Build PyTorch with USE_SYCLTLA=ON and updated torch-xpu-ops
cd /tmp && python /path/to/torch-xpu-ops/test/xpu/test_scaled_grouped_mm_xpu.py -v
```

### After PR 2 (PyTorch)

```bash
# Full PyTorch build with updated xpu.txt
cd /tmp && python /path/to/torch-xpu-ops/test/xpu/test_scaled_grouped_mm_xpu.py -v
```

## Scope

| Item | Decision | Reason |
|------|----------|--------|
| MXFP8 (float8_e8m0fnu scales) | Skip | CUDA-only (USE_MSLK); not needed for Intel BMG |
| `_scaled_grouped_mm_v2` | Skip | Recipe-based API; port v1 first |
| Fused FP8+scale kernel | Skip (future) | sycl-tla lacks combined grouped+FP8+scaling policy |
| Fallback when no SYCLTLA | `TORCH_CHECK(false)` | Matches CUDA pattern (no fallback for scaled) |
| Output dtype | BF16 only | Matches CUDA constraint |

## Key Design Decisions

1. **`_check_scales_fp8_rowwise` copied into XPU dispatch**: This validation function lives in CUDA-only `cuda/GroupedBlas.cpp`, not in the shared `GroupedMMUtils.h`. We copied it into `mkldnn/xpu/ScaledGroupedBlas.cpp` for minimal diff.

2. **CUTLASS kernel types duplicated**: The sycl-tla kernel type definitions (TileShape, TiledMma, etc.) are identical to `GroupedMM.cpp`. Since both `.cpp` files are compiled separately, the types are duplicated in `ScaledGroupedMM.cpp`.

3. **Wider tolerances**: `atol=0.2, rtol=0.05` vs CUDA's `atol=5e-2, rtol=5e-4`. The dequantize-then-GEMM approach truncates to BF16 before the matrix multiply, losing precision compared to a fused kernel.

## Reference: Previous Port

| Kernel | Dev repo | torch-xpu-ops PR | PyTorch PR |
|--------|----------|-------------------|------------|
| `_grouped_mm` | `dev_pytorch_group_mm` | [#3122](https://github.com/intel/torch-xpu-ops/pull/3122) | [#178242](https://github.com/pytorch/pytorch/pull/178242) |
| `_scaled_grouped_mm` | `dev_pytorch_scaled_grouped_mm` | TBD | TBD |
