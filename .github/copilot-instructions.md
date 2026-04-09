# Copilot Instructions

## Build and test commands

Set up the expected environment before building or testing:

```bash
source ~/intel/oneapi/setvars.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch
```

Build the local SYCL extension from `sycl_kernel/`:

```bash
cd sycl_kernel && python setup.py develop
```

Run the local accuracy tests from `/tmp` to avoid `torch/_C` import conflicts:

```bash
cd /tmp && python /home/xu/conda_root/xu_pytorch/dev_pytorch_scaled_grouped_mm/test/test_scaled_grouped_mm.py
```

Run a single test method:

```bash
cd /tmp && python /home/xu/conda_root/xu_pytorch/dev_pytorch_scaled_grouped_mm/test/test_scaled_grouped_mm.py TestScaledGroupedMM.test_3d_3d
```

Run the local benchmark:

```bash
cd /tmp && python /home/xu/conda_root/xu_pytorch/dev_pytorch_scaled_grouped_mm/test/bench_scaled_grouped_mm.py
```

For CUDA-reference behavior, use the upstream PyTorch test from `/tmp`:

```bash
cd /tmp && python /home/xu/xu_github/pytorch/test/test_scaled_matmul_cuda.py -k '_scaled_grouped_mm'
```

## High-level architecture

This repository is a local staging area for porting PyTorch CUDA `_scaled_grouped_mm` to Intel XPU.

- `sycl_kernel/` contains the standalone PyTorch SYCL extension used for local kernel development. `scaled_grouped_mm_ops.sycl` validates the public operator inputs, dequantizes FP8 inputs to BF16 with rowwise scaling, and then dispatches to the grouped GEMM kernel defined in `grouped_mm_kernel.hpp`.
- `test/test_scaled_grouped_mm.py` is the main correctness suite. It covers all four input modes (`3Dx3D`, `2Dx3D`, `3Dx2D`, `2Dx2D`) and checks the extension against a per-group reference implemented as manual dequantization plus matmul.
- `test/bench_scaled_grouped_mm.py` benchmarks the local extension with warmup, timed iterations, and explicit `torch.xpu.synchronize()`.
- `docs/cuda_analysis.md` explains the CUDA operator schema, dispatch chain, validation rules, and the four input layouts that the SYCL port mirrors.
- `docs/pr_submit_plan.md` documents the eventual upstream path: first add the kernel to `torch-xpu-ops`, then wire the XPU dispatch into the main PyTorch tree.

## Key conventions

- Treat `mat_a` as non-transposed row-major input and `mat_b` as transposed input. The local extension validates `mat_a.stride(-1) == 1` and `mat_b.stride(-2) == 1`, matching the CUDA contract.
- The current local implementation is not a fused FP8 grouped GEMM kernel. It dequantizes FP8 inputs to BF16, applies rowwise float32 scales, materializes `mat_b` into logical `(K, N)` layout with `contiguous()`, and then reuses the grouped GEMM kernel.
- The project works around sycl-tla limitations rather than reimplementing the full CUDA fast path. `docs/cuda_analysis.md` and `test/test_report.md` both reflect this dequantize-then-GEMM design and its wider tolerances.
- Always run local tests from `/tmp`. Running from inside the PyTorch source tree can break imports because of `torch/_C` resolution.
- The important external context lives outside this repo: PyTorch source is expected at `/home/xu/xu_github/pytorch`, and `torch-xpu-ops` at `/home/xu/xu_github/torch-xpu-ops`.
- When upstreaming a sycl-tla kernel, PyTorch XPU dispatch code must call a `USE_SYCLTLA`-guarded wrapper in `torch-xpu-ops`, not sycltla symbols directly. `CLAUDE.md` documents this as a hard CI requirement.
- If you adapt upstream XPU device-type tests, `instantiate_device_type_tests(..., allow_xpu=True)` is required or the suite silently generates zero XPU tests.
