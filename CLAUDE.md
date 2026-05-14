# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repository ports PyTorch's CUDA `_scaled_grouped_mm` (and `_scaled_grouped_mm_v2`) kernel to Intel XPU using SYCL.

- **CUDA reference**: `aten/src/ATen/native/cuda/GroupedBlas.cpp`
- **Operator schema**: See `aten/src/ATen/native/native_functions.yaml`, search for `_scaled_grouped_mm` (line ~7344)
- **Validation helpers**: `aten/src/ATen/native/GroupedMMUtils.h` — shared with `_grouped_mm`
- **CUDA tests**: `test/test_scaled_matmul_cuda.py`

## Repository Layout

```
sycl_kernel/          <- local SYCL kernel dev (setup.py extension build)
test/                 <- local accuracy + performance tests
docs/                 <- CUDA analysis, porting plan, test reports
```

## Environment

| Item | Value |
|------|-------|
| PyTorch source | `/home/xu/conda_root/xu_pytorch/pytorch` |
| torch-xpu-ops source | `/home/xu/conda_root/xu_pytorch/pytorch/third_party/torch-xpu-ops` |
| Conda env | `xu_pytorch` |
| oneAPI setup | `source ~/intel/oneapi/setvars.sh` |
| GPU | Intel Arc B580 |
| AOT target | `TORCH_XPU_ARCH_LIST=bmg` |
| Compiler | Intel oneAPI DPC++/C++ 2025.x |

To activate the full build environment:
```bash
source ~/intel/oneapi/setvars.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch
```

## Common Commands

### Build the local SYCL extension
```bash
source ~/intel/oneapi/setvars.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch
cd sycl_kernel && python setup.py develop
```

### Run tests (always from /tmp to avoid torch/_C import conflicts)
```bash
cd /tmp && python /home/xu/conda_root/xu_pytorch/dev_pytorch_scaled_grouped_mm/test/test_scaled_grouped_mm.py
# Run a single test:
cd /tmp && python /home/xu/conda_root/xu_pytorch/dev_pytorch_scaled_grouped_mm/test/test_scaled_grouped_mm.py TestClassName.test_method_name
```

### Run upstream CUDA tests (for reference)
```bash
cd /tmp && python /home/xu/conda_root/xu_pytorch/pytorch/test/test_scaled_matmul_cuda.py -k '_scaled_grouped_mm'
```

## Workflow — Three Phases

### Phase 1: Analyze the CUDA kernel

Before writing any SYCL code, fully understand the CUDA implementation:

1. **Read `native_functions.yaml`** — find the operator schema, dispatch keys, and argument types
2. **Read the CUDA dispatch function** — understand fast-path vs fallback routing (e.g., `_grouped_mm_cuda` in `GroupedBlas.cpp`)
3. **Read validation/utility headers** — find reusable helpers (e.g., `GroupedMMUtils.h` has `_grouped_mm_validate_inputs()`, `create_grouped_gemm_output_tensor()`, `_grouped_mm_fallback()`)
4. **Read the CUDA kernel** — understand the CUTLASS/cuBLAS template assembly, tile shapes, data types
5. **Read CUDA unit tests** — understand test shapes, dtype coverage, gradient checking patterns
6. **Write analysis doc** to `docs/` summarizing the dispatch chain, kernel design, and what to reuse

### Phase 2: Local SYCL kernel development

Build and validate a standalone SYCL kernel before integrating into torch-xpu-ops:

1. **Port the kernel** to SYCL in `sycl_kernel/`
   - For CUTLASS-based kernels: use **sycl-tla** (CUTLASS port for Intel GPUs)
   - For simpler kernels: use plain SYCL with `sycl::nd_range` parallel_for
2. **Build with `setup.py`** using `torch.utils.cpp_extension`
   - For sycl-tla kernels, monkey-patch SPIR-V flags in setup.py (see Pitfalls section)
3. **Test accuracy** against CPU/PyTorch reference across all input modes
4. **Benchmark performance** — warmup + timed iterations with `torch.xpu.synchronize()`
5. **Write test report** to `test/test_report.md`

### Phase 3: Upstream via two PRs

Submit to PyTorch upstream via torch-xpu-ops (kernel) and pytorch (dispatch):

#### torch-xpu-ops PR (commit FIRST)

1. **Add kernel files**:
   - Regular SYCL: `src/ATen/native/xpu/sycl/YourKernel.cpp` + `.h`
   - sycl-tla: `src/ATen/native/xpu/sycltla/YourKernel.cpp` + `.h`
2. **Add USE_SYCLTLA-guarded wrapper** (CRITICAL for CI — see Pitfalls):
   - `src/ATen/native/xpu/YourKernel.h` — public header with `is_*_available()` + wrapper declaration
   - `src/ATen/native/xpu/YourKernel.cpp` — wrapper that `#ifdef USE_SYCLTLA` calls the real kernel, else `TORCH_CHECK(false, ...)`
   - This compiles into `torch_xpu_ops` (always built), NOT the sycltla shared library
   - Follow the Flash Attention pattern: `src/ATen/native/transformers/xpu/flash_attn/flash_api.cpp`
3. **Update CMake** in `src/ATen/CMakeLists.txt`:
   - Add source glob pattern (e.g., `"native/xpu/sycltla/*.cpp"`)
   - Add `install_xpu_headers("native/xpu/sycltla")` if new directory
4. **Add unit tests** to `test/xpu/test_yourkernel_xpu.py`
5. **Commit and note the commit hash**

#### PyTorch PR (commit SECOND)

1. **Add XPU dispatch key** in `aten/src/ATen/native/native_functions.yaml`:
   ```yaml
   XPU: _your_op_xpu
   ```
2. **Add dispatch function** in `aten/src/ATen/native/mkldnn/xpu/YourBlas.cpp`:
   - Include `<ATen/native/YourUtils.h>` for validation helpers
   - Include `<ATen/native/xpu/YourKernel.h>` for the **wrapper** header (NOT sycltla header)
   - Check `is_*_available()` before taking fast path
   - Route to fast-path wrapper for supported dtypes, fallback otherwise
3. **Update `third_party/xpu.txt`** with the torch-xpu-ops commit hash from step above
4. **Commit**

## Key Patterns and Conventions

### File locations

| What | Where |
|------|-------|
| Operator registry | `pytorch/aten/src/ATen/native/native_functions.yaml` |
| CUDA dispatch functions | `pytorch/aten/src/ATen/native/cuda/*.cpp` |
| XPU dispatch functions | `pytorch/aten/src/ATen/native/xpu/*.cpp` (new pattern) or `pytorch/aten/src/ATen/native/mkldnn/xpu/*.cpp` (legacy) |
| torch-xpu-ops regular SYCL kernels | `torch-xpu-ops/src/ATen/native/xpu/sycl/*.cpp` |
| torch-xpu-ops sycl-tla kernels | `torch-xpu-ops/src/ATen/native/xpu/sycltla/*.cpp` |
| torch-xpu-ops CMake | `torch-xpu-ops/src/ATen/CMakeLists.txt` |
| torch-xpu-ops build | `torch-xpu-ops/src/BuildOnLinux.cmake` |
| torch-xpu-ops tests | `torch-xpu-ops/test/xpu/*.py` |
| XPU commit pin | `pytorch/third_party/xpu.txt` |
| XPU fetch logic | `pytorch/caffe2/CMakeLists.txt:1163` |

### Existing XPU BLAS references (study these)

- `aten/src/ATen/native/mkldnn/xpu/Blas.cpp` — mm, bmm, addmm
- `aten/src/ATen/native/mkldnn/xpu/ScaledBlas.cpp` — scaled_mm

### sycl-tla build flags

When using sycl-tla (CUTLASS for Intel GPUs), the build system applies:
- `USE_SYCLTLA` CMake flag
- Compile definitions: `CUTLASS_ENABLE_SYCL`, `SYCL_INTEL_TARGET`
- SPIR-V extensions: `+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`
- sycl-tla version pinned in `torch-xpu-ops/cmake/SYCLTLA.cmake`

### Unit test pattern

```python
from torch.testing._internal.common_device_type import (
    dtypes, instantiate_device_type_tests, onlyXPU,
)
from torch.testing._internal.common_utils import run_tests, TestCase

class TestYourKernelXPU(TestCase):
    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_basic(self, device, dtype):
        # ...

instantiate_device_type_tests(
    TestYourKernelXPU, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
```

## Pitfalls and Lessons Learned

### Tests

- **`allow_xpu=True` is required**: `instantiate_device_type_tests(..., only_for="xpu")` silently generates 0 tests without `allow_xpu=True`.
- **Don't blindly copy CUDA test patterns**: CUDA tests may transpose tensors or use shapes that assume CUDA-specific kernel behavior. Verify the operator's expected input layout (e.g., `A[...,K] x B[K,N]`) independently.
- **Replace CUDA skip guards**: Remove `SM80OrLater`, `SM90OrLater` etc. Use `@onlyXPU` instead.
- **Run tests from `/tmp`**: Running from inside the pytorch source tree causes `torch/_C` import conflicts. Always `cd /tmp` before running test scripts.

### Build and CI

- **USE_SYCLTLA wrapper is mandatory for sycl-tla kernels**: The PyTorch dispatch function in `aten/src/ATen/native/mkldnn/xpu/` must NEVER directly call sycltla symbols (e.g., `at::xpu::detail::bf16bf16_grouped_mm`). sycltla kernels are built as separate shared libraries only when `USE_SYCLTLA=ON`. On CI or builds without sycltla, direct calls cause undefined symbol link errors. Instead, create a wrapper in torch-xpu-ops regular sources (`native/xpu/YourKernel.cpp`) guarded by `#ifdef USE_SYCLTLA`, and have PyTorch call the wrapper. Pattern:
  ```cpp
  // torch-xpu-ops/src/ATen/native/xpu/YourKernel.cpp (always compiled)
  #ifdef USE_SYCLTLA
  #include <ATen/native/xpu/sycltla/YourKernel.h>
  #endif
  bool is_your_kernel_available() {
  #ifdef USE_SYCLTLA
    return true;
  #else
    return false;
  #endif
  }
  void your_kernel(...) {
  #ifdef USE_SYCLTLA
    at::xpu::detail::your_kernel(...);  // forward to sycltla
  #else
    TORCH_CHECK(false, "Not compiled with SYCLTLA support.");
  #endif
  }
  ```
- **Stale installed headers**: After updating torch-xpu-ops, headers in `pytorch/torch/include/ATen/native/xpu/sycl/` may be stale from a previous build. If you see template signature mismatches, copy the updated header from `third_party/torch-xpu-ops/src/` to `torch/include/`.
- **Commit order matters**: Always commit torch-xpu-ops first, get the commit hash, update `pytorch/third_party/xpu.txt`, then commit PyTorch. The build system fetches torch-xpu-ops by this hash.
- **Fork URL required before torch-xpu-ops PR merges**: The PyTorch build fetches torch-xpu-ops from the URL in `caffe2/CMakeLists.txt` (`TORCH_XPU_OPS_REPO_URL`). When `third_party/xpu.txt` points to a commit that only exists on the fork (`xuhancn/torch-xpu-ops`), you **must** change this URL to `https://github.com/xuhancn/torch-xpu-ops.git` — otherwise the build fails with "reference is not a tree". Additionally, add `git remote set-url origin ${TORCH_XPU_OPS_REPO_URL}` before the `git fetch` step — CI runners may cache `third_party/torch-xpu-ops` from a prior run with `origin` pointing to `intel/torch-xpu-ops`, causing `git fetch` to fetch from the wrong remote. Switch the URL back to `intel/torch-xpu-ops.git` only after the torch-xpu-ops PR lands on upstream main:
  ```cmake
  # In caffe2/CMakeLists.txt, add before git fetch:
  execute_process(
    COMMAND git remote set-url origin ${TORCH_XPU_OPS_REPO_URL}
    WORKING_DIRECTORY ${TORCH_XPU_OPS_DIR})
  ```
- **AOTI C shim version guard**: When adding an XPU dispatch key for an operator in `native_functions.yaml`, the auto-generated `c_shim_xpu.h` entry gets a `TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_Z` guard. The version comes from `torchgen/aoti/fallback_ops.py` and tracks *when the operator first entered the stable C ABI* — it matches the CUDA shim version, not when XPU support was added. The version is auto-generated; don't change it manually.
- **sycl-tla SPIR-V flags for local builds**: When building a standalone extension with `setup.py`, monkey-patch the SYCL link flags:
  ```python
  import torch.utils.cpp_extension as _cpp_ext
  _SPIRV_EXT = '+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate'
  _cpp_ext._SYCL_DLINK_FLAGS.extend([
      f'-Xspirv-translator=spir64_gen', f'-spirv-ext={_SPIRV_EXT}',
      f'-Xspirv-translator=spir64', f'-spirv-ext={_SPIRV_EXT}',
  ])
  ```

### Kernel porting

- **sycl-tla ElementC type**: Must match the accumulator type (e.g., `float`), not the output type (e.g., `bfloat16`). The `CollectiveEpilogue` uses `ElementAccumulator` for C pointers.
- **Namespace conflicts**: If the kernel header defines a namespace (e.g., `namespace grouped_mm`), don't name your wrapper function the same thing. Rename to avoid ambiguity.
- **`compat::wait()` is required after every sycl-tla kernel launch**: `GemmUniversalAdapter::run()` submits work **asynchronously** to `compat::get_default_queue()` and returns immediately. `compat::wait()` calls `sycl::queue::wait()` on that queue — the only way to synchronize before reading results. All 14 official sycl-tla examples use this pattern. This is a framework requirement, not a design choice.
  - Source: [`device.hpp`](https://github.com/intel/sycl-tla/blob/357f75c57a962d6ced7e3d5f821276a494ee2aa4/include/cute/util/compat/device.hpp) — `compat::wait()` definition
  - Source: [`gemm_universal_adapter.h`](https://github.com/intel/sycl-tla/blob/357f75c57a962d6ced7e3d5f821276a494ee2aa4/include/cutlass/gemm/device/gemm_universal_adapter.h) — async `run()` 
  - Examples: [`04_bmg_grouped_gemm`](https://github.com/intel/sycl-tla/tree/357f75c57a962d6ced7e3d5f821276a494ee2aa4/examples/04_bmg_grouped_gemm), [`09_bmg_grouped_gemm_f8`](https://github.com/intel/sycl-tla/tree/357f75c57a962d6ced7e3d5f821276a494ee2aa4/examples/09_bmg_grouped_gemm_f8)
- **Use permalinks for sycl-tla references**: Always link to a specific commit hash (e.g., `/blob/357f75c5.../`) rather than `/blob/main/`, since sycl-tla is actively developed and files/paths move frequently. The repo has been force-pushed/rebased in the past, invalidating old commit hashes — always verify permalinks before using them.
  - Current known-good commit: `357f75c57a962d6ced7e3d5f821276a494ee2aa4`
  - Key paths at this commit:
    - `include/cute/util/compat/device.hpp` — `compat::wait()` definition
    - `include/cutlass/gemm/device/gemm_universal_adapter.h` — async `run()`
    - `examples/04_bmg_grouped_gemm/` — grouped GEMM example (was `examples/sycl/04_bmg_grouped_gemm/` at older commits)
    - `examples/09_bmg_grouped_gemm_f8/` — FP8 grouped GEMM example
    - `examples/00_bmg_gemm/00_bmg_gemm.cpp` — basic GEMM with compat docs

## Two-PR Rebase Workflow

When PyTorch `main` advances, rebase all four PR branches in dependency order. The split structure requires rebasing in sequence:

### Branch Structure (Split PRs)

```
torch-xpu-ops:
  origin/main → xpu-grouped-mm (1 commit) → xpu-scaled-grouped-mm (+1 commit)

pytorch:
  upstream/main → xpu-grouped-mm (1 commit) → xpu-scaled-grouped-mm (+1 commit)
```

### Step 1: Rebase torch-xpu-ops (both branches)

```bash
cd pytorch/third_party/torch-xpu-ops
git fetch origin

# Phase 1 branch
git checkout xpu-grouped-mm
git rebase origin/main
# Note new hash: GROUPED_HASH=$(git rev-parse HEAD)

# Phase 2 branch — rebase only its own commit onto updated Phase 1
git rebase --onto xpu-grouped-mm <old-grouped-hash> xpu-scaled-grouped-mm
# Note new hash: SCALED_HASH=$(git rev-parse HEAD)
```

### Step 2: Rebase PyTorch (both branches)

```bash
cd pytorch
git fetch upstream

# Phase 1 branch
git checkout xpu-grouped-mm
git rebase upstream/main
echo "$GROUPED_HASH" > third_party/xpu.txt
git add third_party/xpu.txt && git commit --amend --no-edit

# Phase 2 branch — rebase only its own commit onto updated Phase 1
git rebase --onto xpu-grouped-mm <old-grouped-hash> xpu-scaled-grouped-mm
# Resolve xpu.txt conflict: echo "$SCALED_HASH" > third_party/xpu.txt
git add third_party/xpu.txt && GIT_EDITOR=true git rebase --continue
```

### Step 3: Build, test, and push

```bash
# Build from the Phase 2 branch (includes both operators)
USE_XPU=1 TORCH_XPU_ARCH_LIST=bmg python setup.py develop

# Test both suites from /tmp
cd /tmp
python pytorch/third_party/torch-xpu-ops/test/xpu/test_grouped_mm_xpu.py
python pytorch/third_party/torch-xpu-ops/test/xpu/test_scaled_grouped_mm_xpu.py

# Force-push all 4 branches
cd pytorch/third_party/torch-xpu-ops
git push --force origin xpu-grouped-mm xpu-scaled-grouped-mm
cd pytorch
git push --force origin xpu-grouped-mm xpu-scaled-grouped-mm
```

### Common Rebase Conflicts

| File | Pattern | Resolution |
|------|---------|------------|
| `aten/src/ATen/CMakeLists.txt` | `CONFIGURE_DEPENDS` style changes for XPU globs | Use `CONFIGURE_DEPENDS` with the new glob pattern style |
| `third_party/xpu.txt` | Hash conflict | Replace with the new torch-xpu-ops commit hash |
| `torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h` | `TORCH_FEATURE_VERSION` guards added upstream | Regenerate with the correct version guard for new ops |
| `caffe2/CMakeLists.txt` | Fork URL or submodule changes | Keep the fork URL if torch-xpu-ops PR hasn't merged yet |

### Fixup Commit Pattern

When multiple files need updating after rebase, use interactive rebase with fixup:

```bash
# Make fixes on top of rebase
git commit -m "fixup! <original commit message>"
# Then squash into original
git rebase -i --autosquash upstream/main
```

## Wheel-Level Testing

To test torch-xpu-ops changes through the full PyTorch dispatch chain (like CI does):

### Step 1: Commit and push torch-xpu-ops changes

```bash
cd pytorch/third_party/torch-xpu-ops
git add -A && git commit -m "your changes"
git push origin your-branch
```

### Step 2: Update xpu.txt and rebuild PyTorch

```bash
cd pytorch
# Update xpu.txt with the new commit hash
git log --format='%H' -1 -- third_party/torch-xpu-ops > third_party/xpu.txt

# Rebuild (this fetches torch-xpu-ops from the URL in caffe2/CMakeLists.txt)
source ~/intel/oneapi/setvars.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch
USE_XPU=1 TORCH_XPU_ARCH_LIST=bmg python setup.py develop
```

### Step 3: Sync stale .so files (editable install workaround)

```bash
cp -a pytorch/torch/lib/*.so \
  $(python -c "import site; print(site.getsitepackages()[0])")/torch/lib/
```

### Step 4: Run tests from /tmp

```bash
# torch-xpu-ops tests (through full dispatch)
cd /tmp && python pytorch/third_party/torch-xpu-ops/test/xpu/test_scaled_grouped_mm_xpu.py

# PyTorch upstream tests (CUDA tests run on XPU where applicable)
cd /tmp && python pytorch/test/test_scaled_matmul_cuda.py -k '_scaled_grouped_mm'
```

## Temp Branch Workflow

When testing review changes or experimental modifications before applying to the real PR:

1. **Create temp branch** from the PR branch:
   ```bash
   git checkout xpu-scaled-grouped-mm
   git checkout -b xpu-scaled-grouped-mm-review-test
   ```
2. **Apply and test changes** on the temp branch
3. **Push temp branch** for wheel-level testing (xpu.txt must point to a pushed commit)
4. **After validation**, cherry-pick the commit to the real PR branch:
   ```bash
   git checkout xpu-scaled-grouped-mm
   git cherry-pick <commit-hash>
   git push origin xpu-scaled-grouped-mm --force
   ```
5. **Clean up** the temp branch:
   ```bash
   git branch -D xpu-scaled-grouped-mm-review-test
   git push origin --delete xpu-scaled-grouped-mm-review-test
   ```

## PR Review Comment Handling

Process for evaluating automated review comments (e.g., GitHub Copilot bot):

1. **Fetch all review threads** from the PR via GitHub API
2. **Categorize** each comment:
   - **Already fixed**: Issue was addressed in a later commit
   - **Accept**: Valid suggestion, implement it
   - **Reject**: Suggestion is incorrect (e.g., bot misunderstands domain conventions)
   - **Won't fix**: Valid concern but out of scope or would cause other issues
3. **Create review response doc** (`docs/prNNNN_review_response.md`) with:
   - Each comment's location, suggestion, and category
   - Copy-paste-ready **Reply** text for each thread
4. **Test accepted changes locally** before applying to the PR
5. **Apply via temp branch workflow** (see above)

### Known Bot False Positives

- **Stride B `{N, K, 1}` in grouped GEMM**: The bot repeatedly flags `make_cute_packed_stride(StrideB{}, {N, K, 1})` as incorrect, claiming it should be `{K, N, 1}`. This is **wrong** — `{N, K, 1}` is the correct convention for CUTLASS/sycl-tla grouped GEMM with `LayoutB = RowMajor`. Verified against the [sycl-tla grouped GEMM example](https://github.com/intel/sycl-tla/blob/357f75c57a962d6ced7e3d5f821276a494ee2aa4/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp#L336) which uses the same `{N, K, 1}` pattern. The parameters are problem dimensions, not strides — CUTLASS computes the actual strides from the layout type.
- **Test b shape mismatch**: The bot claims `b = torch.randn(n_groups, k, n)` is wrong and should be `(n_groups, n, k)`. But the tests pass correctly — verify the actual operator contract before changing shapes based on bot suggestions.

### Common Valid Bot Suggestions

These patterns recur across PRs and should be addressed:

| Pattern | Fix |
|---------|-----|
| `offs_host[g]` without `TORCH_CHECK(offs.has_value())` | Add validation block before ragged paths |
| `bias` argument silently ignored | Add `TORCH_CHECK(!bias.has_value(), "not supported")` |
| `out.data_ptr()` without contiguity/dtype check | Add `TORCH_CHECK(out.is_contiguous() && out.scalar_type() == kBFloat16)` |
| `offs` read as `int32` without dtype validation | Add `TORCH_CHECK(offs->scalar_type() == at::kInt)` |
| Missing `#include <optional>` in headers | Add explicit include |
| Tests hard-fail without SYCLTLA | Add try/except availability gate |

## Git Conventions for PRs

- **Run lintrunner before committing**: Always run `lintrunner <files>` on changed files before committing to torch-xpu-ops. The repo has `.lintrunner.toml` with FLAKE8, CLANGTIDY, RUFF, and other linters. Run `lintrunner init` on first use.
  ```bash
  cd pytorch/third_party/torch-xpu-ops
  lintrunner path/to/changed/file.py path/to/changed/file.cpp
  ```
- **No AI Co-authored-by trailers**: Do not include `Co-authored-by: Copilot <...>` or similar AI attribution in PR commits. If accidentally added, strip with:
  ```bash
  git commit --amend  # remove the trailer in editor
  git push --force
  ```
- **Commit order**: Always commit torch-xpu-ops first, then PyTorch (xpu.txt dependency)
- **No force push by default**: Do not force-push PR branches unless explicitly requested by the user. New changes should be added as new commits on top of the existing branch. Force-push is only allowed for rebase operations when the user explicitly asks.
- **Stale cmake cache after branch switch**: After switching between `xpu-grouped-mm` and `xpu-scaled-grouped-mm` branches, the cmake build cache for sycl-tla targets may be stale. Delete the cached target directory and reconfigure:
  ```bash
  rm -rf build/caffe2/aten_xpu/src/CMakeFiles/torch-xpu-ops-sycltla-*.dir
  cd build && cmake .. -DUSE_XPU=ON
  ```
- **Stale `enum_tag.h` after branch switch**: Generated file `build/torch/headeronly/core/enum_tag.h` can conflict with `build/aten/src/ATen/core/enum_tag.h` after switching branches. Fix by deleting both and rebuilding:
  ```bash
  rm -f build/torch/headeronly/core/enum_tag.h build/aten/src/ATen/core/enum_tag.h
  ```

## Building PyTorch from Source with XPU Support

**Official doc**: https://github.com/pytorch/pytorch#intel-gpu-support

### Prerequisites

1. Intel oneAPI Base Toolkit (provides `icpx` compiler and SYCL runtime)
2. The Intel compiler (`icpx`) **must** be on PATH when CMake configures — this is how CMake detects and enables XPU support.

### Build Steps

```bash
# 1. Source Intel oneAPI (CRITICAL — must be done BEFORE cmake configure)
source ~/intel/oneapi/setvars.sh

# 2. Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch

# 3. Set environment variables
export USE_XPU=1
export TORCH_XPU_ARCH_LIST=bmg    # Target architecture (bmg = Battlemage/Arc B-series)
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"

# 4. Build and install (editable mode)
cd /home/xu/conda_root/xu_pytorch/pytorch
python -m pip install --no-build-isolation -v -e .
```

### Common Build Issues

- **`USE_XPU=OFF` despite setting env var**: If CMakeCache.txt already exists with `USE_XPU:BOOL=OFF`, delete it and reconfigure. The env var only takes effect during initial cmake configuration.
  ```bash
  rm build/CMakeCache.txt
  ```
- **Stale shared libraries in site-packages**: After a successful build, editable install may load stale `.so` files from `site-packages/torch/lib/` instead of the freshly built ones in `pytorch/torch/lib/`. Fix by syncing:
  ```bash
  cp -a pytorch/torch/lib/*.so $(python -c "import site; print(site.getsitepackages()[0])")/torch/lib/
  ```
- **sycl-tla `-Werror` failures**: The `torch-xpu-ops/cmake/BuildFlags.cmake` adds `-Werror` to SYCL host flags. sycl-tla headers trigger warnings (`-Wunused-variable`, `-Wunused-local-typedefs`, `-Wreorder`) that cannot be fixed locally. The proper fix is to guard `-Werror` with `REPLACE_FLAGS_FOR_SYCLTLA` in `BuildFlags.cmake`:
  ```cmake
  if(REPLACE_FLAGS_FOR_SYCLTLA)
    list(APPEND SYCL_HOST_FLAGS -Wno-error)
  else()
    list(APPEND SYCL_HOST_FLAGS -Werror)
  endif()
  ```
  This is already applied in PR #3122. For local standalone builds, comment out `-Werror` in `BuildFlags.cmake` line 57.
- **`exmy_base.h` C++20 error**: GCC 14+ in C++20 mode rejects template-id on constructors. Patch `build/_deps/repo-sycl-tla-src/include/cutlass/exmy_base.h`:
  ```cpp
  // Change: explicit float_exmy_base<T, Derived>(float x)
  // To:     explicit float_exmy_base(float x)
  ```
- **Incremental `cmake --build build` fails**: Running `cmake --build build` directly fails with `ModuleNotFoundError: No module named 'tools'`. Always use `python setup.py develop` or `python -m pip install --no-build-isolation -v -e .` for building — these set up the Python path correctly.
- **`target_compile_options()` does NOT work for SYCL targets**: The SYCL compilation pipeline builds its flags from `CMAKE_HOST_FLAGS` (set by `set_build_flags()` in `BuildFlags.cmake`), bypassing cmake's target-level options entirely. Do not try to suppress warnings with `target_compile_options(target PRIVATE -Wno-error)` — it will have no effect. Modify the flag lists in `BuildFlags.cmake` instead.
- **Verify XPU was compiled**: After build, check `torch._C._has_xpu` (not `torch.xpu.is_available()` which also requires hardware). If `_has_xpu` is False but cmake shows `USE_XPU=1`, the issue is likely stale site-packages libraries (see above).

### Verifying the Build

```bash
cd /tmp  # avoid import conflicts
python -c "
import torch
print('_has_xpu:', torch._C._has_xpu)           # True = compiled with XPU
print('XPU available:', torch.xpu.is_available()) # True = hardware detected
print('Device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')
"
```

## Validating Intel XPU Hardware

### Quick Validation (without source build)

Create a fresh conda env and install the nightly XPU wheel — **no Intel oneAPI sourcing needed**:

```bash
# Create clean environment
conda create -y -n xpu_test python=3.11
conda activate xpu_test

# Install nightly XPU wheel (includes bundled SYCL runtime)
python -m pip install --pre torch pytorch-triton-xpu torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu \
  --upgrade --force-reinstall

# Validate
cd /tmp && python -c "
import torch
print('XPU available:', torch.xpu.is_available())
print('Device count:', torch.xpu.device_count())
if torch.xpu.is_available():
    print('Device:', torch.xpu.get_device_name(0))
"
```

### Checking SYCL Devices (with oneAPI)

```bash
source ~/intel/oneapi/setvars.sh
sycl-ls   # Lists all SYCL-visible devices
```

### Hardware Requirements

- Intel discrete GPUs: Arc A-series (Alchemist), Arc B-series (Battlemage), Arc Pro series
- Supported platforms: Linux and Windows
- Driver: Intel GPU driver with Level Zero support
- Device nodes: `/dev/dri/renderD128` (or similar) must be accessible

### Intel Compiler Versions (from PyTorch CI)

The official CI install script is at:
https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_xpu.sh

It installs "Intel Deep Learning Essentials" which bundles the compiler and libraries:

| `XPU_VERSION` | Package URL | Compiler Version |
|---------------|-------------|-----------------|
| `2025.3` | `intel-deep-learning-essentials-2025.3.2.36_offline.sh` | Intel DPC++/C++ 2025.3.x |
| `2025.2` (default) | `intel-deep-learning-essentials-2025.2.1.24_offline.sh` | Intel DPC++/C++ 2025.2.x |

To check your local compiler version:
```bash
source ~/intel/oneapi/setvars.sh
icpx --version
# Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2 (2025.3.2.20260112)
```

The CI script also installs GPU drivers (Level Zero, OpenCL, media) and development headers. On a local dev machine, install the full Intel oneAPI Base Toolkit instead.

### Official Documentation

| Resource | URL |
|----------|-----|
| PyTorch XPU build instructions | https://github.com/pytorch/pytorch#intel-gpu-support |
| PyTorch XPU prerequisites | https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html |
| PyTorch CI XPU install script | https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_xpu.sh |
| PyTorch XPU nightly wheels | https://download.pytorch.org/whl/nightly/xpu |
| Intel oneAPI Base Toolkit | https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html |
| torch-xpu-ops repository | https://github.com/intel/torch-xpu-ops |
| sycl-tla (CUTLASS for Intel) | https://github.com/intel/sycl-tla |

## Model-Level Accuracy Validation

### Why Model-Level Tests

Unit tests validate operator correctness with small shapes (M=16, K=64). Model-level tests validate with **real production shapes** from models like Llama 4 Scout MoE, catching issues that only appear at scale:
- BF16 accumulation error grows with K (reduction dimension)
- Memory/allocation patterns differ at real dimensions
- Numerical stability under realistic value distributions

### How to Write Model-Level Tests

1. **Research the model architecture** to find where the operator is called and extract exact tensor shapes (hidden_size, intermediate_size, num_experts)
2. **Use the 2Dx3D input mode** for MoE — this is the real torchao FP8 training pattern
3. **Scale down M (tokens)** to fit GPU memory while keeping K, N, G at real values
4. **Cross-device comparison**: create tensors on CPU, compute float32 reference on CPU, run kernel on XPU, compare outputs
5. **Report error statistics**: max/mean absolute error, percentile distribution, ULP analysis

### Llama 4 Scout Shape Reference

The torchao MoE FP8 training path (`pytorch/ao`) intercepts `torch._grouped_mm` on `Float8TrainingWeightWrapperTensor` and routes through `torch._scaled_grouped_mm`:

```
Gate/Up:  A=(M, 5120) FP8  x  B=(G, 5120, 8192) FP8  →  (M, 8192) BF16
Down:     A=(M, 8192) FP8  x  B=(G, 8192, 5120) FP8  →  (M, 5120) BF16
```

Where K=5120 (hidden_size), N=8192 (intermediate_size), G=experts per device.

Source: `pytorch/ao:benchmarks/prototype/moe_training/benchmark_scaled_grouped_mm_dq.py`

### Running Model Tests

```bash
# Local SYCL extension
cd /tmp && python <repo>/test/test_llama4_model_shapes.py TestLlama4ExtensionShapes

# Full PyTorch dispatch chain
cd /tmp && python <repo>/test/test_llama4_model_shapes.py TestLlama4Dispatch
```

### BF16 Accumulation Error Budget

When comparing XPU (BF16 accumulation) against CPU (float32 accumulation), errors are bounded by BF16 ULP:

| K | Max Error | Mean Error | Explanation |
|---|---|---|---|
| 64 | 0.25 | 0.005 | 1 ULP at output magnitude ~42 |
| 256 | 0.50 | 0.011 | 1 ULP at output magnitude ~84 |
| 1024 | 1.00 | 0.022 | 1 ULP at output magnitude ~150 |
| 5120 | 2.00 | 0.047 | 1 ULP at output magnitude ~338 |
| 8192 | 2.00 | 0.061 | 1 ULP at output magnitude ~430 |

**All max errors = exactly 1 BF16 ULP.** BF16 has 7-bit mantissa, so ULP = 2^(exponent-7):
- |value| ∈ [128, 256): ULP = 1.0
- |value| ∈ [256, 512): ULP = 2.0

For model-level tests with K=5120–8192, use `atol=4.0, rtol=0.1` (tolerates up to 2 ULPs).

### Test Results

See `docs/llama4_model_validation.md` for detailed results. Summary: 18/18 tests pass, ~50% of elements match exactly, 99.9% within 0.5 absolute error, mean relative error < 0.5%.

## Reference: Completed Ports

| Kernel | Dev repo | torch-xpu-ops PR | PyTorch PR |
|--------|----------|-------------------|------------|
| `_grouped_mm` | `dev_pytorch_group_mm` | [#3122](https://github.com/intel/torch-xpu-ops/pull/3122) | [#178242](https://github.com/pytorch/pytorch/pull/178242) |
| `_scaled_grouped_mm` | `dev_pytorch_scaled_grouped_mm` | [#3172](https://github.com/intel/torch-xpu-ops/pull/3172) | [#178354](https://github.com/pytorch/pytorch/pull/178354) |
