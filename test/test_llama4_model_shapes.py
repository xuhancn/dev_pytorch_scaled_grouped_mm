"""
Model-level validation for _scaled_grouped_mm using real Llama 4 Scout shapes.

Llama 4 Scout (109B/17B-16E) MoE architecture:
  - hidden_size (K) = 5120
  - intermediate_size (N) = 8192
  - num_local_experts (E) = 16
  - num_experts_per_tok = 1 (top-1 routing)

The torchao MoE FP8 training path calls torch._scaled_grouped_mm with:
  Gate/Up:  A=(M, 5120) FP8  x  B=(G, 5120, 8192) FP8  →  (M, 8192) BF16
  Down:     A=(M, 8192) FP8  x  B=(G, 8192, 5120) FP8  →  (M, 5120) BF16

This uses the 2Dx3D input mode (ragged A with offsets along M), matching
the MoE pattern where tokens are sorted by expert.

The full model is too large for Intel Arc B580 (~12GB). We scale down M
(total tokens) while keeping K, N, G at real model values.

Tests create tensors on CPU first, compute a float32 reference on CPU,
then copy to XPU and run the kernel. XPU results are compared against
the CPU reference to validate cross-device correctness.

Run from /tmp to avoid torch/_C import conflicts:
    cd /tmp && python <repo>/test/test_llama4_model_shapes.py

For full dispatch chain testing (requires built PyTorch with XPU):
    cd /tmp && python <repo>/test/test_llama4_model_shapes.py TestLlama4Dispatch
"""

import unittest
import torch
import sys
import time

# Llama 4 Scout 16E configuration
HIDDEN_SIZE = 5120       # K
INTERMEDIATE_SIZE = 8192 # N
NUM_EXPERTS = 16         # E


def reference_scaled_mm_per_group(a_fp8, b_fp8_t, scale_a, scale_b):
    """Reference: dequantize FP8 -> float32, apply scales, matmul, cast to BF16.

    Runs on whatever device the inputs are on (CPU or XPU).

    Args:
        a_fp8: (M, K) FP8 tensor, row-major
        b_fp8_t: (K, N) FP8 tensor — transposed view; physical layout is (N, K)
        scale_a: (M,) float32 rowwise scale for A
        scale_b: (N,) float32 rowwise scale for B
    Returns:
        (M, N) BF16 tensor
    """
    a_f32 = a_fp8.float() * scale_a.unsqueeze(-1)
    b_phys = b_fp8_t.t().contiguous()  # (N, K) row-major
    b_f32 = b_phys.float() * scale_b.unsqueeze(-1)
    out = a_f32 @ b_f32.t()
    return out.to(torch.bfloat16)


def cpu_reference_grouped(a_fp8_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu):
    """Compute full grouped MM reference on CPU in float32.

    Returns: (total_M, N) BF16 tensor on CPU.
    """
    G = b_t_cpu.shape[0]
    N = b_t_cpu.shape[2]
    total_M = a_fp8_cpu.shape[0]
    out = torch.empty(total_M, N, dtype=torch.bfloat16)

    row_start = 0
    for g in range(G):
        row_end = offs_cpu[g].item()
        out[row_start:row_end] = reference_scaled_mm_per_group(
            a_fp8_cpu[row_start:row_end], b_t_cpu[g],
            scale_a_cpu[row_start:row_end], scale_b_cpu[g])
        row_start = row_end
    return out


class TestLlama4ExtensionShapes(unittest.TestCase):
    """Test local SYCL extension with real Llama 4 Scout shapes.

    Creates tensors on CPU, computes float32 reference on CPU, copies to XPU,
    runs kernel, and compares XPU output against CPU reference.
    """

    device = "xpu"
    fp8_dtype = torch.float8_e4m3fn
    # Wider tolerances than unit tests because:
    # 1. CPU reference uses float32 accumulation, XPU uses BF16 GEMM
    # 2. Large K (5120/8192) means ~5K-8K BF16 additions, accumulating error
    # 3. Mean error is typically <0.05; max error ~2-4 at model-scale K
    atol = 4.0
    rtol = 1e-1

    @classmethod
    def setUpClass(cls):
        try:
            import scaled_grouped_mm_sycl
            cls.ext = scaled_grouped_mm_sycl
        except ImportError:
            raise unittest.SkipTest("scaled_grouped_mm_sycl extension not available")

    def _make_fp8_cpu(self, *shape):
        return torch.randn(*shape).to(self.fp8_dtype)

    def _make_scale_cpu(self, *shape):
        return torch.rand(*shape, dtype=torch.float32) + 0.1

    def _make_offsets_cpu(self, total_M, G):
        """Create balanced offsets: tokens evenly split across experts."""
        tokens_per_expert = total_M // G
        return torch.arange(
            tokens_per_expert, total_M + 1, tokens_per_expert, dtype=torch.int32
        )

    def _run_gate_up_test(self, M, G, label=""):
        """Test gate/up projection: A=(M, 5120) x B=(G, 5120, 8192)."""
        K, N = HIDDEN_SIZE, INTERMEDIATE_SIZE

        # Create on CPU
        a_cpu = self._make_fp8_cpu(M, K)
        b_phys_cpu = self._make_fp8_cpu(G, N, K)
        b_t_cpu = b_phys_cpu.transpose(-2, -1)  # (G, K, N) col-major
        scale_a_cpu = self._make_scale_cpu(M)
        scale_b_cpu = self._make_scale_cpu(G, N)
        offs_cpu = self._make_offsets_cpu(M, G)

        # CPU reference (float32 precision)
        t0 = time.perf_counter()
        ref_cpu = cpu_reference_grouped(a_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu)
        cpu_time = time.perf_counter() - t0

        # Copy to XPU and run kernel
        a_xpu = a_cpu.to(self.device)
        b_t_xpu = b_phys_cpu.to(self.device).transpose(-2, -1)
        scale_a_xpu = scale_a_cpu.to(self.device)
        scale_b_xpu = scale_b_cpu.to(self.device)
        offs_xpu = offs_cpu.to(self.device)

        t0 = time.perf_counter()
        out_xpu = self.ext.scaled_grouped_mm(
            a_xpu, b_t_xpu, scale_a_xpu, scale_b_xpu, offs=offs_xpu)
        torch.xpu.synchronize()
        xpu_time = time.perf_counter() - t0

        self.assertEqual(out_xpu.shape, (M, N))
        self.assertEqual(out_xpu.dtype, torch.bfloat16)

        # Compare XPU output against CPU reference
        out_cpu = out_xpu.cpu()
        diff = (out_cpu.float() - ref_cpu.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        torch.testing.assert_close(
            out_cpu, ref_cpu, atol=self.atol, rtol=self.rtol,
            msg=f"{label} gate_up XPU vs CPU mismatch")

        print(f"  {label} gate_up  M={M} G={G} K={K} N={N}  "
              f"CPU={cpu_time*1000:.1f}ms  XPU={xpu_time*1000:.1f}ms  "
              f"max_err={max_err:.4f}  mean_err={mean_err:.4f}")
        return cpu_time, xpu_time

    def _run_down_proj_test(self, M, G, label=""):
        """Test down projection: A=(M, 8192) x B=(G, 8192, 5120)."""
        K, N = INTERMEDIATE_SIZE, HIDDEN_SIZE

        a_cpu = self._make_fp8_cpu(M, K)
        b_phys_cpu = self._make_fp8_cpu(G, N, K)
        b_t_cpu = b_phys_cpu.transpose(-2, -1)
        scale_a_cpu = self._make_scale_cpu(M)
        scale_b_cpu = self._make_scale_cpu(G, N)
        offs_cpu = self._make_offsets_cpu(M, G)

        # CPU reference
        t0 = time.perf_counter()
        ref_cpu = cpu_reference_grouped(a_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu)
        cpu_time = time.perf_counter() - t0

        # XPU kernel
        a_xpu = a_cpu.to(self.device)
        b_t_xpu = b_phys_cpu.to(self.device).transpose(-2, -1)
        scale_a_xpu = scale_a_cpu.to(self.device)
        scale_b_xpu = scale_b_cpu.to(self.device)
        offs_xpu = offs_cpu.to(self.device)

        t0 = time.perf_counter()
        out_xpu = self.ext.scaled_grouped_mm(
            a_xpu, b_t_xpu, scale_a_xpu, scale_b_xpu, offs=offs_xpu)
        torch.xpu.synchronize()
        xpu_time = time.perf_counter() - t0

        self.assertEqual(out_xpu.shape, (M, N))
        self.assertEqual(out_xpu.dtype, torch.bfloat16)

        out_cpu = out_xpu.cpu()
        diff = (out_cpu.float() - ref_cpu.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        torch.testing.assert_close(
            out_cpu, ref_cpu, atol=self.atol, rtol=self.rtol,
            msg=f"{label} down_proj XPU vs CPU mismatch")

        print(f"  {label} down     M={M} G={G} K={K} N={N}  "
              f"CPU={cpu_time*1000:.1f}ms  XPU={xpu_time*1000:.1f}ms  "
              f"max_err={max_err:.4f}  mean_err={mean_err:.4f}")
        return cpu_time, xpu_time

    # --- Gate/Up projection tests ---

    def test_gate_up_G1_M512(self):
        """Gate/Up: 1 expert, 512 tokens (EP=16 scenario)."""
        self._run_gate_up_test(M=512, G=1, label="G1_M512")

    def test_gate_up_G2_M1024(self):
        """Gate/Up: 2 experts, 1024 tokens (EP=8 scenario)."""
        self._run_gate_up_test(M=1024, G=2, label="G2_M1024")

    def test_gate_up_G4_M2048(self):
        """Gate/Up: 4 experts, 2048 tokens (EP=4 scenario)."""
        self._run_gate_up_test(M=2048, G=4, label="G4_M2048")

    def test_gate_up_G16_M512(self):
        """Gate/Up: all 16 experts, 512 tokens (no EP, small batch)."""
        self._run_gate_up_test(M=512, G=16, label="G16_M512")

    # --- Down projection tests ---

    def test_down_proj_G1_M512(self):
        """Down: 1 expert, 512 tokens."""
        self._run_down_proj_test(M=512, G=1, label="G1_M512")

    def test_down_proj_G2_M1024(self):
        """Down: 2 experts, 1024 tokens."""
        self._run_down_proj_test(M=1024, G=2, label="G2_M1024")

    def test_down_proj_G4_M2048(self):
        """Down: 4 experts, 2048 tokens."""
        self._run_down_proj_test(M=2048, G=4, label="G4_M2048")

    def test_down_proj_G16_M512(self):
        """Down: all 16 experts, 512 tokens."""
        self._run_down_proj_test(M=512, G=16, label="G16_M512")

    # --- Unbalanced token distribution ---

    def test_gate_up_unbalanced(self):
        """Gate/Up with unbalanced token distribution across experts."""
        K, N, G = HIDDEN_SIZE, INTERMEDIATE_SIZE, 4
        group_sizes = [128, 256, 512, 128]
        total_M = sum(group_sizes)

        a_cpu = self._make_fp8_cpu(total_M, K)
        b_phys_cpu = self._make_fp8_cpu(G, N, K)
        b_t_cpu = b_phys_cpu.transpose(-2, -1)
        scale_a_cpu = self._make_scale_cpu(total_M)
        scale_b_cpu = self._make_scale_cpu(G, N)

        cumulative = 0
        offs_list = []
        for s in group_sizes:
            cumulative += s
            offs_list.append(cumulative)
        offs_cpu = torch.tensor(offs_list, dtype=torch.int32)

        # CPU reference
        ref_cpu = cpu_reference_grouped(a_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu)

        # XPU kernel
        a_xpu = a_cpu.to(self.device)
        b_t_xpu = b_phys_cpu.to(self.device).transpose(-2, -1)
        scale_a_xpu = scale_a_cpu.to(self.device)
        scale_b_xpu = scale_b_cpu.to(self.device)
        offs_xpu = offs_cpu.to(self.device)

        out_xpu = self.ext.scaled_grouped_mm(
            a_xpu, b_t_xpu, scale_a_xpu, scale_b_xpu, offs=offs_xpu)
        torch.xpu.synchronize()

        self.assertEqual(out_xpu.shape, (total_M, N))

        out_cpu = out_xpu.cpu()
        torch.testing.assert_close(
            out_cpu, ref_cpu, atol=self.atol, rtol=self.rtol,
            msg="Unbalanced gate_up XPU vs CPU mismatch")


class TestLlama4Dispatch(unittest.TestCase):
    """Test via full PyTorch dispatch chain: torch._scaled_grouped_mm.

    Same CPU-first pattern: compute reference on CPU, run dispatch on XPU,
    compare outputs cross-device.
    """

    device = "xpu"
    fp8_dtype = torch.float8_e4m3fn
    atol = 4.0
    rtol = 1e-1

    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, '_scaled_grouped_mm'):
            raise unittest.SkipTest("torch._scaled_grouped_mm not available")
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU not available")

    def _make_fp8_cpu(self, *shape):
        return torch.randn(*shape).to(self.fp8_dtype)

    def _make_scale_cpu(self, *shape):
        return torch.rand(*shape, dtype=torch.float32) + 0.1

    def _make_offsets_cpu(self, total_M, G):
        tokens_per_expert = total_M // G
        return torch.arange(
            tokens_per_expert, total_M + 1, tokens_per_expert, dtype=torch.int32
        )

    def _run_dispatch_test(self, M, K, N, G, label=""):
        """Test through torch._scaled_grouped_mm dispatch with CPU comparison."""
        a_cpu = self._make_fp8_cpu(M, K)
        b_phys_cpu = self._make_fp8_cpu(G, N, K)
        b_t_cpu = b_phys_cpu.transpose(-2, -1)
        scale_a_cpu = self._make_scale_cpu(M)
        scale_b_cpu = self._make_scale_cpu(G, N)
        offs_cpu = self._make_offsets_cpu(M, G)

        # CPU reference
        t0 = time.perf_counter()
        ref_cpu = cpu_reference_grouped(a_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu)
        cpu_time = time.perf_counter() - t0

        # XPU dispatch
        a_xpu = a_cpu.to(self.device)
        b_t_xpu = b_phys_cpu.to(self.device).transpose(-2, -1)
        scale_a_xpu = scale_a_cpu.to(self.device)
        scale_b_xpu = scale_b_cpu.to(self.device)
        offs_xpu = offs_cpu.to(self.device)

        t0 = time.perf_counter()
        out_xpu = torch._scaled_grouped_mm(
            a_xpu, b_t_xpu, scale_a_xpu, scale_b_xpu,
            offs=offs_xpu, out_dtype=torch.bfloat16)
        torch.xpu.synchronize()
        xpu_time = time.perf_counter() - t0

        self.assertEqual(out_xpu.shape, (M, N))
        self.assertEqual(out_xpu.dtype, torch.bfloat16)

        # Compare XPU output against CPU reference
        out_cpu = out_xpu.cpu()
        diff = (out_cpu.float() - ref_cpu.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        torch.testing.assert_close(
            out_cpu, ref_cpu, atol=self.atol, rtol=self.rtol,
            msg=f"{label} dispatch XPU vs CPU mismatch")

        print(f"  {label}  M={M} G={G} K={K} N={N}  "
              f"CPU={cpu_time*1000:.1f}ms  XPU={xpu_time*1000:.1f}ms  "
              f"max_err={max_err:.4f}  mean_err={mean_err:.4f}")
        return cpu_time, xpu_time

    # --- Gate/Up projection via dispatch ---

    def test_dispatch_gate_up_G1_M512(self):
        self._run_dispatch_test(512, HIDDEN_SIZE, INTERMEDIATE_SIZE, 1, "gate_up_G1")

    def test_dispatch_gate_up_G2_M1024(self):
        self._run_dispatch_test(1024, HIDDEN_SIZE, INTERMEDIATE_SIZE, 2, "gate_up_G2")

    def test_dispatch_gate_up_G4_M2048(self):
        self._run_dispatch_test(2048, HIDDEN_SIZE, INTERMEDIATE_SIZE, 4, "gate_up_G4")

    def test_dispatch_gate_up_G16_M512(self):
        self._run_dispatch_test(512, HIDDEN_SIZE, INTERMEDIATE_SIZE, 16, "gate_up_G16")

    # --- Down projection via dispatch ---

    def test_dispatch_down_proj_G1_M512(self):
        self._run_dispatch_test(512, INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, "down_G1")

    def test_dispatch_down_proj_G2_M1024(self):
        self._run_dispatch_test(1024, INTERMEDIATE_SIZE, HIDDEN_SIZE, 2, "down_G2")

    def test_dispatch_down_proj_G4_M2048(self):
        self._run_dispatch_test(2048, INTERMEDIATE_SIZE, HIDDEN_SIZE, 4, "down_G4")

    def test_dispatch_down_proj_G16_M512(self):
        self._run_dispatch_test(512, INTERMEDIATE_SIZE, HIDDEN_SIZE, 16, "down_G16")

    # --- Unbalanced ---

    def test_dispatch_unbalanced(self):
        """Dispatch with unbalanced token distribution."""
        K, N, G = HIDDEN_SIZE, INTERMEDIATE_SIZE, 4
        group_sizes = [128, 256, 512, 128]
        total_M = sum(group_sizes)

        a_cpu = self._make_fp8_cpu(total_M, K)
        b_phys_cpu = self._make_fp8_cpu(G, N, K)
        b_t_cpu = b_phys_cpu.transpose(-2, -1)
        scale_a_cpu = self._make_scale_cpu(total_M)
        scale_b_cpu = self._make_scale_cpu(G, N)

        cumulative = 0
        offs_list = []
        for s in group_sizes:
            cumulative += s
            offs_list.append(cumulative)
        offs_cpu = torch.tensor(offs_list, dtype=torch.int32)

        # CPU reference
        ref_cpu = cpu_reference_grouped(a_cpu, b_t_cpu, scale_a_cpu, scale_b_cpu, offs_cpu)

        # XPU dispatch
        a_xpu = a_cpu.to(self.device)
        b_t_xpu = b_phys_cpu.to(self.device).transpose(-2, -1)
        scale_a_xpu = scale_a_cpu.to(self.device)
        scale_b_xpu = scale_b_cpu.to(self.device)
        offs_xpu = offs_cpu.to(self.device)

        out_xpu = torch._scaled_grouped_mm(
            a_xpu, b_t_xpu, scale_a_xpu, scale_b_xpu,
            offs=offs_xpu, out_dtype=torch.bfloat16)
        torch.xpu.synchronize()

        self.assertEqual(out_xpu.shape, (total_M, N))

        out_cpu = out_xpu.cpu()
        torch.testing.assert_close(
            out_cpu, ref_cpu, atol=self.atol, rtol=self.rtol,
            msg="Dispatch unbalanced XPU vs CPU mismatch")


if __name__ == "__main__":
    # Print test config
    print("=" * 70)
    print("Llama 4 Scout (109B/17B-16E) Model Shape Validation")
    print(f"  hidden_size (K) = {HIDDEN_SIZE}")
    print(f"  intermediate_size (N) = {INTERMEDIATE_SIZE}")
    print(f"  num_experts (E) = {NUM_EXPERTS}")
    print(f"  dtype = float8_e4m3fn → bfloat16")
    print(f"  device = xpu")
    print("=" * 70)
    unittest.main(verbosity=2)
