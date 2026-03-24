"""
Accuracy tests for SYCL scaled_grouped_mm extension.

Verifies FP8 x FP8 -> BF16 grouped GEMM with rowwise float32 scaling
against a per-group reference: manual dequantize + matmul.

Run from /tmp to avoid torch/_C import conflicts:
    cd /tmp && python <repo>/test/test_scaled_grouped_mm.py
"""

import unittest
import torch

# Import our SYCL extension
import scaled_grouped_mm_sycl


def reference_scaled_mm_per_group(a_fp8, b_fp8_t, scale_a, scale_b):
    """Reference: dequantize FP8 -> float32, apply scales, matmul, cast to BF16.

    Args:
        a_fp8: (M, K) FP8 tensor, row-major
        b_fp8_t: (K, N) FP8 tensor — this is the TRANSPOSED view.
                 Physical layout is (N, K) row-major.
        scale_a: (M,) float32 rowwise scale for A
        scale_b: (N,) float32 rowwise scale for B (one per column of logical B)
    Returns:
        (M, N) BF16 tensor
    """
    # Dequantize A: (M, K) * (M, 1) -> (M, K)
    a_f32 = a_fp8.float() * scale_a.unsqueeze(-1)
    # Convert B to physical layout (N, K), dequantize, then transpose for matmul
    b_phys = b_fp8_t.t().contiguous()                  # (N, K) row-major
    b_f32 = b_phys.float() * scale_b.unsqueeze(-1)     # (N, K) * (N, 1)
    # matmul: (M, K) @ (N, K).T = (M, K) @ (K, N)
    out = a_f32 @ b_f32.t()
    return out.to(torch.bfloat16)


class TestScaledGroupedMM(unittest.TestCase):
    """Test all 4 input modes of scaled_grouped_mm."""

    device = "xpu"
    fp8_dtype = torch.float8_e4m3fn
    # Tolerances are wider than CUDA because our approach dequantizes FP8→BF16
    # before the GEMM, losing precision vs the reference which works in float32.
    atol = 2e-1
    rtol = 5e-2

    def _make_fp8(self, *shape):
        """Create a random FP8 tensor on XPU."""
        return torch.randn(*shape, device=self.device).to(self.fp8_dtype)

    def _make_scale(self, *shape):
        """Create a random positive float32 scale on XPU."""
        return torch.rand(*shape, device=self.device, dtype=torch.float32) + 0.1

    def test_3d_3d(self):
        """3D x 3D: batched grouped GEMM, no offsets."""
        m, n, k, G = 16, 32, 64, 4

        # A: (G, M, K) row-major FP8
        a = self._make_fp8(G, m, k)
        # B: (G, N, K) physical, transposed to (G, K, N) logical
        b_phys = self._make_fp8(G, n, k)
        b_t = b_phys.transpose(-2, -1)  # (G, K, N) with col-major strides

        scale_a = self._make_scale(G, m)
        scale_b = self._make_scale(G, n)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        self.assertEqual(out.shape, (G, m, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        # Verify each group
        for g in range(G):
            ref = reference_scaled_mm_per_group(
                a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"3D x 3D group {g} mismatch")

    def test_2d_3d(self):
        """2D x 3D: ragged A (MoE pattern) with offsets along M."""
        m, n, k, G = 16, 32, 64, 4
        total_M = m * G

        # A: (total_M, K) row-major FP8
        a = self._make_fp8(total_M, k)
        # B: (G, N, K) physical, transposed
        b_phys = self._make_fp8(G, n, k)
        b_t = b_phys.transpose(-2, -1)  # (G, K, N)

        scale_a = self._make_scale(total_M)
        scale_b = self._make_scale(G, n)

        offs = torch.arange(m, total_M + 1, m, device=self.device, dtype=torch.int32)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (total_M, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        # Verify each group
        row_start = 0
        for g in range(G):
            row_end = offs[g].item()
            ref = reference_scaled_mm_per_group(
                a[row_start:row_end], b_t[g],
                scale_a[row_start:row_end], scale_b[g])
            torch.testing.assert_close(
                out[row_start:row_end], ref, atol=self.atol, rtol=self.rtol,
                msg=f"2D x 3D group {g} mismatch")
            row_start = row_end

    def test_3d_2d(self):
        """3D x 2D: ragged B with offsets along N."""
        m, n, k, G = 16, 32, 64, 4
        total_N = n * G

        # A: (G, M, K) row-major FP8
        a = self._make_fp8(G, m, k)
        # B: (total_N, K) physical (transposed), logical (K, total_N)
        b_phys = self._make_fp8(total_N, k)
        b_t = b_phys.t()  # (K, total_N) with col-major strides

        scale_a = self._make_scale(G, m)
        scale_b = self._make_scale(total_N)

        offs = torch.arange(n, total_N + 1, n, device=self.device, dtype=torch.int32)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (m, total_N))
        self.assertEqual(out.dtype, torch.bfloat16)

        # Verify each group
        col_start = 0
        for g in range(G):
            col_end = offs[g].item()
            ref = reference_scaled_mm_per_group(
                a[g], b_t[:, col_start:col_end],
                scale_a[g], scale_b[col_start:col_end])
            torch.testing.assert_close(
                out[:, col_start:col_end], ref, atol=self.atol, rtol=self.rtol,
                msg=f"3D x 2D group {g} mismatch")
            col_start = col_end

    def test_2d_2d(self):
        """2D x 2D: ragged K with offsets along K dimension."""
        m, n, k, G = 16, 32, 64, 4
        total_K = k * G

        # A: (M, total_K) row-major FP8
        a = self._make_fp8(m, total_K)
        # B: (N, total_K) physical (transposed), logical (total_K, N)
        b_phys = self._make_fp8(n, total_K)
        b_t = b_phys.t()  # (total_K, N) with col-major strides

        scale_a = self._make_scale(m * G)
        scale_b = self._make_scale(n * G)

        offs = torch.arange(k, total_K + 1, k, device=self.device, dtype=torch.int32)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b, offs=offs)

        self.assertEqual(out.shape, (G, m, n))
        self.assertEqual(out.dtype, torch.bfloat16)

        # Verify each group
        k_start = 0
        for g in range(G):
            k_end = offs[g].item()
            a_g = a[:, k_start:k_end]
            # b_t[:, :] slice: (total_K, N) -> (K_g, N)
            # Physical b_phys: (N, total_K), columns k_start:k_end -> (N, K_g)
            b_g_phys = b_phys[:, k_start:k_end]  # (N, K_g)
            b_g_t = b_g_phys.t()  # (K_g, N) — the transposed view
            sa_g = scale_a[g * m:(g + 1) * m]
            sb_g = scale_b[g * n:(g + 1) * n]
            ref = reference_scaled_mm_per_group(a_g, b_g_t, sa_g, sb_g)
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"2D x 2D group {g} mismatch")
            k_start = k_end

    def test_single_group_3d_3d(self):
        """Edge case: single group."""
        m, n, k = 16, 32, 64

        a = self._make_fp8(1, m, k)
        b_phys = self._make_fp8(1, n, k)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(1, m)
        scale_b = self._make_scale(1, n)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        ref = reference_scaled_mm_per_group(
            a[0], b_t[0], scale_a[0], scale_b[0])
        torch.testing.assert_close(
            out[0], ref, atol=self.atol, rtol=self.rtol)

    def test_min_k_size(self):
        """Edge case: minimum K=16."""
        m, n, k, G = 16, 32, 16, 2

        a = self._make_fp8(G, m, k)
        b_phys = self._make_fp8(G, n, k)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(G, m)
        scale_b = self._make_scale(G, n)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        for g in range(G):
            ref = reference_scaled_mm_per_group(
                a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol)

    def test_larger_shapes(self):
        """Larger shapes for stress testing."""
        m, n, k, G = 64, 128, 128, 8

        a = self._make_fp8(G, m, k)
        b_phys = self._make_fp8(G, n, k)
        b_t = b_phys.transpose(-2, -1)

        scale_a = self._make_scale(G, m)
        scale_b = self._make_scale(G, n)

        out = scaled_grouped_mm_sycl.scaled_grouped_mm(
            a, b_t, scale_a, scale_b)

        for g in range(G):
            ref = reference_scaled_mm_per_group(
                a[g], b_t[g], scale_a[g], scale_b[g])
            torch.testing.assert_close(
                out[g], ref, atol=self.atol, rtol=self.rtol,
                msg=f"Large shape group {g} mismatch")


class TestScaledGroupedMMValidation(unittest.TestCase):
    """Test input validation catches errors."""

    device = "xpu"
    fp8_dtype = torch.float8_e4m3fn

    def test_wrong_a_dtype(self):
        a = torch.randn(4, 16, 64, device=self.device, dtype=torch.bfloat16)
        b = torch.randn(4, 32, 64, device=self.device).to(self.fp8_dtype).transpose(-2, -1)
        sa = torch.rand(4, 16, device=self.device)
        sb = torch.rand(4, 32, device=self.device)
        with self.assertRaises(RuntimeError):
            scaled_grouped_mm_sycl.scaled_grouped_mm(a, b, sa, sb)

    def test_missing_offs_for_2d(self):
        a = torch.randn(64, 64, device=self.device).to(self.fp8_dtype)
        b = torch.randn(4, 32, 64, device=self.device).to(self.fp8_dtype).transpose(-2, -1)
        sa = torch.rand(64, device=self.device)
        sb = torch.rand(4, 32, device=self.device)
        with self.assertRaises(RuntimeError):
            scaled_grouped_mm_sycl.scaled_grouped_mm(a, b, sa, sb)

    def test_bias_not_supported(self):
        a = torch.randn(4, 16, 64, device=self.device).to(self.fp8_dtype)
        b = torch.randn(4, 32, 64, device=self.device).to(self.fp8_dtype).transpose(-2, -1)
        sa = torch.rand(4, 16, device=self.device)
        sb = torch.rand(4, 32, device=self.device)
        bias = torch.randn(4, 32, device=self.device, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            scaled_grouped_mm_sycl.scaled_grouped_mm(a, b, sa, sb, bias=bias)


if __name__ == "__main__":
    unittest.main()
