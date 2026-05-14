"""
Llama 4 MoE production-scale benchmark and correctness test.

Tests both _scaled_grouped_mm and _grouped_mm with real Llama 4 shapes:
  - Scout: 16 experts, d_model=5120, ffn_dim=14336
  - Maverick: 128 experts, d_model=5120, ffn_dim=8192

Run from /tmp:
    cd /tmp && python <repo>/test/bench_llama4.py
"""

import time
import torch
import math

DEVICE = "xpu"
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
WARMUP = 5
ITERS = 20


def reference_scaled_grouped_mm_3d(a_fp8, b_fp8_t, scale_a, scale_b):
    """Per-group dequant + matmul reference for 3D x 3D."""
    G = a_fp8.shape[0]
    results = []
    for g in range(G):
        a_bf16 = a_fp8[g].to(BF16) * scale_a[g].unsqueeze(-1).to(BF16)
        b_bf16 = b_fp8_t[g].to(BF16) * scale_b[g].unsqueeze(0).to(BF16)
        results.append(a_bf16 @ b_bf16)
    return torch.stack(results)


def reference_scaled_grouped_mm_2d3d(a_fp8, b_fp8_t, scale_a, scale_b, offs):
    """Per-group dequant + matmul reference for 2D x 3D (MoE ragged)."""
    G = b_fp8_t.shape[0]
    results = []
    prev = 0
    for g in range(G):
        end = offs[g].item()
        a_g = a_fp8[prev:end].to(BF16) * scale_a[prev:end].unsqueeze(-1).to(BF16)
        b_g = b_fp8_t[g].to(BF16) * scale_b[g].unsqueeze(0).to(BF16)
        results.append(a_g @ b_g)
        prev = end
    return torch.cat(results, dim=0)


def reference_grouped_mm_3d(a, b):
    """Per-group matmul reference for _grouped_mm 3D x 3D."""
    G = a.shape[0]
    return torch.stack([a[g] @ b[g] for g in range(G)])


def check_correctness(out, ref, atol=5e-2, rtol=5e-2):
    """Check correctness and return max absolute error."""
    max_err = (out.float() - ref.float()).abs().max().item()
    mean_err = (out.float() - ref.float()).abs().mean().item()
    # Use relative error metric: max_err / max(|ref|) should be small
    ref_scale = ref.float().abs().max().item()
    rel_max_err = max_err / max(ref_scale, 1e-6)
    # For FP8 dequant path, relative error < 5% is good
    ok = rel_max_err < 0.05 or torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    return status, max_err, mean_err


def bench_scaled_grouped_mm_3d(G, M, N, K):
    """Benchmark _scaled_grouped_mm 3D x 3D (uniform tokens per expert)."""
    a = torch.randn(G, M, K, device=DEVICE).to(FP8)
    b_phys = torch.randn(G, N, K, device=DEVICE).to(FP8)
    b_t = b_phys.transpose(-2, -1)
    scale_a = torch.rand(G, M, device=DEVICE, dtype=torch.float32) + 0.1
    scale_b = torch.rand(G, N, device=DEVICE, dtype=torch.float32) + 0.1

    # Correctness
    out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
    ref = reference_scaled_grouped_mm_3d(a, b_t, scale_a, scale_b)
    status, max_err, mean_err = check_correctness(out, ref)

    # Warmup
    for _ in range(WARMUP):
        torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
    torch.xpu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(ITERS):
        torch._scaled_grouped_mm(a, b_t, scale_a, scale_b)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / ITERS * 1000
    total_flops = G * 2 * M * N * K
    tflops = total_flops / (avg_ms / 1000) / 1e12
    mem_mb = (a.nbytes + b_phys.nbytes + scale_a.nbytes + scale_b.nbytes) / 1e6
    return avg_ms, tflops, mem_mb, status, max_err, mean_err


def bench_scaled_grouped_mm_2d3d(G, tokens_per_expert, N, K):
    """Benchmark _scaled_grouped_mm 2D x 3D (MoE ragged pattern)."""
    total_M = tokens_per_expert * G
    a = torch.randn(total_M, K, device=DEVICE).to(FP8)
    b_phys = torch.randn(G, N, K, device=DEVICE).to(FP8)
    b_t = b_phys.transpose(-2, -1)
    scale_a = torch.rand(total_M, device=DEVICE, dtype=torch.float32) + 0.1
    scale_b = torch.rand(G, N, device=DEVICE, dtype=torch.float32) + 0.1
    offs = torch.arange(tokens_per_expert, total_M + 1, tokens_per_expert,
                        device=DEVICE, dtype=torch.int32)

    # Correctness
    out = torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
    ref = reference_scaled_grouped_mm_2d3d(a, b_t, scale_a, scale_b, offs)
    status, max_err, mean_err = check_correctness(out, ref)

    # Warmup
    for _ in range(WARMUP):
        torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
    torch.xpu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(ITERS):
        torch._scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / ITERS * 1000
    total_flops = G * 2 * tokens_per_expert * N * K
    tflops = total_flops / (avg_ms / 1000) / 1e12
    mem_mb = (a.nbytes + b_phys.nbytes + scale_a.nbytes + scale_b.nbytes) / 1e6
    return avg_ms, tflops, mem_mb, status, max_err, mean_err


def bench_grouped_mm_3d(G, M, N, K):
    """Benchmark _grouped_mm 3D x 3D."""
    a = torch.randn(G, M, K, device=DEVICE, dtype=BF16)
    b = torch.randn(G, K, N, device=DEVICE, dtype=BF16)

    # Correctness
    out = torch._grouped_mm(a, b)
    ref = reference_grouped_mm_3d(a, b)
    status, max_err, mean_err = check_correctness(out, ref, atol=1e-1, rtol=1e-1)

    # Warmup
    for _ in range(WARMUP):
        torch._grouped_mm(a, b)
    torch.xpu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(ITERS):
        torch._grouped_mm(a, b)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / ITERS * 1000
    total_flops = G * 2 * M * N * K
    tflops = total_flops / (avg_ms / 1000) / 1e12
    mem_mb = (a.nbytes + b.nbytes) / 1e6
    return avg_ms, tflops, mem_mb, status, max_err, mean_err


def run_all():
    print("=" * 90)
    print("Llama 4 MoE Production-Scale Benchmark (Intel Arc B580)")
    print("=" * 90)
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    results = []

    # ======== _scaled_grouped_mm (FP8 path) ========
    print("─" * 90)
    print("_scaled_grouped_mm (FP8 → BF16 dequant + grouped GEMM)")
    print("─" * 90)

    # Llama 4 Scout: 16 experts
    scout_configs_3d = [
        # label, G, M_per_expert, N, K
        ("Scout gate/up 3D", 16, 32, 14336, 5120),
        ("Scout gate/up 3D", 16, 64, 14336, 5120),
        ("Scout gate/up 3D", 16, 128, 14336, 5120),
        ("Scout gate/up 3D", 16, 256, 14336, 5120),
        ("Scout down 3D", 16, 32, 5120, 14336),
        ("Scout down 3D", 16, 64, 5120, 14336),
        ("Scout down 3D", 16, 128, 5120, 14336),
        ("Scout down 3D", 16, 256, 5120, 14336),
    ]

    scout_configs_2d3d = [
        # label, G, tokens_per_expert, N, K
        ("Scout gate/up MoE", 16, 32, 14336, 5120),
        ("Scout gate/up MoE", 16, 64, 14336, 5120),
        ("Scout gate/up MoE", 16, 128, 14336, 5120),
        ("Scout gate/up MoE", 16, 256, 14336, 5120),
        ("Scout down MoE", 16, 32, 5120, 14336),
        ("Scout down MoE", 16, 64, 5120, 14336),
        ("Scout down MoE", 16, 128, 5120, 14336),
        ("Scout down MoE", 16, 256, 5120, 14336),
    ]

    # Llama 4 Maverick: 128 experts (only 2D x 3D makes sense for 128 experts)
    # NOTE: Full Maverick shapes (128×8192×5120) need ~10GB+ for B alone.
    # Arc B580 has 12GB VRAM. We test with reduced expert counts to fit.
    # CI servers with more VRAM (e.g., Data Center GPU Max) can run full shapes.
    maverick_configs_2d3d = [
        ("Mav gate/up MoE", 16, 32, 8192, 5120),   # 16 of 128 experts (fits in 12GB)
        ("Mav gate/up MoE", 16, 64, 8192, 5120),
        ("Mav gate/up MoE", 16, 128, 8192, 5120),
        ("Mav down MoE", 16, 32, 5120, 8192),
        ("Mav down MoE", 16, 64, 5120, 8192),
        ("Mav down MoE", 16, 128, 5120, 8192),
    ]

    header = f"{'Label':<22} {'Mode':<8} {'G':>4} {'M':>5} {'N':>6} {'K':>6}  {'ms':>8}  {'TFLOPS':>7}  {'MB':>6}  {'Chk':>4}  {'MaxErr':>8}"
    print(header)
    print("-" * len(header))

    # Scout 3D x 3D
    for label, G, M, N, K in scout_configs_3d:
        try:
            avg_ms, tflops, mem_mb, status, max_err, mean_err = bench_scaled_grouped_mm_3d(G, M, N, K)
            line = f"{label:<22} {'3Dx3D':<8} {G:>4} {M:>5} {N:>6} {K:>6}  {avg_ms:>8.2f}  {tflops:>7.3f}  {mem_mb:>6.1f}  {status:>4}  {max_err:>8.4f}"
            print(line)
            results.append((label, "3Dx3D", G, M, N, K, avg_ms, tflops, mem_mb, status, max_err))
        except Exception as e:
            line = f"{label:<22} {'3Dx3D':<8} {G:>4} {M:>5} {N:>6} {K:>6}  ERROR: {e}"
            print(line)
            results.append((label, "3Dx3D", G, M, N, K, 0, 0, 0, "ERR", 0))

    print()

    # Scout 2D x 3D (MoE pattern)
    for label, G, tpe, N, K in scout_configs_2d3d:
        try:
            avg_ms, tflops, mem_mb, status, max_err, mean_err = bench_scaled_grouped_mm_2d3d(G, tpe, N, K)
            line = f"{label:<22} {'2Dx3D':<8} {G:>4} {tpe:>5} {N:>6} {K:>6}  {avg_ms:>8.2f}  {tflops:>7.3f}  {mem_mb:>6.1f}  {status:>4}  {max_err:>8.4f}"
            print(line)
            results.append((label, "2Dx3D", G, tpe, N, K, avg_ms, tflops, mem_mb, status, max_err))
        except Exception as e:
            line = f"{label:<22} {'2Dx3D':<8} {G:>4} {tpe:>5} {N:>6} {K:>6}  ERROR: {e}"
            print(line)
            results.append((label, "2Dx3D", G, tpe, N, K, 0, 0, 0, "ERR", 0))

    print()

    # Maverick 2D x 3D
    for label, G, tpe, N, K in maverick_configs_2d3d:
        try:
            avg_ms, tflops, mem_mb, status, max_err, mean_err = bench_scaled_grouped_mm_2d3d(G, tpe, N, K)
            line = f"{label:<22} {'2Dx3D':<8} {G:>4} {tpe:>5} {N:>6} {K:>6}  {avg_ms:>8.2f}  {tflops:>7.3f}  {mem_mb:>6.1f}  {status:>4}  {max_err:>8.4f}"
            print(line)
            results.append((label, "2Dx3D", G, tpe, N, K, avg_ms, tflops, mem_mb, status, max_err))
        except Exception as e:
            line = f"{label:<22} {'2Dx3D':<8} {G:>4} {tpe:>5} {N:>6} {K:>6}  ERROR: {e}"
            print(line)
            results.append((label, "2Dx3D", G, tpe, N, K, 0, 0, 0, "ERR", 0))

    # ======== _grouped_mm (BF16 path) ========
    torch.xpu.empty_cache()
    print()
    print("─" * 90)
    print("_grouped_mm (BF16 grouped GEMM, no dequant)")
    print("─" * 90)
    print(header)
    print("-" * len(header))

    grouped_mm_configs = [
        ("Scout gate/up 3D", 16, 32, 14336, 5120),
        ("Scout gate/up 3D", 16, 64, 14336, 5120),
        ("Scout gate/up 3D", 16, 128, 14336, 5120),
        ("Scout down 3D", 16, 32, 5120, 14336),
        ("Scout down 3D", 16, 64, 5120, 14336),
        ("Scout down 3D", 16, 128, 5120, 14336),
    ]

    for label, G, M, N, K in grouped_mm_configs:
        try:
            avg_ms, tflops, mem_mb, status, max_err, mean_err = bench_grouped_mm_3d(G, M, N, K)
            line = f"{label:<22} {'3Dx3D':<8} {G:>4} {M:>5} {N:>6} {K:>6}  {avg_ms:>8.2f}  {tflops:>7.3f}  {mem_mb:>6.1f}  {status:>4}  {max_err:>8.4f}"
            print(line)
            results.append((label + " (bf16)", "3Dx3D", G, M, N, K, avg_ms, tflops, mem_mb, status, max_err))
        except Exception as e:
            line = f"{label:<22} {'3Dx3D':<8} {G:>4} {M:>5} {N:>6} {K:>6}  ERROR: {e}"
            print(line)
            results.append((label + " (bf16)", "3Dx3D", G, M, N, K, 0, 0, 0, "ERR", 0))

    # Summary
    print()
    print("=" * 90)
    total = len(results)
    passed = sum(1 for r in results if r[9] == "PASS")
    failed = sum(1 for r in results if r[9] == "FAIL")
    errors = sum(1 for r in results if r[9] == "ERR")
    print(f"Total: {total} | PASS: {passed} | FAIL: {failed} | ERROR: {errors}")
    if passed == total:
        print("✅ All Llama 4 shapes: CORRECTNESS VERIFIED")
    print("=" * 90)

    return results


if __name__ == "__main__":
    results = run_all()
