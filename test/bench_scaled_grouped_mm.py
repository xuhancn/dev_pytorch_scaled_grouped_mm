"""
Performance benchmark for SYCL scaled_grouped_mm extension.

Run from /tmp to avoid torch/_C import conflicts:
    cd /tmp && python <repo>/test/bench_scaled_grouped_mm.py
"""

import time
import torch
import scaled_grouped_mm_sycl


def bench_3d_3d(G, M, N, K, warmup=10, iters=50):
    """Benchmark 3D x 3D (batched) mode."""
    device = "xpu"
    fp8 = torch.float8_e4m3fn

    a = torch.randn(G, M, K, device=device).to(fp8)
    b_phys = torch.randn(G, N, K, device=device).to(fp8)
    b_t = b_phys.transpose(-2, -1)
    scale_a = torch.rand(G, M, device=device, dtype=torch.float32) + 0.1
    scale_b = torch.rand(G, N, device=device, dtype=torch.float32) + 0.1

    # Warmup
    for _ in range(warmup):
        scaled_grouped_mm_sycl.scaled_grouped_mm(a, b_t, scale_a, scale_b)
    torch.xpu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        scaled_grouped_mm_sycl.scaled_grouped_mm(a, b_t, scale_a, scale_b)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iters * 1000
    total_flops = G * 2 * M * N * K
    tflops = total_flops / (avg_ms / 1000) / 1e12
    return avg_ms, tflops


def bench_2d_3d(G, M, N, K, warmup=10, iters=50):
    """Benchmark 2D x 3D (ragged A / MoE) mode."""
    device = "xpu"
    fp8 = torch.float8_e4m3fn
    total_M = M * G

    a = torch.randn(total_M, K, device=device).to(fp8)
    b_phys = torch.randn(G, N, K, device=device).to(fp8)
    b_t = b_phys.transpose(-2, -1)
    scale_a = torch.rand(total_M, device=device, dtype=torch.float32) + 0.1
    scale_b = torch.rand(G, N, device=device, dtype=torch.float32) + 0.1
    offs = torch.arange(M, total_M + 1, M, device=device, dtype=torch.int32)

    for _ in range(warmup):
        scaled_grouped_mm_sycl.scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        scaled_grouped_mm_sycl.scaled_grouped_mm(a, b_t, scale_a, scale_b, offs=offs)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iters * 1000
    total_flops = G * 2 * M * N * K
    tflops = total_flops / (avg_ms / 1000) / 1e12
    return avg_ms, tflops


if __name__ == "__main__":
    print("=" * 70)
    print("Benchmark: SYCL scaled_grouped_mm (FP8 -> BF16)")
    print("=" * 70)

    configs = [
        # (G, M, N, K)
        (4, 16, 32, 64),
        (4, 64, 128, 128),
        (8, 64, 128, 128),
        (8, 128, 256, 256),
        (16, 128, 256, 256),
    ]

    print(f"\n{'Mode':<10} {'G':>4} {'M':>5} {'N':>5} {'K':>5}  {'ms':>8}  {'TFLOPS':>8}")
    print("-" * 55)

    for G, M, N, K in configs:
        avg_ms, tflops = bench_3d_3d(G, M, N, K)
        print(f"{'3D x 3D':<10} {G:>4} {M:>5} {N:>5} {K:>5}  {avg_ms:>8.3f}  {tflops:>8.4f}")

    print()
    for G, M, N, K in configs:
        avg_ms, tflops = bench_2d_3d(G, M, N, K)
        print(f"{'2D x 3D':<10} {G:>4} {M:>5} {N:>5} {K:>5}  {avg_ms:>8.3f}  {tflops:>8.4f}")
