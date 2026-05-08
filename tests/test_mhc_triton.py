#!/usr/bin/env python3
"""Test Triton mhc_pre/mhc_post kernels against PyTorch reference."""

import torch
import sys
sys.path.insert(0, "/home/majian/vllm-workspace/vllm-project")

from vllm.v1.attention.ops.deepseek_v4_ops.mhc_xpu import (
    mhc_pre_xpu_torch,
    mhc_post_xpu_torch,
)
from vllm.v1.attention.ops.deepseek_v4_ops.mhc_xpu_triton import (
    mhc_pre_xpu_triton,
    mhc_post_xpu_triton,
)


def test_mhc_pre(device="xpu", N=4, hc=4, H=4096, sinkhorn_repeat=20):
    """Test mhc_pre Triton kernel against PyTorch reference."""
    print(f"\n=== test_mhc_pre N={N}, hc={hc}, H={H}, sinkhorn={sinkhorn_repeat} ===")

    torch.manual_seed(42)
    hc3 = hc * 2 + hc * hc
    hcH = hc * H

    # Create inputs
    residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc3, hcH, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn(hc3, dtype=torch.float32, device=device) * 0.01

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 2.0

    # Reference
    post_ref, comb_ref, li_ref = mhc_pre_xpu_torch(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    )

    # Triton
    post_tri, comb_tri, li_tri = mhc_pre_xpu_triton(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    )

    # Compare
    def check(name, ref, tri, atol=1e-4, rtol=1e-3):
        diff = (ref.float() - tri.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_err = (diff / (ref.float().abs() + 1e-8)).max().item()
        ok = torch.allclose(ref.float(), tri.float(), atol=atol, rtol=rtol)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  rel_err={rel_err:.2e}")
        if not ok:
            print(f"    ref range: [{ref.float().min().item():.4f}, {ref.float().max().item():.4f}]")
            print(f"    tri range: [{tri.float().min().item():.4f}, {tri.float().max().item():.4f}]")
        return ok

    all_ok = True
    all_ok &= check("post_mix", post_ref, post_tri)
    all_ok &= check("comb_mix", comb_ref, comb_tri, atol=1e-3, rtol=1e-2)
    all_ok &= check("layer_input", li_ref, li_tri, atol=5e-3, rtol=5e-3)

    return all_ok


def test_mhc_post(device="xpu", N=4, hc=4, H=4096):
    """Test mhc_post Triton kernel against PyTorch reference."""
    print(f"\n=== test_mhc_post N={N}, hc={hc}, H={H} ===")

    torch.manual_seed(123)

    x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
    residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn(N, hc, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(N, hc, hc, dtype=torch.float32, device=device) * 0.1

    # Reference
    out_ref = mhc_post_xpu_torch(x, residual, post_layer_mix, comb_res_mix)

    # Triton
    out_tri = mhc_post_xpu_triton(x, residual, post_layer_mix, comb_res_mix)

    diff = (out_ref.float() - out_tri.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = (diff / (out_ref.float().abs() + 1e-8)).max().item()
    ok = torch.allclose(out_ref.float(), out_tri.float(), atol=5e-3, rtol=5e-3)
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  output: {status}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  rel_err={rel_err:.2e}")
    if not ok:
        print(f"    ref range: [{out_ref.float().min().item():.4f}, {out_ref.float().max().item():.4f}]")
        print(f"    tri range: [{out_tri.float().min().item():.4f}, {out_tri.float().max().item():.4f}]")

    return ok


def bench_mhc_pre(device="xpu", N=4, hc=4, H=4096, sinkhorn_repeat=20, warmup=10, iters=100):
    """Benchmark mhc_pre: Triton vs PyTorch eager."""
    print(f"\n=== bench_mhc_pre N={N}, hc={hc}, H={H} ===")

    torch.manual_seed(42)
    hc3 = hc * 2 + hc * hc
    hcH = hc * H

    residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc3, hcH, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn(hc3, dtype=torch.float32, device=device) * 0.01

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 2.0

    args = (residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)

    # Warmup
    for _ in range(warmup):
        mhc_pre_xpu_torch(*args)
        mhc_pre_xpu_triton(*args)
    torch.xpu.synchronize()

    # Bench PyTorch
    torch.xpu.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        mhc_pre_xpu_torch(*args)
    torch.xpu.synchronize()
    t_torch = (time.perf_counter() - t0) / iters * 1000

    # Bench Triton
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mhc_pre_xpu_triton(*args)
    torch.xpu.synchronize()
    t_triton = (time.perf_counter() - t0) / iters * 1000

    print(f"  PyTorch eager: {t_torch:.3f} ms/call")
    print(f"  Triton fused:  {t_triton:.3f} ms/call")
    print(f"  Speedup: {t_torch/t_triton:.2f}x")


if __name__ == "__main__":
    device = "xpu"
    print(f"Device: {device}")
    print(f"torch.xpu.is_available(): {torch.xpu.is_available()}")

    ok1 = test_mhc_pre(device, N=4)
    ok2 = test_mhc_pre(device, N=1)  # single token
    ok3 = test_mhc_post(device, N=4)
    ok4 = test_mhc_post(device, N=1)

    print("\n" + "="*60)
    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")

    if all_ok:
        bench_mhc_pre(device, N=4)

    sys.exit(0 if all_ok else 1)
