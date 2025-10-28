import time, json, argparse
import torch
from ops.cuda_ops import layernorm as cuda_ln, gelu as cuda_gelu, swish as cuda_swish, fused_ln_gelu as cuda_ln_gelu
from ops.triton_ops import layernorm as tri_ln, gelu as tri_gelu, swish as tri_swish, fused_ln_gelu as tri_ln_gelu

device = "cuda" if torch.cuda.is_available() else "cpu"

OPS = {
  "layernorm": {"cuda": lambda x,g,b: cuda_ln(x,g,b,1e-5),
                "triton": lambda x,g,b: tri_ln(x,g,b,1e-5)},
  "gelu": {"cuda": lambda x, *_: cuda_gelu(x),
           "triton": lambda x, *_: tri_gelu(x)},
  "swish": {"cuda": lambda x, *_: cuda_swish(x),
            "triton": lambda x, *_: tri_swish(x)},
  "ln_gelu": {"cuda": lambda x,g,b: cuda_ln_gelu(x,g,b,1e-5),
              "triton": lambda x,g,b: tri_ln_gelu(x,g,b,1e-5)},
}

def bench(op, impl, rows, cols, iters=100, warmup=10):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)
    g = torch.randn(cols, device=device, dtype=torch.float32)
    b = torch.randn(cols, device=device, dtype=torch.float32)

    fn = OPS[op][impl]
    # warmup
    for _ in range(warmup):
        y = fn(x, g, b)
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        y = fn(x, g, b)
    torch.cuda.synchronize()
    t1 = time.time()
    return 1000.0 * (t1 - t0) / iters  # ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=OPS.keys(), required=True)
    ap.add_argument("--impl", choices=["cuda","triton"], required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--cols", type=int, nargs="+", default=[256,512,1024])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=str, default="profiling/cuda_vs_triton_metrics.json")
    args = ap.parse_args()

    results = []
    for r in args.rows:
        for c in args.cols:
            ms = bench(args.op, args.impl, r, c, args.iters)
            results.append({"op": args.op, "impl": args.impl, "rows": r, "cols": c, "kernel_ms": ms})
            print(f"{args.op}/{args.impl}  rows={r} cols={c}  {ms:.3f} ms")

    # append to JSON log
    import os, json
    os.makedirs("profiling", exist_ok=True)
    path = args.out
    existing = []
    if os.path.exists(path):
        try:
            existing = json.load(open(path))
        except Exception:
            existing = []
    existing.extend(results)
    json.dump(existing, open(path, "w"), indent=2)

if __name__ == "__main__":
    main()
