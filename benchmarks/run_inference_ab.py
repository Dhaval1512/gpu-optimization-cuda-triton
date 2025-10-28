import time, torch, torch.nn.functional as F
from ops.cuda_ops import fused_ln_gelu as cuda_ln_gelu
from ops.triton_ops import fused_ln_gelu as tri_ln_gelu

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_once(batch=128, ch=32, h=28, w=28, impl="cuda"):
    x = torch.randn(batch, ch, h, w, device=device)
    cols = ch*h*w
    X = x.view(batch, -1)
    gamma = torch.randn(cols, device=device)
    beta  = torch.randn(cols, device=device)

    # unfused
    torch.cuda.synchronize(); t0=time.time()
    y_ref = F.gelu(F.layer_norm(X, (cols,), gamma, beta, 1e-5))
    torch.cuda.synchronize(); t1=time.time()

    # fused
    fn = cuda_ln_gelu if impl=="cuda" else tri_ln_gelu
    torch.cuda.synchronize(); t2=time.time()
    y_fused = fn(X, gamma, beta, 1e-5)
    torch.cuda.synchronize(); t3=time.time()

    print(f"unfused_ms={(t1-t0)*1000:.3f}  fused_{impl}_ms={(t3-t2)*1000:.3f}  close={torch.allclose(y_ref,y_fused,1e-4,1e-4)}")

if __name__=="__main__":
    run_once(128,32,28,28,"cuda")
    run_once(128,32,28,28,"triton")
