#!/usr/bin/env python3
"""
Quick verification - Check if cuda_ops is compiled and list functions
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("CUDA_OPS VERIFICATION")
print("="*80 + "\n")

# Try to import cuda_ops
try:
    import cuda_ops
    print("✅ cuda_ops extension imported successfully!\n")
except ImportError as e:
    print(f"❌ Failed to import cuda_ops: {e}\n")
    print("💡 You need to compile the extension first:")
    print("\n   Option 1 (install):")
    print("   python setup.py install\n")
    print("   Option 2 (build in-place):")
    print("   python setup.py build_ext --inplace\n")
    sys.exit(1)

# List all available functions
print("📋 Available functions in cuda_ops module:")
print("-" * 60)

functions = []
for attr in dir(cuda_ops):
    if not attr.startswith('_'):
        obj = getattr(cuda_ops, attr)
        if callable(obj):
            functions.append(attr)
            print(f"   ✓ {attr}")

print(f"\n✅ Found {len(functions)} functions\n")

# Categorize functions
print("🎯 Function Categories:")
print("-" * 60)

gelu_funcs = [f for f in functions if 'gelu' in f.lower()]
swish_funcs = [f for f in functions if 'swish' in f.lower()]
ln_funcs = [f for f in functions if 'layer' in f.lower() or 'ln' in f.lower()]
loss_funcs = [f for f in functions if 'loss' in f.lower()]
fusion_funcs = [f for f in functions if 'fuse' in f.lower()]

if gelu_funcs:
    print(f"\n   GELU functions:")
    for f in gelu_funcs:
        print(f"      • {f}")

if swish_funcs:
    print(f"\n   Swish functions:")
    for f in swish_funcs:
        print(f"      • {f}")

if ln_funcs:
    print(f"\n   LayerNorm functions:")
    for f in ln_funcs:
        print(f"      • {f}")

if loss_funcs:
    print(f"\n   Loss functions:")
    for f in loss_funcs:
        print(f"      • {f}")

if fusion_funcs:
    print(f"\n   ⭐ FUSION functions:")
    for f in fusion_funcs:
        print(f"      • {f}")
        # Highlight Fusion 3
        if 'ln' in f.lower() and 'gelu' in f.lower() and 'swish' in f.lower():
            print(f"         👉 This is FUSION 3! (2.94× speedup)")

print("\n" + "="*80)
print("✅ VERIFICATION COMPLETE")
print("="*80)
print("\n📝 Next steps:")
print("   1. Use the function names shown above in your profiling scripts")
print("   2. Run profiling with: python profiling/nsight_profiling_wsl.py benchmarks/profile_fusion3_fixed.py")
print()