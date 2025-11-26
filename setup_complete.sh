#!/bin/bash

echo "========================================="
echo "  Setting up CUDA Environment"
echo "========================================="

# Activate virtual environment
if [ -f ~/gpu_env/bin/activate ]; then
    source ~/gpu_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Get Python site-packages path
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "📦 Site packages: $SITE_PACKAGES"

# Build library path
LIB_PATHS=(
    "$SITE_PACKAGES/torch/lib"
    "$SITE_PACKAGES/nvidia/cuda_nvrtc/lib"
    "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
    "$SITE_PACKAGES/nvidia/cudnn/lib"
    "$SITE_PACKAGES/nvidia/cublas/lib"
    "$SITE_PACKAGES/nvidia/cusparse/lib"
    "$SITE_PACKAGES/nvidia/cufft/lib"
    "$SITE_PACKAGES/nvidia/nvjitlink/lib"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-12/lib64"
)

# Add existing paths
for lib_path in "${LIB_PATHS[@]}"; do
    if [ -d "$lib_path" ]; then
        export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
        echo "  ✅ Added: $lib_path"
    fi
done

echo ""
echo "========================================="
echo "  Testing CUDA Import"
echo "========================================="

# Test import
python -c "
import sys
try:
    import cuda_ops
    print('✅ SUCCESS: cuda_ops imported')
    
    # Check available functions
    if hasattr(cuda_ops, 'cuda_fused_gelu_swish'):
        print('  ✅ Fusion 1: cuda_fused_gelu_swish')
    if hasattr(cuda_ops, 'cuda_fused_ln_swish_dropout'):
        print('  ✅ Fusion 2: cuda_fused_ln_swish_dropout')
        
except ImportError as e:
    print(f'❌ FAILED: {e}')
    print()
    print('Troubleshooting:')
    print('  1. Check LD_LIBRARY_PATH is set correctly')
    print('  2. Verify PyTorch CUDA version matches system CUDA')
    print('  3. Try: pip install torch --force-reinstall')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "  ✅ Environment Ready!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "  ❌ Setup Failed - See errors above"
    echo "========================================="
fi