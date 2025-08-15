#!/bin/bash
# setup_4gpu.sh - Complete setup for 4x H100 training

echo "🚀 Setting up 4x H100 training environment..."

# 1. Configure accelerate interactively (safest)
echo "📋 Configuring accelerate..."
accelerate config --config_file ~/.cache/huggingface/accelerate/4gpu_config.yaml

# Alternative: Quick non-interactive setup
# cat > ~/.cache/huggingface/accelerate/4gpu_config.yaml << EOF
# compute_environment: LOCAL_MACHINE
# distributed_type: MULTI_GPU
# downcast_bf16: 'no'
# enable_cpu_affinity: false
# gpu_ids: 0,1,2,3
# machine_rank: 0
# main_training_function: main
# mixed_precision: 'no'
# num_machines: 1
# num_processes: 4
# rdzv_backend: static
# same_network: true
# tpu_env: []
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: false
# EOF

# 2. Set environment variables
echo "🔧 Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_COMPILE_MODE=reduce-overhead

# 3. Verify HF token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set! Please run: export HF_TOKEN='your_token'"
    echo "You can get your token from: https://huggingface.co/settings/tokens"
else
    echo "✅ HF_TOKEN is set"
fi

# 4. Test GPU setup
echo "🔍 Testing GPU setup..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 5. Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs/euclidean_baseline
mkdir -p outputs/euclidean_full  
mkdir -p outputs/mixed_small
mkdir -p outputs/mixed_full

echo "✅ Setup complete!"
echo ""
echo "🚀 Ready to launch training:"
echo "   Small comparison (30 min):"
echo "     accelerate launch scripts/train.py --model euclid --config configs/euclidean_baseline.yaml"
echo "     accelerate launch scripts/train.py --model mixed --config configs/mixed_small.yaml"
echo ""
echo "   Full comparison (90 min):"  
echo "     accelerate launch scripts/train.py --model euclid --config configs/euclidean_full.yaml"
echo "     accelerate launch scripts/train.py --model mixed --config configs/mixed_full.yaml"