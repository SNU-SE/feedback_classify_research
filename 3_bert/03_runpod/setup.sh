#!/bin/bash
# RunPod BERT 실험 환경 설정 스크립트
# Usage: bash setup.sh

set -e

echo "=============================================="
echo "  RunPod BERT Experiment - Environment Setup"
echo "=============================================="

# 0. .env 파일에서 환경 변수 자동 로딩
if [ -f .env ]; then
    echo ""
    echo "[0/3] Loading environment variables from .env..."
    set -a
    source .env
    set +a
    echo "  -> HF_TOKEN: ${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
    echo "  -> HF_REPO:  ${HF_REPO:-not set}"
else
    echo ""
    echo "[0/3] No .env file found. To auto-load env vars:"
    echo "      cp .env.example .env && nano .env"
fi

# 1. Python 패키지 설치
echo ""
echo "[1/3] Installing Python packages..."
pip install -q -r requirements.txt
echo "  -> Packages installed."

# 2. GPU 확인
echo ""
echo "[2/3] Checking GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM: {mem:.1f} GB')
else:
    print('  WARNING: No GPU detected!')
    print('  Make sure you selected a GPU pod.')
"

# 3. 디렉토리 생성
echo ""
echo "[3/3] Creating directories..."
mkdir -p /workspace/data/question_data
mkdir -p /workspace/data/student_splits
mkdir -p /workspace/results/predictions
echo "  -> /workspace/data/question_data/"
echo "  -> /workspace/data/student_splits/"
echo "  -> /workspace/results/predictions/"

# 완료 안내
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  === Test Experiment (pipeline verification) ==="
echo "  1. Upload: train_set_80.xlsx, test_set_20.xlsx -> /workspace/data/"
echo "  2. python3 bert_test.py"
echo ""
echo "  === Main Experiment ==="
echo "  1. Upload: Q1.xlsx~Q8.xlsx -> /workspace/data/question_data/"
echo "     Upload: train_*.xlsx, test_*.xlsx -> /workspace/data/student_splits/"
echo "  2. Question combos:  python3 bert_experiment.py --mode question --pod N --total-pods 4"
echo "     Student splits:   python3 bert_experiment.py --mode student"
echo ""
