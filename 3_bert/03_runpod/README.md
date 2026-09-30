# RunPod BERT Test Experiment

254개 문항 조합 BERT 실험을 RunPod에서 병렬 실행하기 전, 파이프라인 검증을 위한 테스트 실험입니다.
KLUE-BERT 모델로 기존 80:20 데이터를 10-fold CV로 훈련하고, 결과를 검증합니다.

## 사전 준비

### 1. Hugging Face 설정

1. https://huggingface.co 계정 생성 (또는 로그인)
2. Settings → Access Tokens → **New token** 클릭
   - Token name: `runpod-bert` (아무 이름)
   - Type: **Write**
   - Generate token → 복사해서 보관 (예: `hf_AbCdEfGh...`)
3. **New Model** → Repository 생성
   - Repository name: `feedback-bert-klue` (아무 이름)
   - **Private** 선택
   - Create model
   - URL 형식: `your_username/feedback-bert-klue`

### 2. RunPod 설정

1. https://runpod.io 계정 생성
2. Billing → 크레딧 **$5** 충전 (테스트 비용 ~$0.08)
3. **GPU Cloud** → **Community Cloud** 클릭
4. GPU 선택: **RTX A5000** ($0.16/hr) — 24GB VRAM, 가성비 최고
   - 대안: RTX 4090 ($0.34/hr), RTX 3090 ($0.16/hr)
5. Template: **RunPod Pytorch 2.x** 선택
6. Container Disk: **20GB** (기본값)
7. **Deploy On-Demand** 클릭
8. Pod이 Running 상태가 되면 (약 1분) **Connect** → **JupyterLab** 클릭

### 3. 로컬 파일 준비

다음 파일들을 준비합니다:

```
runpod_bert/
├── requirements.txt
├── setup.sh
└── bert_test.py

01_Analysis/
├── train_set_80.xlsx
└── test_set_20.xlsx
```

## 실행 절차

### Step 1: 파일 업로드 (JupyterLab)

JupyterLab에서 `/workspace/` 디렉토리로 이동 후:

1. **스크립트 파일 업로드**: 왼쪽 파일 브라우저에서 Upload 버튼(위쪽 화살표)을 클릭하여 3개 파일 업로드
   - `requirements.txt` → `/workspace/requirements.txt`
   - `setup.sh` → `/workspace/setup.sh`
   - `bert_test.py` → `/workspace/bert_test.py`

2. **데이터 파일 업로드**: `/workspace/data/` 폴더를 만들고 업로드
   - 먼저 터미널에서 `mkdir -p /workspace/data` 실행
   - `train_set_80.xlsx` → `/workspace/data/train_set_80.xlsx`
   - `test_set_20.xlsx` → `/workspace/data/test_set_20.xlsx`

> 팁: JupyterLab에서 New → Terminal 클릭하면 터미널을 열 수 있습니다.

### Step 2: 환경 설정

터미널에서:

```bash
cd /workspace
bash setup.sh
```

GPU가 정상 감지되는지 확인합니다.

### Step 3: 환경 변수 설정

```bash
export HF_TOKEN=hf_your_token_here
export HF_REPO=your_username/feedback-bert-klue
```

> HF 업로드를 건너뛰려면 이 단계를 생략해도 됩니다. 실험은 정상 실행됩니다.

### Step 4: 실험 실행

```bash
cd /workspace
python3 bert_test.py
```

예상 소요시간: **약 20-25분** (RTX A5000 기준)

### Step 5: 결과 확인

실험 완료 후 `/workspace/results/`에 다음 파일들이 생성됩니다:

| 파일 | 내용 |
|------|------|
| `cv_results.csv` | 10-fold CV 성능 (accuracy, f1, kappa 평균/표준편차) |
| `test_results.csv` | Test set 성능 |
| `predictions.xlsx` | Test set 개별 예측 결과 |
| `experiment_info.json` | 실험 설정, 결과 요약, GPU 정보, 소요 시간 |
| `model/` | 학습된 모델 (HF 업로드 설정 시) |

### Step 6: 결과 다운로드

JupyterLab에서 `/workspace/results/` 내 파일을 우클릭 → **Download**

또는 터미널에서:
```bash
cd /workspace
zip -r results.zip results/
```
이후 JupyterLab에서 `results.zip` 다운로드.

### Step 7: Pod 종료 (중요!)

**과금을 중지하려면 Pod을 반드시 종료해야 합니다.**

1. RunPod 대시보드 → My Pods
2. 해당 Pod의 **Stop** 버튼 클릭 (일시정지, 데이터 유지)
3. 또는 **Terminate** 클릭 (완전 삭제, 데이터 삭제)

> Stop은 디스크 비용만 발생 (~$0.10/GB/월). Terminate는 완전 무료.

## 검증 기준

| 항목 | 기대값 | 허용 범위 |
|------|--------|----------|
| CV Accuracy | ~0.946 | ±0.03 |
| CV Macro-F1 | ~0.914 | ±0.03 |
| Test Accuracy | ~0.923 | ±0.03 |
| Test Macro-F1 | ~0.904 | ±0.03 |
| Test Kappa | ~0.900 | ±0.03 |
| HF 업로드 | 성공 | Private 레포에 모델 존재 |
| 총 소요시간 | ~20-25분 | RTX A5000 기준 |
| 비용 | ~$0.08 | < $0.15 |

> 기존 seed=42 MPS 결과와 CUDA 결과는 부동소수점 차이로 1-3% 차이가 날 수 있습니다.

## 트러블슈팅

### GPU가 감지되지 않음
```
ERROR: CUDA not available
```
- RunPod Pod 생성 시 GPU를 선택했는지 확인
- `nvidia-smi` 명령어로 GPU 상태 확인

### 모듈 import 에러
```
ModuleNotFoundError: No module named 'transformers'
```
- `bash setup.sh`를 먼저 실행했는지 확인
- 수동 설치: `pip install -r requirements.txt`

### 데이터 파일 없음
```
ERROR: Train file not found
```
- `/workspace/data/` 디렉토리에 `train_set_80.xlsx`, `test_set_20.xlsx`가 있는지 확인
- 파일명이 정확한지 확인 (대소문자 구분)

### HF 업로드 실패
```
Upload failed: ...
```
- `HF_TOKEN`이 올바른지 확인: `echo $HF_TOKEN`
- 토큰에 Write 권한이 있는지 확인
- `HF_REPO` 형식이 `username/repo-name`인지 확인
- HF 업로드 없이도 실험 결과는 정상 저장됩니다

### Out of Memory (OOM)
```
CUDA out of memory
```
- `bert_test.py`에서 `batch_size`를 16으로 줄이기
- RTX A5000 (24GB)에서는 batch_size=32로 충분합니다

## 비용 계산

| 항목 | 시간 | 비용 (RTX A5000) |
|------|------|------------------|
| 환경 설정 | ~5분 | $0.01 |
| 10-Fold CV | ~15-20분 | $0.04-0.05 |
| 최종 모델 학습 | ~2-3분 | $0.01 |
| Test 평가 + 업로드 | ~1분 | < $0.01 |
| **합계** | **~25분** | **~$0.07** |

## 향후 확장 (본 실험)

이 테스트 검증 후 `bert_question_combo.py`를 작성하여:
- `--start-combo`, `--end-combo` 인자로 Pod별 조합 범위 지정
- 254 조합을 4개 Pod에 분배하여 병렬 실행
- 각 조합 완료 즉시 CSV 저장 (중단 시 재개 가능)
- rclone으로 Google Drive 자동 업로드
