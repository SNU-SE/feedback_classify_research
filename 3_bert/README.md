# 3_bert: BERT 계열 사전학습 모델 실험

한국어 사전학습 BERT 모델을 미세조정(fine-tuning)하여 피드백을 분류하는 실험입니다.

## 실험 개요

- **모델**: 4개 한국어 BERT 모델
  - mBERT (Google, 104개 언어)
  - KLUE-BERT (한국어 특화)
  - KLUE-RoBERTa (개선된 BERT 아키텍처)
  - KoELECTRA (효율적 판별 사전학습)
- **평가**: 층화 10-fold 교차검증, 3개 시드 반복
- **학습 전략**: 클래스 가중 손실함수 + Early Stopping (patience=2)
- **최적 모델**: KLUE-BERT (Kappa: 0.91, F1: 0.89)

## 노트북 및 스크립트

| 파일명 | 설명 |
|--------|------|
| `01_experiment.ipynb` | 8:2 기본 실험 — 로컬 환경(M2 Max MPS) 실행용 |
| `02_experiment.py` | 8:2 기본 실험 — Python 스크립트 버전 |
| `03_runpod/` | RunPod GPU 클라우드 실행 환경 (일반화 실험용) |
| `04_result_analysis.ipynb` | 결과 분석 — 기본·일반화 실험 결과 종합 시각화 |
| `05_recall_analysis.ipynb` | Recall 심층 분석 — 레이블별 재현율 상세 분석 |

## 분석 순서

### Step 1: 8:2 기본 실험
`01_experiment.ipynb` 또는 `02_experiment.py`
- 4개 BERT 모델의 10-fold 교차검증 성능 비교
- 재현성 검증 (3회 반복 일치율)
- 교차 시드 일반화 (시드 간 성능 변화)
- 결과: `results/baseline/`

### Step 2: A형↔B형 5:5 구조적 일반화
`03_runpod/bert_experiment.py`
- 반(form) 기반 전이 실험
- RunPod GPU 클라우드에서 실행 (로컬 GPU 없는 환경 대응)
- 결과: `results/generalization/bert_form_transfer_analysis.csv`

### Step 3: 문항 유형별 조합
`03_runpod/bert_experiment.py`
- 같은 유형 vs 다른 유형 문항 전이 성능
- 8×8 문항 쌍별 성능 매트릭스
- 결과: `results/generalization/question_type/`

### Step 4: 문항 수별 조합
`03_runpod/bert_experiment.py`
- 훈련 문항 수(1~7개)별 성능 변화 추이
- 결과: `results/generalization/bert_question_all.csv`

### Step 5: 학생 비율별 조합
`03_runpod/bert_experiment.py`
- 학생 비율(40%~80%)별 성능 변화
- 결과: `results/generalization/bert_student_all.csv`

### Step 6: 결과 종합
`04_result_analysis.ipynb` → `05_recall_analysis.ipynb`
- 기본 실험 + 일반화 실험 전체 시각화
- 레이블별 Recall 심층 분석

## 사용 도구

- **딥러닝**: PyTorch, HuggingFace Transformers
- **실행 환경**: 로컬(M2 Max MPS) 또는 RunPod(NVIDIA GPU)
- **평가**: scikit-learn metrics

## RunPod 실행 방법

```bash
cd 03_runpod/
bash setup.sh              # 환경 설정 (의존성 설치)
python bert_experiment.py  # 일반화 실험 실행
python bert_test.py        # 테스트 실행
```

## 폴더 구조

```
3_bert/
├── results/
│   ├── baseline/              # 8:2 기본 실험 결과
│   │   └── figures/           # 시각화 그래프
│   └── generalization/        # 일반화 실험 결과
│       └── question_type/     # 문항 유형별 전이
├── 03_runpod/                 # GPU 클라우드 실행 환경
│   ├── bert_experiment.py     # 일반화 실험 스크립트
│   ├── bert_test.py           # 테스트 스크립트
│   ├── setup.sh               # 환경 설정
│   └── requirements.txt       # 의존성
├── 01_experiment.ipynb
├── 02_experiment.py
├── 04_result_analysis.ipynb
└── 05_recall_analysis.ipynb
```
