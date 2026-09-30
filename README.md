# 동료 피드백 자동 분류 시스템 (Automated Peer Feedback Classification)

고등학교 과학 수업에서 수집한 2,080개의 한국어 동료 피드백 텍스트를 7단계 코딩 체계(0~6)로 자동 분류하는 연구 프로젝트입니다.

## 연구 개요

### 연구 질문
- **RQ1**: 전통적 ML, BERT, LLM 세 가지 접근법의 분류 신뢰도와 타당도 비교
- **RQ2**: 최적 모델의 새로운 맥락에 대한 일반화 성능 검증

### 데이터
- **출처**: 고등학교 과학 수업 동료평가
- **규모**: 2,080개 피드백 텍스트
- **형태**: A형/B형 평가지, 8개 문항
- **코딩**: 7단계 (0=무관, 1=무응답, 2=확인, 3=설명요청, 4=지적, 5=대안제시, 6=대안+근거)

### 데이터 파일
분석에 필요한 데이터는 `data/feedback_data.xlsx` 하나에 모두 들어 있습니다.

| 시트 | 행 | 열 |
|------|----|----|
| `main` | 2,080 | `id`, `student`, `form`(A/B), `question`(Q1~Q8), `feedback_text`, `label`(0~6), `split_seed42/43/44`, `S1_form`, `S1S4_form_student`, `S2_question`, `S2S4_question_student`, `S3_cross`, `S3S4_cross_student` |
| `new_context` | 193 | `id`, `feedback_text`, `label` (3차 실험: 새로운 맥락 데이터) |

분할 열의 값은 `train` / `test`이며, 빈 칸은 해당 분할에서 제외된 샘플입니다.

```python
import pandas as pd
df = pd.read_excel("data/feedback_data.xlsx", sheet_name="main")
train, test = df[df.split_seed42 == "train"], df[df.split_seed42 == "test"]
```

**개인정보 보호**: 학생 식별 번호는 시드 없는 무작위 코드로 치환했고, 모든 표의 행 순서를 코드 기준으로 재정렬했습니다. 저장소 내용만으로는 원래 학생을 역추적할 수 없습니다. 이에 따라 train/test 구성은 논문과 동일하지만, 행 순서에 의존하는 교차검증 폴드와 학생 비율별 분할(40~80%)은 재실행 시 원 결과와 수치가 약간 다를 수 있습니다.

> 노트북의 데이터 경로는 원 분석 환경 기준입니다. 재실행 시 위 코드처럼 `data/feedback_data.xlsx`를 읽도록 경로를 바꿔 주세요.

### 주요 결과
| 접근법 | Cohen's Kappa | F1 Score |
|--------|--------------|----------|
| BERT (KLUE-BERT) | 0.90 | 0.89 |
| ML (최적 앙상블) | 0.80 | 0.76 |
| LLM (GPT-4o) | 0.64 | 0.63 |

## 폴더 구조

```
Git/
├── data/             # 분석 데이터 (feedback_data.xlsx 1개)
├── 1_split/          # 데이터 분할 (8:2 랜덤, 구조적, 문항별, 학생별)
├── 2_ml/             # 전통적 ML 실험 (24개 모델, 앙상블)
├── 3_bert/           # BERT 계열 실험 (4개 모델, RunPod GPU)
├── 4_llm/            # LLM 실험 (GPT-4o, 프롬프트 엔지니어링)
└── 5_result/         # 비교 분석 및 시각화
```

## 실험 흐름

### 1차 실험: 기본 성능 평가
전체 데이터를 8:2로 분할(3개 시드)하여 ML(24개 모델), BERT(4개 모델), LLM(GPT-4o)의 기본 분류 성능을 비교합니다.

### 2차 실험: 일반화 성능 평가
교육 현장의 실제 상황을 반영한 다양한 조건에서 일반화 성능을 검증합니다:
- **A형↔B형 전이**: 다른 수업 형태로의 전이 가능성
- **문항 유형 조합**: 254가지 문항 조합별 성능 변화
- **문항 수 효과**: 훈련 문항 수(1~7개)에 따른 성능 변화
- **학생 비율**: 훈련 학생 비율(40%~80%)에 따른 성능 변화

### 3차 실험: 새로운 데이터 평가
완전히 새로운 맥락의 데이터에 대한 분류 성능을 검증합니다.

## 실행 환경

### ML 실험
- Python 3.10, scikit-learn, pandas
- **병렬 처리**: 멀티코어 CPU 전체 활용 (24개 모델 동시 훈련)

### BERT 실험
- Python 3.10, transformers, PyTorch
- **GPU 클라우드**: RunPod을 이용한 병렬 pod 구성 (GPU 없는 환경 대응)

### LLM 실험
- Python 3.10, OpenAI API (GPT-4o)
- **프롬프트 엔지니어링**: Zero-shot → CoT+Few-shot → Hard Tiebreaker 3단계 개선

## 설치 및 실행

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# LLM 실험을 위한 API 키 설정
cp 4_llm/.env.example 4_llm/.env
# .env 파일에 OpenAI API 키 입력
```

## 라이선스

이 연구 코드는 학술 연구 목적으로 공개됩니다.
