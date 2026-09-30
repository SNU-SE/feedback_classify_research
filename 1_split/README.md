# 1_split: 데이터 전처리 및 분할

동료 피드백 원시 데이터를 실험 목적에 맞게 다양한 방식으로 분할하는 모듈입니다.

## 데이터 개요

- **데이터**: 저장소 루트의 `data/feedback_data.xlsx` (분할 결과는 이 파일의 열로 제공, 루트 README 참고)
- **전체 규모**: 2,080개 피드백 텍스트, 7단계 코딩 (0~6)
- **문항 수**: 8개 문항 (Q1~Q8), 4개 문항 유형

## 노트북 및 스크립트

| 파일명 | 설명 |
|--------|------|
| `01_data_exploration.ipynb` | 데이터 탐색 및 8:2 기본 분할 |
| `02_structural_split.ipynb` | A형↔B형 5:5 구조적 분할 (반·형태·문항·학생 조합) |
| `03_question_split.ipynb` | 문항 유형별·문항 수별 분할 |
| `04_student_split.ipynb` | 학생 비율별 분할 (40%~80%) |
| `generate_paper_tables.py` | 논문용 표 생성 스크립트 |

## 분석 순서

### Step 1: 8:2 기본 분할
`01_data_exploration.ipynb`에서 전체 데이터를 80:20으로 층화 랜덤 분할합니다.
- 3개 시드(42, 43, 44)로 반복하여 분할 안정성 확인
- 결과: `data/feedback_data.xlsx`의 `split_seed42/43/44` 열

### Step 2: A형↔B형 5:5 구조적 분할
`02_structural_split.ipynb`에서 교육 현장의 구조적 차이를 반영한 분할을 수행합니다.
- S1: 검사지(form) 기반 분할 — A형으로 훈련, B형으로 테스트
- S2: 문항(question) 기반 분할
- S3: 교차(cross) 분할
- S1S4~S3S4: 위 조건 + 학생 분할 결합
- 결과: `data/feedback_data.xlsx`의 `S1_form` ~ `S3S4_cross_student` 열 (실험 결과는 `splits/structural/results/`)

### Step 3: 문항 유형별·문항 수별 분할
`03_question_split.ipynb`에서 문항 단위로 데이터를 분리합니다.
- 8개 문항(Q1~Q8)으로 분리
- 결과: `data/feedback_data.xlsx`의 `question` 열

### Step 4: 학생 비율별 분할
`04_student_split.ipynb`에서 훈련에 참여하는 학생 비율을 조절합니다.
- 학생 비율: 40%, 50%, 60%, 70%, 80%
- 학생 단위로 분할하여 데이터 누수(leakage) 방지
- 결과: ML/BERT/LLM 실험에서 직접 활용

## 사용 도구

- **Python**: pandas, openpyxl
- **분할 전략**: scikit-learn의 StratifiedShuffleSplit (클래스 비율 유지)

## 폴더 구조

```
1_split/
├── splits/structural/results/  # 구조적 분할 실험 결과
├── 01_data_exploration.ipynb
├── 02_structural_split.ipynb
├── 03_question_split.ipynb
├── 04_student_split.ipynb
└── generate_paper_tables.py
```
