# 5_result: 비교 분석 및 시각화

ML, BERT, LLM 세 접근법의 결과를 종합 비교하고, 논문용 시각화를 생성하는 모듈입니다.

## 분석 개요

- **비교 대상**: ML(Soft Voting), BERT(KLUE-BERT), LLM(GPT-4o)
- **평가 지표**: Cohen's Kappa, F1-Macro, Accuracy, Recall
- **시각화**: 논문 삽입용 고품질 그래프 생성

## 노트북

| 파일명 | 설명 |
|--------|------|
| `01_comparison.ipynb` | 8:2 기본 성능 비교 — 3개 접근법 종합 |
| `02_confusion_analysis.ipynb` | 혼동 행렬 분석 — 오분류 패턴 파악 |
| `03_misclassified.ipynb` | 오분류 텍스트 분석 — 오분류 원인 탐색 |
| `04_5050_comparison.ipynb` | A형↔B형 5:5 비교 — 구조적 전이 성능 |
| `05_type_boxplot.ipynb` | 문항 유형별 박스플롯 — 유형 전이 성능 비교 |
| `06_question_count_boxplot.ipynb` | 문항 수별 박스플롯 — 훈련 문항 수 효과 |
| `07_student_ratio_boxplot.ipynb` | 학생 비율별 박스플롯 — 훈련 학생 비율 효과 |
| `08_question_type_analysis.ipynb` | 문항 유형 심층 분석 — 유형 간 전이 매트릭스 |
| `09_new_data_eval.ipynb` | 새로운 데이터 평가 — Phase 3 일반화 검증 |
| `10_generate_figures.ipynb` | 논문용 그래프 생성 — 최종 출판 품질 시각화 |

## 분석 순서

### Phase 1: 8:2 기본 성능 비교
`01_comparison.ipynb` → `02_confusion_analysis.ipynb` → `03_misclassified.ipynb`
1. 세 접근법(ML, BERT, LLM)의 Kappa, F1, Accuracy 종합 비교
2. 혼동 행렬로 어떤 레이블 쌍이 혼동되는지 패턴 분석
3. 실제 오분류 텍스트를 확인하여 오분류 원인 탐색
- 결과: `results/` 폴더 (predictions, metrics, variability CSV)

### Phase 2: 일반화 성능 비교
`04_5050_comparison.ipynb` → `05_type_boxplot.ipynb` → `06_question_count_boxplot.ipynb` → `07_student_ratio_boxplot.ipynb` → `08_question_type_analysis.ipynb`

#### Step 1: A형↔B형 5:5 구조적 전이
`04_5050_comparison.ipynb`
- 반(form) 기반 전이에서 ML vs BERT 성능 비교
- 결과: `figures/generalization/comparison_5050_*.png`

#### Step 2: 문항 유형별 조합
`05_type_boxplot.ipynb` + `08_question_type_analysis.ipynb`
- 같은 유형 vs 다른 유형 전이 성능 박스플롯
- 유형 수(1~4)별 성능 변화
- 유형 간 전이 매트릭스 심층 분석
- 결과: `figures/generalization/type_boxplot_*.png`

#### Step 3: 문항 수별 조합
`06_question_count_boxplot.ipynb`
- 훈련 문항 수(1~7)별 ML vs BERT 성능 추이
- 최소 몇 문항이면 충분한 성능에 도달하는지 탐색
- 결과: `figures/generalization/question_count_boxplot_*.png`

#### Step 4: 학생 비율별 조합
`07_student_ratio_boxplot.ipynb`
- 훈련 학생 비율(40%~80%)별 ML vs BERT 성능 추이
- LLM 베이스라인과의 비교 (학습 불필요한 LLM 기준선)
- 결과: `figures/generalization/student_ratio_boxplot_*.png`

### Phase 3: 완전 새로운 데이터 평가
`09_new_data_eval.ipynb`
- 훈련 데이터에 포함되지 않은 완전히 새로운 맥락의 데이터 평가
- ML, BERT, LLM 세 접근법의 새 데이터 적용 성능 검증
- Hybrid 평가 (모델 조합) 포함
- 결과: `results/new_data/`

### 최종: 논문용 그래프 생성
`10_generate_figures.ipynb`
- Phase 1~3 결과를 출판 품질의 그래프로 생성
- 결과: `figures/paper/fig*.png`

## 사용 도구

- **시각화**: matplotlib, seaborn
- **데이터 처리**: pandas, numpy
- **평가 지표**: scikit-learn (Cohen's Kappa, F1, confusion matrix)

## 폴더 구조

```
5_result/
├── results/
│   ├── *_predictions.csv          # 접근법별 예측 결과
│   ├── *_metric_summary.csv       # 접근법별 성능 지표
│   ├── misclassified_texts/       # 오분류 텍스트 분석
│   └── new_data/                  # Phase 3 새 데이터 평가 결과
├── figures/
│   ├── confusion/                 # 혼동 행렬 시각화
│   ├── generalization/            # 일반화 실험 시각화
│   └── paper/                     # 논문 삽입용 최종 그래프
├── 01_comparison.ipynb
├── 02_confusion_analysis.ipynb
├── 03_misclassified.ipynb
├── 04_5050_comparison.ipynb
├── 05_type_boxplot.ipynb
├── 06_question_count_boxplot.ipynb
├── 07_student_ratio_boxplot.ipynb
├── 08_question_type_analysis.ipynb
├── 09_new_data_eval.ipynb
└── 10_generate_figures.ipynb
```
