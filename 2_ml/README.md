# 2_ml: 전통적 머신러닝 실험

TF-IDF 벡터화 + 14개 분류 알고리즘을 이용한 전통적 머신러닝 기반 피드백 분류 실험입니다.

## 실험 개요

- **벡터화**: 3가지 (V1_Word 형태소 TF-IDF, V2_Char 문자 TF-IDF, V3_Chi2 특성 선택)
- **분류기**: 14개 (LR, SVM-L, SVM-RBF, MNB, CNB, KNN, DT, RF, GB, XGB, LGBM, Hard Voting, Soft Voting, Stacking)
- **평가**: 층화 10-fold 교차검증, 3개 시드 반복
- **총 조합**: 126개 (3 벡터화 × 14 분류기 × 3 시드)

## 노트북

| 파일명 | 설명 |
|--------|------|
| `01_experiment.ipynb` | 8:2 기본 실험 — 126개 조합 교차검증 및 테스트 |
| `02_ensemble.ipynb` | 8:2 앙상블 최적화 — 최적 앙상블 조합 탐색 |
| `03_generalization.ipynb` | 일반화 실험 — 문항·학생 조건별 성능 평가 |
| `04_results_analysis.ipynb` | 결과 분석 — 일반화 실험 결과 종합 시각화 |

## 분석 순서

### Step 1: 8:2 기본 실험
`01_experiment.ipynb` → `02_ensemble.ipynb`
- 126개 벡터화-분류기 조합의 교차검증 성능 비교
- 재현성 검증 (동일 조건 3회 반복 → 일치율 확인)
- 교차 시드 일반화 (시드 간 성능 하락폭 측정)
- 최적 조합: **V1_Word + Soft Voting** (F1-Macro: 0.769)
- 결과: `results/baseline/`

### Step 2: A형↔B형 5:5 구조적 일반화
`03_generalization.ipynb`
- 반(form) 기반 전이 실험 포함
- 결과: `results/generalization/` 내 관련 CSV

### Step 3: 문항 유형별 조합
`03_generalization.ipynb`
- 같은 유형 vs 다른 유형 문항 간 전이 성능 비교
- 8×8 문항 쌍별 F1/Kappa 매트릭스
- 4×4 유형 단위 Kappa 매트릭스
- 결과: `results/generalization/question_type/`

### Step 4: 문항 수별 조합
`03_generalization.ipynb`
- 254가지 문항 조합 (1~7개 훈련 문항)별 성능 변화
- 훈련 문항 수 증가에 따른 성능 향상 추이
- 결과: `results/generalization/question_combo/`

### Step 5: 학생 비율별 조합
`03_generalization.ipynb`
- 훈련 학생 비율(40%~80%)에 따른 성능 변화
- 20:80 비율의 선형 외삽 추정
- 결과: `results/generalization/student_split/`

## 사용 도구

- **텍스트 처리**: KoNLPy(형태소 분석), scikit-learn TfidfVectorizer
- **분류**: scikit-learn, XGBoost, LightGBM
- **병렬 처리**: joblib (멀티코어 CPU 전체 활용)

## 폴더 구조

```
2_ml/
├── results/
│   ├── baseline/              # 8:2 기본 실험 결과
│   │   ├── ensemble/          # 앙상블 최적화 결과
│   │   └── optimal/           # 최적 모델 예측 결과
│   └── generalization/        # 일반화 실험 결과
│       ├── question_type/     # 문항 유형별 전이
│       ├── question_combo/    # 문항 수별 조합
│       └── student_split/     # 학생 비율별
├── 01_experiment.ipynb
├── 02_ensemble.ipynb
├── 03_generalization.ipynb
└── 04_results_analysis.ipynb
```
