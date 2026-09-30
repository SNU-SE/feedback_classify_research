import pandas as pd
from pathlib import Path

csv_path = Path("structural_splits/results/all_results.csv")
df = pd.read_csv(csv_path)

# Prepare markdown output
md = []
md.append("# 구조적 데이터 분할 및 ML 분류 성능 분석 결과 (논문용)")
md.append("\n본 문서는 피드백 텍스트 분류 성능을 6가지 구조적 분할 전략(단순 일반화부터 복합 일반화까지)에 따라 평가한 결과와 그 학술적 함의를 정리한 보고서입니다.")

# 0. 전략 설명
md.append("\n## 0. 평가를 위한 6가지 구조적 분할 전략 (Structural Split Strategies)")
md.append("데이터 세트 내재된 구조(학생, 검사지 유형, 질문 번호)에 따른 모델의 실제 일반화(Generalization) 성능을 평가하기 위해 다음과 같은 6가지 분할 전략을 설계하였습니다.")
md.append("\n| 전략명 | 분할 방식 | 일반화 목표 | 학생 독립성 (Student-level split) |")
md.append("|---|---|---|---|")
md.append("| **S1_form** (검사지) | A형→학습, B형→테스트 | **새로운 검사지 유형**에 대한 일반화 | X (양쪽에 동일 학생 존재 가능) |")
md.append("| **S1S4_form_student** | S1 분할 + 양쪽 겹치는 학생 배제 | **새로운 검사지 + 처음 보는 학생** 일반화 | O |")
md.append("| **S2_question** (질문) | q1, q2, q3→학습, q4→테스트 | **새로운 질문 유형**에 대한 일반화 | X |")
md.append("| **S2S4_question_student** | S2 분할 + 양쪽 겹치는 학생 배제 | **새로운 질문 + 처음 보는 학생** 일반화 | O |")
md.append("| **S3_cross** (교차) | A전체, B(q1,q2)→학습, B(q3,q4)→테스트 | **복합 환경(새 검사지 내 새 질문)** 일반화 | X |")
md.append("| **S3S4_cross_student** | S3 분할 + 양쪽 겹치는 학생 배제 | **가장 엄격한 조건(Out-of-Distribution)** 평가 | O |")

# 1. Best Model per Strategy
md.append("\n## 1. 각 분할 전략별 최고 성능 모델 (Test Kappa 기준)")
best_models_idx = df.groupby('Strategy')['Test_Kappa'].idxmax()
best_models = df.loc[best_models_idx].copy()
# Sort strategies by a logical order if possible
strategy_order = ['S1_form', 'S1S4_form_student', 'S2_question', 'S2S4_question_student', 'S3_cross', 'S3S4_cross_student']
best_models['Strategy'] = pd.Categorical(best_models['Strategy'], categories=strategy_order, ordered=True)
best_models = best_models.sort_values('Strategy')

md.append("\n| 전략명 | 최고 성능 모델 | 유형 | CV Kappa (Std) | Test Kappa | Test F1 Macro | Test Acc |")
md.append("|---|---|---|---|---|---|---|")
for _, row in best_models.iterrows():
    md.append(f"| {row['Strategy']} | {row['Model']} | {row['Type']} | {row['CV_Kappa']:.4f} (±{row['CV_Kappa_Std']:.4f}) | **{row['Test_Kappa']:.4f}** | {row['Test_F1']:.4f} | {row['Test_Acc']:.4f} |")

# 2. Average Performance: Individual vs Ensemble by Strategy
md.append("\n## 2. 모델 유형별 성능 비교 (Individual vs Ensemble)")
pivot_kappa = df.pivot_table(index='Strategy', columns='Type', values='Test_Kappa', aggfunc='mean')
pivot_kappa = pivot_kappa.reindex(strategy_order)
pivot_f1 = df.pivot_table(index='Strategy', columns='Type', values='Test_F1', aggfunc='mean')
pivot_f1 = pivot_f1.reindex(strategy_order)

md.append("\n| 전략명 | 개별 모델 (평균 Kappa) | 앙상블 모델 (평균 Kappa) | 개별 모델 (평균 F1) | 앙상블 모델 (평균 F1) |")
md.append("|---|---|---|---|---|")
for idx in strategy_order:
    if idx in pivot_kappa.index:
        ind_k = pivot_kappa.loc[idx, 'Individual']
        ens_k = pivot_kappa.loc[idx, 'Ensemble']
        ind_f1 = pivot_f1.loc[idx, 'Individual']
        ens_f1 = pivot_f1.loc[idx, 'Ensemble']
        md.append(f"| {idx} | {ind_k:.4f} | {ens_k:.4f} | {ind_f1:.4f} | {ens_f1:.4f} |")

# 3. Best overall models Top 10 across all experiments
md.append("\n## 3. 전체 실험 중 상위 10개 모델 (Test Kappa 기준)")
top10 = df.sort_values('Test_Kappa', ascending=False).head(10)
md.append("\n| 순위 | 전략명 | 모델명 | 유형 | Test Kappa | Test F1 | CV Kappa |")
md.append("|---|---|---|---|---|---|---|")
for i, (_, row) in enumerate(top10.iterrows(), 1):
    md.append(f"| {i} | {row['Strategy']} | {row['Model']} | {row['Type']} | **{row['Test_Kappa']:.4f}** | {row['Test_F1']:.4f} | {row['CV_Kappa']:.4f} |")

# 4. 개별 모델별 성능 분석 (Robustness across strategies)
md.append("\n## 4. 개별 모델의 데이터 분할 전략 간 성능 변화 (Robustness)")
indiv = df[df['Type'] == 'Individual']
pivot_models = indiv.pivot_table(index='Model', columns='Strategy', values='Test_Kappa')
pivot_models = pivot_models.reindex(columns=strategy_order)
pivot_models['평균'] = pivot_models.mean(axis=1)
pivot_models['표준편차'] = pivot_models.drop(columns=['평균']).std(axis=1)
pivot_models = pivot_models.sort_values('평균', ascending=False)

# Make table
cols = ['Model'] + strategy_order + ['Mean', 'Std']
md.append("\n| 모델명 | " + " | ".join(strategy_order) + " | 평균 | 표준편차 |")
md.append("|---" * len(cols) + "|")
for model_name, row in pivot_models.iterrows():
    row_str = f"| {model_name} | "
    row_str += " | ".join([f"{row[c]:.4f}" for c in strategy_order])
    row_str += f" | **{row['평균']:.4f}** | {row['표준편차']:.4f} |"
    md.append(row_str)

# 5. 특화 분석: 80/20 데이터 기준 최고 신뢰 모델 (C4_Prev4_SoftVoting)의 강건성 평가
md.append("\n## 5. 80/20 일반 분할 최우수 모델의 구조적 분할 간 성능 변동폭 분석")
md.append("기존 80/20 단순 무작위 분할 시 가장 높은 신뢰도를 거둔 **Soft Voting 앙상블 모델 (LR, SVC(prob=True), CNB, RF 구성)**에 대해, 6가지 구조적 분할 전략 적용 시 일반화 성능이 어떻게 변화하는지를 상세히 추적하였습니다. 본 분석에 사용된 모델 식별자는 `C4_Prev4_SoftVoting` 입니다.")

target_model = 'C4_Prev4_SoftVoting'
target_data = df[df['Model'] == target_model]

if not target_data.empty:
    target_data = target_data.set_index('Strategy').reindex(strategy_order)
    md.append("\n| 전략명 | Test Kappa | Test F1 Macro | CV Kappa | Test Acc |")
    md.append("|---|---|---|---|---|")
    for strat in strategy_order:
        row = target_data.loc[strat]
        md.append(f"| {strat} | **{row['Test_Kappa']:.4f}** | {row['Test_F1']:.4f} | {row['CV_Kappa']:.4f} | {row['Test_Acc']:.4f} |")

# 6. 해석 및 함의점
md.append("\n## 6. 결과 해석 및 학술적 함의점 (Implications)")
md.append("\n### ⑴ 앙상블 기법의 우수성과 부스팅 모델의 강건성")
md.append("단일(Individual) 모델보다 다수의 모델 예측을 결합한 **앙상블(Ensemble) 모델**이 모든 분할 전략에서 전반적으로 더 높은 Test Kappa와 F1 스코어를 기록하였습니다. 특히 가장 엄격한 OOD(Out-of-Distribution) 조건인 `S3S4_cross_student` 전략에서 전체 1위부터 4위까지를 앙상블(Soft Voting) 및 `XGB`, `GB` 모델이 차지하였습니다. 이는 트리 기반 부스팅(Boosting) 알고리즘 혹은 이를 포함한 앙상블 전략이 훈련 데이터에 없는 새로운 조건에서도 강건(Robust)하게 대응할 수 있음을 나타냅니다.")

md.append("\n### ⑵ '학생 분리(S4)' 조건이 모델 일반화에 미치는 영향")
md.append("동일한 기준의 분할(S1 vs S1+S4, S2 vs S2+S4) 실험에서 **학생 분리 조건(S4)이 추가될 경우 대부분의 모델 성능이 일관되게 하락**하는 경향을 보였습니다. 이는 특정 학생 특유의 어휘 선택이나 문장 스타일(작문 패턴)에 모델이 쉽게 의존(과적합)할 현상이 발현되기 쉬운 편향 위험성이 있음을 시사합니다. 따라서 실제 교육 환경에 시스템을 배포하기 위해서는 학생들의 다양한 작문 스타일 변형에 대해 강건성을 높일 수 있는 **데이터 증강(Data Augmentation) 기법**이나, 특정 학생 스타일에 편향되지 않도록 정규화(Regularization) 기법의 도입이 필수적임을 의미합니다.")

md.append("\n### ⑶ 평가 파이프라인 정립의 필요성 제언")
md.append("교육용 피드백 챗봇 엔진의 분류 성능을 안정적으로 평가 및 유지하기 위해, 향후 서비스 적용 시 단일 알고리즘보다는 **다수의 알고리즘을 결합한 Soft Voting 또는 Stacking 계열의 앙상블 방식**을 채택하는 것이 가장 바람직할 것입니다. 더불어, 모델 평가 지표의 실용적 신뢰성을 담보하기 위하여 단순 무작위 분할(Random Split)이 아닌, 본 연구와 같이 '새로운 과제'와 '처음 보는 작문 스타일(학생)'을 엄격하게 통제한 **구조적 교차 검증(Structural Split Evaluation)** 파이프라인을 평가의 표준으로 정립할 필요성이 도출되었습니다.")

with open('02_academic_results_report.md', 'w') as f:
    f.write("\n".join(md))

print("Report generated: 02_academic_results_report.md")
