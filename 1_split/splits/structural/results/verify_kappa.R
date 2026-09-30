# Cohen's Kappa 검증 (R irr 패키지 사용)
# Python sklearn의 cohen_kappa_score와 일치 여부 확인

if (!require("irr", quietly = TRUE)) install.packages("irr", repos = "https://cran.r-project.org")
if (!require("readxl", quietly = TRUE)) install.packages("readxl", repos = "https://cran.r-project.org")

library(irr)
library(readxl)

# Python 결과 로드
python_results <- read.csv("all_results.csv")

# 예측 파일 목록
pred_files <- list.files(".", pattern = "^predictions_.*\\.xlsx$", full.names = TRUE)

cat("=" , rep("=", 89), "\n", sep = "")
cat("Cohen's Kappa 검증: Python(sklearn) vs R(irr)\n")
cat("=", rep("=", 89), "\n\n", sep = "")

all_comparisons <- data.frame()

for (pred_file in pred_files) {
  # 전략 이름 추출: predictions_S1_form.xlsx -> S1_form
  strategy <- sub("^\\./predictions_(.*)\\.xlsx$", "\\1", pred_file)

  cat("#", rep("#", 69), "\n", sep = "")
  cat("# Strategy:", strategy, "\n")
  cat("#", rep("#", 69), "\n", sep = "")

  # 예측 데이터 로드
  pred_df <- read_excel(pred_file)
  true_label <- pred_df$true_label

  # 모델 예측 컬럼 (true_label, id, feedback_text 제외)
  model_cols <- setdiff(names(pred_df), c("id", "feedback_text", "true_label"))

  for (model_col in model_cols) {
    pred_label <- pred_df[[model_col]]

    # R에서 Cohen's Kappa 계산
    r_kappa <- kappa2(data.frame(true_label, pred_label), weight = "unweighted")$value

    # 모델 이름 정리 (_pred 접미사 제거)
    model_name <- sub("_pred$", "", model_col)

    # Python 결과에서 해당 값 찾기
    py_row <- python_results[python_results$Strategy == strategy & python_results$Model == model_name, ]

    if (nrow(py_row) == 1) {
      py_kappa <- py_row$Test_Kappa
      diff <- abs(r_kappa - py_kappa)
      match <- ifelse(diff < 1e-4, "OK", "MISMATCH")

      cat(sprintf("  [%s] %-30s  Python=%.6f  R=%.6f  diff=%.2e\n",
                  match, model_name, py_kappa, r_kappa, diff))

      all_comparisons <- rbind(all_comparisons, data.frame(
        Strategy = strategy,
        Model = model_name,
        Python_Kappa = py_kappa,
        R_Kappa = r_kappa,
        Diff = diff,
        Match = match,
        stringsAsFactors = FALSE
      ))
    }
  }
  cat("\n")
}

# 요약
cat("=", rep("=", 89), "\n", sep = "")
cat("검증 요약\n")
cat("=", rep("=", 89), "\n", sep = "")
cat(sprintf("  전체: %d개\n", nrow(all_comparisons)))
cat(sprintf("  일치(diff < 1e-4): %d개\n", sum(all_comparisons$Match == "OK")))
cat(sprintf("  불일치: %d개\n", sum(all_comparisons$Match == "MISMATCH")))
cat(sprintf("  최대 차이: %.2e\n", max(all_comparisons$Diff)))
cat(sprintf("  평균 차이: %.2e\n", mean(all_comparisons$Diff)))

if (sum(all_comparisons$Match == "MISMATCH") > 0) {
  cat("\n[불일치 항목]\n")
  mismatches <- all_comparisons[all_comparisons$Match == "MISMATCH", ]
  print(mismatches)
}

# 결과 CSV 저장
write.csv(all_comparisons, "kappa_verification.csv", row.names = FALSE)
cat(sprintf("\n결과 저장: kappa_verification.csv (%d행)\n", nrow(all_comparisons)))
