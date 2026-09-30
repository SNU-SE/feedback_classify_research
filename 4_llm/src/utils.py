# ============================================================
# utils.py - 유틸리티 함수 모음
# ============================================================
# 이 파일은 LLM 실험에서 공통으로 사용되는 유틸리티 함수를 제공합니다.
#
# 주요 기능:
#   1. 시드 고정 (set_seed): 실험 재현성 보장
#   2. 성능 지표 계산 (compute_metrics): Accuracy, F1, Kappa 등
#   3. 시각화 (plot_*): 혼동 행렬, 모델 비교, 히트맵
#   4. 토큰 추정 (estimate_tokens): API 비용 사전 예측
#
# 7단계 피드백 코딩 체계 (레이블 0~6):
#   0: 무관(비피드백) - 피드백과 관련 없는 내용
#   1: 평가만 - 단순 긍정/부정 평가
#   2: 기술 - 구체적 내용 기술
#   3: 문제 지적 - 오류나 문제점 지적
#   4: 해결 방향 제시 - 개선 방향 제안
#   5: 구체적 대안 제시 - 구체적 수정 방안 제시
#   6: 대안적 관점 제시 - 새로운 관점이나 접근법 제안
# ============================================================

"""
유틸리티 함수
"""

import random
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# 랜덤 시드 고정 함수
# - 실험 재현성을 위해 모든 난수 생성기의 시드를 고정
# - Python, NumPy, PyTorch(GPU 포함)의 시드를 모두 설정
# ============================================================
def set_seed(seed: int = 42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================
# 분류 성능 지표 계산 함수
# - LLM 예측 결과와 실제 레이블을 비교하여 다양한 지표 산출
# - 파싱 실패(LLM이 유효한 레이블을 반환하지 못한 경우) 처리 포함
# - 소수 클래스(레이블 0, 6)의 성능도 별도 추적
#   (이 클래스들은 데이터가 적어 LLM이 잘 맞추지 못하는 경향)
# ============================================================
def compute_metrics(y_true: List[int], y_pred: List[int], parsed_ok_list: List[bool] = None) -> Dict[str, float]:
    """
    분류 성능 지표 계산 (minority class 메트릭 포함)

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블 (None은 파싱 실패)
        parsed_ok_list: 파싱 성공 여부 리스트 (optional)

    Returns:
        메트릭 딕셔너리
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        cohen_kappa_score,
        precision_score,
        recall_score
    )

    # 유효한 예측만 사용 (None, -1 제외)
    # LLM이 JSON 형식의 레이블을 반환하지 못하면 None이 됨 → 성능 계산에서 제외
    valid_mask = [p is not None and p != -1 for p in y_pred]
    y_true_valid = [y for y, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

    # n_fail 계산 (파싱 실패 수)
    if parsed_ok_list is not None:
        n_fail = sum(1 for ok in parsed_ok_list if not ok)
    else:
        n_fail = sum(1 for p in y_pred if p is None or p == -1)

    if len(y_pred_valid) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "kappa": 0.0,
            "valid_predictions": 0,
            "total_samples": len(y_true),
            "n_fail": n_fail,
            "error_rate": 1.0,
            # Minority class metrics (label 0 and 6)
            "recall_0": 0.0,
            "f1_0": 0.0,
            "recall_6": 0.0,
            "f1_6": 0.0
        }

    # 클래스별 재현율(Recall)과 F1 점수 (레이블 0~6)
    # Recall: 실제 해당 레이블인 샘플 중 올바르게 예측한 비율
    # F1: 정밀도(Precision)와 재현율의 조화 평균
    all_labels = list(range(7))  # 0~6 (7단계 피드백 코딩)
    recall_per_class = recall_score(
        y_true_valid, y_pred_valid,
        labels=all_labels, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true_valid, y_pred_valid,
        labels=all_labels, average=None, zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),  # 전체 정확도
        "f1_macro": f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),  # 클래스 균등 가중 F1
        "f1_weighted": f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0),  # 샘플 수 가중 F1
        "precision_macro": precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true_valid, y_pred_valid),  # Cohen's Kappa: 우연 일치를 보정한 일치도
        "valid_predictions": len(y_pred_valid),
        "total_samples": len(y_true),
        "n_fail": n_fail,
        "error_rate": 1 - (len(y_pred_valid) / len(y_true)),
        # Minority class metrics (label 0 and 6)
        "recall_0": float(recall_per_class[0]),
        "f1_0": float(f1_per_class[0]),
        "recall_6": float(recall_per_class[6]),
        "f1_6": float(f1_per_class[6])
    }

    return metrics


# ============================================================
# 시각화 함수들
# - 혼동 행렬: 어떤 레이블이 어떤 레이블로 잘못 예측되는지 확인
# - 모델 비교: 모델별 F1 점수를 가로 막대 그래프로 비교
# - 히트맵: 모델×프롬프트 조합의 성능을 한눈에 파악
# ============================================================
def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None
):
    """혼동 행렬 시각화"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 한글 폰트 설정
    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 유효한 예측만 사용
    valid_mask = [p != -1 for p in y_pred]
    y_true_valid = [y for y, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [y for y, v in zip(y_pred, valid_mask) if v]

    if len(y_pred_valid) == 0:
        print("유효한 예측이 없습니다.")
        return

    cm = confusion_matrix(y_true_valid, y_pred_valid)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels or range(cm.shape[1]),
        yticklabels=labels or range(cm.shape[0])
    )
    plt.title(title)
    plt.xlabel('예측 레이블')
    plt.ylabel('실제 레이블')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(
    results_df,
    metric: str = "f1_macro",
    title: str = "모델별 성능 비교",
    save_path: str = None
):
    """모델별 성능 비교 차트"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 한글 폰트 설정
    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6))

    # 모델별 평균 및 표준편차 계산
    summary = results_df.groupby("model_name")[metric].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    bars = plt.barh(summary["model_name"], summary["mean"], xerr=summary["std"], capsize=5)
    plt.xlabel(metric)
    plt.title(title)
    plt.xlim(0, 1)

    # 값 표시
    for bar, mean_val in zip(bars, summary["mean"]):
        plt.text(mean_val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean_val:.3f}', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_prompt_heatmap(
    results_df,
    metric: str = "f1_macro",
    title: str = "모델-프롬프트별 성능 히트맵",
    save_path: str = None
):
    """모델-프롬프트별 성능 히트맵"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 한글 폰트 설정
    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 피벗 테이블 생성
    pivot = results_df.pivot_table(
        index="model_name",
        columns="prompt_version",
        values=metric,
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        vmin=0,
        vmax=1,
        linewidths=0.5
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 토큰 추정 유틸리티
# - API 호출 전에 비용을 사전 예측하여 예산 초과를 방지
# - 토큰(Token): LLM이 텍스트를 처리하는 최소 단위
#   영어 1단어 ≈ 1~2토큰, 한글 1글자 ≈ 2토큰
# - GPT-4o 비용: 입력 $2.50/1M토큰, 출력 $10.00/1M토큰
# ============================================================

def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    텍스트의 토큰 수 추정 (tiktoken 사용)

    Args:
        text: 토큰 수를 계산할 텍스트
        model: 모델명 (기본값: gpt-4o)

    Returns:
        추정 토큰 수
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # GPT-4 계열 기본 인코딩
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # tiktoken 미설치시 대략적 추정 (한글은 글자당 ~2토큰)
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        other_chars = len(text) - korean_chars
        return korean_chars * 2 + other_chars // 4


# ============================================================
# 전체 실험 토큰 사용량 추정 함수
# - 실험 실행 전 총 비용을 미리 계산하여 예산 계획 수립
# - 프롬프트 길이 + 피드백 텍스트 길이 + 출력 토큰을 모두 고려
# - 예: 416샘플 × 2모델 × 4프롬프트 × 3반복 = 9,984회 호출
# ============================================================
def estimate_experiment_tokens(
    test_df: pd.DataFrame,
    prompt_loader,
    prompt_versions: List[str],
    n_models: int = 2,
    n_seeds: int = 3,
    n_repeats: int = 3,
    max_output_tokens: int = 50,
    text_column: str = "feedback_text"
) -> Dict[str, any]:
    """
    전체 실험의 토큰 사용량 추정

    Args:
        test_df: 테스트 데이터프레임
        prompt_loader: PromptLoader 인스턴스
        prompt_versions: 프롬프트 버전 리스트
        n_models: 모델 수
        n_seeds: seed 수
        n_repeats: 반복 횟수
        max_output_tokens: 최대 출력 토큰 수
        text_column: 텍스트 컬럼명

    Returns:
        토큰 추정 결과 딕셔너리
    """
    results = {
        "prompts": {},
        "summary": {}
    }

    # 피드백 텍스트 평균 토큰 수
    feedback_tokens = test_df[text_column].apply(estimate_tokens)
    avg_feedback_tokens = feedback_tokens.mean()
    min_feedback_tokens = feedback_tokens.min()
    max_feedback_tokens = feedback_tokens.max()

    total_input_tokens = 0
    total_output_tokens = 0

    for prompt_version in prompt_versions:
        try:
            prompt = prompt_loader.load(prompt_version)
            system_tokens = estimate_tokens(prompt.system_prompt)
            user_template_tokens = estimate_tokens(prompt.user_prompt_template)

            # 템플릿에서 {{feedback_text}} 제외한 토큰
            template_overhead = user_template_tokens - estimate_tokens("{{feedback_text}}")

            # 1회 호출당 입력 토큰
            input_per_call = system_tokens + template_overhead + avg_feedback_tokens

            results["prompts"][prompt_version] = {
                "system_tokens": system_tokens,
                "user_template_tokens": user_template_tokens,
                "avg_input_per_call": input_per_call
            }

            # 해당 프롬프트의 총 호출 수
            calls_for_prompt = len(test_df) * n_models * n_seeds * n_repeats
            total_input_tokens += input_per_call * calls_for_prompt
            total_output_tokens += max_output_tokens * calls_for_prompt

        except Exception as e:
            results["prompts"][prompt_version] = {"error": str(e)}

    n_samples = len(test_df)
    total_calls = n_samples * n_models * len(prompt_versions) * n_seeds * n_repeats

    results["summary"] = {
        "n_samples": n_samples,
        "n_models": n_models,
        "n_prompts": len(prompt_versions),
        "n_seeds": n_seeds,
        "n_repeats": n_repeats,
        "total_api_calls": total_calls,
        "avg_feedback_tokens": round(avg_feedback_tokens, 1),
        "min_feedback_tokens": min_feedback_tokens,
        "max_feedback_tokens": max_feedback_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_input_tokens": int(total_input_tokens),
        "estimated_output_tokens": int(total_output_tokens),
        "estimated_total_tokens": int(total_input_tokens + total_output_tokens)
    }

    return results


def print_token_estimation(estimation: Dict) -> None:
    """토큰 추정 결과 출력"""
    print("=" * 70)
    print("토큰 사용량 추정")
    print("=" * 70)

    summary = estimation["summary"]

    print(f"\n[실험 설정]")
    print(f"  샘플 수: {summary['n_samples']:,}개")
    print(f"  모델 수: {summary['n_models']}개")
    print(f"  프롬프트 수: {summary['n_prompts']}개")
    print(f"  Seed 수: {summary['n_seeds']}개")
    print(f"  반복 횟수: {summary['n_repeats']}회")
    print(f"  총 API 호출: {summary['total_api_calls']:,}회")

    print(f"\n[피드백 텍스트 토큰]")
    print(f"  평균: {summary['avg_feedback_tokens']:.1f} tokens")
    print(f"  최소: {summary['min_feedback_tokens']} tokens")
    print(f"  최대: {summary['max_feedback_tokens']} tokens")

    print(f"\n[프롬프트별 토큰]")
    for prompt_name, info in estimation["prompts"].items():
        if "error" in info:
            print(f"  {prompt_name}: 로드 실패 - {info['error']}")
        else:
            print(f"  {prompt_name}:")
            print(f"    시스템 프롬프트: {info['system_tokens']:,} tokens")
            print(f"    사용자 템플릿: {info['user_template_tokens']:,} tokens")
            print(f"    1회 호출 입력: ~{info['avg_input_per_call']:,.0f} tokens")

    print(f"\n[총 토큰 추정]")
    print(f"  입력 토큰: ~{summary['estimated_input_tokens']:,} tokens")
    print(f"  출력 토큰: ~{summary['estimated_output_tokens']:,} tokens")
    print(f"  총 토큰: ~{summary['estimated_total_tokens']:,} tokens")
    print("=" * 70)
