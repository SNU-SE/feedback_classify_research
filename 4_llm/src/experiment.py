# ============================================================
# experiment.py - LLM 실험 실행기 (메인 파이프라인)
# ============================================================
# 이 파일은 LLM 기반 피드백 분류 실험의 핵심 실행 모듈입니다.
#
# 전체 실험 흐름:
#   1. 테스트 데이터 로드 (416개 또는 312개 피드백 텍스트)
#   2. 각 피드백에 대해 LLM API 호출 → 레이블(0~6) 예측
#   3. LLM 응답 텍스트에서 레이블 파싱 (JSON 형식: {"label": N})
#   4. 실제 레이블과 비교하여 성능 지표 계산 (F1, Kappa 등)
#   5. 결과 저장 (predictions.csv, metrics.csv, config.json)
#
# 프롬프트 엔지니어링(Prompt Engineering) 전략:
#   - 제로샷(Zero-shot): 예시 없이 루브릭(채점 기준)만 제공하여 분류
#   - 퓨샷(Few-shot): 소수의 예시를 포함하여 분류 정확도 향상
#   - CoT(Chain-of-Thought): 단계별 사고 과정을 유도하는 프롬프트 기법
#   - Hard Tiebreaker: 경계 레이블(3/4/5) 구분을 위한 명시적 규칙 추가
#
# 주요 클래스:
#   - ExperimentResult: 단일 실험의 결과를 담는 데이터 클래스
#   - ExperimentRunner: 실험 실행, 결과 저장, 배치 처리를 관리
#
# API 비용 주의:
#   - 전체 실험(2모델 × 4프롬프트 × 3반복) = 약 $60~120
#   - 한 번의 run_full_experiment() 호출로 모든 조합이 자동 실행됨
# ============================================================

"""
실험 실행기
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm

from config import Config, ModelConfig
from prompt_loader import PromptLoader, Prompt
from models.base import BaseLLM, LLMResponse
from models.openai_model import OpenAIModel
from models.hf_model import HuggingFaceModel
from models.hf_inference_api import HuggingFaceInferenceAPI
from utils import compute_metrics, set_seed


# ============================================================
# 실험 결과 데이터 클래스
# - 한 번의 실험(1모델 × 1프롬프트 × 1회)의 모든 정보를 저장
# - 샘플별 예측값, 원본 응답, 토큰 사용량 등을 포함
# ============================================================
@dataclass
class ExperimentResult:
    """실험 결과 데이터 클래스 (새 CSV 스키마 지원)"""
    # 기본 식별자 - 실험을 고유하게 식별하는 정보
    run_id: str  # 실행 ID, 예: "2026-01-25_P2_v1.3_gpt-4o_80_20"
    split: str  # 데이터 분할 비율: "80_20" 또는 "85_15"
    model: str  # 모델 표시명
    prompt_version: str  # 프롬프트 버전, 예: "P9_v1.0"

    # 레거시 호환 (이전 버전 결과 파일과의 호환성 유지)
    model_id: str  # 모델의 전체 ID (API용)
    model_name: str  # 모델 표시명
    timestamp: str  # 실험 실행 시각 (ISO 형식)

    # 예측 결과 (샘플별) - 각 피드백 텍스트에 대한 LLM 응답 상세 정보
    sample_ids: List  # 원본 데이터의 ID 목록
    predictions: List[Optional[int]]  # 예측 레이블 (0~6, 파싱 실패 시 None)
    true_labels: List[int]  # 실제 레이블 (0~6, 인간 코더가 부여)
    parsed_ok_list: List[bool]  # LLM 응답에서 레이블 파싱 성공 여부
    retries_list: List[int]  # API 호출 재시도 횟수 (네트워크 오류 등)
    raw_responses: List[str]  # LLM의 원본 응답 텍스트

    # 요약 메트릭 - 전체 성능 지표
    metrics: Dict[str, float]  # accuracy, f1_macro, kappa 등
    config: Dict  # 실험 설정 요약

    # 토큰 사용량 (비용 추적용 - API 비용 = 입력토큰 + 출력토큰)
    prompt_tokens_list: List[int] = None  # 샘플별 입력 토큰 수
    completion_tokens_list: List[int] = None  # 샘플별 출력 토큰 수
    total_prompt_tokens: int = 0  # 전체 입력 토큰 합계
    total_completion_tokens: int = 0  # 전체 출력 토큰 합계
    total_tokens: int = 0  # 전체 토큰 합계 (비용 계산에 사용)


# ============================================================
# 실험 실행기 클래스
# - 모델 로드, 데이터 준비, 예측 실행, 결과 저장을 총괄
# - run_single_experiment(): 1모델 × 1프롬프트 × 1회 실행
# - run_full_experiment(): 모든 조합을 자동으로 순차 실행
# ============================================================
class ExperimentRunner:
    """실험 실행기"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.prompt_loader = PromptLoader(self.config.prompts_dir)
        self.models: Dict[str, BaseLLM] = {}

    def load_model(self, model_config: ModelConfig) -> BaseLLM:
        """
        모델 로드 (캐싱)

        Args:
            model_config: 모델 설정 객체
                - model_type='openai': OpenAI API
                - model_type='friendliai': FriendliAI API (K-EXAONE)
                - model_type='upstage': Upstage API (Solar)
                - model_type='huggingface': 로컬 실행 (GPU 필요)
                - model_type='hf_inference_api': HuggingFace API (GPU 불필요)

        Returns:
            BaseLLM 인스턴스
        """
        if model_config.model_id in self.models:
            return self.models[model_config.model_id]

        if model_config.model_type == "openai":
            model = OpenAIModel(
                model_id=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
        elif model_config.model_type == "friendliai":
            model = OpenAIModel(
                model_id=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                base_url="https://api.friendli.ai/serverless/v1"
            )
        elif model_config.model_type == "upstage":
            model = OpenAIModel(
                model_id=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                base_url="https://api.upstage.ai/v1/solar",
                reasoning_effort=model_config.reasoning_effort
            )
        elif model_config.model_type == "huggingface":
            model = HuggingFaceModel(
                model_id=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
        elif model_config.model_type == "hf_inference_api":
            model = HuggingFaceInferenceAPI(
                model_id=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.model_type}")

        self.models[model_config.model_id] = model
        return model

    def load_test_data(self) -> pd.DataFrame:
        """테스트 데이터 로드"""
        df = pd.read_excel(self.config.test_data_path)
        return df[df['split_seed42'] == 'test'].reset_index(drop=True)

    def _get_rate_limit_delay(self, model_config: ModelConfig) -> float:
        """모델 타입별 rate limit delay (초)"""
        if model_config.model_type == "friendliai":
            key = "FRIENDLI_RATE_LIMIT_DELAY"
        elif model_config.model_type == "upstage":
            key = "UPSTAGE_RATE_LIMIT_DELAY"
        else:
            key = "OPENAI_RATE_LIMIT_DELAY"

        try:
            return float(os.getenv(key, "0"))
        except ValueError:
            return 0.0

    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        응답 텍스트에서 JSON 배열 문자열 추출

        LLM이 배치 모드에서 반환하는 JSON 배열을 추출합니다.
        예: ```json [{"id": "001", "label": 3}, ...] ``` → 배열 부분만 추출
        """
        if not text:
            return None

        fence_match = re.search(r"```(?:json)?\\s*(\\[.*?\\])\\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return None

    def _parse_batch_response(self, text: str) -> tuple[dict, list]:
        """
        배치 응답(JSON 리스트) 파싱 -> (labels_by_id, order_labels)

        LLM의 배치 응답에서 레이블을 추출하는 두 가지 방식:
        1. labels_by_id: {"id": "001", "label": 3} 형식 → ID로 매칭
        2. order_labels: [3, 1, 4, ...] 형식 → 순서로 매칭
        """
        labels_by_id: dict = {}
        order_labels: list = []

        json_array = self._extract_json_array(text)
        if not json_array:
            return labels_by_id, order_labels

        try:
            data = json.loads(json_array)
        except Exception:
            return labels_by_id, order_labels

        if not isinstance(data, list):
            return labels_by_id, order_labels

        for item in data:
            if isinstance(item, dict):
                label_val = item.get("label", None)
                try:
                    label_int = int(label_val)
                except Exception:
                    continue

                if 0 <= label_int <= 6:
                    if "id" in item:
                        labels_by_id[str(item["id"])] = label_int
                    else:
                        order_labels.append(label_int)
            else:
                # [0,1,2] 형태도 지원
                try:
                    label_int = int(item)
                except Exception:
                    continue
                if 0 <= label_int <= 6:
                    order_labels.append(label_int)

        return labels_by_id, order_labels

    # ============================================================
    # 배치 예측 메서드
    # - 여러 샘플을 한 번의 API 호출로 처리 (비용 절감)
    # - FriendliAI (K-EXAONE) 등 RPM(분당 요청수) 제한이 있는 API에 유용
    # - 단, 배치 응답의 파싱이 더 복잡하고 오류 가능성이 높음
    # ============================================================
    def _predict_json_batch(
        self,
        model: BaseLLM,
        prompt: Prompt,
        test_df: pd.DataFrame,
        text_column: str = "feedback_text",
        id_column: str = "id",
        batch_size: int = 1,
        model_config: Optional[ModelConfig] = None
    ):
        """배치 예측 (id 포함 JSON 리스트 반환 요구)"""
        batch_size = int(batch_size)
        n_samples = len(test_df)

        predictions = [None] * n_samples
        parsed_ok_list = [False] * n_samples
        retries_list = [0] * n_samples
        raw_responses = [""] * n_samples

        if id_column in test_df.columns:
            sample_ids = test_df[id_column].tolist()
        else:
            sample_ids = test_df.index.tolist()

        if model_config is None:
            model_config = ModelConfig(
                model_id=model.model_id,
                model_type="openai",
                display_name=model.name
            )

        delay = self._get_rate_limit_delay(model_config)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_df = test_df.iloc[start:end]
            batch_items = []
            for idx, row in batch_df.iterrows():
                item_id = row[id_column] if id_column in batch_df.columns else idx
                batch_items.append({
                    "id": item_id,
                    "feedback_text": str(row[text_column])
                })

            batch_items_json = json.dumps(batch_items, ensure_ascii=False, indent=2)
            if "{{batch_items}}" not in prompt.user_prompt_template:
                raise ValueError(
                    "배치 프롬프트 템플릿에 {{batch_items}}가 없습니다. "
                    f"프롬프트 파일을 확인하세요: {prompt.version}"
                )
            user_prompt = prompt.user_prompt_template.replace("{{batch_items}}", batch_items_json)

            resp = None
            raw_text = ""
            labels_by_id = {}
            order_labels = []

            try:
                resp = model.predict(
                    system_prompt=prompt.system_prompt,
                    user_prompt=user_prompt
                )
                raw_text = resp.text or resp.raw_response
                labels_by_id, order_labels = self._parse_batch_response(raw_text)
            except Exception as e:
                raw_text = f"ERROR: {str(e)}"

            # 배치 응답 검증
            expected_ids = [str(x) for x in sample_ids[start:end]]
            if labels_by_id:
                missing_ids = [eid for eid in expected_ids if eid not in labels_by_id]
                extra_ids = [eid for eid in labels_by_id.keys() if eid not in expected_ids]
                if missing_ids:
                    print(f"    [경고] 배치 응답 누락 ID {len(missing_ids)}개: {missing_ids[:5]}")
                if extra_ids:
                    print(f"    [경고] 배치 응답 추가 ID {len(extra_ids)}개: {extra_ids[:5]}")
            elif order_labels:
                if len(order_labels) != (end - start):
                    print(f"    [경고] 배치 응답 길이 불일치: {len(order_labels)} != {end - start}")
            else:
                print("    [경고] 배치 응답 파싱 실패 (JSON 리스트 없음)")

            for local_idx, (row_idx, sample_id) in enumerate(zip(range(start, end), sample_ids[start:end])):
                label_val = None
                parsed_ok = False
                key = str(sample_id)

                if key in labels_by_id:
                    label_val = labels_by_id[key]
                    parsed_ok = True
                elif order_labels and local_idx < len(order_labels):
                    label_val = order_labels[local_idx]
                    parsed_ok = True

                predictions[row_idx] = label_val
                parsed_ok_list[row_idx] = parsed_ok
                retries_list[row_idx] = getattr(resp, "retries", 0) if resp else 0
                raw_responses[row_idx] = raw_text

            if delay > 0:
                time.sleep(delay)

        return predictions, parsed_ok_list, retries_list, raw_responses

    # ============================================================
    # 단일 실험 실행 메서드
    # - 하나의 (모델, 프롬프트) 조합으로 전체 테스트셋 예측
    # - 각 피드백 텍스트를 프롬프트에 삽입 → LLM API 호출 → 응답 파싱
    # - 결과: 예측 레이블 목록 + 성능 메트릭 (Accuracy, F1, Kappa)
    # ============================================================
    def run_single_experiment(
        self,
        model: BaseLLM,
        prompt: Prompt,
        test_df: pd.DataFrame,
        run_number: int = 1,
        text_column: str = "feedback_text",
        label_column: str = "label",
        id_column: str = "id",
        model_config: Optional[ModelConfig] = None
    ) -> ExperimentResult:
        """
        단일 실험 실행

        Args:
            model: LLM 모델
            prompt: 프롬프트
            test_df: 테스트 데이터프레임
            run_number: 실행 번호
            text_column: 텍스트 컬럼명
            label_column: 레이블 컬럼명
            id_column: 샘플 ID 컬럼명

        Returns:
            ExperimentResult 객체
        """
        # FriendliAI 배치 모드 (RPM 제한 대응)
        # RPM = Requests Per Minute: API 제공자가 분당 호출 횟수를 제한함
        # 배치 모드로 여러 샘플을 한 번에 보내면 호출 횟수를 줄일 수 있음
        batch_size = getattr(self.config, "batch_size", 1)
        if model_config is not None and model_config.model_type == "friendliai":
            friendli_bs = getattr(self.config, "friendli_batch_size", 1)
            if friendli_bs > 1:
                batch_size = friendli_bs

        use_batch = model_config is not None and batch_size > 1

        if use_batch:
            (
                predictions,
                parsed_ok_list,
                retries_list,
                raw_responses
            ) = self._predict_json_batch(
                model=model,
                prompt=prompt,
                test_df=test_df,
                text_column=text_column,
                id_column=id_column,
                batch_size=batch_size,
                model_config=model_config
            )
            # 배치 모드에서는 토큰 추적 미지원 (추후 개선 가능)
            prompt_tokens_list = [0] * len(predictions)
            completion_tokens_list = [0] * len(predictions)
        else:
            # 사용자 프롬프트 생성
            user_prompts = [
                prompt.format_user_prompt(str(text))
                for text in test_df[text_column]
            ]

            # 예측 실행
            responses = model.predict_batch(
                system_prompt=prompt.system_prompt,
                user_prompts=user_prompts,
                show_progress=True
            )

            # 결과 추출
            predictions = [r.label for r in responses]  # None if parsing failed
            parsed_ok_list = [r.parsed_ok for r in responses]
            retries_list = [r.retries for r in responses]
            raw_responses = [r.raw_response for r in responses]
            # 토큰 사용량 추출
            prompt_tokens_list = [r.prompt_tokens for r in responses]
            completion_tokens_list = [r.completion_tokens for r in responses]

        true_labels = test_df[label_column].tolist()

        # 샘플 ID 추출 (컬럼이 없으면 인덱스 사용)
        if id_column in test_df.columns:
            sample_ids = test_df[id_column].tolist()
        else:
            sample_ids = test_df.index.tolist()

        # 메트릭 계산 (parsed_ok_list 전달)
        metrics = compute_metrics(true_labels, predictions, parsed_ok_list)

        # run_id 생성: "YYYY-MM-DD_prompt_model_split"
        timestamp = datetime.now()
        run_id = f"{timestamp.strftime('%Y-%m-%d')}_{prompt.version}_{model.name}_{self.config.split}"

        # 토큰 사용량 합계 계산
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)
        total_tokens = total_prompt_tokens + total_completion_tokens

        # 결과 객체 생성
        result = ExperimentResult(
            run_id=run_id,
            split=self.config.split,
            model=model.name,
            prompt_version=prompt.version,
            model_id=model.model_id,
            model_name=model.name,
            timestamp=timestamp.isoformat(),
            sample_ids=sample_ids,
            predictions=predictions,
            true_labels=true_labels,
            parsed_ok_list=parsed_ok_list,
            retries_list=retries_list,
            raw_responses=raw_responses,
            metrics=metrics,
            config={
                "temperature": model.temperature,
                "max_tokens": model.max_tokens,
                "prompt_name": prompt.name,
                "split": self.config.split
            },
            # 토큰 사용량
            prompt_tokens_list=prompt_tokens_list,
            completion_tokens_list=completion_tokens_list,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens
        )

        return result

    # ============================================================
    # 전체 실험 실행 메서드
    # - 모든 (모델 × 프롬프트 × 반복) 조합을 순차적으로 실행
    # - 예: 2모델 × 4프롬프트 × 3반복 = 24개 실험 자동 실행
    # - 각 실험 결과는 즉시 저장 (중단 시에도 이전 결과 보존)
    # - HuggingFace 모델은 실험 후 GPU 메모리 해제 (모델 언로드)
    # ============================================================
    def run_full_experiment(
        self,
        model_configs: List[ModelConfig] = None,
        prompt_versions: List[str] = None,
        n_repeats: int = None,
        save_results: bool = True
    ) -> List[ExperimentResult]:
        """
        전체 실험 실행

        Args:
            model_configs: 모델 설정 리스트 (None이면 config의 기본값)
            prompt_versions: 프롬프트 버전 리스트 (None이면 모든 버전)
            n_repeats: 반복 횟수 (None이면 config의 기본값)
            save_results: 결과 저장 여부

        Returns:
            ExperimentResult 리스트
        """
        # 기본값 설정
        model_configs = model_configs or self.config.models
        prompt_versions = prompt_versions or self.prompt_loader.list_versions()
        n_repeats = n_repeats or self.config.n_repeats

        # 시드 설정
        set_seed(self.config.random_seed)

        # 테스트 데이터 로드
        test_df = self.load_test_data()
        print(f"테스트 데이터 로드 완료: {len(test_df)} 샘플")

        # 실험 실행
        all_results = []
        total_experiments = len(model_configs) * len(prompt_versions) * n_repeats

        print(f"\n총 {total_experiments}개 실험 시작")
        print(f"- 모델: {len(model_configs)}개")
        print(f"- 프롬프트: {len(prompt_versions)}개")
        print(f"- 반복: {n_repeats}회")
        print("=" * 50)

        for model_config in model_configs:
            print(f"\n[모델] {model_config.display_name}")

            try:
                model = self.load_model(model_config)
            except Exception as e:
                print(f"  모델 로드 실패: {e}")
                continue

            for prompt_version in prompt_versions:
                print(f"  [프롬프트] {prompt_version}")

                # 배치 모드면 _batch 프롬프트 우선 사용
                prompt_to_load = prompt_version
                prompt = None
                batch_size = getattr(self.config, "batch_size", 1)
                if model_config.model_type == "friendliai":
                    friendli_bs = getattr(self.config, "friendli_batch_size", 1)
                    if friendli_bs > 1:
                        batch_size = friendli_bs

                use_batch = batch_size > 1

                if use_batch:
                    candidate = f"{prompt_version}_batch"
                    try:
                        prompt = self.prompt_loader.load(candidate)
                        prompt_to_load = candidate
                    except Exception:
                        prompt = None

                if use_batch and prompt is None:
                    print(f"    배치 프롬프트 없음: {prompt_version}_batch")
                    continue

                if prompt is None:
                    try:
                        prompt = self.prompt_loader.load(prompt_version)
                    except Exception as e:
                        print(f"    프롬프트 로드 실패: {e}")
                        continue

                for run_number in range(1, n_repeats + 1):
                    print(f"    [실행] {run_number}/{n_repeats}")

                    try:
                        result = self.run_single_experiment(
                            model=model,
                            prompt=prompt,
                            test_df=test_df,
                            run_number=run_number,
                            model_config=model_config
                        )
                        all_results.append(result)

                        print(f"      Accuracy: {result.metrics['accuracy']:.4f}")
                        print(f"      F1 Macro: {result.metrics['f1_macro']:.4f}")

                        # 결과 즉시 저장
                        if save_results:
                            self._save_result(result)

                    except Exception as e:
                        print(f"    실험 실패: {e}")
                        continue

            # HuggingFace 모델인 경우 메모리 해제
            if model_config.model_type == "huggingface":
                if hasattr(model, 'unload'):
                    model.unload()
                del self.models[model_config.model_id]

        print("\n" + "=" * 50)
        print(f"총 {len(all_results)}개 실험 완료")

        return all_results

    # ============================================================
    # 결과 저장 메서드
    # - 각 실험 결과를 개별 디렉토리에 저장하여 추적 용이
    # - predictions.csv: 샘플별 예측 상세 (원본 응답 포함)
    # - metrics.csv: 성능 요약 (Accuracy, F1, Kappa 등)
    # - config.json: 실험 설정 (모델, 프롬프트, 파라미터 등)
    # ============================================================
    def _save_result(self, result: ExperimentResult):
        """
        단일 결과 저장 (새 디렉토리 구조)

        runs/
          run_2026-01-25_P2_v1.3_gpt-4o_80_20/
            predictions.csv    ← 샘플별 예측 상세
            metrics.csv        ← 성능 요약 지표
            config.json        ← 실험 설정 정보
        """
        # run 디렉토리 생성
        run_dir = self.config.runs_dir / f"run_{result.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. predictions.csv (per-sample)
        predictions_data = []
        for i in range(len(result.sample_ids)):
            predictions_data.append({
                "run_id": result.run_id,
                "split": result.split,
                "model": result.model,
                "prompt_version": result.prompt_version,
                "sample_id": result.sample_ids[i],
                "y_true": result.true_labels[i],
                "y_pred": result.predictions[i] if result.predictions[i] is not None else "NA",
                "parsed_ok": result.parsed_ok_list[i],
                "retries": result.retries_list[i],
                "response_text": result.raw_responses[i]
            })

        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        # 2. metrics.csv (summary)
        metrics_data = {
            "run_id": result.run_id,
            "split": result.split,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "n": result.metrics.get("total_samples", len(result.sample_ids)),
            "n_fail": result.metrics.get("n_fail", 0),
            "accuracy": result.metrics.get("accuracy", 0.0),
            "macro_f1": result.metrics.get("f1_macro", 0.0),
            "kappa": result.metrics.get("kappa", 0.0),
            "recall_0": result.metrics.get("recall_0", 0.0),
            "f1_0": result.metrics.get("f1_0", 0.0),
            "recall_6": result.metrics.get("recall_6", 0.0),
            "f1_6": result.metrics.get("f1_6", 0.0)
        }

        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")

        # 3. config.json
        config_data = {
            "run_id": result.run_id,
            "split": result.split,
            "model": result.model,
            "model_id": result.model_id,
            "prompt_version": result.prompt_version,
            "timestamp": result.timestamp,
            **result.config
        }

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # 레거시 raw JSON도 저장 (선택적)
        raw_dir = self.config.results_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        result_dict = asdict(result)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(raw_dir / f"{result.run_id}_{timestamp_str}.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 전체 결과 통합 저장 메서드
    # - 모든 실험 결과를 하나의 파일로 통합하여 분석 편의성 제공
    # - 모델-프롬프트별 평균 성능 요약 테이블도 자동 생성
    # ============================================================
    def save_all_results(self, results: List[ExperimentResult]):
        """
        전체 결과 요약 저장 (새 스키마 적용)

        - llm_predictions.csv: 모든 per-sample 예측 통합
        - llm_metrics.csv: 모든 실험의 요약 메트릭 통합
        - summary.csv: 모델-프롬프트별 평균±표준편차 요약
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = self.config.results_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # 1. llm_predictions.csv (per-sample 통합)
        all_predictions = []
        for r in results:
            for i in range(len(r.sample_ids)):
                all_predictions.append({
                    "run_id": r.run_id,
                    "split": r.split,
                    "model": r.model,
                    "prompt_version": r.prompt_version,
                    "sample_id": r.sample_ids[i],
                    "y_true": r.true_labels[i],
                    "y_pred": r.predictions[i] if r.predictions[i] is not None else "NA",
                    "parsed_ok": r.parsed_ok_list[i],
                    "retries": r.retries_list[i],
                    "response_text": r.raw_responses[i]
                })

        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(
            metrics_dir / f"llm_predictions_{timestamp}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # 2. llm_metrics.csv (summary per experiment)
        metrics_data = []
        for r in results:
            row = {
                "run_id": r.run_id,
                "split": r.split,
                "model": r.model,
                "prompt_version": r.prompt_version,
                "n": r.metrics.get("total_samples", len(r.sample_ids)),
                "n_fail": r.metrics.get("n_fail", 0),
                "accuracy": r.metrics.get("accuracy", 0.0),
                "macro_f1": r.metrics.get("f1_macro", 0.0),
                "kappa": r.metrics.get("kappa", 0.0),
                "recall_0": r.metrics.get("recall_0", 0.0),
                "f1_0": r.metrics.get("f1_0", 0.0),
                "recall_6": r.metrics.get("recall_6", 0.0),
                "f1_6": r.metrics.get("f1_6", 0.0)
            }
            metrics_data.append(row)

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(
            metrics_dir / f"llm_metrics_{timestamp}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # 3. 모델-프롬프트별 평균 계산 (summary)
        # 3회 반복의 평균과 표준편차를 계산하여 결과의 안정성을 확인
        summary_df = metrics_df.groupby(["model", "prompt_version", "split"]).agg({
            "accuracy": ["mean", "std"],
            "macro_f1": ["mean", "std"],
            "kappa": ["mean", "std"],
            "recall_0": ["mean", "std"],
            "f1_0": ["mean", "std"],
            "recall_6": ["mean", "std"],
            "f1_6": ["mean", "std"],
            "n_fail": ["sum"]
        }).round(4)

        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df.to_csv(
            metrics_dir / f"summary_{timestamp}.csv",
            encoding="utf-8-sig"
        )

        print(f"\n결과 저장 완료:")
        print(f"  - 예측 상세: {metrics_dir / f'llm_predictions_{timestamp}.csv'}")
        print(f"  - 메트릭 상세: {metrics_dir / f'llm_metrics_{timestamp}.csv'}")
        print(f"  - 요약: {metrics_dir / f'summary_{timestamp}.csv'}")

        return metrics_df, summary_df, predictions_df
