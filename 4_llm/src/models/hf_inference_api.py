# ============================================================
# hf_inference_api.py - HuggingFace Inference API 래퍼
# ============================================================
# 이 파일은 HuggingFace 서버에서 오픈소스 LLM을 실행하는 API를 래핑합니다.
#
# 왜 Inference API를 사용하는가?
#   - 로컬 GPU 없이도 대형 모델(7B~70B 파라미터)을 사용할 수 있음
#   - EXAONE-3.5, Qwen2.5, Llama-3.1, Gemma-2 등 오픈소스 모델 테스트에 활용
#   - GPT-4o와의 성능 비교를 위해 다양한 모델을 실험하는 데 사용
#
# 비용:
#   - 무료 티어: 분당 요청 제한 있음 (실험 속도 느림)
#   - Pro 구독 ($9/월): 더 많은 모델, 더 빠른 응답
#   - 상용 API(GPT-4o) 대비 훨씬 저렴하지만 성능도 낮음
#
# 실험 결과 참고:
#   - 오픈소스 모델들은 GPT-4o 대비 성능이 크게 낮음 (F1 ≈ 0.3~0.5)
#   - 한국어 이해 능력의 차이가 주요 원인으로 분석됨
# ============================================================

"""
HuggingFace Inference API 래퍼

HuggingFace 서버에서 모델을 실행하는 API 방식입니다.
로컬 GPU 없이도 대형 모델을 사용할 수 있습니다.

사용 방법:
    1. HuggingFace 토큰 발급: https://huggingface.co/settings/tokens
    2. .env 파일에 HF_TOKEN=hf_xxx... 추가
    3. 모델 초기화 후 predict() 호출

지원 모델 예시:
    - meta-llama/Llama-3.1-8B-Instruct (Pro 구독 필요)
    - mistralai/Mistral-7B-Instruct-v0.3
    - Qwen/Qwen2.5-7B-Instruct
    - google/gemma-2-9b-it
    - LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct

참고:
    - 무료 티어: 분당 요청 제한 있음
    - Pro 구독 ($9/월): 더 많은 모델, 더 빠른 응답
    - 일부 모델은 Inference API 미지원 (로컬 실행 필요)
"""

import os
import time
from typing import List, Optional
from huggingface_hub import InferenceClient
from huggingface_hub.inference._client import InferenceTimeoutError

from models.base import BaseLLM, LLMResponse


# ============================================================
# HuggingFace Inference API 모델 클래스
# - BaseLLM을 상속받아 HuggingFace 서버 기반 예측 구현
# - InferenceClient를 통해 chat_completion API 호출
# - Rate limit, 모델 로딩 대기, 타임아웃 등 자동 처리
# ============================================================
class HuggingFaceInferenceAPI(BaseLLM):
    """
    HuggingFace Inference API를 사용한 모델 래퍼

    로컬 GPU 없이 HuggingFace 서버에서 모델을 실행합니다.

    Args:
        model_id: HuggingFace 모델 ID (예: "meta-llama/Llama-3.1-8B-Instruct")
        temperature: 생성 온도 (0.0 = 결정론적)
        max_tokens: 최대 출력 토큰 수
        token: HuggingFace API 토큰 (None이면 환경변수에서 로드)
        timeout: API 타임아웃 (초)
        retry_count: 재시도 횟수
        retry_delay: 재시도 대기 시간 (초)

    Example:
        >>> model = HuggingFaceInferenceAPI("Qwen/Qwen2.5-7B-Instruct")
        >>> response = model.predict(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="What is 2+2?"
        ... )
        >>> print(response.text)
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 50,
        token: Optional[str] = None,
        timeout: int = 120,
        retry_count: int = 3,
        retry_delay: float = 5.0
    ):
        super().__init__(model_id, temperature, max_tokens)

        # HuggingFace 토큰 로드
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError(
                "HuggingFace 토큰이 필요합니다.\n"
                "1. https://huggingface.co/settings/tokens 에서 토큰 발급\n"
                "2. .env 파일에 HF_TOKEN=hf_xxx... 추가"
            )

        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # Inference Client 초기화
        self.client = InferenceClient(
            model=model_id,
            token=self.token,
            timeout=timeout
        )

        print(f"HuggingFace Inference API 초기화: {model_id}")

    def predict(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        단일 텍스트 예측

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트

        Returns:
            LLMResponse 객체
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        last_error = None
        for attempt in range(self.retry_count):
            try:
                # Chat Completion API 호출
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else 0.01,
                    # temperature=0(완전 결정론적)은 일부 HuggingFace 모델에서 오류 발생
                    # 0.01로 대체하여 거의 동일한 결과를 얻으면서 오류 방지
                )

                # 응답 텍스트 추출
                text = response.choices[0].message.content.strip()
                return LLMResponse.from_text(text)

            except InferenceTimeoutError as e:
                last_error = e
                print(f"  타임아웃 (시도 {attempt + 1}/{self.retry_count})")
                time.sleep(self.retry_delay)

            except Exception as e:
                last_error = e
                error_msg = str(e).strip()
                if not error_msg:
                    error_msg = repr(e)

                # HTTP 응답 정보가 있으면 함께 출력
                if hasattr(e, "response") and e.response is not None:
                    try:
                        status_code = getattr(e.response, "status_code", "unknown")
                        resp_text = getattr(e.response, "text", "")
                        resp_text = resp_text.strip().replace("\n", " ")
                        if resp_text:
                            error_msg = f"{error_msg} | HTTP {status_code}: {resp_text[:200]}"
                        else:
                            error_msg = f"{error_msg} | HTTP {status_code}"
                    except Exception:
                        pass

                # Rate limit 처리
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"  Rate limit, {wait_time}초 대기...")
                    time.sleep(wait_time)

                # 모델 로딩 중
                elif "loading" in error_msg.lower():
                    print(f"  모델 로딩 중, 30초 대기...")
                    time.sleep(30)

                # 기타 오류
                else:
                    print(f"  오류: {error_msg}")
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)

        # 모든 재시도 실패
        return LLMResponse(
            text=f"ERROR: {str(last_error) or repr(last_error)}",
            label=None,
            parsed_ok=False,
            raw_response=str(last_error) or repr(last_error),
            retries=self.retry_count
        )

    def predict_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        show_progress: bool = True,
        delay_between_requests: float = 0.5
    ) -> List[LLMResponse]:
        """
        배치 예측

        Args:
            system_prompt: 시스템 프롬프트
            user_prompts: 사용자 프롬프트 리스트
            show_progress: 진행률 표시 여부
            delay_between_requests: 요청 간 대기 시간 (Rate limit 방지)

        Returns:
            LLMResponse 리스트
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(user_prompts, desc=f"[API] {self.name}") if show_progress else user_prompts

        for prompt in iterator:
            response = self.predict(system_prompt, prompt)
            results.append(response)

            # Rate limit 방지를 위한 대기
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        return results

    def check_model_status(self) -> dict:
        """
        모델 상태 확인

        Returns:
            모델 상태 정보 딕셔너리
        """
        try:
            status = self.client.get_model_status()
            return {
                "model_id": self.model_id,
                "loaded": status.loaded,
                "state": status.state,
                "framework": status.framework,
                "compute_type": getattr(status, 'compute_type', 'unknown')
            }
        except Exception as e:
            return {
                "model_id": self.model_id,
                "error": str(e)
            }


# ============================================================
# 편의 함수
# ============================================================

# ============================================================
# 편의 함수
# - 사용 가능한 모델 목록 조회 및 API 연결 테스트
# ============================================================

def list_available_models() -> List[str]:
    """
    Inference API에서 사용 가능한 주요 한국어 지원 모델 목록

    한국어 피드백 분류를 위해 한국어 이해 능력이 중요합니다.
    EXAONE-3.5은 한국어 특화, Qwen2.5는 다국어 지원으로 한국어 성능이 양호합니다.

    Returns:
        모델 ID 리스트
    """
    return [
        # 한국어 특화
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",

        # 다국어 지원 (한국어 포함)
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",

        # Meta Llama
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",

        # Google Gemma
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",

        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]


def test_inference_api(model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> bool:
    """
    Inference API 연결 테스트

    Args:
        model_id: 테스트할 모델 ID

    Returns:
        성공 여부
    """
    try:
        model = HuggingFaceInferenceAPI(model_id, max_tokens=20)
        response = model.predict(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello' in Korean."
        )
        print(f"테스트 성공: {response.text}")
        return True
    except Exception as e:
        print(f"테스트 실패: {e}")
        return False


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    # 환경 변수 로드
    from dotenv import load_dotenv
    load_dotenv()

    print("=== HuggingFace Inference API 테스트 ===\n")

    # 사용 가능한 모델 목록
    print("사용 가능한 모델:")
    for model in list_available_models():
        print(f"  - {model}")

    print("\n" + "=" * 50)

    # 간단한 테스트
    print("\n테스트 실행...")
    test_inference_api("Qwen/Qwen2.5-7B-Instruct")
