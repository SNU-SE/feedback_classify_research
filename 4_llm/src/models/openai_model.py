# ============================================================
# openai_model.py - OpenAI API 호환 모델 래퍼
# ============================================================
# 이 파일은 OpenAI API 형식을 사용하는 모든 LLM 서비스를 지원합니다.
#
# 지원 서비스 및 모델:
#   1. OpenAI: GPT-4o, GPT-4o-mini (가장 높은 성능, Kappa ≈ 0.65)
#   2. FriendliAI: K-EXAONE-236B (한국어 특화 대형 모델, Kappa ≈ 0.32)
#   3. Upstage: Solar Pro (한국 기업 모델)
#
# OpenAI-compatible API란?
#   OpenAI가 정의한 chat.completions API 형식을 다른 서비스도 동일하게
#   지원하는 것을 말합니다. base_url만 변경하면 동일한 코드로 다른 모델 사용 가능.
#
# API 비용 참고:
#   - GPT-4o: 입력 $2.50/1M토큰, 출력 $10.00/1M토큰
#   - 416개 테스트셋 1회 실행 ≈ $3~5 (프롬프트 길이에 따라 다름)
#   - K-EXAONE: FriendliAI 크레딧 기반 과금
#   - Solar: Upstage 크레딧 기반 과금
#
# Rate Limiting (요청 속도 제한):
#   - 각 API 제공자는 분당 요청 수(RPM)와 토큰 수(TPM)를 제한
#   - 지수 백오프(Exponential Backoff): 재시도 시 대기 시간을 2배씩 증가
#   - 병렬 처리: ThreadPoolExecutor를 사용하여 동시 요청 수 제어
# ============================================================

"""
OpenAI API 모델 래퍼

OpenAI-compatible API를 지원합니다:
- OpenAI (GPT-4o, GPT-4o-mini)
- FriendliAI (K-EXAONE)
- Upstage (Solar)
"""

import os
import time
from typing import List, Optional

from models.base import BaseLLM, LLMResponse


class OpenAIModel(BaseLLM):
    """
    OpenAI API 래퍼 (OpenAI-compatible API 지원)

    Args:
        model_id: 모델 ID
        temperature: 생성 온도
        max_tokens: 최대 출력 토큰 수
        api_key: API 키 (None이면 환경변수에서 로드)
        base_url: API 기본 URL (None이면 OpenAI 기본값)
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 대기 시간
        extra_body: API 호출 시 추가 파라미터 (FriendliAI의 reasoning 등)

    Example:
        # OpenAI
        model = OpenAIModel("gpt-4o")

        # FriendliAI (K-EXAONE with reasoning)
        model = OpenAIModel(
            model_id="LGAI-EXAONE/K-EXAONE-236B-A23B",
            base_url="https://api.friendli.ai/serverless/v1"
        )
        # FRIENDLI_TOKEN 환경변수 사용

        # Upstage
        model = OpenAIModel(
            model_id="solar-open-100b",
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 50,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        extra_body: Optional[dict] = None,
        reasoning_effort: Optional[str] = None
    ):
        super().__init__(model_id, temperature, max_tokens)

        # API 키 설정 (base_url에 따라 환경변수 선택)
        # base_url로 서비스 제공자를 자동 판별하여 적절한 API 키를 로드
        self.is_friendliai = base_url and "friendli" in base_url.lower()
        self.is_upstage = base_url and "upstage" in base_url.lower()

        if api_key:
            self.api_key = api_key
        elif self.is_friendliai:
            # FriendliAI: FRIENDLI_TOKEN 사용
            self.api_key = os.getenv("FRIENDLI_TOKEN")
        elif self.is_upstage:
            self.api_key = os.getenv("UPSTAGE_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            env_var = "FRIENDLI_TOKEN" if self.is_friendliai else (
                "UPSTAGE_API_KEY" if self.is_upstage else "OPENAI_API_KEY"
            )
            raise ValueError(f"API 키가 필요합니다. {env_var} 환경변수를 설정하세요.")

        self.base_url = base_url
        # provider별 환경변수로 재시도 설정 오버라이드
        if self.is_friendliai:
            max_retries = int(os.getenv("FRIENDLI_MAX_RETRIES", str(max_retries)))
            retry_delay = float(os.getenv("FRIENDLI_RETRY_DELAY", str(retry_delay)))
        elif self.is_upstage:
            max_retries = int(os.getenv("UPSTAGE_MAX_RETRIES", str(max_retries)))
            retry_delay = float(os.getenv("UPSTAGE_RETRY_DELAY", str(retry_delay)))
        else:
            max_retries = int(os.getenv("OPENAI_MAX_RETRIES", str(max_retries)))
            retry_delay = float(os.getenv("OPENAI_RETRY_DELAY", str(retry_delay)))

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # FriendliAI K-EXAONE: 기본 extra_body 설정 (reasoning mode)
        # K-EXAONE은 사고 과정(thinking)을 분리하여 반환하는 기능을 지원
        # enable_thinking=False: 사고 과정을 비활성화하여 응답 속도 향상 및 비용 절감
        if extra_body is not None:
            self.extra_body = extra_body
        elif self.is_friendliai and "K-EXAONE" in model_id:
            self.extra_body = {
                "parse_reasoning": True,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        else:
            self.extra_body = None

        # Upstage Solar Pro 3: reasoning_effort 저장
        self.reasoning_effort = reasoning_effort

        # OpenAI 클라이언트 초기화
        from openai import OpenAI
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # 로깅
        provider = "OpenAI"
        if self.is_friendliai:
            provider = "FriendliAI"
        elif self.is_upstage:
            provider = "Upstage"

        extra_info = " (reasoning mode)" if self.extra_body else ""
        print(f"[{provider}] {model_id} 초기화 완료{extra_info}")

    def predict(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        단일 텍스트 예측 (retries 추적)

        하나의 피드백 텍스트에 대해 LLM API를 호출하고 응답을 파싱합니다.
        네트워크 오류나 API 오류 시 지수 백오프(대기시간 2배 증가)로 재시도합니다.
        """
        retries_count = 0

        for attempt in range(self.max_retries):
            try:
                # API 호출 파라미터 구성
                api_kwargs = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                }
                if self.temperature is not None:
                    api_kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    api_kwargs["max_tokens"] = self.max_tokens

                # FriendliAI K-EXAONE: extra_body 추가
                if self.extra_body:
                    api_kwargs["extra_body"] = self.extra_body

                # Upstage Solar Pro 3: reasoning_effort 추가
                if self.reasoning_effort is not None:
                    if "extra_body" not in api_kwargs:
                        api_kwargs["extra_body"] = {}
                    api_kwargs["extra_body"]["reasoning_effort"] = self.reasoning_effort

                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                except Exception as e:
                    # 일부 서버는 temperature 고정 -> 제거 후 재시도
                    msg = str(e)
                    body_msg = ""
                    if hasattr(e, "body"):
                        try:
                            body_msg = str(e.body)
                        except Exception:
                            body_msg = ""

                    full_msg = f"{msg} {body_msg}".lower()
                    if "cannot be overridden" in full_msg:
                        if "temperature" in full_msg:
                            api_kwargs.pop("temperature", None)
                        if "max_tokens" in full_msg:
                            api_kwargs.pop("max_tokens", None)
                        response = self.client.chat.completions.create(**api_kwargs)
                    else:
                        raise

                # 응답 텍스트 추출
                message = response.choices[0].message
                text = message.content.strip() if message.content else ""

                # FriendliAI K-EXAONE: reasoning_content도 저장 (디버깅용)
                reasoning = getattr(message, 'reasoning_content', None)

                # 토큰 사용량 추출
                usage = getattr(response, 'usage', None)
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
                completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
                total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0

                return LLMResponse.from_text(
                    text,
                    retries=retries_count,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

            except Exception as e:
                retries_count += 1
                if attempt < self.max_retries - 1:
                    # 지수 백오프(Exponential Backoff): 1초 → 2초 → 4초로 대기시간 증가
                    # API 서버 부하를 줄이면서 일시적 오류를 자동 복구
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise e

    def predict_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        show_progress: bool = True,
        rate_limit_delay: float = 0.1,
        max_concurrency: Optional[int] = None
    ) -> List[LLMResponse]:
        """
        배치 예측 (Rate limiting 적용)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompts: 사용자 프롬프트 리스트
            show_progress: 진행률 표시 여부
            rate_limit_delay: API 호출 간 딜레이 (초)
            max_concurrency: 병렬 처리 개수 (None이면 환경변수에서 로드)

        Returns:
            LLMResponse 리스트
        """
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 환경변수로 기본 병렬 수/딜레이 설정
        if max_concurrency is None:
            if self.is_friendliai:
                env_key = "FRIENDLI_MAX_CONCURRENCY"
                delay_key = "FRIENDLI_RATE_LIMIT_DELAY"
            elif self.is_upstage:
                env_key = "UPSTAGE_MAX_CONCURRENCY"
                delay_key = "UPSTAGE_RATE_LIMIT_DELAY"
            else:
                env_key = "OPENAI_MAX_CONCURRENCY"
                delay_key = "OPENAI_RATE_LIMIT_DELAY"

            try:
                max_concurrency = int(os.getenv(env_key, "1"))
            except ValueError:
                max_concurrency = 1

            # rate_limit_delay 환경변수 오버라이드
            if delay_key in os.environ:
                try:
                    rate_limit_delay = float(os.environ[delay_key])
                except ValueError:
                    pass

        if max_concurrency <= 1:
            results = []
            iterator = tqdm(user_prompts, desc=self.model_id) if show_progress else user_prompts

            for prompt in iterator:
                try:
                    response = self.predict(system_prompt, prompt)
                    results.append(response)

                    # Rate limiting
                    if rate_limit_delay > 0:
                        time.sleep(rate_limit_delay)

                except Exception as e:
                    results.append(LLMResponse(
                        text=f"ERROR: {str(e)}",
                        label=None,
                        parsed_ok=False,
                        raw_response=f"ERROR: {str(e)}",
                        retries=self.max_retries
                    ))

            return results

        # 병렬 처리 (chunk 단위로 제출)
        # ThreadPoolExecutor를 사용하여 max_concurrency개의 요청을 동시에 실행
        # 이를 통해 순차 처리 대비 실험 시간을 크게 단축 (예: 5배 빠름)
        results: List[Optional[LLMResponse]] = [None] * len(user_prompts)
        pbar = tqdm(total=len(user_prompts), desc=self.model_id) if show_progress else None

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for start in range(0, len(user_prompts), max_concurrency):
                end = min(start + max_concurrency, len(user_prompts))
                futures = {
                    executor.submit(self.predict, system_prompt, user_prompts[i]): i
                    for i in range(start, end)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = LLMResponse(
                            text=f"ERROR: {str(e)}",
                            label=None,
                            parsed_ok=False,
                            raw_response=f"ERROR: {str(e)}",
                            retries=self.max_retries
                        )
                    if pbar:
                        pbar.update(1)

                # Rate limiting between chunks
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)

        if pbar:
            pbar.close()

        return results
