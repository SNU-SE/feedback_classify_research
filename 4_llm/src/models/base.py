# ============================================================
# base.py - LLM 모델 추상 베이스 클래스
# ============================================================
# 이 파일은 모든 LLM 모델 래퍼의 공통 인터페이스를 정의합니다.
#
# 설계 패턴: 추상 클래스(Abstract Base Class) 패턴
#   - BaseLLM을 상속받아 각 LLM 제공자별 구현 클래스를 작성
#   - OpenAIModel, HuggingFaceModel, HuggingFaceInferenceAPI가 이를 상속
#   - predict() 메서드만 구현하면 다른 기능은 자동으로 사용 가능
#
# LLM 응답 파싱 전략:
#   LLM은 항상 텍스트로 응답하므로, 이를 정수 레이블(0~6)로 변환해야 합니다.
#   파싱 우선순위:
#     1. JSON 형식: {"label": 3} → 가장 정확한 형식
#     2. CoT 형식: "최종 레이블: 3" → Chain-of-Thought 추론 후 결론
#     3. 단순 숫자: "3" → 텍스트가 거의 숫자만 포함할 때
#     4. 첫 번째 숫자 (fallback): "레이블은 3입니다" → 불확실한 파싱
# ============================================================

"""
LLM 모델 추상 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


# ============================================================
# LLM 응답 데이터 클래스
# - LLM이 반환한 텍스트에서 레이블을 추출하고 메타데이터를 저장
# - parsed_ok: 응답이 정상적으로 파싱되었는지 여부 (품질 지표)
# - retries: API 호출 재시도 횟수 (네트워크 안정성 지표)
# ============================================================
@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    text: str  # 파싱된 텍스트 (레이블 추출 대상)
    label: Optional[int] = None  # 추출된 레이블 (0~6, 파싱 실패 시 None)
    confidence: Optional[float] = None  # 신뢰도 (JSON에 포함된 경우)
    raw_response: str = ""  # LLM의 원본 응답 텍스트 (디버깅용)
    parsed_ok: bool = False  # 파싱 성공 여부 (False면 불확실한 파싱)
    retries: int = 0  # API 호출 재시도 횟수
    # 토큰 사용량 추적 (API 비용 계산에 사용)
    prompt_tokens: int = 0  # 입력 토큰 수 (프롬프트 + 피드백 텍스트)
    completion_tokens: int = 0  # 출력 토큰 수 (LLM 응답)
    total_tokens: int = 0  # 총 토큰 수

    # ============================================================
    # 응답 텍스트에서 레이블을 자동 파싱하는 팩토리 메서드
    # 파싱 순서 (우선순위가 높은 것부터):
    #   1. {"label": N} JSON 형식 → parsed_ok = True
    #   2. "최종 레이블: N" CoT 형식 → parsed_ok = True
    #   3. 텍스트가 거의 숫자만 포함 → parsed_ok = True
    #   4. 첫 번째 숫자 fallback → parsed_ok = False (불확실)
    # ============================================================
    @classmethod
    def from_text(
        cls,
        text: str,
        retries: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0
    ) -> 'LLMResponse':
        """텍스트에서 레이블 파싱 ({"label": N} 형식 우선)"""
        import re
        import json

        response = cls(
            text=text,
            raw_response=text,
            retries=retries,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        # [1순위] JSON 형식 파싱 시도 - {"label": N} 형식
        # 프롬프트에서 JSON 형식으로 응답하도록 지시하므로 가장 먼저 시도
        try:
            # JSON 블록 추출
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                if 'label' in data:
                    label_val = int(data['label'])
                    # 유효 범위 체크 (0-6)
                    if 0 <= label_val <= 6:
                        response.label = label_val
                        response.parsed_ok = True
                        response.confidence = float(data.get('confidence', 0))
                        return response
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # [2순위] "최종 레이블: X" 형식 파싱 (CoT 추론 결과)
        cot_match = re.search(r'최종\s*레이블[:\s]+(\d)', text)
        if cot_match:
            label_val = int(cot_match.group(1))
            if 0 <= label_val <= 6:
                response.label = label_val
                response.parsed_ok = True
                return response

        # [3순위] 단순 숫자 파싱 (텍스트가 거의 숫자만 포함할 때)
        number_match = re.search(r'^[^\d]*(\d)[^\d]*$', text.strip())
        if number_match:
            label_val = int(number_match.group(1))
            if 0 <= label_val <= 6:
                response.label = label_val
                response.parsed_ok = True
                return response

        # [4순위] 첫 번째 숫자 찾기 (fallback - parsed_ok = False로 유지하여 불확실성 표시)
        first_digit = re.search(r'\d', text)
        if first_digit:
            label_val = int(first_digit.group())
            if 0 <= label_val <= 6:
                response.label = label_val
                # parsed_ok는 False로 유지 (불확실한 파싱)

        return response


# ============================================================
# LLM 추상 베이스 클래스
# - 모든 LLM 모델이 구현해야 하는 공통 인터페이스 정의
# - predict(): 단일 텍스트 예측 (각 모델별로 구현 필수)
# - predict_batch(): 배치 예측 (기본 구현: 순차 처리, 오버라이드 가능)
# ============================================================
class BaseLLM(ABC):
    """LLM 추상 베이스 클래스"""

    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: Optional[int] = 50):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def predict(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        단일 텍스트 예측

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트

        Returns:
            LLMResponse 객체
        """
        pass

    def predict_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        show_progress: bool = True
    ) -> List[LLMResponse]:
        """
        배치 예측 (기본 구현: 순차 처리)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompts: 사용자 프롬프트 리스트
            show_progress: 진행률 표시 여부

        Returns:
            LLMResponse 리스트
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(user_prompts, desc=self.model_id) if show_progress else user_prompts

        for prompt in iterator:
            try:
                response = self.predict(system_prompt, prompt)
                results.append(response)
            except Exception as e:
                # 에러 발생 시 빈 응답 (parsed_ok=False, label=None)
                results.append(LLMResponse(
                    text=f"ERROR: {str(e)}",
                    label=None,
                    parsed_ok=False,
                    raw_response=f"ERROR: {str(e)}"
                ))

        return results

    @property
    def name(self) -> str:
        """모델 이름"""
        return self.model_id.split("/")[-1]

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
