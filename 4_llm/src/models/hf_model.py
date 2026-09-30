# ============================================================
# hf_model.py - 로컬 HuggingFace 모델 래퍼
# ============================================================
# 이 파일은 로컬 GPU에서 오픈소스 LLM을 직접 실행하는 래퍼입니다.
#
# Inference API(hf_inference_api.py)와의 차이점:
#   - Inference API: HuggingFace 서버에서 실행 (GPU 불필요, 느림)
#   - 로컬 모델: 자체 GPU에서 직접 실행 (GPU 필요, 빠름)
#
# GPU 요구사항:
#   - 7B 모델: VRAM 16GB 이상 (RTX 4090, A100 등)
#   - bfloat16 정밀도로 메모리 절약 (FP32 대비 절반)
#   - MPS(Apple Silicon) 지원: M1/M2 Mac에서도 실행 가능
#
# 지연 로딩(Lazy Loading) 패턴:
#   - 모델을 처음 predict() 호출 시에만 로드 (초기화 시 로드하지 않음)
#   - 모델 로드에 수십 초~수 분이 소요되므로 필요할 때만 로드
#   - unload()로 GPU 메모리를 명시적으로 해제 (다른 모델 로드 전)
# ============================================================

"""
HuggingFace 모델 래퍼
"""

from typing import List, Optional
import torch

from models.base import BaseLLM, LLMResponse


def get_device() -> str:
    """
    사용 가능한 디바이스 반환

    우선순위: CUDA(NVIDIA GPU) > MPS(Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# 로컬 HuggingFace 모델 클래스
# - BaseLLM을 상속받아 로컬 GPU 기반 예측 구현
# - 지연 로딩: predict() 최초 호출 시 모델 로드
# - 메모리 관리: unload()로 GPU 메모리 해제
# ============================================================
class HuggingFaceModel(BaseLLM):
    """HuggingFace 모델 래퍼 (로컬 GPU 실행)"""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 50,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True
    ):
        super().__init__(model_id, temperature, max_tokens)

        self.device = device or get_device()  # 자동 디바이스 선택 (CUDA > MPS > CPU)
        self.torch_dtype = torch_dtype  # bfloat16: FP32 대비 메모리 절반, 성능 유사
        self.trust_remote_code = trust_remote_code  # 일부 모델은 커스텀 코드 실행 필요

        # 지연 로딩을 위한 상태 변수
        self.tokenizer = None  # 토크나이저 (텍스트 → 토큰 변환)
        self.model = None  # 언어 모델 본체
        self._loaded = False  # 로드 완료 여부

    def load(self):
        """모델 로드 (지연 로딩)"""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"모델 로드 중: {self.model_id}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code
        )

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=self.trust_remote_code
        )

        # MPS의 경우 수동으로 디바이스 이동
        if self.device == "mps":
            self.model = self.model.to(self.device)

        self._loaded = True
        print(f"모델 로드 완료: {self.model_id} (device: {self.device})")

    def unload(self):
        """모델 언로드 (메모리 해제)"""
        if self._loaded:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self._loaded = False

    def predict(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """단일 텍스트 예측"""

        # 모델이 로드되지 않았으면 로드
        if not self._loaded:
            self.load()

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 토크나이즈: 텍스트를 모델이 이해할 수 있는 토큰 ID 시퀀스로 변환
        # apply_chat_template: 모델별 대화 형식(system/user/assistant)을 자동 적용
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            )
        else:
            # chat_template이 없는 경우 수동 포맷
            text = f"{system_prompt}\n\n{user_prompt}"
            input_ids = self.tokenizer.encode(text, return_tensors="pt")

        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]

        # 생성 파라미터 설정
        # max_new_tokens: 최대 출력 토큰 수 (레이블 하나만 출력하므로 50이면 충분)
        # do_sample=False: 가장 확률 높은 토큰만 선택 (결정론적 출력)
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # temperature 처리
        if self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # 디코딩: 생성된 토큰 ID를 텍스트로 변환 (입력 부분은 제외하고 새로 생성된 부분만)
        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return LLMResponse.from_text(text.strip())

    def predict_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        show_progress: bool = True,
        batch_size: int = 1
    ) -> List[LLMResponse]:
        """
        배치 예측

        Args:
            system_prompt: 시스템 프롬프트
            user_prompts: 사용자 프롬프트 리스트
            show_progress: 진행률 표시 여부
            batch_size: 배치 크기 (메모리에 따라 조절)

        Returns:
            LLMResponse 리스트
        """
        from tqdm import tqdm

        # 모델 로드
        if not self._loaded:
            self.load()

        results = []
        iterator = tqdm(user_prompts, desc=self.name) if show_progress else user_prompts

        for prompt in iterator:
            try:
                response = self.predict(system_prompt, prompt)
                results.append(response)
            except Exception as e:
                results.append(LLMResponse(text=f"ERROR: {str(e)}", label=-1))

        return results

    def __del__(self):
        """소멸자 - 메모리 정리"""
        self.unload()
