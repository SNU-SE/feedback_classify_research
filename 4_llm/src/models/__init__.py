"""
LLM 모델 래퍼

사용 가능한 모델 클래스:
    - OpenAIModel: OpenAI API (GPT-4o, GPT-4o-mini)
    - HuggingFaceModel: 로컬 실행 (GPU 필요)
    - HuggingFaceInferenceAPI: HuggingFace API (GPU 불필요, 권장)
"""

from .base import BaseLLM, LLMResponse
from .openai_model import OpenAIModel
from .hf_model import HuggingFaceModel
from .hf_inference_api import HuggingFaceInferenceAPI

__all__ = [
    'BaseLLM',
    'LLMResponse',
    'OpenAIModel',
    'HuggingFaceModel',
    'HuggingFaceInferenceAPI'
]
