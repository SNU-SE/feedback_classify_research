# ============================================================
# config.py - LLM 실험 설정 관리 모듈
# ============================================================
# 이 파일은 LLM 기반 피드백 분류 실험의 전체 설정을 관리합니다.
#
# 주요 역할:
#   1. 모델 설정 (ModelConfig): 어떤 LLM을 사용할지 정의
#   2. 실험 설정 (Config): 데이터 경로, 반복 횟수, 분할 비율 등
#
# 프롬프트 엔지니어링(Prompt Engineering) 배경:
#   - LLM에 전달하는 지시문(프롬프트)을 최적화하여 분류 성능을 높이는 과정
#   - 본 연구에서는 P0(제로샷) → P3(CoT+퓨샷) → P9(Hard Tiebreaker)로 진화
#
# 실험 대상 모델:
#   - 상용 API: GPT-4o (OpenAI), K-EXAONE (FriendliAI), Solar (Upstage)
#   - 오픈소스: EXAONE-3.5, Qwen2.5, Llama-3.1, Gemma-2 (HuggingFace)
#
# API 비용 참고:
#   - GPT-4o: 입력 $2.50/1M토큰, 출력 $10.00/1M토큰
#   - 416개 테스트 샘플 × 3회 반복 = 1,248회 API 호출 (약 $5~10/프롬프트)
#   - 전체 실험 (4 프롬프트 × 3회) = 약 $60~120
# ============================================================

"""
실험 설정 관리
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv


# ============================================================
# 모델 설정 (ModelConfig)
# - 각 LLM의 API 접속 정보와 생성 파라미터를 정의
# - model_type에 따라 다른 API 클라이언트가 사용됨
# ============================================================
@dataclass
class ModelConfig:
    """
    모델 설정

    Args:
        model_id: 모델 식별자 (예: "gpt-4o", "Qwen/Qwen2.5-7B-Instruct")
        model_type: 모델 타입
            - 'openai': OpenAI API (GPT-4o, GPT-4o-mini)
            - 'friendliai': FriendliAI API (K-EXAONE)
            - 'upstage': Upstage API (Solar)
            - 'huggingface': 로컬 실행 (GPU 필요)
            - 'hf_inference_api': HuggingFace Inference API (GPU 불필요)
        display_name: 표시용 이름
        temperature: 생성 온도 (0.0 = 결정론적)
        max_tokens: 최대 출력 토큰 수
        reasoning_effort: Upstage Solar Pro 3 모델의 reasoning 강도 ("low", "medium", "high")
    """
    model_id: str  # 모델 식별자 (API에 전달되는 정확한 모델명)
    model_type: str  # 'openai', 'friendliai', 'upstage', 'huggingface', 'hf_inference_api'
    display_name: str  # 결과 출력 시 표시되는 이름
    temperature: float = 0.0  # 생성 온도: 0.0이면 항상 동일한 출력 (결정론적), 높을수록 다양한 출력
    max_tokens: int = 50  # 최대 출력 토큰 수: 레이블(0~6) 하나만 출력하므로 50이면 충분
    reasoning_effort: Optional[str] = None  # Upstage Solar Pro 3 전용: 추론 강도 설정


# ============================================================
# 전체 실험 설정 (Config)
# - 디렉토리 구조, 데이터 분할, 반복 횟수 등 실험 전반을 관리
# - __post_init__에서 환경변수(.env)를 자동 로드하여 API 키 설정
# ============================================================
@dataclass
class Config:
    """전체 실험 설정"""
    # 경로 설정
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    prompts_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    data_dir: Path = field(init=False)

    # 실험 설정
    n_repeats: int = 3  # 재현성 검증을 위한 반복 횟수 (동일 조건에서 3회 반복하여 결과 안정성 확인)
    random_seed: int = 42  # 랜덤 시드 고정 (데이터 분할의 재현성 보장)
    split: str = "80_20"  # 훈련/테스트 분할 비율: "80_20" 또는 "85_15"
    batch_size: int = 1  # 배치 처리 크기 (1이면 한 번에 하나씩 API 호출, 비용 제어에 유리)
    friendli_batch_size: int = 1  # FriendliAI 전용 배치 크기 (RPM 제한 대응)

    # 모델 설정
    models: List[ModelConfig] = field(default_factory=list)

    def __post_init__(self):
        self.prompts_dir = self.base_dir / "prompts"
        self.results_dir = self.base_dir / "results"
        self.runs_dir = self.base_dir / "runs"
        self.data_dir = self.base_dir.parent / "data"

        # 환경 변수 로드
        load_dotenv(self.base_dir / ".env")

        # 기본 모델 설정
        if not self.models:
            self.models = self._get_default_models()

    def _get_default_models(self) -> List[ModelConfig]:
        """
        기본 모델 목록 반환

        연구에서 비교한 7개 모델:
        - GPT-4o: OpenAI의 최신 플래그십 모델 (최고 성능, Kappa ≈ 0.65)
        - GPT-4o-mini: GPT-4o의 경량 버전 (비용 절감용)
        - EXAONE-3.5: LG AI Research의 한국어 특화 오픈소스 모델
        - Qwen2.5: Alibaba의 다국어 오픈소스 모델
        - Llama-3.1: Meta의 오픈소스 모델
        - Gemma-2: Google의 오픈소스 모델
        """
        return [
            # OpenAI 모델 (상용 API - 가장 높은 성능, 비용 발생)
            ModelConfig(
                model_id="gpt-4o",
                model_type="openai",
                display_name="GPT-4o"
            ),
            ModelConfig(
                model_id="gpt-4o-mini",
                model_type="openai",
                display_name="GPT-4o-mini"
            ),
            # HuggingFace 오픈소스 모델 (무료, 로컬 GPU 필요 - VRAM 16GB 이상 권장)
            ModelConfig(
                model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                model_type="huggingface",
                display_name="EXAONE-3.5-7.8B"
            ),
            ModelConfig(
                model_id="Qwen/Qwen2.5-7B-Instruct",
                model_type="huggingface",
                display_name="Qwen2.5-7B"
            ),
            ModelConfig(
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                model_type="huggingface",
                display_name="Llama-3.1-8B"
            ),
            ModelConfig(
                model_id="google/gemma-2-9b-it",
                model_type="huggingface",
                display_name="Gemma-2-9B"
            ),
        ]

    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API 키 반환"""
        return os.getenv("OPENAI_API_KEY")

    @property
    def train_data_path(self) -> Path:
        """학습 데이터 경로 (split에 따라 결정)"""
        return self.data_dir / "feedback_data.xlsx"  # split_seed42 열로 구분

    @property
    def test_data_path(self) -> Path:
        """테스트 데이터 경로 (split에 따라 결정)"""
        return self.data_dir / "feedback_data.xlsx"  # split_seed42 열로 구분

    def validate_split(self) -> bool:
        """split 값 유효성 검사"""
        return self.split in ("80_20", "85_15")

    def get_prompt_versions(self) -> List[str]:
        """
        사용 가능한 프롬프트 버전 목록

        프롬프트 진화 단계:
        - P0: 제로샷(Zero-shot) - 루브릭만 제공, 예시 없음 (Kappa ≈ 0.34)
        - P1~P2: 퓨샷(Few-shot) + CoT - 예시와 사고 과정 포함 (Kappa ≈ 0.53)
        - P9: Hard Tiebreaker - 3/4/5 경계 레이블 구분 규칙 추가 (Kappa ≈ 0.65)
        """
        versions = []
        for f in self.prompts_dir.glob("P*.md"):
            if f.stem.endswith("_batch"):
                continue
            versions.append(f.stem)
        return sorted(versions)
