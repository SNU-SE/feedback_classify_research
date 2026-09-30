# ============================================================
# prompt_loader.py - 프롬프트 파일 관리 모듈
# ============================================================
# 이 파일은 마크다운(.md) 형식의 프롬프트 파일을 파싱하여
# LLM에 전달할 시스템 프롬프트와 사용자 프롬프트를 로드합니다.
#
# 프롬프트 엔지니어링(Prompt Engineering) 배경:
#   LLM에 전달하는 지시문을 최적화하여 분류 성능을 높이는 과정입니다.
#   본 연구에서는 프롬프트를 단계적으로 개선했습니다:
#
# 프롬프트 진화 과정 (왜 각 전환이 일어났는가):
#
#   P0 (제로샷, Zero-shot): 루브릭만 제공, 예시 없음
#     → 문제: Kappa ≈ 0.34로 매우 낮음. LLM이 레이블 경계를 이해하지 못함
#     → 전환 이유: 예시가 없으면 한국어 피드백의 미묘한 차이를 구분 못함
#
#   P1~P3 (CoT + 퓨샷, Few-shot): 사고 과정(Chain-of-Thought)과 예시 포함
#     → 개선: Kappa ≈ 0.53으로 상승. 기본적인 분류 패턴은 학습함
#     → 문제: 레이블 3(문제 지적), 4(해결 방향), 5(구체적 대안) 간 혼동 심함
#     → 전환 이유: 이 세 레이블의 경계가 모호하여 추가 규칙이 필요
#
#   P9 (Hard Tiebreaker): 경계 레이블 구분을 위한 명시적 규칙 추가
#     → 개선: Kappa ≈ 0.65로 최종 달성. 3/4/5 경계 혼동이 크게 감소
#     → 핵심: "3은 문제만 지적, 4는 방향만 제시, 5는 구체적 방법 포함" 규칙
#
# 프롬프트 파일 구조 (마크다운 형식):
#   # Prompt: P9_P6_hard_tiebreak
#   - prompt_version: P9_v1.0
#   - task: 피드백 분류
#   ## SYSTEM
#   (시스템 프롬프트 내용 - LLM의 역할과 루브릭 정의)
#   ## USER
#   (사용자 프롬프트 템플릿 - {{feedback_text}}가 실제 텍스트로 치환됨)
# ============================================================

"""
프롬프트 로더 - MD 파일에서 프롬프트 로드
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================
# 프롬프트 데이터 클래스
# - 하나의 프롬프트 파일에서 추출된 정보를 담는 구조체
# - system_prompt: LLM의 역할 정의 (채점 기준, 분류 규칙 등)
# - user_prompt_template: 개별 피드백을 삽입할 템플릿
# ============================================================
@dataclass
class Prompt:
    """프롬프트 데이터 클래스"""
    version: str
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    metadata: Dict = field(default_factory=dict)

    def format_user_prompt(self, feedback_text: str, student_id: str = "") -> str:
        """
        사용자 프롬프트에 피드백 텍스트와 학생 ID 삽입

        템플릿의 {{feedback_text}}를 실제 피드백 텍스트로 치환합니다.
        예: "다음 피드백을 분류하세요: {{feedback_text}}"
            → "다음 피드백을 분류하세요: 실험 결과가 잘못되었어요"
        """
        result = self.user_prompt_template.replace("{{feedback_text}}", feedback_text)
        result = result.replace("{{id}}", student_id)
        return result

    def __repr__(self):
        return f"Prompt(version='{self.version}', name='{self.name}')"


# ============================================================
# 프롬프트 파일 로더 클래스
# - prompts/ 디렉토리에서 마크다운 파일을 읽어 Prompt 객체로 변환
# - 캐싱을 통해 동일 프롬프트의 중복 파싱 방지
# - _batch 접미사가 붙은 파일은 배치 모드 전용 프롬프트
# ============================================================
class PromptLoader:
    """프롬프트 파일 로더"""

    def __init__(self, prompts_dir: str | Path):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Prompt] = {}

    def load(self, version: str) -> Prompt:
        """
        버전명으로 프롬프트 로드

        Args:
            version: 프롬프트 버전 (예: 'P0_zero-shot_rubric')

        Returns:
            Prompt 객체
        """
        if version in self._cache:
            return self._cache[version]

        # 파일 경로 결정
        file_path = self.prompts_dir / f"{version}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")

        # 파일 파싱
        prompt = self._parse_prompt_file(file_path)
        self._cache[version] = prompt

        return prompt

    def _parse_prompt_file(self, file_path: Path) -> Prompt:
        """
        MD 파일 파싱 (새 양식: ## SYSTEM, ## USER 섹션)

        마크다운의 ## 레벨 헤더를 기준으로 섹션을 분리합니다.
        SYSTEM 섹션 = 시스템 프롬프트 (LLM 역할 + 루브릭 + 분류 규칙)
        USER 섹션 = 사용자 프롬프트 템플릿 ({{feedback_text}} 포함)
        """
        content = file_path.read_text(encoding='utf-8')

        # 메타데이터 파싱 (헤더 아래 불릿 리스트)
        metadata = self._parse_metadata(content)

        # 프롬프트 이름 추출 (# Prompt: {name})
        name_match = re.search(r'^#\s+Prompt:\s*(.+)$', content, re.MULTILINE)
        name = name_match.group(1).strip() if name_match else file_path.stem

        # 섹션별 내용 추출
        sections = self._parse_sections(content)

        return Prompt(
            version=metadata.get('prompt_version', file_path.stem),
            name=name,
            description=metadata.get('task', ''),
            system_prompt=sections.get('SYSTEM', ''),
            user_prompt_template=sections.get('USER', ''),
            metadata=metadata
        )

    def _parse_metadata(self, content: str) -> Dict:
        """헤더 아래 불릿 리스트 형식 메타데이터 파싱"""
        metadata = {}

        # 불릿 리스트 항목 파싱 (- key: value 형식)
        pattern = r'^-\s*(\w+):\s*(.+)$'
        for match in re.finditer(pattern, content, re.MULTILINE):
            key = match.group(1).strip()
            value = match.group(2).strip()
            metadata[key] = value

        return metadata

    def _parse_sections(self, content: str) -> Dict[str, str]:
        """
        마크다운 섹션 파싱 (## 레벨 섹션)

        ## SYSTEM
        content...

        ## USER
        content...
        """
        sections = {}

        # ## 레벨 섹션 분리
        pattern = r'^##\s+(.+?)$'
        parts = re.split(pattern, content, flags=re.MULTILINE)

        # 첫 번째는 섹션 이전 내용 (버림)
        parts = parts[1:]

        # 섹션명과 내용 쌍으로 처리
        for i in range(0, len(parts) - 1, 2):
            section_name = parts[i].strip()
            section_content = parts[i + 1].strip() if i + 1 < len(parts) else ''
            # --- 구분자 제거
            section_content = re.sub(r'^---\s*$', '', section_content, flags=re.MULTILINE).strip()
            sections[section_name] = section_content

        return sections

    def list_versions(self) -> list:
        """
        사용 가능한 프롬프트 버전 목록

        prompts/ 디렉토리에서 P로 시작하는 .md 파일을 검색합니다.
        _batch 접미사 파일은 배치 모드 전용이므로 제외합니다.
        """
        versions = []
        for f in self.prompts_dir.glob("P*.md"):
            if f.stem.endswith("_batch"):
                continue
            versions.append(f.stem)
        return sorted(versions)

    def load_all(self) -> Dict[str, Prompt]:
        """모든 프롬프트 로드"""
        prompts = {}
        for version in self.list_versions():
            prompts[version] = self.load(version)
        return prompts
