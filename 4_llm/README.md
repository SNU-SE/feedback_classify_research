# 4_llm: 대규모 언어모델(LLM) 실험

GPT-4o 등 상용 LLM API를 이용한 프롬프트 기반 피드백 분류 실험입니다.

## 실험 개요

- **모델**: GPT-4o (OpenAI), Solar-Pro2 (Upstage), EXAONE (LG AI Research)
- **프롬프트 전략**: 3단계 진화
  - P0: Zero-shot + 루브릭 제공
  - P3: Chain-of-Thought + Few-shot + 맥락 정보
  - P9: Hard Tiebreaker (경계 사례 명시적 판별 규칙)
- **평가**: 3회 반복 실행으로 재현성 검증
- **비용**: 프롬프트당 약 $5~10, 전체 실험 약 $60~120

## 노트북

| 파일명 | 설명 |
|--------|------|
| `01_experiment.ipynb` | 8:2 기본 실험 — 모델·프롬프트 조합별 성능 비교 |
| `02_analysis.ipynb` | 기본 실험 결과 분석 — 재현성·일반화 시각화 |
| `03_full_evaluation.ipynb` | 일반화 실험 — 문항 유형별 조합 성능 평가 |
| `04_kexaone_analysis.ipynb` | EXAONE 모델 별도 분석 |

## 분석 순서

### Step 1: 8:2 기본 실험
`01_experiment.ipynb` → `02_analysis.ipynb`
- GPT-4o + P0/P3/P9 프롬프트 조합별 성능 비교
- 3회 반복으로 재현성(일관성) 검증
- 최적 조합: GPT-4o + P9 (Kappa: 0.65, F1: 0.63)
- 결과: `results/baseline/`

### Step 2: 문항 유형별 조합
`03_full_evaluation.ipynb`
- 같은 유형 vs 다른 유형 문항 전이 성능
- 문항 조합별 성능 추이
- 결과: `results/full_evaluation/type_combo/`

### Step 3: EXAONE 모델 분석
`04_kexaone_analysis.ipynb`
- 한국형 LLM(EXAONE)의 성능 별도 평가
- 결과: `results/baseline/kexaone/`

## 프롬프트 진화 과정

```
P0 (Zero-shot)   →  루브릭만 제공, 기본 분류 지시
P3 (CoT+Few-shot) →  단계적 추론 + 예시 제공 + 맥락 정보
P9 (Hard Tiebreak) → 경계 사례(3↔4, 4↔5) 명시적 판별 규칙 추가
```

프롬프트 원문: `prompts/` 폴더 참조

## 사용 도구

- **API**: OpenAI API (GPT-4o), FriendliAI, Upstage Solar
- **프레임워크**: 자체 실험 프레임워크 (`src/`)
  - `config.py`: 모델·실험 설정
  - `experiment.py`: 실험 실행 엔진
  - `prompt_loader.py`: 프롬프트 로딩
  - `models/`: 모델별 API 어댑터

## 실행 방법

```bash
# API 키 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 등 입력

# 의존성 설치
pip install -r requirements.txt
```

## 폴더 구조

```
4_llm/
├── prompts/                   # 프롬프트 원문 (P0, P3, P9)
├── results/
│   ├── baseline/              # 8:2 기본 실험 결과
│   │   ├── confidence/        # 신뢰도 기반 분석
│   │   └── kexaone/           # EXAONE 모델 결과
│   └── full_evaluation/       # 일반화 실험 결과
│       └── type_combo/        # 문항 유형별 조합
├── src/                       # 실험 프레임워크 코드
│   ├── models/                # 모델별 API 어댑터
│   ├── config.py
│   ├── experiment.py
│   ├── prompt_loader.py
│   └── utils.py
├── 01_experiment.ipynb
├── 02_analysis.ipynb
├── 03_full_evaluation.ipynb
├── 04_kexaone_analysis.ipynb
├── .env.example
└── requirements.txt
```
