"""
RunPod BERT 일반화 실험 (Generalization Experiment)
=====================================================

목적:
    KLUE-BERT 모델의 일반화 성능을 체계적으로 검증합니다.
    새로운 문항이나 새로운 학생 집단에서도 피드백 분류가 잘 작동하는지 확인합니다.

실험 구성:
    Part 1 - 문항 조합 실험 (254개 조합):
        8개 문항(Q1~Q8) 중 k개(1~7)를 선택하여 학습하고,
        나머지 문항의 피드백을 분류합니다.
        → 어떤 문항 조합이 가장 일반적인 분류 모델을 만드는지 확인

    Part 2 - 학생 분할 실험 (15개 분할):
        학생 비율(80:20, 70:30, 60:40, 50:50, 40:60)과
        3개 시드(42, 43, 44)로 학습/테스트 분할하여 실험
        → 학습 데이터 양에 따른 성능 변화 확인

RunPod 클라우드 GPU 사용 이유:
    - BERT 미세조정은 GPU가 필수 (CPU 대비 10~50배 빠름)
    - 254개 조합 × 10-fold CV = 약 2,540회 학습 필요
    - 로컬 GPU 1대로는 수일 소요 → RunPod에서 4~5개 Pod 병렬 실행
    - 각 Pod는 독립적인 GPU 서버로, 실험을 분배하여 동시 처리

Pod 분배 방식:
    - Pod 1~4: 문항 조합 254개를 GPU 시간 기준으로 균등 분배 (bin-packing)
    - Pod 5: 학생 분할 15개 실행
    - 각 Pod는 독립적으로 CSV에 결과를 기록하며, 중단 시 재개 가능(resume)

사용법:
    # 문항 조합 실험 (Pod 1~4에서 각각 실행)
    python3 bert_experiment.py --mode question --pod 1 --total-pods 4

    # 학생 분할 실험 (Pod 5에서 실행)
    python3 bert_experiment.py --mode student

    # 특정 조합 범위만 수동 실행
    python3 bert_experiment.py --mode question --start 0 --end 63

    # 모든 실험을 한 머신에서 실행
    python3 bert_experiment.py --mode all
"""

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

from tqdm import tqdm


# ============================================================
# Section 1: 실험 설정 (Configuration)
# ============================================================
# KLUE-BERT 모델의 하이퍼파라미터 및 RunPod 환경 설정입니다.
# RunPod의 /workspace/ 경로는 Pod 내 영구 저장소입니다.
# CUDA 최적화: FP16 혼합정밀도(Mixed Precision)로 학습 속도 향상
# ============================================================

CONFIG = {
    # Model
    "model_name": "KLUE-BERT",
    "model_path": "klue/bert-base",  # 사전학습 모델(Pre-trained Model): 대규모 한국어 텍스트로 미리 학습된 언어 모델

    # Data
    "data_dir": "/workspace/data",
    "text_column": "feedback_text",
    "label_column": "label",

    # Training
    "max_length": 128,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clip": 1.0,

    # CUDA 최적화 설정
    "use_fp16": True,       # FP16 혼합정밀도: 메모리 절약 + 학습 속도 2배 향상
    "num_workers": 2,       # 데이터 로딩 병렬 워커 수 (CUDA 환경에서 효과적)
    "pin_memory": True,     # CPU→GPU 데이터 전송 속도 향상

    # Early stopping
    "patience": 2,

    # Cross-validation
    "n_folds": 10,
    "seed": 42,

    # Labels
    "num_labels": 7,

    # Output
    "output_dir": "/workspace/results",
}

# Question mapping (same as ML experiment)
Q_MAP = {
    'Q1': ('a', 'q1'), 'Q2': ('a', 'q2'), 'Q3': ('a', 'q3'), 'Q4': ('a', 'q4'),
    'Q5': ('b', 'q1'), 'Q6': ('b', 'q2'), 'Q7': ('b', 'q3'), 'Q8': ('b', 'q4'),
}
ALL_QUESTIONS = list(Q_MAP.keys())


# ============================================================
# Section 2: 유틸리티 클래스 및 함수
# ============================================================
# 재현성 보장, 데이터셋 변환, 조기 종료, 학습/평가 루프 등
# 모든 실험에서 공통으로 사용되는 핵심 컴포넌트입니다.
# ============================================================

def set_seed(seed=42):
    """재현성을 위한 시드 고정 (CUDA 환경)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeedbackDataset(Dataset):
    """PyTorch Dataset for feedback classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience=2, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif (self.mode == 'max' and score <= self.best_score) or \
             (self.mode == 'min' and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                gradient_clip=1.0, use_fp16=False, scaler=None):
    """한 에폭 학습 (FP16 혼합정밀도 지원).

    FP16 혼합정밀도(Mixed Precision)란?
        일부 연산을 32비트 대신 16비트 부동소수점으로 수행하여
        메모리 사용량을 줄이고 학습 속도를 높이는 기법입니다.
        GradScaler가 작은 기울기의 underflow를 방지합니다.
    """
    model.train()  # 학습 모드 활성화 (dropout, batch norm 등 활성)
    total_loss = 0
    predictions, true_labels = [], []

    for batch in dataloader:
        optimizer.zero_grad()  # 이전 배치의 기울기 초기화
        input_ids = batch['input_ids'].to(device)       # 토큰 ID → GPU
        attention_mask = batch['attention_mask'].to(device)  # 어텐션 마스크 → GPU
        labels = batch['label'].to(device)              # 정답 레이블 → GPU

        if use_fp16 and scaler is not None:
            # FP16 혼합정밀도 학습 경로 (GPU에서만 사용)
            with autocast():  # 자동으로 FP16/FP32 혼합 적용
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()       # 스케일링된 역전파
            scaler.unscale_(optimizer)           # 기울기 원래 스케일로 복원
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)               # 가중치 갱신
            scaler.update()                      # 스케일러 상태 업데이트
        else:
            # 일반(FP32) 학습 경로
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(true_labels, predictions)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return metrics + predictions."""
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'macro_f1': f1_score(true_labels, predictions, average='macro'),
        'weighted_f1': f1_score(true_labels, predictions, average='weighted'),
        'kappa': cohen_kappa_score(true_labels, predictions),
    }
    return metrics, predictions, true_labels


# ============================================================
# Section 3: 핵심 실험 실행기 (Core Experiment Runner)
# ============================================================
# 하나의 실험 단위(문항 조합 1개 또는 학생 분할 1개)를 완전히 실행합니다.
# 10-Fold CV → 전체 데이터 재학습 → 테스트 평가의 전체 파이프라인입니다.
# ============================================================

def run_single_experiment(train_texts, train_labels, test_texts, test_labels,
                          config, device, experiment_name=""):
    """단일 실험 실행: 10-fold CV + 전체 학습 + 테스트 평가.

    하나의 실험(예: Q1+Q2로 학습 → Q3~Q8 테스트)의 전체 파이프라인을 수행합니다.

    반환: (CV 요약, 테스트 지표, 테스트 예측값, 모델, 토크나이저)
    """
    # --- 10-Fold CV ---
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        set_seed(config["seed"] + fold)

        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

        train_dataset = FeedbackDataset(fold_train_texts, fold_train_labels, tokenizer, config["max_length"])
        val_dataset = FeedbackDataset(fold_val_texts, fold_val_labels, tokenizer, config["max_length"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=config["num_workers"], pin_memory=config["pin_memory"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                                num_workers=config["num_workers"], pin_memory=config["pin_memory"])

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_path"], num_labels=config["num_labels"])
        model.to(device)

        # 클래스 가중치 계산 (불균형 데이터 핵심 처리)
        # 문항 조합에 따라 특정 클래스가 아예 없을 수 있으므로,
        # 없는 클래스는 가중치 1.0으로 설정하고, 있는 클래스만 balanced 계산
        all_classes = np.arange(config["num_labels"])
        present_classes = np.unique(fold_train_labels)
        present_weights = compute_class_weight('balanced', classes=present_classes, y=fold_train_labels)
        class_weights = np.ones(config["num_labels"], dtype=np.float32)
        for cls, w in zip(present_classes, present_weights):
            class_weights[cls] = w
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # AdamW 옵티마이저 + 학습률 워밍업 스케줄러
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        scaler = GradScaler() if config["use_fp16"] else None  # FP16용 기울기 스케일러
        early_stopping = EarlyStopping(patience=config["patience"], mode='max')

        for epoch in range(config["epochs"]):
            train_loss, _ = train_epoch(model, train_loader, optimizer, scheduler, criterion,
                                        device, config["gradient_clip"], config["use_fp16"], scaler)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            early_stopping(val_metrics['macro_f1'], model)
            if early_stopping.early_stop:
                break

        early_stopping.load_best_model(model)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        fold_results.append(val_metrics)

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    cv_summary = {
        'CV_Accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'CV_Accuracy_Std': np.std([r['accuracy'] for r in fold_results]),
        'CV_F1': np.mean([r['macro_f1'] for r in fold_results]),
        'CV_F1_Std': np.std([r['macro_f1'] for r in fold_results]),
        'CV_Weighted_F1': np.mean([r['weighted_f1'] for r in fold_results]),
        'CV_Weighted_F1_Std': np.std([r['weighted_f1'] for r in fold_results]),
        'CV_Kappa': np.mean([r['kappa'] for r in fold_results]),
        'CV_Kappa_Std': np.std([r['kappa'] for r in fold_results]),
    }

    # --- Final Model Training ---
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer, config["max_length"])
    test_dataset = FeedbackDataset(test_texts, test_labels, tokenizer, config["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_path"], num_labels=config["num_labels"])
    model.to(device)

    all_classes = np.arange(config["num_labels"])
    present_classes = np.unique(train_labels)
    present_weights = compute_class_weight('balanced', classes=present_classes, y=train_labels)
    class_weights = np.ones(config["num_labels"], dtype=np.float32)
    for cls, w in zip(present_classes, present_weights):
        class_weights[cls] = w
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler() if config["use_fp16"] else None

    for epoch in range(config["epochs"]):
        train_epoch(model, train_loader, optimizer, scheduler, criterion,
                    device, config["gradient_clip"], config["use_fp16"], scaler)

    # --- Test Evaluation ---
    criterion_eval = nn.CrossEntropyLoss()
    test_metrics, predictions, true_labels = evaluate(model, test_loader, criterion_eval, device)

    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return cv_summary, test_metrics, predictions, model, tokenizer


# ============================================================
# Section 4: HuggingFace Hub 모델 업로드
# ============================================================
# 학습된 모델을 HuggingFace Hub에 업로드하여 중앙 관리합니다.
# 각 실험 결과(모델 가중치 + 토크나이저 + 성능 정보)를
# 서브폴더 구조로 저장하여 나중에 쉽게 불러올 수 있습니다.
# ============================================================

def upload_model_to_hf(model, tokenizer, hf_repo, hf_token, subfolder,
                       cv_summary, test_metrics, train_info):
    """학습된 모델을 HuggingFace Hub의 서브폴더에 업로드합니다.

    Repo structure:
        {hf_repo}/
        ├── question_combos/Q1/
        ├── question_combos/Q1_Q2/
        ├── student_splits/80_20_seed42/
        └── ...

    Each model can be loaded with:
        AutoModelForSequenceClassification.from_pretrained(
            "{hf_repo}", subfolder="question_combos/Q1_Q2_Q3")
    """
    if not hf_token or not hf_repo:
        return False

    import shutil
    from huggingface_hub import HfApi

    try:
        # Save to temp directory
        tmp_dir = Path("/workspace/_tmp_model")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        # Create model card
        model_card = f"""---
language: ko
tags:
  - text-classification
  - bert
  - korean
  - peer-feedback
---

# {subfolder}

KLUE-BERT feedback classifier trained on: **{train_info}**

| Metric | CV (10-fold) | Test |
|--------|:---:|:---:|
| Accuracy | {cv_summary['CV_Accuracy']:.4f} | {test_metrics['accuracy']:.4f} |
| Macro-F1 | {cv_summary['CV_F1']:.4f} | {test_metrics['macro_f1']:.4f} |
| Kappa | {cv_summary['CV_Kappa']:.4f} | {test_metrics['kappa']:.4f} |

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "{hf_repo}", subfolder="{subfolder}")
tokenizer = AutoTokenizer.from_pretrained(
    "{hf_repo}", subfolder="{subfolder}")
```
"""
        (tmp_dir / "README.md").write_text(model_card)

        # Upload
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=hf_repo,
            path_in_repo=subfolder,
            commit_message=f"Add {subfolder} (F1={test_metrics['macro_f1']:.4f})",
        )

        # Cleanup temp
        shutil.rmtree(tmp_dir)
        return True

    except Exception as e:
        print(f"    HF upload failed: {e}")
        return False


# ============================================================
# Section 5: 문항 조합 실험 (Question Combination Experiment)
# ============================================================
# 8개 문항(Q1~Q8)에서 k개(1~7)를 선택하는 모든 조합(C(8,1)+...+C(8,7)=254)에 대해
# 선택된 문항으로 학습하고 나머지 문항으로 테스트합니다.
#
# 이를 통해 확인하는 것:
#   - 최소 몇 개 문항이 있어야 일반화 성능이 확보되는지
#   - 같은 시험지(Form) 내 문항 vs 다른 시험지 문항의 전이 효과
#   - 어떤 문항이 분류 모델 학습에 가장 유용한지
#
# Pod 간 작업 분배:
#   bin-packing 알고리즘으로 각 Pod의 예상 GPU 시간이 균등하도록 분배합니다.
#   큰 조합(7문항)은 학습 데이터가 많아 시간이 오래 걸리므로
#   이를 고려하여 Pod 간 부하를 균형 맞춥니다.
# ============================================================

def generate_all_combos():
    """254개 문항 조합 생성 (k=1~7). ML 실험과 동일한 순서를 보장합니다."""
    all_combos = []
    for k in range(1, 8):
        for combo in combinations(ALL_QUESTIONS, k):
            all_combos.append(combo)
    return all_combos


def estimate_combo_time(combo, samples_per_q):
    """조합의 예상 GPU 시간(분) 추정 (Pod 간 부하 균형용)."""
    train_n = sum(samples_per_q.get(q, 260) for q in combo)
    return 1.0 + (train_n / 1664) * 19.0  # ~20 min for full dataset


def assign_combos_to_pods(all_combos, n_pods, samples_per_q):
    """탐욕 bin-packing: 예상 시간 기준으로 조합을 Pod에 균등 분배합니다."""
    # Estimate time for each combo
    combo_times = [(i, estimate_combo_time(c, samples_per_q)) for i, c in enumerate(all_combos)]

    # Sort by time descending for better packing
    sorted_combos = sorted(combo_times, key=lambda x: x[1], reverse=True)

    pods = [[] for _ in range(n_pods)]
    pod_times = [0.0] * n_pods

    for combo_idx, t in sorted_combos:
        min_pod = pod_times.index(min(pod_times))
        pods[min_pod].append(combo_idx)
        pod_times[min_pod] += t

    # Sort each pod's combos by original index
    for p in range(n_pods):
        pods[p].sort()

    return pods, pod_times


def load_question_data(data_dir):
    """Load Q1-Q8 data files from question_data directory."""
    q_dir = Path(data_dir) / "question_data"
    question_data = {}

    for q_label in ALL_QUESTIONS:
        filepath = q_dir / f"{q_label}.xlsx"
        if not filepath.exists():
            print(f"  ERROR: {filepath} not found")
            sys.exit(1)
        df = pd.read_excel(filepath)
        question_data[q_label] = df

    return question_data


def get_completed_experiments(csv_path):
    """이미 완료된 실험명을 CSV에서 로드 (중단 후 재개 지원)."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df['Combo'].tolist())
    except Exception:
        return set()


def run_question_experiments(combo_indices, config, device, hf_repo=None, hf_token=None):
    """지정된 조합 인덱스에 대한 문항 조합 실험을 실행합니다."""
    data_dir = config["data_dir"]
    output_dir = Path(config["output_dir"])
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load question data
    print("\n  Loading question data (Q1-Q8)...")
    question_data = load_question_data(data_dir)
    samples_per_q = {q: len(df) for q, df in question_data.items()}
    for q, n in samples_per_q.items():
        print(f"    {q}: {n} samples")

    # Generate all combos (deterministic order)
    all_combos = generate_all_combos()
    print(f"\n  Total combos: {len(all_combos)}, assigned to this pod: {len(combo_indices)}")

    # Resume support
    csv_path = output_dir / "bert_question_results.csv"
    completed = get_completed_experiments(csv_path)
    if completed:
        print(f"  Resuming: {len(completed)} experiments already completed")

    total = len(combo_indices)
    start_time = time.time()

    for progress_idx, combo_idx in enumerate(combo_indices):
        combo = all_combos[combo_idx]
        combo_name = '_'.join(combo)
        test_qs = tuple(q for q in ALL_QUESTIONS if q not in combo)

        # Skip if already done
        if combo_name in completed:
            print(f"  [{progress_idx+1}/{total}] {combo_name} - SKIP (already done)")
            continue

        combo_start = time.time()

        # Build train/test DataFrames
        train_df = pd.concat([question_data[q] for q in combo], ignore_index=True)
        test_df = pd.concat([question_data[q] for q in test_qs], ignore_index=True)

        train_texts = train_df[config["text_column"]].tolist()
        train_labels = train_df[config["label_column"]].tolist()
        test_texts = test_df[config["text_column"]].tolist()
        test_labels = test_df[config["label_column"]].tolist()

        # Run experiment
        cv_summary, test_metrics, predictions, model, tokenizer = run_single_experiment(
            train_texts, train_labels, test_texts, test_labels, config, device, combo_name)

        # Upload to HF Hub
        hf_ok = False
        if hf_repo and hf_token:
            subfolder = f"question_combos/{combo_name}"
            train_info = f"Questions {', '.join(combo)} ({len(train_df)} samples)"
            hf_ok = upload_model_to_hf(
                model, tokenizer, hf_repo, hf_token, subfolder,
                cv_summary, test_metrics, train_info)

        del model, tokenizer
        torch.cuda.empty_cache()

        combo_time = time.time() - combo_start
        elapsed = time.time() - start_time
        remaining = (elapsed / (progress_idx + 1)) * (total - progress_idx - 1)
        hf_mark = " [HF]" if hf_ok else ""

        print(f"  [{progress_idx+1}/{total}] {combo_name} "
              f"(k={len(combo)}, train={len(train_df)}, test={len(test_df)}) "
              f"CV_F1={cv_summary['CV_F1']:.4f} Test_F1={test_metrics['macro_f1']:.4f} "
              f"Test_Kappa={test_metrics['kappa']:.4f} "
              f"({combo_time:.0f}s, ETA {remaining/60:.0f}min){hf_mark}")

        # Save result row
        result = {
            'Combo': combo_name,
            'Train_Qs': ','.join(combo),
            'Test_Qs': ','.join(test_qs),
            'N_Train_Qs': len(combo),
            'Train_N': len(train_df),
            'Test_N': len(test_df),
            'Model': config["model_name"],
            **cv_summary,
            'Test_Accuracy': test_metrics['accuracy'],
            'Test_F1': test_metrics['macro_f1'],
            'Test_Weighted_F1': test_metrics['weighted_f1'],
            'Test_Kappa': test_metrics['kappa'],
            'Time_Seconds': round(combo_time, 1),
        }

        # Append to CSV (atomic)
        result_df = pd.DataFrame([result])
        if csv_path.exists():
            result_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(csv_path, index=False)

        # Save predictions
        pred_df = pd.DataFrame({
            'id': test_df['id'].tolist() if 'id' in test_df.columns else range(len(test_texts)),
            'feedback_text': test_texts,
            'true_label': test_labels,
            'predicted_label': predictions,
        })
        pred_df.to_excel(pred_dir / f"qc_pred_{combo_name}.xlsx", index=False)

    return csv_path


# ============================================================
# Section 5-2: 학생 분할 실험 (Student Split Experiment)
# ============================================================
# 학생 단위로 학습/테스트를 분할하여, 한 번도 보지 못한 학생의
# 피드백에 대한 분류 성능을 평가합니다.
#
# 분할 비율: 80:20, 70:30, 60:40, 50:50, 40:60
# 각 비율마다 3개 시드(42, 43, 44)로 다른 학생 조합을 테스트합니다.
# → 총 5 비율 × 3 시드 = 15개 실험
# ============================================================

def get_completed_student_experiments(csv_path):
    """Load already-completed student split names from CSV."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df['Split'].tolist())
    except Exception:
        return set()


def discover_student_splits(data_dir):
    """Discover student split files from the student_splits directory."""
    ss_dir = Path(data_dir) / "student_splits"
    if not ss_dir.exists():
        print(f"  ERROR: {ss_dir} not found")
        sys.exit(1)

    # Find all train files and match with test files
    splits = []
    for train_file in sorted(ss_dir.glob("train_*.xlsx")):
        # Parse: train_80_20_seed42.xlsx → ratio=(80,20), seed=42
        match = re.match(r'train_(\d+)_(\d+)_seed(\d+)\.xlsx', train_file.name)
        if not match:
            continue
        train_pct, test_pct, seed = int(match.group(1)), int(match.group(2)), int(match.group(3))
        test_file = ss_dir / f"test_{train_pct}_{test_pct}_seed{seed}.xlsx"

        if not test_file.exists():
            print(f"  WARNING: {test_file} not found, skipping")
            continue

        split_name = f"{train_pct}:{test_pct}_seed{seed}"
        splits.append({
            'name': split_name,
            'ratio': f"{train_pct}:{test_pct}",
            'train_pct': train_pct,
            'seed': seed,
            'train_file': train_file,
            'test_file': test_file,
        })

    return splits


def run_student_experiments(config, device, hf_repo=None, hf_token=None):
    """Run student-based split experiments."""
    data_dir = config["data_dir"]
    output_dir = Path(config["output_dir"])
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Discover splits
    print("\n  Discovering student split files...")
    splits = discover_student_splits(data_dir)
    if not splits:
        print("  ERROR: No student split files found!")
        print(f"  Expected files in: {Path(data_dir) / 'student_splits'}/")
        print("  Format: train_80_20_seed42.xlsx, test_80_20_seed42.xlsx")
        sys.exit(1)

    print(f"  Found {len(splits)} split configurations:")
    for s in splits:
        print(f"    {s['name']}")

    # Load split_info.json for student counts (optional)
    info_path = Path(data_dir) / "student_splits" / "split_info.json"
    split_info = {}
    if info_path.exists():
        with open(info_path, 'r') as f:
            split_info = json.load(f)

    # Resume support
    csv_path = output_dir / "bert_student_results.csv"
    completed = get_completed_student_experiments(csv_path)
    if completed:
        print(f"  Resuming: {len(completed)} experiments already completed")

    total = len(splits)
    start_time = time.time()

    for idx, split in enumerate(splits):
        split_name = split['name']

        if split_name in completed:
            print(f"  [{idx+1}/{total}] {split_name} - SKIP (already done)")
            continue

        split_start = time.time()

        train_df = pd.read_excel(split['train_file'])
        test_df = pd.read_excel(split['test_file'])

        train_texts = train_df[config["text_column"]].tolist()
        train_labels = train_df[config["label_column"]].tolist()
        test_texts = test_df[config["text_column"]].tolist()
        test_labels = test_df[config["label_column"]].tolist()

        # Get student counts from split_info if available
        info_key = split_name
        train_students = len(split_info.get(info_key, {}).get('train_students', []))
        test_students = len(split_info.get(info_key, {}).get('test_students', []))

        # Run experiment
        cv_summary, test_metrics, predictions, model, tokenizer = run_single_experiment(
            train_texts, train_labels, test_texts, test_labels, config, device, split_name)

        # Upload to HF Hub
        hf_ok = False
        if hf_repo and hf_token:
            safe_name = split_name.replace(':', '_')
            subfolder = f"student_splits/{safe_name}"
            train_info = f"Student split {split['ratio']} seed={split['seed']} ({len(train_df)} samples)"
            hf_ok = upload_model_to_hf(
                model, tokenizer, hf_repo, hf_token, subfolder,
                cv_summary, test_metrics, train_info)

        del model, tokenizer
        torch.cuda.empty_cache()

        split_time = time.time() - split_start
        elapsed = time.time() - start_time
        remaining_count = total - idx - 1
        remaining = (elapsed / (idx + 1)) * remaining_count if remaining_count > 0 else 0
        hf_mark = " [HF]" if hf_ok else ""

        print(f"  [{idx+1}/{total}] {split_name} "
              f"(train={len(train_df)}, test={len(test_df)}) "
              f"CV_F1={cv_summary['CV_F1']:.4f} Test_F1={test_metrics['macro_f1']:.4f} "
              f"Test_Kappa={test_metrics['kappa']:.4f} "
              f"({split_time:.0f}s, ETA {remaining/60:.0f}min){hf_mark}")

        # Save result row
        result = {
            'Split': split_name,
            'Ratio': split['ratio'],
            'Train_Pct': split['train_pct'],
            'Seed': split['seed'],
            'Train_Students': train_students,
            'Test_Students': test_students,
            'Train_N': len(train_df),
            'Test_N': len(test_df),
            'Model': config["model_name"],
            **cv_summary,
            'Test_Accuracy': test_metrics['accuracy'],
            'Test_F1': test_metrics['macro_f1'],
            'Test_Weighted_F1': test_metrics['weighted_f1'],
            'Test_Kappa': test_metrics['kappa'],
            'Time_Seconds': round(split_time, 1),
        }

        result_df = pd.DataFrame([result])
        if csv_path.exists():
            result_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(csv_path, index=False)

        # Save predictions
        safe_name = split_name.replace(':', '_')
        pred_df = pd.DataFrame({
            'id': test_df['id'].tolist() if 'id' in test_df.columns else range(len(test_texts)),
            'feedback_text': test_texts,
            'true_label': test_labels,
            'predicted_label': predictions,
        })
        pred_df.to_excel(pred_dir / f"ss_pred_{safe_name}.xlsx", index=False)

    return csv_path


# ============================================================
# Section 6: 메인 실행부 (Main)
# ============================================================
# 명령줄 인자를 파싱하고, 모드(question/student/all)에 따라
# 적절한 실험을 실행합니다.
#
# RunPod에서의 실행 예시:
#   Pod 1: python3 bert_experiment.py --mode question --pod 1 --total-pods 4
#   Pod 2: python3 bert_experiment.py --mode question --pod 2 --total-pods 4
#   Pod 3: python3 bert_experiment.py --mode question --pod 3 --total-pods 4
#   Pod 4: python3 bert_experiment.py --mode question --pod 4 --total-pods 4
#   Pod 5: python3 bert_experiment.py --mode student
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RunPod BERT Generalization Experiment")
    parser.add_argument('--mode', choices=['question', 'student', 'all'], required=True,
                        help='Experiment mode: question (254 combos), student (15 splits), or all')
    parser.add_argument('--pod', type=int, default=None,
                        help='Pod number (1-based) for automatic combo assignment')
    parser.add_argument('--total-pods', type=int, default=4,
                        help='Total number of pods for question experiments (default: 4)')
    parser.add_argument('--start', type=int, default=None,
                        help='Start combo index (0-based, inclusive)')
    parser.add_argument('--end', type=int, default=None,
                        help='End combo index (0-based, inclusive)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory path (default: /workspace/data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory path (default: /workspace/results)')
    parser.add_argument('--hf-repo', type=str, default=None,
                        help='HF Hub repo (default: $HF_REPO env var)')
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Override config with CLI args
    if args.data_dir:
        CONFIG["data_dir"] = args.data_dir
    if args.output_dir:
        CONFIG["output_dir"] = args.output_dir

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(Path(CONFIG["output_dir"]) / "predictions", exist_ok=True)

    print("=" * 60)
    print("  RunPod BERT Generalization Experiment")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {CONFIG['model_name']} ({CONFIG['model_path']})")
    print("=" * 60)

    # Device check
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  Device: {gpu_name} ({gpu_mem:.1f} GB)")

    # HF Hub credentials
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = args.hf_repo or os.environ.get("HF_REPO")
    if hf_token and hf_repo:
        print(f"  HF Hub: {hf_repo} (upload enabled)")
    else:
        print("  HF Hub: disabled (set HF_TOKEN & HF_REPO to enable)")

    # ---- Question Combination Experiment ----
    if args.mode in ('question', 'all'):
        print(f"\n{'=' * 60}")
        print("  Part 1: Question Combination Experiment")
        print(f"{'=' * 60}")

        all_combos = generate_all_combos()

        if args.start is not None and args.end is not None:
            # Manual range
            combo_indices = list(range(args.start, args.end + 1))
            print(f"  Manual range: #{args.start} - #{args.end} ({len(combo_indices)} combos)")
        elif args.pod is not None:
            # Auto-assign via bin-packing
            question_data = load_question_data(CONFIG["data_dir"])
            samples_per_q = {q: len(df) for q, df in question_data.items()}
            pods, pod_times = assign_combos_to_pods(all_combos, args.total_pods, samples_per_q)
            pod_idx = args.pod - 1  # 1-based to 0-based

            if pod_idx < 0 or pod_idx >= args.total_pods:
                print(f"  ERROR: --pod must be 1-{args.total_pods}")
                sys.exit(1)

            combo_indices = pods[pod_idx]
            print(f"  Pod {args.pod}/{args.total_pods}: {len(combo_indices)} combos "
                  f"(est. {pod_times[pod_idx]:.0f} min = {pod_times[pod_idx]/60:.1f}h)")

            # Show all pod assignments
            print(f"\n  Pod assignment summary:")
            for p in range(args.total_pods):
                marker = " <-- THIS POD" if p == pod_idx else ""
                print(f"    Pod {p+1}: {len(pods[p])} combos, "
                      f"~{pod_times[p]:.0f} min ({pod_times[p]/60:.1f}h){marker}")
        else:
            # Run all
            combo_indices = list(range(len(all_combos)))
            print(f"  Running all {len(combo_indices)} combos")

        run_question_experiments(combo_indices, CONFIG, device, hf_repo, hf_token)

    # ---- Student Split Experiment ----
    if args.mode in ('student', 'all'):
        print(f"\n{'=' * 60}")
        print("  Part 2: Student Split Experiment")
        print(f"{'=' * 60}")

        run_student_experiments(CONFIG, device, hf_repo, hf_token)

    # ---- Summary ----
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Experiment Complete!")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.1f}h)")
    print(f"  GPU: {gpu_name}")
    print(f"  Output: {CONFIG['output_dir']}/")
    print(f"{'=' * 60}")

    # Save experiment info
    info = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': args.mode,
        'pod': args.pod,
        'total_pods': args.total_pods,
        'gpu': gpu_name,
        'total_time_minutes': round(total_time / 60, 1),
        'config': {k: v for k, v in CONFIG.items() if k not in ('data_dir', 'output_dir')},
    }
    info_path = Path(CONFIG["output_dir"]) / f"experiment_info_{args.mode}"
    if args.pod:
        info_path = Path(f"{info_path}_pod{args.pod}.json")
    else:
        info_path = Path(f"{info_path}.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
