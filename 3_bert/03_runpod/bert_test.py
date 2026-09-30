"""
RunPod BERT 파이프라인 검증 테스트 스크립트
=============================================

목적:
    본격적인 일반화 실험(bert_experiment.py) 전에,
    KLUE-BERT 기본 파이프라인이 RunPod GPU 환경에서 정상 작동하는지
    검증하는 테스트 스크립트입니다.

    로컬(M2 Max MPS)에서 얻은 기준 성능과 RunPod(CUDA)에서의 성능을
    비교하여 파이프라인 정합성을 확인합니다.

검증 항목:
    - 10-fold CV 성능이 기대치(Kappa ~0.91, F1 ~0.914) 범위 내인지
    - 테스트 세트 성능이 기대치(Kappa ~0.90, F1 ~0.904) 범위 내인지
    - MPS vs CUDA 간 1~3% 차이는 정상 (하드웨어 차이로 인한 부동소수점 오차)

RunPod 사용 이유:
    - BERT 미세조정에는 GPU가 필수적입니다
    - RunPod은 GPU 클라우드 플랫폼으로, 필요할 때만 GPU를 대여하여 사용합니다
    - RTX 3090/4090 등의 고성능 GPU를 시간당 요금으로 이용 가능합니다

실행 방법:
    # RunPod Pod 내 터미널에서:
    export HF_TOKEN=hf_xxxxx           # HuggingFace 토큰 (모델 업로드용)
    export HF_REPO=username/model-name  # 업로드할 HF 저장소
    python3 bert_test.py
"""

import os
import sys
import json
import random
import time
import warnings
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm

# ============================================================
# Section 1: 실험 설정 (Configuration)
# ============================================================
# KLUE-BERT 모델의 학습 하이퍼파라미터와 RunPod 환경 경로를 정의합니다.
# /workspace/data/에 학습/테스트 데이터(Excel)가 미리 업로드되어 있어야 합니다.
# ============================================================

CONFIG = {
    # 모델 설정: KLUE-BERT (한국어 특화 BERT)
    "model_name": "KLUE-BERT",
    "model_path": "klue/bert-base",  # 사전학습 모델(Pre-trained Model): HuggingFace에서 자동 다운로드

    # Data
    "train_path": "/workspace/data/train_set_80.xlsx",
    "test_path": "/workspace/data/test_set_20.xlsx",
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

    # CUDA optimization
    "use_fp16": True,
    "num_workers": 2,
    "pin_memory": True,

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

# 기대 성능값 (로컬 M2 Max MPS, seed=42 실험 결과 기준)
# 이 값과 RunPod 결과를 비교하여 파이프라인 정합성을 검증합니다.
# tolerance=0.03: MPS/CUDA 간 3% 이내 차이는 정상으로 판단
EXPECTED = {
    "cv_accuracy": 0.946,
    "cv_macro_f1": 0.914,
    "test_accuracy": 0.923,
    "test_macro_f1": 0.904,
    "test_kappa": 0.900,
    "tolerance": 0.03,
}


# ============================================================
# Section 2: 유틸리티 함수 및 클래스
# ============================================================
# 재현성, 데이터셋, 조기종료, 학습/평가 루프 등
# BERT 미세조정에 필요한 공통 컴포넌트입니다.
# bert_experiment.py와 동일한 구조를 사용합니다.
# ============================================================

def set_seed(seed=42):
    """재현성을 위한 시드 고정 (CUDA 버전)."""
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
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
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
    """Train one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for batch in tqdm(dataloader, desc="  Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        if use_fp16 and scaler is not None:
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return metrics + predictions."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(true_labels, predictions),
        'macro_f1': f1_score(true_labels, predictions, average='macro'),
        'weighted_f1': f1_score(true_labels, predictions, average='weighted'),
        'kappa': cohen_kappa_score(true_labels, predictions),
    }

    return metrics, predictions, true_labels


# ============================================================
# Section 3: 실험 함수
# ============================================================
# 10-Fold CV, 최종 모델 학습, 테스트 평가의 세 단계로 구성됩니다.
# ============================================================

def run_cross_validation(train_texts, train_labels, config, device):
    """Stratified 10-fold 교차검증을 실행합니다."""
    print(f"\n{'=' * 60}")
    print(f"  10-Fold Cross-Validation: {config['model_name']}")
    print(f"{'=' * 60}")

    skf = StratifiedKFold(
        n_splits=config["n_folds"], shuffle=True, random_state=config["seed"]
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        fold_start = time.time()
        print(f"\n--- Fold {fold + 1}/{config['n_folds']} ---")
        set_seed(config["seed"] + fold)

        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

        train_dataset = FeedbackDataset(fold_train_texts, fold_train_labels, tokenizer, config["max_length"])
        val_dataset = FeedbackDataset(fold_val_texts, fold_val_labels, tokenizer, config["max_length"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_path"],
            num_labels=config["num_labels"],
        )
        model.to(device)

        # 클래스 가중 손실 함수 (Class-weighted Loss)
        # 클래스 불균형 처리: 데이터가 적은 클래스에 높은 가중치를 부여하여
        # 모델이 소수 클래스도 올바르게 학습하도록 유도합니다.
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(fold_train_labels),
            y=fold_train_labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = GradScaler() if config["use_fp16"] else None
        early_stopping = EarlyStopping(patience=config["patience"], mode='max')

        for epoch in range(config["epochs"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, config["gradient_clip"], config["use_fp16"], scaler,
            )
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

            print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                  f"Val Macro-F1={val_metrics['macro_f1']:.4f}")

            early_stopping(val_metrics['macro_f1'], model)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best model and evaluate
        early_stopping.load_best_model(model)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        fold_results.append(val_metrics)
        fold_time = time.time() - fold_start
        print(f"  Fold {fold + 1} Best: Macro-F1={val_metrics['macro_f1']:.4f}, "
              f"Kappa={val_metrics['kappa']:.4f} ({fold_time:.0f}s)")

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # Summary
    cv_summary = {
        'model': config["model_name"],
        'seed': config["seed"],
        'accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'macro_f1_mean': np.mean([r['macro_f1'] for r in fold_results]),
        'macro_f1_std': np.std([r['macro_f1'] for r in fold_results]),
        'weighted_f1_mean': np.mean([r['weighted_f1'] for r in fold_results]),
        'weighted_f1_std': np.std([r['weighted_f1'] for r in fold_results]),
        'kappa_mean': np.mean([r['kappa'] for r in fold_results]),
        'kappa_std': np.std([r['kappa'] for r in fold_results]),
    }

    print(f"\n[CV Summary] {config['model_name']}")
    print(f"  Accuracy: {cv_summary['accuracy_mean']:.4f} (+/- {cv_summary['accuracy_std']:.4f})")
    print(f"  Macro-F1: {cv_summary['macro_f1_mean']:.4f} (+/- {cv_summary['macro_f1_std']:.4f})")
    print(f"  Cohen's Kappa: {cv_summary['kappa_mean']:.4f} (+/- {cv_summary['kappa_std']:.4f})")

    return cv_summary, fold_results


def train_final_model(train_texts, train_labels, config, device):
    """전체 학습 데이터로 최종 모델을 학습합니다 (CV 없이)."""
    print(f"\n{'=' * 60}")
    print(f"  Final Model Training (full train set)")
    print(f"{'=' * 60}")

    set_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer, config["max_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_path"],
        num_labels=config["num_labels"],
    )
    model.to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler() if config["use_fp16"] else None

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, config["gradient_clip"], config["use_fp16"], scaler,
        )
        print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

    return model, tokenizer


def evaluate_on_test(model, tokenizer, test_texts, test_labels, config, device):
    """테스트 세트에서 모델을 평가하고 상세 결과를 반환합니다."""
    print(f"\n{'=' * 60}")
    print(f"  Test Set Evaluation")
    print(f"{'=' * 60}")

    test_dataset = FeedbackDataset(test_texts, test_labels, tokenizer, config["max_length"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Dummy criterion for loss computation
    criterion = nn.CrossEntropyLoss()

    test_metrics, predictions, true_labels = evaluate(model, test_loader, criterion, device)

    print(f"\n  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Kappa:       {test_metrics['kappa']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, predictions))

    return test_metrics, predictions, true_labels


# ============================================================
# Section 4: 업로드 함수
# ============================================================
# 학습된 모델을 HuggingFace Hub에 업로드하고,
# Google Drive 연동(rclone) 가능 여부를 확인합니다.
# ============================================================

def upload_to_hf(model, tokenizer, config, test_metrics, cv_summary):
    """학습된 모델을 HuggingFace Hub에 업로드합니다."""
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_REPO")

    if not hf_token:
        print("\n[HF Upload] Skipped - HF_TOKEN not set")
        return False

    if not hf_repo:
        print("\n[HF Upload] Skipped - HF_REPO not set")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Uploading to Hugging Face Hub: {hf_repo}")
    print(f"{'=' * 60}")

    try:
        from huggingface_hub import HfApi

        # Save model locally first
        save_dir = Path(config["output_dir"]) / "model"
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Create model card
        model_card = f"""---
language: ko
license: mit
tags:
  - text-classification
  - bert
  - korean
  - peer-feedback
datasets:
  - custom
metrics:
  - accuracy
  - f1
---

# KLUE-BERT Feedback Classifier

Korean peer feedback classification model (7 classes) fine-tuned from `klue/bert-base`.

## Performance

| Metric | CV (10-fold) | Test Set |
|--------|-------------|----------|
| Accuracy | {cv_summary['accuracy_mean']:.4f} (+/- {cv_summary['accuracy_std']:.4f}) | {test_metrics['accuracy']:.4f} |
| Macro-F1 | {cv_summary['macro_f1_mean']:.4f} (+/- {cv_summary['macro_f1_std']:.4f}) | {test_metrics['macro_f1']:.4f} |
| Weighted-F1 | {cv_summary['weighted_f1_mean']:.4f} (+/- {cv_summary['weighted_f1_std']:.4f}) | {test_metrics['weighted_f1']:.4f} |
| Kappa | {cv_summary['kappa_mean']:.4f} (+/- {cv_summary['kappa_std']:.4f}) | {test_metrics['kappa']:.4f} |

## Training Details

- Base model: `klue/bert-base`
- Epochs: {config['epochs']}
- Batch size: {config['batch_size']}
- Learning rate: {config['learning_rate']}
- Max length: {config['max_length']}
- Trained on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- Date: {datetime.now().strftime('%Y-%m-%d')}
"""
        (save_dir / "README.md").write_text(model_card)

        # Upload
        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=hf_repo,
            commit_message=f"KLUE-BERT test experiment (F1={test_metrics['macro_f1']:.4f})",
        )

        print(f"  -> Model uploaded to https://huggingface.co/{hf_repo}")
        return True

    except Exception as e:
        print(f"  -> Upload failed: {e}")
        return False


def check_gdrive_upload(config):
    """Check if rclone is available for Google Drive upload, otherwise guide manual download."""
    rclone_available = shutil.which("rclone") is not None

    if rclone_available:
        print("\n[Google Drive] rclone detected. To upload results:")
        print(f"  rclone copy {config['output_dir']} gdrive:RunPod_BERT_Results/test/")
    else:
        print("\n[Google Drive] rclone not installed.")
        print("  Download results manually from JupyterLab:")
        print(f"  1. Navigate to {config['output_dir']}/")
        print("  2. Right-click files -> Download")
        print("  3. Or use: zip -r /workspace/results.zip /workspace/results/")


# ============================================================
# Section 5: 메인 실행부
# ============================================================
# 6단계로 구성된 실험 파이프라인:
#   Step 1: 데이터 로드 (Excel → DataFrame)
#   Step 2: 10-Fold CV 실행 (내부 검증 성능 측정)
#   Step 3: 전체 학습 데이터로 최종 모델 학습
#   Step 4: 테스트 세트 평가 (외부 일반화 성능 측정)
#   Step 5: 실험 정보 JSON 저장
#   Step 6: HuggingFace Hub 업로드 + Google Drive 안내
#
# 마지막에 기대 성능과의 비교를 통해 파이프라인 정합성을 검증합니다.
# ============================================================

def main():
    start_time = time.time()

    print("=" * 60)
    print("  RunPod BERT Test Experiment")
    print("  Model: KLUE-BERT (klue/bert-base)")
    print("  Task: 7-class Korean Peer Feedback Classification")
    print("=" * 60)

    # Device
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ---- Step 1: Load Data ----
    print(f"\n[Step 1/6] Loading data...")

    if not os.path.exists(CONFIG["train_path"]):
        print(f"  ERROR: Train file not found: {CONFIG['train_path']}")
        print(f"  Upload train_set_80.xlsx to /workspace/data/")
        sys.exit(1)
    if not os.path.exists(CONFIG["test_path"]):
        print(f"  ERROR: Test file not found: {CONFIG['test_path']}")
        print(f"  Upload test_set_20.xlsx to /workspace/data/")
        sys.exit(1)

    df_train = pd.read_excel(CONFIG["train_path"])
    df_test = pd.read_excel(CONFIG["test_path"])

    train_texts = df_train[CONFIG["text_column"]].tolist()
    train_labels = df_train[CONFIG["label_column"]].tolist()
    test_texts = df_test[CONFIG["text_column"]].tolist()
    test_labels = df_test[CONFIG["label_column"]].tolist()

    print(f"  Train: {len(train_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")
    print(f"  Labels: {sorted(set(train_labels))}")

    # ---- Step 2: Cross-Validation ----
    print(f"\n[Step 2/6] Running 10-Fold Cross-Validation...")
    cv_start = time.time()

    cv_summary, fold_results = run_cross_validation(
        train_texts, train_labels, CONFIG, device,
    )

    cv_time = time.time() - cv_start

    # Save CV results
    cv_df = pd.DataFrame([cv_summary])
    cv_path = Path(CONFIG["output_dir"]) / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\n  -> Saved: {cv_path} ({cv_time:.0f}s)")

    # ---- Step 3: Train Final Model ----
    print(f"\n[Step 3/6] Training final model on full train set...")
    final_start = time.time()

    model, tokenizer = train_final_model(train_texts, train_labels, CONFIG, device)

    final_time = time.time() - final_start
    print(f"  -> Final model trained ({final_time:.0f}s)")

    # ---- Step 4: Test Set Evaluation ----
    print(f"\n[Step 4/6] Evaluating on test set...")

    test_metrics, predictions, true_labels = evaluate_on_test(
        model, tokenizer, test_texts, test_labels, CONFIG, device,
    )

    # Save test results
    test_result = {
        'model': CONFIG["model_name"],
        'seed': CONFIG["seed"],
        'accuracy': test_metrics['accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'weighted_f1': test_metrics['weighted_f1'],
        'kappa': test_metrics['kappa'],
    }
    test_df = pd.DataFrame([test_result])
    test_path = Path(CONFIG["output_dir"]) / "test_results.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  -> Saved: {test_path}")

    # Save predictions
    pred_df = pd.DataFrame({
        'feedback_text': test_texts,
        'true_label': true_labels,
        'predicted_label': predictions,
    })
    pred_path = Path(CONFIG["output_dir"]) / "predictions.xlsx"
    pred_df.to_excel(pred_path, index=False)
    print(f"  -> Saved: {pred_path}")

    # ---- Step 5: Save Experiment Info ----
    print(f"\n[Step 5/6] Saving experiment info...")

    total_time = time.time() - start_time
    experiment_info = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": CONFIG["model_name"],
            "model_path": CONFIG["model_path"],
            "seed": CONFIG["seed"],
            "n_folds": CONFIG["n_folds"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "max_length": CONFIG["max_length"],
            "use_fp16": CONFIG["use_fp16"],
        },
        "cv_results": {
            "accuracy_mean": cv_summary["accuracy_mean"],
            "accuracy_std": cv_summary["accuracy_std"],
            "macro_f1_mean": cv_summary["macro_f1_mean"],
            "macro_f1_std": cv_summary["macro_f1_std"],
            "kappa_mean": cv_summary["kappa_mean"],
            "kappa_std": cv_summary["kappa_std"],
        },
        "test_results": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "kappa": test_metrics["kappa"],
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timing": {
            "cv_seconds": round(cv_time, 1),
            "final_train_seconds": round(final_time, 1),
            "total_seconds": round(total_time, 1),
            "total_minutes": round(total_time / 60, 1),
        },
        "data": {
            "train_samples": len(train_texts),
            "test_samples": len(test_texts),
            "num_labels": CONFIG["num_labels"],
        },
    }

    info_path = Path(CONFIG["output_dir"]) / "experiment_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved: {info_path}")

    # ---- Step 6: Upload ----
    print(f"\n[Step 6/6] Uploading results...")

    hf_success = upload_to_hf(model, tokenizer, CONFIG, test_metrics, cv_summary)
    check_gdrive_upload(CONFIG)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Experiment Complete!")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  CV  Accuracy: {cv_summary['accuracy_mean']:.4f} (expected ~{EXPECTED['cv_accuracy']:.3f})")
    print(f"  CV  Macro-F1: {cv_summary['macro_f1_mean']:.4f} (expected ~{EXPECTED['cv_macro_f1']:.3f})")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f} (expected ~{EXPECTED['test_accuracy']:.3f})")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f} (expected ~{EXPECTED['test_macro_f1']:.3f})")
    print(f"  Test Kappa:    {test_metrics['kappa']:.4f} (expected ~{EXPECTED['test_kappa']:.3f})")
    print(f"  HF Upload:    {'Success' if hf_success else 'Skipped/Failed'}")

    # Validation check
    tol = EXPECTED["tolerance"]
    checks = [
        ("CV Accuracy", cv_summary["accuracy_mean"], EXPECTED["cv_accuracy"]),
        ("CV Macro-F1", cv_summary["macro_f1_mean"], EXPECTED["cv_macro_f1"]),
        ("Test Accuracy", test_metrics["accuracy"], EXPECTED["test_accuracy"]),
        ("Test Macro-F1", test_metrics["macro_f1"], EXPECTED["test_macro_f1"]),
        ("Test Kappa", test_metrics["kappa"], EXPECTED["test_kappa"]),
    ]

    print(f"\n  Validation (tolerance +/- {tol}):")
    all_pass = True
    for name, actual, expected in checks:
        status = "PASS" if abs(actual - expected) <= tol else "WARN"
        if status == "WARN":
            all_pass = False
        print(f"    {status}: {name} = {actual:.4f} (expected {expected:.3f})")

    if all_pass:
        print(f"\n  All checks passed! Pipeline verified for RunPod.")
    else:
        print(f"\n  Some checks outside tolerance. Review results.")
        print(f"  Note: MPS vs CUDA differences of 1-3% are normal.")

    print(f"\n  Output files: {CONFIG['output_dir']}/")
    print(f"    - cv_results.csv")
    print(f"    - test_results.csv")
    print(f"    - predictions.xlsx")
    print(f"    - experiment_info.json")
    if hf_success:
        print(f"    - model/ (uploaded to HF Hub)")


if __name__ == "__main__":
    main()
