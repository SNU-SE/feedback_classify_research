#!/usr/bin/env python
# coding: utf-8

# ============================================================
# BERT 기반 한국어 동료 피드백 자동 분류 실험
# ============================================================
#
# 목적:
#   고등학교 과학 수업에서 수집된 2,080개의 한국어 동료 피드백 텍스트를
#   7단계 코딩 체계(0=무관~6=대안제시)로 자동 분류하는 딥러닝 모델을 구축합니다.
#
# 사전학습 모델(Pre-trained Model)이란?
#   대규모 텍스트 데이터로 미리 학습된 언어 모델로, 언어의 문법/의미 구조를
#   이미 이해하고 있어 소규모 데이터에서도 높은 성능을 달성할 수 있습니다.
#
# 미세조정(Fine-tuning)이란?
#   사전학습된 모델을 특정 작업(여기서는 피드백 분류)에 맞게
#   추가 학습하는 과정입니다. 전체 모델을 처음부터 학습하는 것보다
#   훨씬 적은 데이터와 시간으로 높은 성능을 얻을 수 있습니다.
#
# 사용 모델 4종:
#   1. mBERT: 104개 언어로 학습된 다국어 BERT (Google)
#   2. KLUE-BERT: 한국어 특화 BERT (KLUE 벤치마크용)
#   3. KLUE-RoBERTa: BERT 개선 모델의 한국어 버전 (더 많은 데이터로 학습)
#   4. KoELECTRA: 한국어 특화 ELECTRA (효율적 사전학습 방식)
#
# 실험 설계:
#   - Class-weighted Cross-Entropy Loss: 클래스 불균형 처리
#     (소수 클래스에 더 큰 가중치를 부여하여 모델이 골고루 학습)
#   - Stratified 10-fold Cross-Validation: 각 fold에서 클래스 비율 유지
#   - Fixed Test Set: ML 실험과 동일한 테스트 세트로 공정 비교
#   - Early Stopping: 과적합 방지 (patience=2)
#
# 이 파일은 01_experiment.ipynb 노트북을 Python 스크립트로 변환한 버전입니다.
# 로컬 환경(M2 Max MPS)에서 실행하도록 최적화되어 있습니다.
# ============================================================

# # BERT 기반 한국어 피드백 분류 실험
#
# 이 노트북은 4개의 사전학습 언어모델을 사용하여 한국어 동료 피드백을 분류합니다.
#
# ## 실험 개요
# - **모델**: mBERT, KLUE-BERT, KLUE-RoBERTa, KoELECTRA
# - **불균형 처리**: Class-weighted Cross-Entropy Loss
# - **검증**: Stratified 10-fold Cross-Validation
# - **평가**: Fixed Test Set (ML 실험과 동일)

# ============================================================
# 1. 환경 설정
# ============================================================
# 딥러닝 학습에 필요한 라이브러리를 불러옵니다.
# - torch: PyTorch 딥러닝 프레임워크
# - transformers: HuggingFace의 사전학습 모델 라이브러리
# - sklearn: 교차검증, 평가지표 등 머신러닝 유틸리티
# ============================================================

# In[1]:


import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # PyTorch native AdamW (transformers deprecated theirs)

# HuggingFace transformers: 사전학습된 BERT 계열 모델을 불러오고 미세조정하는 라이브러리
from transformers import (
    AutoTokenizer,                          # 텍스트를 토큰(숫자)으로 변환하는 토크나이저
    AutoModelForSequenceClassification,     # 분류 작업용 사전학습 모델
    get_linear_schedule_with_warmup         # 학습률 스케줄러 (워밍업 후 선형 감소)
)

# scikit-learn: 교차검증 및 평가지표 계산용
from sklearn.model_selection import StratifiedKFold          # 클래스 비율 유지 K-Fold 분할
from sklearn.utils.class_weight import compute_class_weight  # 클래스 불균형 가중치 자동 계산
from sklearn.metrics import (
    accuracy_score,           # 정확도
    f1_score,                 # F1 점수 (정밀도와 재현율의 조화평균)
    cohen_kappa_score,        # Cohen's Kappa (우연 일치를 보정한 일치도)
    classification_report,    # 클래스별 상세 성능 보고서
    confusion_matrix          # 혼동 행렬
)

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")


# In[2]:


# 재현성을 위한 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 디바이스 설정 (M2 Max MPS 최적화)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # MPS 사용 시 메모리 최적화
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# M2 Max MPS 최적화 설정
if DEVICE.type == "mps":
    # MPS 메모리 관리 활성화
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 메모리 제한 해제
    print("✅ MPS 최적화 설정 완료 (M2 Max 36GB RAM)")


# ============================================================
# 2. 실험 설정 (Configuration)
# ============================================================
# 모든 하이퍼파라미터와 경로를 한 곳에서 관리합니다.
# M2 Max (36GB RAM) 로컬 환경에 맞게 최적화되어 있습니다.
# RunPod GPU 환경에서 실행할 경우 03_runpod/ 스크립트를 사용하세요.
# ============================================================

# In[3]:


# 실험 설정 (M2 Max 36GB RAM 최적화)
CONFIG = {
    # 데이터 경로
    "data_path": "../data/feedback_data.xlsx",  # split_seed42 사용

    # 컬럼 이름 (데이터에 맞게 수정)
    "text_column": "feedback_text",  # 텍스트 컬럼명
    "label_column": "label",  # 레이블 컬럼명

    # 모델 설정 (4종)
    "models": {
        "mBERT": "bert-base-multilingual-cased",
        "KLUE-BERT": "klue/bert-base",
        "KLUE-RoBERTa": "klue/roberta-base",
        "KoELECTRA": "monologg/koelectra-base-v3-discriminator"
    },

    # 학습 설정 (M2 Max 최적화)
    "max_length": 128,
    "batch_size": 32,  # M2 Max 36GB RAM -> batch_size 32 가능
    "epochs": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clip": 1.0,

    # DataLoader 설정 (MPS 최적화)
    "num_workers": 0,  # MPS에서는 0 필수
    "pin_memory": False,  # MPS에서는 False

    # Early stopping
    "patience": 2,

    # Cross-validation
    "n_folds": 10,

    # 결과 저장
    "output_dir": "./results"
}

# 출력 디렉토리 생성
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("Configuration loaded (M2 Max Optimized)")
print(f"Models: {list(CONFIG['models'].keys())}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Device: {DEVICE}")


# ============================================================
# 3. 데이터 로드 및 확인
# ============================================================
# 80:20으로 분할된 학습/테스트 데이터를 로드합니다.
# 테스트 세트는 ML 실험과 동일한 세트를 사용하여 공정한 비교를 보장합니다.
# 7개 클래스(0~6)의 분포가 불균형하므로, 이후 class-weighted loss로 보정합니다.
# ============================================================

# In[4]:


# 데이터 로드
all_df = pd.read_excel(CONFIG["data_path"])
df_train = all_df[all_df["split_seed42"] == "train"].reset_index(drop=True)
df_test = all_df[all_df["split_seed42"] == "test"].reset_index(drop=True)

print(f"Train set: {len(df_train)} samples")
print(f"Test set: {len(df_test)} samples")
print(f"\nColumns: {list(df_train.columns)}")


# In[5]:


# 컬럼명 확인 및 매핑
# 실제 데이터에 맞게 수정하세요
print("\n=== 데이터 샘플 ===")
print(df_train.head())

# 텍스트/레이블 컬럼 자동 탐지
text_cols = [c for c in df_train.columns if 'text' in c.lower() or 'feedback' in c.lower() or 'content' in c.lower()]
label_cols = [c for c in df_train.columns if 'label' in c.lower() or 'class' in c.lower() or 'category' in c.lower()]

print(f"\n탐지된 텍스트 컬럼: {text_cols}")
print(f"탐지된 레이블 컬럼: {label_cols}")

# 컬럼명 설정 (자동 탐지 또는 수동 지정)
if text_cols:
    CONFIG["text_column"] = text_cols[0]
if label_cols:
    CONFIG["label_column"] = label_cols[0]

print(f"\n사용할 텍스트 컬럼: {CONFIG['text_column']}")
print(f"사용할 레이블 컬럼: {CONFIG['label_column']}")


# In[6]:


# 클래스 분포 확인
print("\n=== 클래스 분포 ===")
print("\n[Train Set]")
print(df_train[CONFIG["label_column"]].value_counts().sort_index())

print("\n[Test Set]")
print(df_test[CONFIG["label_column"]].value_counts().sort_index())

# 클래스 수 확인
NUM_LABELS = df_train[CONFIG["label_column"]].nunique()
print(f"\n총 클래스 수: {NUM_LABELS}")


# In[7]:


# 클래스 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Train set
train_dist = df_train[CONFIG["label_column"]].value_counts().sort_index()
axes[0].bar(train_dist.index.astype(str), train_dist.values)
axes[0].set_title("Train Set Class Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

# Test set
test_dist = df_test[CONFIG["label_column"]].value_counts().sort_index()
axes[1].bar(test_dist.index.astype(str), test_dist.values)
axes[1].set_title("Test Set Class Distribution")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/class_distribution.png", dpi=150)
plt.show()


# ============================================================
# 4. Dataset 클래스 정의
# ============================================================
# PyTorch Dataset 클래스: 텍스트 데이터를 BERT 모델이 이해할 수 있는
# 형태(토큰 ID, 어텐션 마스크)로 변환합니다.
#
# 토크나이저 처리 과정:
#   "좋은 발표였습니다" → [CLS] 좋 ##은 발표 ##였 ##습니다 [SEP] [PAD]...
#   → input_ids: [2, 3421, 1102, 5567, ...] (숫자로 변환)
#   → attention_mask: [1, 1, 1, 1, ..., 0, 0] (실제 토큰=1, 패딩=0)
# ============================================================

# In[8]:


class FeedbackDataset(Dataset):
    """피드백 분류를 위한 PyTorch Dataset"""

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
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

print("FeedbackDataset class defined.")


# ============================================================
# 5. 학습 및 평가 함수 정의
# ============================================================
# train_epoch(): 모델을 한 에폭(전체 데이터 1회 순회) 동안 학습
#   - 순전파(forward) → 손실 계산 → 역전파(backward) → 가중치 갱신
#   - gradient clipping: 기울기 폭발 방지 (max_norm=1.0)
#
# evaluate(): 검증/테스트 데이터에 대한 성능 평가
#   - torch.no_grad()로 기울기 계산 비활성화 (메모리 절약)
#   - Accuracy, Macro-F1, Weighted-F1, Cohen's Kappa 계산
#
# EarlyStopping: 검증 성능이 patience(2 에폭) 동안 개선되지 않으면
#   학습을 조기 중단하여 과적합을 방지합니다.
# ============================================================

# In[9]:


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, gradient_clip=1.0):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)  # 클래스 가중치가 적용된 손실 계산
        total_loss += loss.item()

        loss.backward()  # 역전파: 각 파라미터의 기울기 계산
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)  # 기울기 크기 제한
        optimizer.step()   # 가중치 갱신
        scheduler.step()   # 학습률 조정

        preds = torch.argmax(outputs.logits, dim=1)  # 가장 높은 확률의 클래스를 예측값으로
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy


# In[10]:


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

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
        'kappa': cohen_kappa_score(true_labels, predictions)
    }

    return metrics, predictions, true_labels

print("Training and evaluation functions defined.")


# In[11]:


class EarlyStopping:
    """Early stopping 구현"""

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
            self.best_model_state = model.state_dict().copy()
        elif (self.mode == 'max' and score <= self.best_score) or \
             (self.mode == 'min' and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

print("EarlyStopping class defined.")


# ============================================================
# 6. Stratified 10-Fold Cross-Validation 함수
# ============================================================
# Stratified K-Fold CV: 데이터를 10등분하여 9개로 학습, 1개로 검증을 반복합니다.
# 'Stratified'는 각 fold에서 원본 데이터의 클래스 비율이 유지됨을 의미합니다.
# 이는 불균형 데이터에서 특히 중요합니다.
#
# 각 fold에서 수행되는 작업:
#   1. 사전학습 모델을 새로 로드 (매 fold마다 초기화)
#   2. 학습 데이터의 클래스 가중치 계산 → CrossEntropyLoss에 적용
#      (클래스 가중치: 소수 클래스일수록 높은 가중치 → 균형 학습)
#   3. AdamW 옵티마이저 + 학습률 워밍업 스케줄러 설정
#   4. Early Stopping으로 최적 에폭에서 학습 중단
#   5. Best 모델로 검증 성능 기록
#
# 결과: 10개 fold의 평균/표준편차 → 모델 성능의 안정성 평가
# ============================================================

# In[12]:


def run_cross_validation(model_name, model_path, train_texts, train_labels, config):
    """Stratified K-Fold Cross-Validation 실행 (MPS 최적화)"""

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        print(f"\n--- Fold {fold + 1}/{config['n_folds']} ---")
        set_seed(42 + fold)

        # 데이터 분할
        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        # 토크나이저 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Dataset 생성
        train_dataset = FeedbackDataset(fold_train_texts, fold_train_labels, tokenizer, config["max_length"])
        val_dataset = FeedbackDataset(fold_val_texts, fold_val_labels, tokenizer, config["max_length"])

        # DataLoader (MPS 최적화 설정 적용)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False)
        )

        # 모델 로드
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=NUM_LABELS,
                trust_remote_code=True
            )
        except:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=NUM_LABELS
            )
        model.to(DEVICE)

        # Class weight 계산 (불균형 데이터 처리의 핵심)
        # 'balanced' 옵션: n_samples / (n_classes * n_samples_per_class)
        # 예) 클래스 0이 50개, 클래스 3이 500개면 → 클래스 0에 10배 높은 가중치
        # 이를 통해 소수 클래스의 오분류에 더 큰 페널티를 부여합니다.
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(fold_train_labels),
            y=fold_train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # 가중치 적용 손실 함수

        # Optimizer: AdamW (가중치 감쇠가 적용된 Adam 옵티마이저)
        optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],    # 학습률 2e-5 (BERT 미세조정 표준값)
            weight_decay=config["weight_decay"]  # L2 정규화로 과적합 방지
        )

        # 학습률 스케줄러: 처음에 학습률을 천천히 올리고(warmup), 이후 선형 감소
        # warmup은 사전학습된 가중치를 급격히 변경하지 않기 위한 안전장치
        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config["warmup_ratio"])  # 전체의 10%를 워밍업
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=config["patience"], mode='max')

        # 학습
        for epoch in range(config["epochs"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, DEVICE, config["gradient_clip"]
            )

            val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Macro-F1={val_metrics['macro_f1']:.4f}")

            early_stopping(val_metrics['macro_f1'], model)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Best 모델 로드 후 최종 평가
        early_stopping.load_best_model(model)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        fold_results.append(val_metrics)
        print(f"  Fold {fold+1} Best: Macro-F1={val_metrics['macro_f1']:.4f}, Kappa={val_metrics['kappa']:.4f}")

        # 메모리 정리 (MPS 최적화)
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # CV 결과 요약
    cv_summary = {
        'model': model_name,
        'accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'macro_f1_mean': np.mean([r['macro_f1'] for r in fold_results]),
        'macro_f1_std': np.std([r['macro_f1'] for r in fold_results]),
        'weighted_f1_mean': np.mean([r['weighted_f1'] for r in fold_results]),
        'weighted_f1_std': np.std([r['weighted_f1'] for r in fold_results]),
        'kappa_mean': np.mean([r['kappa'] for r in fold_results]),
        'kappa_std': np.std([r['kappa'] for r in fold_results])
    }

    print(f"\n[CV Summary] {model_name}")
    print(f"  Accuracy: {cv_summary['accuracy_mean']:.4f} (+/- {cv_summary['accuracy_std']:.4f})")
    print(f"  Macro-F1: {cv_summary['macro_f1_mean']:.4f} (+/- {cv_summary['macro_f1_std']:.4f})")
    print(f"  Cohen's Kappa: {cv_summary['kappa_mean']:.4f} (+/- {cv_summary['kappa_std']:.4f})")

    return cv_summary, fold_results

print("Cross-validation function defined (MPS optimized).")


# ============================================================
# 7. Test Set 최종 평가 함수
# ============================================================
# CV가 완료된 후, 전체 학습 데이터(10-fold 분할 없이)로 최종 모델을 학습하고
# 고정된 테스트 세트에서 최종 성능을 평가합니다.
#
# CV 성능 = 모델의 일반적 성능 추정치 (내부 검증)
# Test 성능 = 모델의 실제 일반화 성능 (외부 검증)
#
# 학습 완료 후 모델과 토크나이저를 디스크에 저장하여 재사용할 수 있습니다.
# ============================================================

# In[13]:


def train_and_evaluate_on_test(model_name, model_path, train_texts, train_labels,
                                test_texts, test_labels, config):
    """전체 학습 데이터로 학습 후 Test Set 평가 (MPS 최적화)"""

    print(f"\n{'='*60}")
    print(f"Final Training & Test Evaluation: {model_name}")
    print(f"{'='*60}")

    set_seed(42)

    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Dataset 생성
    train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer, config["max_length"])
    test_dataset = FeedbackDataset(test_texts, test_labels, tokenizer, config["max_length"])

    # DataLoader (MPS 최적화 설정 적용)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False)
    )

    # 모델 로드
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=NUM_LABELS,
            trust_remote_code=True
        )
    except:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=NUM_LABELS
        )
    model.to(DEVICE)

    # Class weight 계산
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 학습
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE, config["gradient_clip"]
        )
        print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

    # Test Set 평가
    test_metrics, predictions, true_labels = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n[Test Set Results] {model_name}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa: {test_metrics['kappa']:.4f}")

    # Classification Report
    print(f"\n[Classification Report]")
    print(classification_report(true_labels, predictions))

    # 모델 저장
    save_dir = f"{config['output_dir']}/models/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    return test_metrics, predictions, true_labels, model

print("Test evaluation function defined (MPS optimized).")


# ============================================================
# 8. 실험 실행
# ============================================================
# 4개 모델 각각에 대해:
#   1) 10-Fold CV로 내부 검증 성능 측정
#   2) 전체 학습 데이터로 최종 모델 학습 → 테스트 세트 평가
#
# 총 실행 시간: M2 Max 기준 약 2~3시간 (모델당 30~45분)
# GPU(CUDA) 환경에서는 훨씬 빠릅니다.
# ============================================================

# In[14]:


# 데이터 준비
train_texts = df_train[CONFIG["text_column"]].tolist()
train_labels = df_train[CONFIG["label_column"]].tolist()
test_texts = df_test[CONFIG["text_column"]].tolist()
test_labels = df_test[CONFIG["label_column"]].tolist()

print(f"Train samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")


# In[15]:


# 모든 모델에 대해 CV 실행
cv_results = {}

for model_name, model_path in CONFIG["models"].items():
    try:
        cv_summary, fold_results = run_cross_validation(
            model_name, model_path, train_texts, train_labels, CONFIG
        )
        cv_results[model_name] = {
            'summary': cv_summary,
            'folds': fold_results
        }
    except Exception as e:
        print(f"\n❌ Error with {model_name}: {e}")
        continue


# In[16]:


# CV 결과 테이블
cv_df = pd.DataFrame([r['summary'] for r in cv_results.values()])
print("\n=== Cross-Validation Results ===")
print(cv_df.to_string())

# 저장
cv_df.to_csv(f"{CONFIG['output_dir']}/cv_results.csv", index=False)
print(f"\nCV results saved to {CONFIG['output_dir']}/cv_results.csv")


# In[17]:


# Test Set 최종 평가
test_results = {}
best_model = None
best_predictions = None
best_true_labels = None

for model_name, model_path in CONFIG["models"].items():
    try:
        test_metrics, predictions, true_labels, model = train_and_evaluate_on_test(
            model_name, model_path, train_texts, train_labels, 
            test_texts, test_labels, CONFIG
        )
        test_results[model_name] = test_metrics

        # Best 모델 저장 (Macro-F1 기준)
        if best_model is None or test_metrics['macro_f1'] > test_results.get(best_model, {}).get('macro_f1', 0):
            best_model = model_name
            best_predictions = predictions
            best_true_labels = true_labels

    except Exception as e:
        print(f"\n❌ Error with {model_name}: {e}")
        continue


# In[18]:


# Test 결과 테이블
test_df = pd.DataFrame([
    {'model': name, **metrics} 
    for name, metrics in test_results.items()
])
print("\n=== Test Set Results ===")
print(test_df.to_string())

# 저장
test_df.to_csv(f"{CONFIG['output_dir']}/test_results.csv", index=False)
print(f"\nTest results saved to {CONFIG['output_dir']}/test_results.csv")


# ============================================================
# 9. 결과 시각화
# ============================================================
# Best 모델의 혼동 행렬(Confusion Matrix)과 모델 간 성능 비교 차트를 생성합니다.
# 혼동 행렬: 실제 클래스 vs 예측 클래스를 격자로 보여줌
#   → 대각선 = 정확한 예측, 비대각선 = 오분류 패턴 파악 가능
# ============================================================

# In[19]:


# Best 모델 Confusion Matrix
if best_model and best_predictions is not None:
    cm = confusion_matrix(best_true_labels, best_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/confusion_matrix_{best_model}.png", dpi=150)
    plt.show()

    print(f"\nBest model: {best_model}")


# In[20]:


# 모델 성능 비교 차트
if test_results:
    models = list(test_results.keys())
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'kappa']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [test_results[m][metric] for m in models]
        axes[idx].bar(models, values)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim(0, 1)
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/model_comparison.png", dpi=150)
    plt.show()


# ============================================================
# 10. 실험 설정 저장
# ============================================================
# 실험 재현을 위해 모든 하이퍼파라미터를 JSON 파일로 저장합니다.
# ============================================================

# In[21]:


import json

# 설정 저장
config_to_save = CONFIG.copy()
config_to_save['num_labels'] = NUM_LABELS
config_to_save['device'] = str(DEVICE)

with open(f"{CONFIG['output_dir']}/experiment_config.json", 'w') as f:
    json.dump(config_to_save, f, indent=2, ensure_ascii=False)

print(f"Experiment config saved to {CONFIG['output_dir']}/experiment_config.json")
print("\n=== Experiment Complete ===")


# ============================================================
# 11. 모델 로드 및 예측
# ============================================================
# 학습이 완료된 모델을 디스크에서 불러와 새로운 텍스트에 대해
# 예측을 수행하는 유틸리티 함수입니다.
# 저장된 모델은 HuggingFace 형식으로, 토크나이저와 모델 가중치가 함께 저장됩니다.
# ============================================================

# In[22]:


def load_bert_model(model_name, model_dir='./results/models', device=None):
    """저장된 BERT 모델 로드"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model_path = f"{model_dir}/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, tokenizer, device

def predict_with_bert(texts, model, tokenizer, device, max_length=128, batch_size=32):
    """로드된 BERT 모델로 예측 수행"""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

    dataset = SimpleDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

print("Model loading and prediction functions defined.")


# In[23]:


# 모델 로드 테스트
print("=== 모델 로드 테스트 ===")
test_model_name = "KLUE-RoBERTa"  # 또는 best_model
model, tokenizer, device = load_bert_model(test_model_name)
print(f"모델 로드 완료: {test_model_name}")

# 예측 테스트
sample_texts = test_texts[:5]
preds = predict_with_bert(sample_texts, model, tokenizer, device)
print(f"\n샘플 예측 결과:")
for text, pred in zip(sample_texts, preds):
    print(f"  '{text[:30]}...' → {pred}")

