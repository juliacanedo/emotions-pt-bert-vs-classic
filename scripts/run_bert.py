# -*- coding: utf-8 -*-
"""
BERT multilabel + CB-Loss — EXPORTA arrays e MÉTRICAS BASELINE (corte 0.5).
Sem calibração e sem retune

Salva em outputs/metrics/<tag>_*:
- *_val_y_true.npy, *_val_y_prob.npy, *_val_logits.npy
- *_test_y_true.npy, *_test_y_prob.npy, *_test_logits.npy
- <tag>.json (com metrics_baseline, timing, config, cb_loss)
- <tag>_per_class_test.csv (métricas por classe no TEST)
- <tag>_classes.txt (ordem das classes)
"""

import argparse, os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from src.utils.io import Paths, ensure_dir, save_json, set_global_seed
from src.utils.timing import TimerVRAM
from src.utils.metrics import compute_all_metrics



# ---------- Device ----------

def get_device() -> str:
    """Escolhe o melhor device disponível."""
    if torch.cuda.is_available():
        device = "cuda"
    # Mac com Apple Silicon
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Usando device: {device}")
    return device

# ---------- Collator & Dataset ----------
class MultiLabelCollator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, features):
        import torch
        labels = [torch.tensor(f["labels"], dtype=torch.float32) for f in features]
        features_no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(features_no_labels, return_tensors="pt")
        batch["labels"] = torch.stack(labels)
        return batch

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d["input_ids"])
    def __getitem__(self, i):
        item = {k: self.d[k][i] for k in ("input_ids", "attention_mask")}
        item["labels"] = self.d["labels"][i]
        return item

# ---------- Métricas p/ Trainer (logging) ----------
def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    y_true = p.label_ids
    probs = sigmoid_np(logits)
    y_pred = (probs > 0.5).astype(int)
    f1_mic = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    return {"f1_micro": f1_mic}

# ---------- IO ----------
def read_fixed_csvs(cfg_data):
    import pandas as pd
    train = pd.read_csv(cfg_data.dataset.train_file)
    val   = pd.read_csv(cfg_data.dataset.val_file)
    test  = pd.read_csv(cfg_data.dataset.test_file)
    text_col   = cfg_data.dataset.text_col
    labels_col = cfg_data.dataset.labels_col
    sep        = cfg_data.dataset.label_sep or ";"
    def to_list(s: str):
        if isinstance(s, str) and s.strip():
            return [t for t in s.split(sep) if t]
        return []
    for df in (train, val, test):
        df[text_col]   = df[text_col].astype(str)
        df[labels_col] = df[labels_col].astype(str).apply(to_list)
    all_labels = set()
    for df in (train, val, test):
        for labs in df[labels_col]:
            all_labels.update(labs)
    classes = sorted(all_labels)
    return train, val, test, classes

def df_to_torch_dataset(df, tokenizer, text_col: str, classes: List[str], max_len: int):
    import numpy as np
    label_to_idx = {c: i for i, c in enumerate(classes)}
    texts = df[text_col].tolist()
    y = np.zeros((len(df), len(classes)), dtype=np.float32)
    for i, labs in enumerate(df["labels"]):
        for l in labs:
            if l in label_to_idx:
                y[i, label_to_idx[l]] = 1.0
    enc = tokenizer(texts, truncation=True, padding=False, max_length=max_len)
    d = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": y.tolist()}
    return d

# ---------- Config e Modelo ----------
from dataclasses import dataclass
@dataclass
class BertConfig:
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    freeze_encoder: bool = False
    unfreeze_last_n: int = 0
    gradient_checkpointing: bool = True
    temperature_scaling: bool = False
    max_length: int = 128
    batch_size: int = 4
    grad_accum: int = 8
    num_epochs: float = 10.0
    lr: float = 1.5e-5
    weight_decay: float = 0.01
    fp16: bool = True
    warmup_ratio: float = 0.06
    eval_steps: int = 0
    save_strategy: str = "epoch"
    output_tag: str = "bert_full_ft_cb"
    cb_beta: float = 0.999
    lr_scheduler_type: str = "linear"
    dataloader_num_workers: int = 4
    pin_memory: bool = True

def build_model(cfg_bert: BertConfig, num_labels: int, device: str = "cpu"):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg_bert.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        use_safetensors=True,
    )

    base = getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "distilbert", None)
    if cfg_bert.freeze_encoder and base is not None:
        for p in base.parameters():
            p.requires_grad = False
        if cfg_bert.unfreeze_last_n > 0 and hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            layers = base.encoder.layer
            for layer in layers[-int(cfg_bert.unfreeze_last_n):]:
                for p in layer.parameters():
                    p.requires_grad = True

    if cfg_bert.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    return model

# ---------- CB-Loss ----------
class CBTrainer(Trainer):
    def __init__(self, *args, cb_pos_weight: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs); self.cb_pos_weight = cb_pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = labels.float().to(logits.device)
        if self.cb_pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.cb_pos_weight.to(logits.device))
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_cb_pos_weight(label_matrix: np.ndarray, beta: float = 0.999, normalize: bool = True) -> torch.Tensor:
    n_c = label_matrix.sum(axis=0)
    n_c = np.clip(n_c, 1, None)
    effective_num = 1.0 - np.power(beta, n_c)
    alpha = (1.0 - beta) / effective_num
    if normalize: alpha = alpha * (len(alpha) / alpha.sum())
    return torch.tensor(alpha, dtype=torch.float32)

# ---------- Main ----------
def main(cfg_paths):
    base = OmegaConf.load(cfg_paths[0])
    data = OmegaConf.load(cfg_paths[1])
    bert = OmegaConf.load(cfg_paths[2])

    set_global_seed(int(base.seed))
    paths = Paths.from_cfg(dict(base.paths))
    ensure_dir(paths.outputs_metrics); ensure_dir(paths.outputs_curves)

    bc = BertConfig(**OmegaConf.to_container(bert, resolve=True))
    
    #Device
    device = get_device()

    # Dados/tokenizer
    train_df, val_df, test_df, classes = read_fixed_csvs(data)
    text_col = data.dataset.text_col
    tokenizer = AutoTokenizer.from_pretrained(bc.model_name, use_fast=True)
    train_dict = df_to_torch_dataset(train_df, tokenizer, text_col, classes, bc.max_length)
    val_dict   = df_to_torch_dataset(val_df,   tokenizer, text_col, classes, bc.max_length)
    test_dict  = df_to_torch_dataset(test_df,  tokenizer, text_col, classes, bc.max_length)
    ds_train, ds_val, ds_test = DictDataset(train_dict), DictDataset(val_dict), DictDataset(test_dict)
    collator = MultiLabelCollator(tokenizer)

    # Modelo
    model = build_model(bc, num_labels=len(classes), device=device)

    # CB weights
    beta = float(bc.cb_beta)
    y_train = np.array(train_dict["labels"], dtype=np.float64)
    cb_weights = compute_cb_pos_weight(y_train, beta=beta, normalize=True)
    n_pos = y_train.sum(axis=0)

    # Trainer args
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    train_args = TrainingArguments(
        output_dir=os.path.join(paths.outputs_models, bc.output_tag),
        per_device_train_batch_size=bc.batch_size,
        per_device_eval_batch_size=max(1, bc.batch_size * 2),
        optim="adamw_torch", gradient_accumulation_steps=bc.grad_accum,
        num_train_epochs=bc.num_epochs, learning_rate=bc.lr, weight_decay=bc.weight_decay,
        warmup_ratio=bc.warmup_ratio, lr_scheduler_type=bc.lr_scheduler_type,
        eval_strategy="epoch" if bc.eval_steps == 0 else "steps",
        eval_steps=bc.eval_steps if bc.eval_steps > 0 else None,
        logging_strategy="epoch" if bc.eval_steps == 0 else "steps",
        save_strategy=bc.save_strategy, fp16=bool(bc.fp16),
        report_to=[], load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_pin_memory=bc.pin_memory,
        dataloader_num_workers=bc.dataloader_num_workers
    )

    trainer = CBTrainer(
        model=model, args=train_args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=collator, processing_class=tokenizer,
        callbacks=[early_stopping], compute_metrics=compute_metrics, cb_pos_weight=cb_weights,
    )

    # Treino
    with TimerVRAM(track_vram=bool(base.track_vram)) as t_fit:
        trainer.train()
        t_train = t_fit.snapshot()
        
    # Salva o melhor modelo (carregado automaticamente pelo 'load_best_model_at_end=True')
    final_model_path = os.path.join(paths.outputs_models, f"{bc.output_tag}_final")
    trainer.save_model(final_model_path)
    print(f"Melhor modelo final salvo em: {final_model_path}")

    # Predições
    val_pred  = trainer.predict(ds_val)
    test_pred = trainer.predict(ds_test)
    logits_val, logits_test = val_pred.predictions, test_pred.predictions
    y_val  = np.array(val_dict["labels"], dtype=np.float32)
    y_test = np.array(test_dict["labels"], dtype=np.float32)
    proba_val, proba_test = sigmoid_np(logits_val), sigmoid_np(logits_test)

    # === EXPORT arrays ===
    tag = getattr(bert, "output_tag", bc.output_tag)
    out_dir = Path(paths.outputs_metrics)
    np.save(out_dir / f"{tag}_val_y_true.npy",  y_val.astype(np.float32))
    np.save(out_dir / f"{tag}_val_y_prob.npy",  proba_val.astype(np.float32))
    np.save(out_dir / f"{tag}_val_logits.npy",  logits_val.astype(np.float32))
    np.save(out_dir / f"{tag}_test_y_true.npy", y_test.astype(np.float32))
    np.save(out_dir / f"{tag}_test_y_prob.npy", proba_test.astype(np.float32))
    np.save(out_dir / f"{tag}_test_y_logits.npy", logits_test.astype(np.float32))

    # === MÉTRICAS BASELINE (corte 0.5) ===
    yhat_val  = (proba_val  >= 0.5).astype(int)
    yhat_test = (proba_test >= 0.5).astype(int)
    metrics_val_baseline  = compute_all_metrics(y_val,  proba_val,  yhat_val,  class_names=classes)
    metrics_test_baseline = compute_all_metrics(y_test, proba_test, yhat_test, class_names=classes)

    # JSON de saída (sem thresholds; baseline somente)
    out_json = out_dir / f"{tag}.json"
    result = {
        "tag": tag,
        "labels": classes,
        "metrics_val_baseline": metrics_val_baseline,
        "metrics_test_baseline": metrics_test_baseline,
        "timing": {"fit_seconds": t_train.seconds, "peak_vram_mb_fit": t_train.peak_mb},
        "config": {
            "base": OmegaConf.to_container(base, resolve=True),
            "data": OmegaConf.to_container(data, resolve=True),
            "bert": OmegaConf.to_container(bert, resolve=True),
        },
        "cb_loss": {"beta": beta, "n_pos_per_class": n_pos.tolist()},
    }
    save_json(result, str(out_json))

    # CSV por classe (TEST)
    per_class_test = metrics_test_baseline.get("per_class", {})
    if per_class_test:
        import pandas as pd
        df_pc = pd.DataFrame.from_dict(per_class_test, orient="index")
        df_pc.index.name = "label"
        (out_dir / f"{tag}_per_class_test.csv").write_text(df_pc.to_csv(), encoding="utf-8")

    # Classes
    with open(out_dir / f"{tag}_classes.txt", "w", encoding="utf-8") as f:
        for c in classes: f.write(c + "\n")

    # Print resumo
    print("\n[BASELINE 0.5] BERT —", tag)
    g = metrics_test_baseline

    f1mic = float(g.get("f1_micro", float("nan")))
    f1mac = float(g.get("f1_macro", float("nan")))
    mapv  = g.get("map_macro", None)

    print(
        " Test — F1_micro: {:.4f} | F1_macro: {:.4f} | mAP: {}".format(
            f1mic,
            f1mac,
            f"{float(mapv):.4f}" if mapv is not None else "n/a"
        )
    )
    
    print(" Arrays + JSON salvos em:", str(out_dir))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="+", required=True, help="Arquivos de config: base, data, bert")
    args = ap.parse_args()
    main(args.cfg)
