# scripts/preprocess.py

# -*- coding: utf-8 -*-
import re, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import ftfy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from datasets import Dataset
from transformers import AutoTokenizer
import torch

# ====== CONFIG ======

# NOVO: Dicionário de tradução de classes de Inglês para Português
EN_PT_LABELS = {
    "admiration": "admiração", "amusement": "diversão", "anger": "raiva",
    "annoyance": "irritação", "approval": "aprovação", "caring": "cuidado",
    "confusion": "confusão", "curiosity": "curiosidade", "desire": "desejo",
    "disappointment": "decepção", "disapproval": "desaprovação", "disgust": "nojo",
    "embarrassment": "constrangimento", "excitement": "empolgação", "fear": "medo",
    "gratitude": "gratidão", "grief": "luto", "joy": "alegria", "love": "amor",
    "nervousness": "nervosismo", "optimism": "otimismo", "pride": "orgulho",
    "realization": "percepção", "relief": "alívio", "remorse": "remorso",
    "sadness": "tristeza", "surprise": "surpresa", "neutral": "neutro"
}

MIN_POS_PER_LABEL = 20
KEEP_NEUTRAL = True
CANONICAL_27 = True

DEDUP_POLICY = "union"
VAL_SIZE = 0.10
TEST_SIZE = 0.10
RANDOM_STATE = 42

# Lista de classes em português
CANONICAL_LABELS_27_EN = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise"
]
CANONICAL_LABELS_27 = [EN_PT_LABELS[en] for en in CANONICAL_LABELS_27_EN]
NEUTRAL = EN_PT_LABELS["neutral"]

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML = re.compile(r"<[^>]+>")
_USER = re.compile(r"@\w+")
_HASH = re.compile(r"#(\w+)")
_MULTISPACE = re.compile(r"\s+")
_MULTIPUNCT = re.compile(r"([!?.,;:])\1{1,}")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("[nome]", " ").replace("[name]", " ")
    s = ftfy.fix_text(s)
    s = _URL.sub(" ", s)
    s = _USER.sub(" ", s)
    s = _HASH.sub(r"\1", s)
    s = _HTML.sub(" ", s)
    s = _MULTIPUNCT.sub(r"\1\1", s)
    s = _MULTISPACE.sub(" ", s).strip().lower()
    return s

def infer_columns(df: pd.DataFrame, preferred_text_col: str | None = None):
    cols = list(df.columns)
    text_col = preferred_text_col if (preferred_text_col and preferred_text_col in cols) else ("texto" if "texto" in cols else ("text" if "text" in cols else cols[0]))
    label_cols = [c for c in cols if c != text_col]
    good = []
    for c in label_cols:
        try:
            arr = df[c].dropna().astype(int).unique()
            if set(arr).issubset({0,1}):
                good.append(c)
        except Exception:
            pass
    return text_col, good

def preprocess_dataframe(df, text_col: str, label_cols: list[str]):
    df = df.copy()
    df[text_col] = df[text_col].astype(str).map(clean_text)

    if CANONICAL_27:
        keep_set = set(CANONICAL_LABELS_27 + ([NEUTRAL] if KEEP_NEUTRAL else []))
        label_cols = [c for c in label_cols if c in keep_set]

    Y = df[label_cols].values.astype(int)
    mask_any = Y.sum(axis=1) > 0
    df = df.loc[mask_any].reset_index(drop=True)

    if df.duplicated(subset=[text_col]).any():
        def merge_block(g):
            labs = g[label_cols].values
            if DEDUP_POLICY == "union":
                merged = (labs.sum(axis=0) > 0).astype(int)
            elif DEDUP_POLICY == "majority":
                merged = (labs.sum(axis=0) >= (len(g)/2.0)).astype(int)
            elif DEDUP_POLICY == "drop_conflict":
                merged = labs[0] if (labs == labs[0]).all() else None
            else:
                merged = (labs.sum(axis=0) > 0).astype(int)
            if merged is None: return None
            row = g.iloc[0].copy()
            row[label_cols] = merged
            return row

        rows = []
        for _, g in df.groupby(text_col, as_index=False):
            r = merge_block(g)
            if r is not None:
                if isinstance(r, pd.DataFrame):
                    rows.append(r.iloc[0])
                else:
                    rows.append(r)
        df = pd.DataFrame(rows).reset_index(drop=True)

    Y = df[label_cols].values.astype(int)
    sums = Y.sum(axis=0)
    keep_mask = sums >= MIN_POS_PER_LABEL
    kept = [c for c,k in zip(label_cols, keep_mask) if k]
    dropped = [c for c,k in zip(label_cols, keep_mask) if not k]
    if dropped:
        print(f"[info] dropping labels with < {MIN_POS_PER_LABEL} positives:", dropped)
    df = df[[text_col] + kept].copy()

    Y = df[kept].values.astype(int)
    mask_any = Y.sum(axis=1) > 0
    df = df.loc[mask_any].reset_index(drop=True)

    return df, kept

def multilabel_stratified_splits(df, text_col: str, label_cols: list[str], test_size=0.10, val_size=0.10, seed=42):
    X = df[[text_col]].copy()
    Y = df[label_cols].values.astype(int)

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(msss1.split(X, Y))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size/(1.0-test_size), random_state=seed)
    tr2_idx, va_idx = next(msss2.split(X_tr, Y_tr))
    X_tr2, X_va = X_tr.iloc[tr2_idx], X_tr.iloc[va_idx]
    Y_tr2, Y_va = Y_tr[tr2_idx], Y_tr[va_idx]

    def pack(X_, Y_):
        out = X_.copy()
        out[label_cols] = Y_
        return out.reset_index(drop=True)

    return pack(X_tr2, Y_tr2), pack(X_va, Y_va), pack(X_te, Y_te)

def label_report(df_train, df_val, df_test, text_col, label_cols, out_dir: Path):
    def counts(df):
        Y = df[label_cols].values.astype(int)
        c = Y.sum(axis=0)
        return pd.DataFrame({"label": label_cols, "positives": c, "freq_pct": (c/len(df)*100).round(3)}).sort_values("positives", ascending=False)

    rep = {
        "labels_kept": label_cols,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "train_counts": counts(df_train).to_dict(orient="records"),
        "val_counts": counts(df_val).to_dict(orient="records"),
        "test_counts": counts(df_test).to_dict(orient="records"),
    }
    (out_dir / "label_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep

def df_to_hfds(df: pd.DataFrame, text_col: str, label_cols: list[str]) -> Dataset:
    df2 = df.copy()
    df2["labels"] = df2[label_cols].values.tolist()
    df2 = df2.rename(columns={text_col: "text"})[["text", "labels"]]
    return Dataset.from_pandas(df2, preserve_index=False)

def tokenize_and_save(out_dir: Path, train_df, val_df, test_df, text_col, label_cols, model_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds_tr = df_to_hfds(train_df, text_col, label_cols)
    ds_va = df_to_hfds(val_df,   text_col, label_cols)
    ds_te = df_to_hfds(test_df,  text_col, label_cols)
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=max_length)
    print("Tokenizando…")
    ds_tr = ds_tr.map(tok, batched=True, batch_size=1000)
    ds_va = ds_va.map(tok, batched=True, batch_size=1000)
    ds_te = ds_te.map(tok, batched=True, batch_size=1000)
    ds_tr = ds_tr.remove_columns(["text"])
    ds_va = ds_va.remove_columns(["text"])
    ds_te = ds_te.remove_columns(["text"])
    ds_tr.set_format("torch", columns=["input_ids","attention_mask","labels"])
    ds_va.set_format("torch", columns=["input_ids","attention_mask","labels"])
    ds_te.set_format("torch", columns=["input_ids","attention_mask","labels"])
    (out_dir / "tokenized_train_dataset").mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenized_val_dataset").mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenized_test_dataset").mkdir(parents=True, exist_ok=True)
    ds_tr.save_to_disk((out_dir / "tokenized_train_dataset").as_posix())
    ds_va.save_to_disk((out_dir / "tokenized_val_dataset").as_posix())
    ds_te.save_to_disk((out_dir / "tokenized_test_dataset").as_posix())
    tokenizer.save_pretrained(out_dir / "tokenizer")
    meta = {
        "train_size": len(ds_tr), "val_size": len(ds_va), "test_size": len(ds_te),
        "num_labels": len(label_cols), "labels": label_cols,
        "model_name": model_name, "max_length": max_length,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(meta, out_dir / "metadata.pt")
    print("Tokenização salva em:", out_dir)

def save_text_labels_csvs(out_dir: Path, train_df, val_df, test_df, label_cols, suffix: str, text_col: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    def _sv(df, name):
        d = pd.DataFrame({text_col: df[text_col].values})
        lab = df[label_cols]
        d["labels"] = lab.apply(lambda r: ";".join([c for c,v in zip(label_cols, r.values) if int(v)==1]), axis=1)
        d.to_csv(out_dir / f"{name}{suffix}.csv", index=False, encoding="utf-8")
    
    _sv(train_df, "train")
    _sv(val_df,   "val")
    _sv(test_df,  "test")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=None)
    ap.add_argument("--val",   type=str, default=None)
    ap.add_argument("--test",  type=str, default=None)
    ap.add_argument("--single_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--text_col", type=str, default=None)
    ap.add_argument("--tokenize", action="store_true", help="Tokenizar e salvar datasets/tokenizer/metadata")
    ap.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--save_csv_text_labels", action="store_true", help="Salvar train/val/test.csv (text;labels) também", default=True)

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_csv:
        df = pd.read_csv(args.single_csv)
    else:
        if not (args.train and args.val and args.test):
            raise SystemExit("Provide either --single_csv or all of --train --val --test")
        df_tr = pd.read_csv(args.train)
        df_va = pd.read_csv(args.val)
        df_te = pd.read_csv(args.test)
        df = pd.concat([df_tr, df_va, df_te], ignore_index=True)

    # Renomeia as colunas do DataFrame
    rename_map = {en: pt for en, pt in EN_PT_LABELS.items() if en in df.columns}
    df = df.rename(columns=rename_map)
    print(f"Renomeadas {len(rename_map)} colunas de labels para português.")

    text_col, label_cols = infer_columns(df, preferred_text_col=args.text_col)
    df_clean, kept_cols = preprocess_dataframe(df, text_col, label_cols)
    train_df, val_df, test_df = multilabel_stratified_splits(df_clean, text_col, kept_cols, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=RANDOM_STATE)

    label_report(train_df, val_df, test_df, text_col, kept_cols, out_dir)

    print("Salvando a lista de emoções em selected_emotions.txt...")
    with open(out_dir / 'selected_emotions.txt', 'w', encoding='utf-8') as f:
        for emotion in kept_cols:
            f.write(f"{emotion}\n")

    suffix = f"_trunc{args.max_length}"
    if args.save_csv_text_labels:
        print(f"Salvando CSVs no formato (texto;labels) com sufixo {suffix}.csv...")
        save_text_labels_csvs(out_dir, train_df, val_df, test_df, kept_cols, suffix, text_col)

    if args.tokenize:
        tokenize_and_save(out_dir, train_df, val_df, test_df, text_col, kept_cols, model_name=args.model_name, max_length=args.max_length)

    print("\nOK! Pré-processamento concluído.")
    print("Saída:", out_dir.resolve())
    print("Labels mantidos:", kept_cols)

if __name__ == "__main__":
    main()