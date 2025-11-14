#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline clássico (TF-IDF + OneVsRest(LogReg)) — EXPORT + MÉTRICAS BASELINE 0.5

- Lê train/val/test do configs/data.yaml
- TF-IDF (word/char) conforme classic.yaml + busca de hiperparâmetros via RandomizedSearchCV
- Refit final no train completo
- Exporta arrays (VAL/TEST): y_true, y_prob, e (se disponível) logits
- Calcula métricas baseline (corte fixo 0.5) e salva JSON + CSV per-class

"""

import argparse
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import loguniform

from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV

try:
    # Intel sklearnex (opcional)
    from sklearnex import patch_sklearn; patch_sklearn()
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearnex._utils')
except Exception:
    pass

from joblib import dump, load

from src.utils.io import Paths, ensure_dir, save_json, set_global_seed
from src.utils.metrics import compute_all_metrics
from src.features.tfidf import TFIDFConfig, TFIDFFeaturizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import warnings
warnings.filterwarnings("ignore")


# ---------- Config & Dados

def read_cfgs(cfg_paths):
    base = OmegaConf.load(cfg_paths[0])
    data = OmegaConf.load(cfg_paths[1])
    exp  = OmegaConf.load(cfg_paths[2]) if len(cfg_paths) > 2 else None
    return base, data, exp


def read_splits(data_cfg):
    text_col   = data_cfg.dataset.text_col
    labels_col = data_cfg.dataset.labels_col
    sep        = (data_cfg.dataset.label_sep or ";")

    def to_list(s):
        if isinstance(s, str) and s.strip():
            return [t for t in s.split(sep) if t]
        return []

    train = pd.read_csv(data_cfg.dataset.train_file)
    val   = pd.read_csv(data_cfg.dataset.val_file)
    test  = pd.read_csv(data_cfg.dataset.test_file)

    for df in (train, val, test):
        df[text_col] = df[text_col].astype(str)
        df["labels_list"] = df[labels_col].astype(str).apply(to_list)

    classes_file = getattr(data_cfg.dataset, "classes_file", None)
    if classes_file and Path(classes_file).exists():
        classes = [l.strip() for l in Path(classes_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        classes = sorted({l for L in train["labels_list"] for l in L})

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return train, val, test, mlb, text_col


def build_featurizer(exp_cfg):
    # defaults; sobrescreve via classic.yaml se existir
    w_ng = (1, 2); w_min=2; w_max=0.9; w_feat=None
    c_ng = (3, 5); c_min=2; c_max=0.95; c_an="char_wb"; c_feat=None
    if exp_cfg is not None and hasattr(exp_cfg, "tfidf"):
        w = exp_cfg.tfidf.word; c = exp_cfg.tfidf.char
        w_ng = tuple(w.ngram_range); w_min=int(w.min_df); w_max=float(w.max_df)
        w_feat = None if w.max_features is None else int(w.max_features)
        c_ng = tuple(c.ngram_range); c_min=int(c.min_df); c_max=float(c.max_df)
        c_an = str(c.analyzer); c_feat = None if c.max_features is None else int(c.max_features)
    cfg = TFIDFConfig(
        word_ngram_range=w_ng, word_min_df=w_min, word_max_df=w_max, word_max_features=w_feat,
        char_ngram_range=c_ng, char_min_df=c_min, char_max_df=c_max, char_analyzer=c_an, char_max_features=c_feat
    )
    return TFIDFFeaturizer(cfg)


# ---------- Main

def main(cfg_paths):
    base, data, exp = read_cfgs(cfg_paths)
    set_global_seed(int(base.seed))

    paths = Paths.from_cfg(dict(base.paths))
    ensure_dir(paths.outputs_models)
    ensure_dir(paths.outputs_metrics)
    ensure_dir(paths.outputs_curves)

    tag = getattr(exp, "output_tag", "classic_tfidf_logreg")
    out_dir = Path(paths.outputs_metrics)
    out_dir.mkdir(parents=True, exist_ok=True)  # não cria subpasta; segue padrão do BERT

    cache_path = getattr(exp, "cache_path", None)
    if not cache_path:
        cache_path = os.path.join("data", "cache", f"tfidf_logreg_cache_{tag}.joblib")
    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
    Xy_cache = Path(cache_path)

    train, val, test, mlb, text_col = read_splits(data)
    fe = build_featurizer(exp)

    if Xy_cache.exists():
        print(f"✓ Recarregando TF-IDF de {Xy_cache}")
        X_train, X_val, X_test, y_train, y_val, y_test = load(Xy_cache)
    else:
        print("→ Calculando TF-IDF do zero...")
        X_train = fe.fit_transform(train[text_col].tolist())
        X_val   = fe.transform(val[text_col].tolist())
        X_test  = fe.transform(test[text_col].tolist())
        y_train = mlb.transform(train["labels_list"])
        y_val   = mlb.transform(val["labels_list"])
        y_test  = mlb.transform(test["labels_list"])
        dump((X_train, X_val, X_test, y_train, y_val, y_test), Xy_cache, compress=3)
        print(f"✓ Cache salvo em {Xy_cache}")

    assert y_train.shape[1] == len(mlb.classes_), \
        f"Descompasso: y_train tem {y_train.shape[1]} colunas, mas mlb tem {len(mlb.classes_)} classes."

 
 
    # ---------- Estimador + espaço de busca

    model_cfg = getattr(exp, "model", None) if exp is not None else None
    # 'balanced' para lidar com classes desbalanceadas
    class_weight    = getattr(model_cfg, "class_weight", "balanced") if model_cfg else "balanced"
    logreg_max_iter = int(getattr(model_cfg, "logreg_max_iter", 2000)) if model_cfg else 2000

    # O solver 'liblinear' é excelente e um padrão seguro para problema de classificação multirotulo com linear regression
    base_lr = LogisticRegression(
        max_iter=logreg_max_iter,
        n_jobs=-1,
        tol=1e-4,
        solver="liblinear",
        class_weight=class_weight
    )
    ovr_lr  = OneVsRestClassifier(base_lr, n_jobs=-1)

    # Grid de busca
    
    logreg_space = {
        # Foca em uma faixa mais provável de bons valores para C
        "estimator__C": loguniform(1e-2, 10.0),
        # 'l2' é o padrão mais robusto e geralmente suficiente
        "estimator__penalty": ["l2"],
        # Fixa o solver, simplificando a busca
        "estimator__solver": ["liblinear"],
    }

    scorer = make_scorer(f1_score, average="micro")
    cv_splitter = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)

        search = RandomizedSearchCV(
            estimator=ovr_lr,
            param_distributions=logreg_space,
            n_iter=100,
            cv=cv_splitter,
            n_jobs=-1,
            verbose=1,
            random_state=int(base.seed),
            refit=False,
            return_train_score=False,
            scoring=scorer,
        )
        search.fit(X_train, y_train)

    best_cv = float(search.best_score_)
    best_params = search.best_params_
    params = {k.replace("estimator__", ""): v for k, v in best_params.items()}

    # contra combinações inválidas
    solver  = params.get("solver", "liblinear")
    penalty = params.get("penalty", "l2")
    valid = (
        (solver == "liblinear" and penalty in {"l1","l2"}) or
        (solver == "saga"      and penalty in {"l1","l2"})
    )
    if not valid:
        params["penalty"] = "l2"

    best_est = OneVsRestClassifier(
        LogisticRegression(
            max_iter=logreg_max_iter,
            n_jobs=-1,
            random_state=int(base.seed),
            class_weight=class_weight,
            **params
        ),
        n_jobs=-1
    )
    best_est.fit(X_train, y_train)

    
    #------------ Probabilidades (sem calibração)
    # Obs.: OneVsRestClassifier.predict_proba às vezes retorna lista de arrays, então padronizamos para matriz [N, C].
    
    proba_val  = best_est.predict_proba(X_val)
    proba_test = best_est.predict_proba(X_test)
    if isinstance(proba_val, list):
        proba_val  = np.column_stack([p[:, 1] for p in proba_val])
        proba_test = np.column_stack([p[:, 1] for p in proba_test])

    # Logits (quando disponíveis)
    logits_val = logits_test = None
    if hasattr(best_est, "decision_function"):
        try:
            logits_val  = best_est.decision_function(X_val)
            logits_test = best_est.decision_function(X_test)
            # garante 2D
            if logits_val.ndim == 1:  logits_val  = logits_val.reshape(-1, 1)
            if logits_test.ndim == 1: logits_test = logits_test.reshape(-1, 1)
        except Exception:
            pass

    # y_true
    y_val  = mlb.transform(val["labels_list"])
    y_test = mlb.transform(test["labels_list"])

    # EXPORT: arrays (VAL/TEST)
    np.save(out_dir / f"{tag}_val_y_true.npy",  y_val.astype(np.float32))
    np.save(out_dir / f"{tag}_val_y_prob.npy",  proba_val.astype(np.float32))
    np.save(out_dir / f"{tag}_test_y_true.npy", y_test.astype(np.float32))
    np.save(out_dir / f"{tag}_test_y_prob.npy", proba_test.astype(np.float32))
    if logits_val is not None:
        np.save(out_dir / f"{tag}_val_logits.npy",  logits_val.astype(np.float32))
    if logits_test is not None:
        np.save(out_dir / f"{tag}_test_logits.npy", logits_test.astype(np.float32))

    # BASELINE 0.5

    yhat_val  = (proba_val  >= 0.5).astype(int)
    yhat_test = (proba_test >= 0.5).astype(int)

    class_names = mlb.classes_.tolist()
    metrics_val_baseline  = compute_all_metrics(y_val,  proba_val,  yhat_val,  class_names=class_names)
    metrics_test_baseline = compute_all_metrics(y_test, proba_test, yhat_test, class_names=class_names)

    # Modelo
    dump(best_est, Path(paths.outputs_models) / f"{tag}_logreg_ovr.joblib")

    # JSON
    out = {
        "algo": "logreg_ovr",
        "tag": tag,
        "labels": class_names,
        "best_cv_f1_micro": best_cv,
        "best_params": best_params,
        "metrics_val_baseline": metrics_val_baseline,
        "metrics_test_baseline": metrics_test_baseline,
        "tfidf_config": OmegaConf.to_container(exp.tfidf, resolve=True) if (exp and hasattr(exp, "tfidf")) else None,
        "calibration": None,  # calibração fica para o retune
    }
    out_file = out_dir / f"{tag}.json"
    save_json(out, str(out_file))

    # per-class CSV (TEST)
    per_class_test = metrics_test_baseline.get("per_class", {})
    if per_class_test:
        df_pc = pd.DataFrame.from_dict(per_class_test, orient="index")
        df_pc.index.name = "label"
        (out_dir / f"{tag}_per_class_test.csv").write_text(df_pc.to_csv(), encoding="utf-8")

    # classes.txt (ordem)
    with open(out_dir / f"{tag}_classes.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    # console
    print("\n[BASELINE 0.5] CLASSIC —", tag)
    
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
    ap.add_argument("--cfg", nargs="+", required=True,
                    help="Arquivos de config em ordem: base, data, classic(optional tfidf)")
    args = ap.parse_args()
    try:
        main(args.cfg)
    except Exception as e:
        print("--- ERRO NA EXECUÇÃO ---", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)