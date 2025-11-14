# src/utils/metrics.py
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score

def compute_all_metrics(y_true, y_proba, y_pred, class_names=None):
    """
    Retorna dicionário com métricas agregadas + por classe .
    y_true, y_proba, y_pred: arrays (N, C)
    """
    # --- agregadas ---
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # mAP macro (AP por classe e média)
    ap_per_class = []
    for c in range(y_true.shape[1]):
        try:
            ap_per_class.append(average_precision_score(y_true[:, c], y_proba[:, c]))
        except Exception:
            ap_per_class.append(np.nan)
    map_macro = float(np.nanmean(ap_per_class))

    # ECE
    ece = None # Sem calibração nos baselines
    
    # --- por classe ---
    # f1/prec/recall por classe (vetores tamanho C)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "map_macro": float(map_macro),
        "ece": ece,
        "ap_per_class": [None if np.isnan(x) else float(x) for x in ap_per_class],
        "f1_per_class": [float(x) for x in f1_c],
        "precision_per_class": [float(x) for x in prec_c],
        "recall_per_class": [float(x) for x in rec_c],
    }

    if class_names is not None:
        # também retorna um dicionário nome->métrica, útil para salvar legível
        metrics["per_class"] = {
            cls: {
                "f1": float(f1_c[i]),
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "ap": None if np.isnan(ap_per_class[i]) else float(ap_per_class[i]),
            }
            for i, cls in enumerate(class_names)
        }

    return metrics