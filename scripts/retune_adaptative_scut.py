
# Esse script realiza o retune de thresholds para modelos multilabel,
# com foco em otimização adaptativa (Adaptative SCUT), para ser usado no run_retune_scut.ps1

import argparse
import json
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from calibration_utils import (
    fit_temperature_per_class, apply_temperature_per_class,
    fit_platt_per_class, apply_platt_per_class,
    fit_isotonic_per_class, apply_isotonic_per_class,
    ece_binary_mean, sigmoid, sweep_thresholds, evaluate,
    per_class_report, export_bundle, plot_calibration_curves_per_class
)

def main():
    ap = argparse.ArgumentParser(description="Retune Adaptative SCUT - Otimizador de Thresholds Focado.")
    
    # Argumentos principais
    ap.add_argument("model_tag", type=str, help="O 'tag' do modelo")
    ap.add_argument("--kind", required=True, choices=["bert", "classic"], help="Tipo de modelo")
    ap.add_argument("--export_dir", type=str, required=True, help="Pasta de saída")

    # Parâmetros de experimentação
    ap.add_argument("--calibration", type=str, default="none", choices=["none", "temperature", "platt", "isotonic"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--precision_floor", type=float, default=None)
    ap.add_argument("--cv_thresholds", type=int, default=0)
    ap.add_argument("--regularize_lambda", type=float, default=None)
    ap.add_argument("--topk_hybrid", type=int, default=0)
    
    # Parâmetros de robustez
    ap.add_argument("--precision_smooth_alpha", type=float, default=0.0, help="Laplace smoothing (ex: 1.0). 0=off")
    ap.add_argument("--prevalence_cap", type=float, default=None, help="Limite de taxa de predição (ex: 3.0)")
    
    args = ap.parse_args()

    # --- 1. Montar caminhos e carregar dados ---
    # Raiz do projeto = pasta acima de scripts/
    ROOT = Path(__file__).resolve().parent.parent
    metrics_dir = ROOT / "outputs" / "metrics"

    paths = {
        "classes":      metrics_dir / f"{args.model_tag}_classes.txt",
        "val_y_true":   metrics_dir / f"{args.model_tag}_val_y_true.npy",
        "val_y_prob":   metrics_dir / f"{args.model_tag}_val_y_prob.npy",
        "val_logits":   metrics_dir / f"{args.model_tag}_val_logits.npy",
        "test_y_true":  metrics_dir / f"{args.model_tag}_test_y_true.npy",
        "test_y_prob":  metrics_dir / f"{args.model_tag}_test_y_prob.npy",
        "test_logits":  metrics_dir / f"{args.model_tag}_test_y_logits.npy",
    }
    print(f"Carregando dados para o modelo: {args.model_tag} ({args.kind})")
    classes = [line.strip() for line in paths["classes"].read_text(encoding="utf-8").splitlines()]
    y_true_val = np.load(paths["val_y_true"])
    y_true_test = np.load(paths["test_y_true"])

    # --- 2. Lógica de Calibração ---

    print(f"Modo de calibração: {args.calibration}")
    calib = {"mode": args.calibration}
    yprob_info = {}
    logits_val = None
    if args.kind == 'bert':
        logits_val = np.load(paths["val_logits"]); logits_test = np.load(paths["test_logits"])
        y_prob_val_raw, y_prob_test_raw = sigmoid(logits_val), sigmoid(logits_test)
        yprob_info = {"val_ylogits": str(paths["val_logits"]), "test_ylogits": str(paths["test_logits"])}
    else:
        y_prob_val_raw, y_prob_test_raw = np.load(paths["val_y_prob"]), np.load(paths["test_y_prob"])
        yprob_info = {"val_yprob": str(paths["val_y_prob"]), "test_yprob": str(paths["test_y_prob"])}
    y_prob_val, y_prob_test = y_prob_val_raw, y_prob_test_raw
    if args.calibration == "temperature":
        T = fit_temperature_per_class(logits_val, y_true_val); y_prob_val = apply_temperature_per_class(logits_val, T); y_prob_test = apply_temperature_per_class(logits_test, T); calib["artifacts"] = {"T": T}
    elif args.calibration == "platt":
        params = fit_platt_per_class(logits_val, y_true_val); y_prob_val = apply_platt_per_class(logits_val, params); y_prob_test = apply_platt_per_class(logits_test, params); calib["artifacts"] = {k: {"a": v[0], "b": v[1]} for k, v in params.items()}
    elif args.calibration == "isotonic":
        models = fit_isotonic_per_class(y_prob_val_raw, y_true_val); y_prob_val = apply_isotonic_per_class(y_prob_val_raw, models); y_prob_test = apply_isotonic_per_class(y_prob_test_raw, models); calib["artifacts"] = models
    yprob_info["calibration_mode"] = args.calibration
    
    if args.calibration != "none":
        plot_dir = Path(args.export_dir) / "calibration_curves"
        plot_calibration_curves_per_class(
            y_true=y_true_val,
            y_prob_raw=y_prob_val_raw,
            y_prob_calibrated=y_prob_val,
            classes=classes,
            calib_mode=args.calibration,
            out_dir=plot_dir,
        )

    # --- 3. Busca de Thresholds ---
    print(f"Buscando thresholds (beta={args.beta}, cv={args.cv_thresholds})...")

    sweep_kwargs = {
        "beta": args.beta,
        "precision_floor": args.precision_floor,
        "min_positives": 1,
        "fallback": 0.5,
        "precision_smooth_alpha": args.precision_smooth_alpha,
        "prevalence_cap": args.prevalence_cap,
    }

    if args.cv_thresholds and args.cv_thresholds > 1:
        kf = KFold(n_splits=int(args.cv_thresholds), shuffle=True, random_state=42)
        thr_list = [sweep_thresholds(y_true_val[idx], y_prob_val[idx], **sweep_kwargs)[0] for _, idx in kf.split(y_true_val)]
        thresholds = np.median(np.vstack(thr_list), axis=0)
    else:
        thresholds, _ = sweep_thresholds(y_true_val, y_prob_val, **sweep_kwargs)
    
    if args.regularize_lambda is not None:
        lam = float(args.regularize_lambda)
        thresholds = lam * thresholds + (1.0 - lam) * 0.5

    # --- 4. Avaliação e Export ---

    print("Avaliando métricas finais...")
    metrics_val = evaluate(y_true_val, y_prob_val, thresholds, topk_hybrid=args.topk_hybrid)
    metrics_test = evaluate(y_true_test, y_prob_test, thresholds, topk_hybrid=args.topk_hybrid)
    metrics_val['ece'] = ece_binary_mean(y_true_val, y_prob_val)
    metrics_test['ece'] = ece_binary_mean(y_true_test, y_prob_test)
    per_class_val = per_class_report(y_true_val, y_prob_val, classes, thresholds, topk_hybrid=args.topk_hybrid)
    per_class_test = per_class_report(y_true_test, y_prob_test, classes, thresholds, topk_hybrid=args.topk_hybrid)
    metrics_test["per_class_test"] = per_class_test
    sweep_cfg = {k: v for k, v in vars(args).items() if k not in ["model_tag", "kind", "export_dir", "calibration"]}
    sweep_cfg["beta"] = args.beta
    export_bundle(export_dir=Path(args.export_dir), tag=args.model_tag, classes=classes, thresholds=thresholds, calib=calib, 
                  metrics_val=metrics_val, per_class_val=per_class_val, y_true_path=paths["val_y_true"],
                  y_prob_or_logits_info=yprob_info, sweep_cfg=sweep_cfg, metrics_test=metrics_test)
    print(f"\n Experimento concluído para {args.model_tag}!")

if __name__ == "__main__":
    main()