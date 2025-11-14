# -*- coding: utf-8 -*-
"""
calibration_utils.py

Caixa de ferramentas para o pós-processamento de modelos de classificação multi-label.

Este módulo contém funções para:
1.  **Calibração de Probabilidades:** Métodos como Temperature Scaling, Platt Scaling e 
    Isotonic Regression para ajustar as saídas do modelo e torná-las mais confiáveis.
2.  **Otimização de Thresholds:** Uma função robusta (`sweep_thresholds`) para encontrar
    o ponto de corte ideal por classe que maximiza uma métrica de interesse (F-beta),
    incluindo várias salvaguardas contra overfitting.
3.  **Avaliação:** Funções para calcular métricas de desempenho agregadas e por classe,
    além de métricas de calibração como o ECE.
4.  **Exportação:** Utilitários para salvar todos os artefatos de um experimento
    (configurações, resultados, thresholds) de forma organizada.
"""

# Imports

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
import re
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

# =============================================================================
# MÉTODOS DE CALIBRAÇÃO DE PROBABILIDADES
# =============================================================================
# Estas funções ajustam as saídas brutas do modelo para que a confiança do modelo reflita melhor a acurácia real.

def fit_temperature_per_class(logits_val: np.ndarray, y_val: np.ndarray, grid: np.ndarray = np.linspace(0.8, 2.0, 25)) -> Dict[str, float]:
    """
    Ajusta um parâmetro de Temperatura (T) por classe para minimizar a log-loss.
    Ideal para redes neurais que tendem a ser superconfiantes usnndo logits como entrada.
    """
    n_classes = y_val.shape[1]
    T_by_class: Dict[str, float] = {}
    for i in range(n_classes):
        y = y_val[:, i]
        # Se uma classe não tem variação (só 0s ou só 1s), não há o que calibrar.
        if len(np.unique(y)) < 2:
            T_by_class[str(i)] = 1.0
            continue
        best_T, best_loss = 1.0, math.inf
        z = logits_val[:, i]
        # Procura o melhor valor de T em uma grade pré-definida
        for T in grid:
            p = sigmoid(z / T).clip(1e-12, 1 - 1e-12)
            loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            if loss < best_loss:
                best_loss, best_T = loss, T
        T_by_class[str(i)] = float(best_T)
    return T_by_class

def apply_temperature_per_class(logits: np.ndarray, T_by_class: Dict[str, float]) -> np.ndarray:
    """Aplica a Temperatura (T) aprendida aos logits para obter probabilidades calibradas."""
    out = np.empty_like(logits, dtype=float)
    for i in range(logits.shape[1]):
        # Divide os logits pela temperatura antes de aplicar a sigmoide
        T = T_by_class.get(str(i), 1.0)
        out[:, i] = sigmoid(logits[:, i] / T)
    return out

def fit_platt_per_class(logits_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Ajusta uma Regressão Logística por classe (Platt Scaling).
    Aprende dois parâmetros (a, b) para a transformação sigmoidal: σ(a*z + b) e usa logits.
    """
    n_classes = y_val.shape[1]
    params: Dict[str, Tuple[float, float]] = {}
    for i in range(n_classes):
        y = y_val[:, i]
        if len(np.unique(y)) < 2:
            params[str(i)] = (1.0, 0.0)  # Fallback: sigmoide padrão
            continue
        # Treina um modelo de regressão logística nos logits
        lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, class_weight=None)
        lr.fit(logits_val[:, i].reshape(-1, 1), y)
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        params[str(i)] = (a, b)
    return params

def apply_platt_per_class(logits: np.ndarray, params: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Aplica os parâmetros (a, b) aprendidos para obter probabilidades calibradas."""
    out = np.empty_like(logits, dtype=float)
    for i in range(logits.shape[1]):
        a, b = params.get(str(i), (1.0, 0.0))
        out[:, i] = sigmoid(a * logits[:, i] + b)
    return out

def fit_isotonic_per_class(prob_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
    """
    Ajusta uma Regressão Isotônica por classe.
    É um método não-paramétrico que aprende uma função "escada" monotônica.
    Muito flexível, mas pode sofrer overfit com poucos dados. Usa probabilidades.
    """
    n_classes = y_val.shape[1]
    models: Dict[str, Dict[str, List[float]]] = {}
    for i in range(n_classes):
        y = y_val[:, i]
        p = prob_val[:, i]
        if len(np.unique(y)) < 2:
            # Fallback: identidade
            models[str(i)] = {"x_": [0.0, 1.0], "y_": [0.0, 1.0]}
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        grid = np.linspace(0, 1, 101)
        mapped = ir.predict(grid)
        models[str(i)] = {"x_": grid.tolist(), "y_": mapped.tolist()}
    return models

def apply_isotonic_per_class(prob: np.ndarray, models: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
    """Aplica a Regressão Isotônica aprendida usando interpolação linear nos pontos salvos."""
    out = np.empty_like(prob, dtype=float)
    for i in range(prob.shape[1]):
        m = models.get(str(i))
        if m is None:
            out[:, i] = prob[:, i] # Se não houver modelo, mantém a prob original
            continue
        x = np.asarray(m["x_"])
        y = np.asarray(m["y_"])
        # Interpola as novas probabilidades com base na curva aprendida
        out[:, i] = np.interp(prob[:, i], x, y)
    return out

# =============================================================================
# OTIMIZAÇÃO DE THRESHOLDS
# =============================================================================

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    start: float = 0.01,
    stop: float = 0.99,
    step: float = 0.01,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
    min_positives: int = 0,
    fallback: float = 0.5,
    precision_smooth_alpha: float = 0.0,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    prevalence_cap: float | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Principal da otimização. Testa uma grade de thresholds 
    para cada classe e seleciona aquele que maximiza o F-beta Score, sujeito
    a várias restrições de robustez para evitar overfitting.
    """
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, fallback, dtype=float)
    per_class: List[Dict[str, Any]] = []
    grid = np.arange(start, stop + 1e-12, step)

    # Calcula a frequência real de cada classe no conjunto de validação
    prevalence = np.clip(np.mean(y_true, axis=0), 1e-12, 1.0)

    for c in range(n_classes):
        y_t, y_p = y_true[:, c], y_prob[:, c]
        best_f, best_t = -1.0, fallback
        best_tuple = (0.0, 0.0, 0.0)
        best_counts = (0, 0, 0)

        for t in grid:
            # Aplica restrições de limite rígido, se houver
            if (min_threshold is not None and t < min_threshold) or (max_threshold is not None and t > max_threshold):
                continue

            y_hat = (y_p >= t).astype(int)

            # Ignora thresholds que resultam em poucas predições positivas
            if min_positives and y_hat.sum() < min_positives:
                continue

            # PCap Impede que a taxa de predição exceda X vezes a prevalência real
            if prevalence_cap is not None:
                pred_rate = y_hat.mean() + 1e-12
                if pred_rate > prevalence_cap * prevalence[c]:
                    continue

            tp, fp, fn = _counts(y_t, y_hat)

            # Usa Laplace Smoothing na precisão para estabilizar a busca
            if precision_smooth_alpha and precision_smooth_alpha > 0.0:
                p = _laplace_precision(tp, fp, alpha=precision_smooth_alpha)
            else:
                p = tp / (tp + fp + 1e-12)

            r = tp / (tp + fn + 1e-12)

            # PFlorr: ignora thresholds que não atingem um piso mínimo de P ou R
            if (precision_floor is not None and p < precision_floor) or (recall_floor is not None and r < recall_floor):
                continue
            
            # Calcula o F-beta Score
            b2 = beta * beta
            f = (1 + b2) * p * r / (b2 * p + r + 1e-12)

            # Atualiza o melhor threshold se o F-score atual for maior
            if f > best_f:
                best_f, best_t, best_tuple, best_counts = f, t, (p, r, f), (tp, fp, fn)

        thresholds[c] = best_t
        per_class.append({"precision": float(best_tuple[0]),"recall": float(best_tuple[1]),"f1": float(best_tuple[2]),"threshold": float(best_t),"tp": int(best_counts[0]),"fp": int(best_counts[1]),"fn": int(best_counts[2])})
    
    return thresholds, per_class

# =============================================================================
# AVALIAÇÃO E RELATÓRIOS
# =============================================================================

def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray, topk_hybrid: int = 0) -> np.ndarray:
    """
    Converte probabilidades em predições binárias (0/1) usando os thresholds otimizados.
    Inclui uma lógica opcional de 'top-k híbrido' que força as k classes mais prováveis
    a serem 1, garantindo um recall mínimo.
    """
    y_hat = (y_prob >= thresholds).astype(int)
    if topk_hybrid and topk_hybrid > 0:
        topk = min(topk_hybrid, y_prob.shape[1])
        # Encontra os índices das 'k' maiores probabilidades para cada amostra
        idx = np.argpartition(-y_prob, kth=topk - 1, axis=1)[:, :topk]
        rows = np.arange(y_prob.shape[0])[:, None]
        # Define essas posições como 1
        y_hat[rows, idx] = 1
    return y_hat

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, topk_hybrid: int = 0) -> Dict[str, float]:
    """Calcula métricas de desempenho agregadas (micro, macro, mAP) após aplicar os thresholds."""
    y_hat = apply_thresholds(y_prob, thresholds, topk_hybrid=topk_hybrid)
    f1_micro = f1_score(y_true, y_hat, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_hat, average="macro", zero_division=0)
    
    # mAP (mean Average Precision) avalia o ranking das probabilidades
    ap_per_class = [average_precision_score(y_true[:, c], y_prob[:, c]) if np.any(y_true[:, c]) else float("nan") for c in range(y_true.shape[1])]
    map_macro = float(np.nanmean(ap_per_class))
    return dict(f1_micro=f1_micro, f1_macro=f1_macro, map_macro=map_macro)

def per_class_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
    thresholds: np.ndarray,
    topk_hybrid: int = 0,
) -> dict[str, dict]:
    """Calcula um relatório detalhado de métricas para CADA CLASSE individualmente."""
    y_hat = apply_thresholds(y_prob, thresholds, topk_hybrid=topk_hybrid)
    out: dict[str, dict] = {}
    for c, cls in enumerate(classes):
        y_t, y_p, y_h = y_true[:, c], y_prob[:, c], y_hat[:, c]
        tp, fp, fn = _counts(y_t, y_h)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ap = average_precision_score(y_t, y_p) if np.any(y_t) else float("nan")
        out[cls] = dict(precision=float(prec), recall=float(rec), f1=float(f1), ap=float(ap), threshold=float(thresholds[c]), tp=tp, fp=fp, fn=fn)
    return out


def plot_calibration_curves_per_class(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
    classes: list[str],
    calib_mode: str,
    out_dir: Path,
):
    """
    Gera e salva um gráfico de calibração para cada classe, comparando as
    probabilidades antes e depois da calibração, com fontes ampliadas
    para uso em publicações.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = y_true.shape[1]
    print(f"Gerando curvas de calibração em: {out_dir.resolve()}")

    # --- Ajuste global de fontes e estilo ---
    plt.rcParams.update({
        "font.size": 20,           # fonte base
        "axes.titlesize": 24,      # título do gráfico
        "axes.labelsize": 20,      # rótulos dos eixos
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "figure.titlesize": 24,
    })

    for c in range(n_classes):
        cls_name = classes[c]
        y_t = y_true[:, c]
        y_p_raw = y_prob_raw[:, c]
        y_p_calib = y_prob_calibrated[:, c]

        fig, ax = plt.subplots(figsize=(12, 11))

        # --- Curva antes da calibração ---
        CalibrationDisplay.from_predictions(
            y_t, y_p_raw,
            name="Antes (Raw)",
            n_bins=15,
            ax=ax,
            strategy="uniform"
        )

        # --- Curva após calibração ---
        CalibrationDisplay.from_predictions(
            y_t, y_p_calib,
            name=f"Depois ({calib_mode.capitalize()})",
            n_bins=15,
            ax=ax,
            strategy="uniform"
        )

        # --- Estilo e legendas ---
        ax.set_title(f"Curva de Calibração — {cls_name}", pad=15)
        ax.set_xlabel("Probabilidade prevista", labelpad=10)
        ax.set_ylabel("Frequência observada", labelpad=10)
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        # Linhas de grade e limites visuais
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # Nome seguro do arquivo
        safe_cls_name = re.sub(r'[\\/*?:"<>|]', "", cls_name)
        out_path = out_dir / f"calib_curve_{safe_cls_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"{n_classes} curvas salvas em {out_dir}")

def ece_binary_mean(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Calcula o Expected Calibration Error (ECE), a principal métrica para avaliar a calibração.
    Mede a diferença média entre a confiança do modelo e a acurácia real.
    Quanto menor o ECE, melhor calibrado está o modelo.
    """
    eces = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_prob[:, i]
        if len(np.unique(yt)) < 2: continue
        
        # Agrupa as predições em 'bins' (faixas de confiança)
        idx = np.digitize(yp, bins) - 1
        err_sum, cnt_sum = 0.0, 0.0
        for b in range(n_bins):
            m = idx == b
            if not np.any(m): continue
            acc = yt[m].mean() # Acurácia real no bin
            conf = yp[m].mean() # Confiança média no bin
            err_sum += abs(acc - conf) * m.sum()
            cnt_sum += m.sum()
        if cnt_sum > 0:
            eces.append(err_sum / cnt_sum)
    return float(np.mean(eces)) if eces else float("nan")

# =============================================================================
# FUNÇÕES AUXILIARES E DE EXPORTAÇÃO
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Converte logits em probabilidades."""
    return 1.0 / (1.0 + np.exp(-x))

def _counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[int, int, int]:
    """Calcula contagens de TP, FP, FN."""
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    return tp, fp, fn

def _laplace_precision(tp: int, fp: int, alpha: float = 1.0) -> float:
    """Calcula precisão com Laplace Smoothing para estabilizar a busca."""
    return (tp + alpha) / (tp + fp + 2.0 * alpha + 1e-12)

def export_bundle(
    export_dir: Path,
    tag: str,
    classes: List[str],
    thresholds: np.ndarray,
    calib: Dict[str, Any],
    metrics_val: Dict[str, float],
    per_class_val: Dict[str, Any],
    y_true_path: Path,
    y_prob_or_logits_info: Dict[str, str],
    sweep_cfg: Dict[str, Any],
    metrics_test: Dict[str, float] | None = None,
):
    """
    Coleta todos os resultados de um experimento (métricas, configs, thresholds, etc.)
    e os salva em uma pasta de forma organizada para garantir a reprodutibilidade.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    # Salva thresholds em .npy e .json
    np.save(export_dir / f"{tag}_thresholds.npy", thresholds.astype(np.float32))
    (export_dir / f"{tag}_thresholds.json").write_text(json.dumps({"thresholds": thresholds.tolist(), "classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Salva configurações do experimento
    (export_dir / f"{tag}_config.json").write_text(json.dumps({"tag": tag,"classes": classes,"calibration": calib,"sweep_cfg": sweep_cfg,"inputs": {"val_ytrue": str(y_true_path),**y_prob_or_logits_info,},},ensure_ascii=False,indent=2,),encoding="utf-8",)
    # Salva métricas de validação e teste
    (export_dir / f"{tag}_metrics_val.json").write_text(json.dumps({"metrics_val": metrics_val, "per_class_val": per_class_val}, ensure_ascii=False, indent=2), encoding="utf-8")
    if metrics_test is not None:
        (export_dir / f"{tag}_metrics_test.json").write_text(json.dumps({"metrics_test": metrics_test}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] bundle salvo em: {export_dir}")

def save_json(obj, path: str):
    """Função auxiliar para salvar objetos em arquivos JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    """Função auxiliar para carregar objetos de arquivos JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)