# scripts/predict_bert_calibrated.py
# -*- coding: utf-8 -*-

# Script para fazer predições com um modelo BERT multilabel calibrado
# Suporta calibração por Platt Scaling ou Temperature Scaling, a partir do set do RETUNE_DIR

import json
from pathlib import Path
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= AJUSTE AQUI =========
MODEL_DIR   = Path(r"outputs/models/bert_base_cb_loss_final")
CLASSES_TXT = Path(r"outputs/metrics/bert_base_cb_loss_classes.txt")


# aponte para a PASTA do retune escolhido (platt OU temperature)
RETUNE_DIR  = Path(r"outputs/retunes/bert_base_cb_loss/calib=platt.beta=2.pfNone.topk=0.cv=5.lam0.85.alpha1.0.pcap3.0")
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Carrega classes
CLASSES = [l.strip() for l in CLASSES_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]

# ---- Acha thresholds (json ou npy) e carrega de forma robusta
def load_thresholds(th_json_path, th_npy_path, classes):
    if th_json_path and th_json_path.exists():
        obj = json.loads(th_json_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "thresholds" in obj:
            th = np.array(obj["thresholds"], dtype=float)
            # se o JSON trouxer as classes, faça só um aviso
            if "classes" in obj:
                cls_in = [str(c) for c in obj["classes"]]
                if len(cls_in) == len(classes) and cls_in != classes:
                    print("[aviso] Ordem de classes difere entre thresholds.json e CLASSES_TXT.")
            return th
        if isinstance(obj, list):
            return np.array(obj, dtype=float)
        raise ValueError(f"Formato inesperado em {th_json_path}")
    if th_npy_path and th_npy_path.exists():
        return np.load(th_npy_path).astype(float)
    raise FileNotFoundError("Não encontrei thresholds em JSON/NPY.")

# ---- carregar calibração (temperature/platt) em vários formatos

def load_calibration(config_json_path, n_classes):
    mode = "none"
    T = None; A = None; B = None
    
    if config_json_path and config_json_path.exists():
        cfg = json.loads(config_json_path.read_text(encoding="utf-8"))
        
        if "calibration" in cfg:
            cal = cfg["calibration"]
            mode = cal.get("mode", "none")

            if mode == "platt":
                artifacts = cal.get("artifacts", {})
                
                # Inicializar listas para A e B
                A_list = [0.0] * n_classes
                B_list = [0.0] * n_classes
                
                # Extrair parâmetros para cada classe
                valid_params = True
                for i in range(n_classes):
                    class_key = str(i)
                    if class_key in artifacts:
                        class_artifacts = artifacts[class_key]
                        if "a" in class_artifacts and "b" in class_artifacts:
                            A_list[i] = class_artifacts["a"]
                            B_list[i] = class_artifacts["b"]
                        else:
                            valid_params = False
                    else:
                        valid_params = False
                
                if valid_params:
                    A = A_list
                    B = B_list
                else:
                    mode = "none"

            elif mode == "temperature":
                artifacts = cal.get("artifacts", {})
                T_dict = artifacts.get("T", {})
                
                if isinstance(T_dict, dict) and len(T_dict) == n_classes:
                    T = np.array([T_dict[str(i)] for i in range(n_classes)], dtype=float).reshape(1, -1)
                else:
                    mode = "none"
                    
    return mode, T, A, B

# ---- localizar arquivos no RETUNE_DIR
th_json = next(RETUNE_DIR.glob("*_thresholds.json"), None)
th_npy  = next(RETUNE_DIR.glob("*_thresholds.npy"),  None)
cfg_json = next(RETUNE_DIR.glob("*_config.json"),    None)

# carregar thresholds e calibração
thresholds = load_thresholds(th_json, th_npy, CLASSES)
assert len(thresholds) == len(CLASSES), "Mismatch: thresholds vs classes"
cal_mode, T, A, B = load_calibration(cfg_json, len(CLASSES))

print(f"[info] calibração detectada: {cal_mode}")
if cal_mode == "temperature" and T is None:
    print("[aviso] temperature sem vetor T; seguindo sem calibração.")
if cal_mode == "platt" and (A is None or B is None):
    print("[aviso] platt sem A/B; seguindo sem calibração.")

# ---- Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

@torch.inference_mode()
def predict(text: str, max_length: int = 128, topk_fallback: int = 3):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
    logits = model(**enc).logits  # [1, C]

    # aplica calibração
    if cal_mode == "temperature" and T is not None:
        logits = logits / torch.tensor(T, dtype=logits.dtype, device=logits.device)
        probs = torch.sigmoid(logits)
    elif cal_mode == "platt" and A is not None and B is not None:
        A_t = torch.tensor(np.array(A).reshape(1, -1), dtype=logits.dtype, device=logits.device)
        B_t = torch.tensor(np.array(B).reshape(1, -1), dtype=logits.dtype, device=logits.device)
        probs = torch.sigmoid(logits * A_t + B_t)
    else:
        probs = torch.sigmoid(logits)

    probs = probs.squeeze(0).detach().cpu().numpy()  # [C]
    pred  = (probs >= thresholds).astype(int)
    picked = [(CLASSES[i], float(probs[i])) for i in range(len(CLASSES)) if pred[i] == 1]

    if not picked and topk_fallback > 0:
        idx = np.argsort(probs)[-topk_fallback:][::-1]
        picked = [(CLASSES[i], float(probs[i])) for i in idx]
    return picked, probs

if __name__ == "__main__":
    print(f"\n=== BERT calibrado — modo={cal_mode} | retune={RETUNE_DIR.name} ===")
    while True:
        try:
            s = input("\nDigite uma frase (ou 'sair'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s or s.lower() in {"sair", "exit", "quit"}:
            break
        
        labels, probs = predict(s)
        sorted_labels = sorted(labels, key=lambda item: item[1], reverse=True)
        
        print("→ Emoções (ordenado por probabilidade):")
        # Altera o loop para usar a lista ordenada
        for lbl, p in sorted_labels:
            print(f"  - {lbl:<15} {p:.3f}")