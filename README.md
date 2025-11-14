# 🎭 Emotions-PT — BERT vs. Classic ML para Classificação de Emoções em Português  
**Multilabel Emotion Classification • BERTimbau + Logistic Regression • SCut Threshold Optimization**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

</div>

---

## 📝 Sobre o Projeto

Este repositório implementa um pipeline **completo e reprodutível** para classificação multilabel de emoções em português, comparando o desempenho de um modelo BERTimbau e um algoritmo Clássico (TF-IDF + Regressão Logística).

### 🔥 Modelos BERT (HuggingFace)
- BERTimbau Base / Large  
- Fine-tuning com **Class-Balanced Loss (CB-Loss)**  
- Avaliação completa (F1-micro/macro, mAP, ECE)  
- **Calibração + Otimização de thresholds (SCut / Fβ)**  

### ⚙️ Baseline Clássico (TF-IDF + Logistic Regression)
- Extração híbrida TF-IDF (words + char-ngrams)
- One-vs-Rest Logistic Regression
- Otimização de thresholds classe a classe

### 🎯 Objetivo
Comparar abordagens clássicas vs deep learning no dataset **GoEmotions-PT (28 emoções)** traduzido automaticamente e limpo.

---

## 📂 Estrutura do Repositório

```txt
.
├── configs/
├── data/
│   ├── raw/ * Arquivos originais do dataset GoEmotions-PTBR  
│   │       Fonte: https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr/tree/main 
├── outputs/
│   ├── metrics/
│   ├── retunes/
│   └── models/
├── notebooks/ # Análise do projeto + Interpretabilidade + script para gerar imagens
├── scripts/
├── src/
│    ├── features/
│    └── utils/
├── run_pipeline.ps1
├── run_pipeline.sh
├── requirements_base.txt # Bibliotecas base
├── requirements_cpu.txt # Padrão de instalação para CPU
├── requirements_gpu.txt # Padrão de instalação para GPU (*recomendado*)
└── README.md
```

---

## 🚀 Pipeline Completo (One-Command)

### Windows
```powershell
.\run_pipeline.ps1
```

### Linux
```bash
./run_pipeline.sh
```

O pipeline executa:

1. Criação/remoção do venv  
2. Checagem de GPU  
3. Instalação CPU/GPU  
4. Preprocessamento  
5. EDA  
6. Treino BERT  
7. Treino Clássico  
8. SCut + Calibração  
9. Inicialização do modelo de classificação com melhor BERT Treinado 

---

## 💻 Instalação Manual

Clone o repositório:

```bash
git clone https://github.com/seu-user/emotions-pt-bert-vs-classic.git
cd emotions-pt-bert-vs-classic
```

Criar venv:

```bash
python -m venv venv
source venv/bin/activate      # Linux
venv\Scripts\Activate.ps1   # Windows
```

Instalar dependências GPU:

```bash
pip install -r requirements_gpu.txt
```

Ou CPU:

```bash
pip install -r requirements_cpu.txt
```

---

## Treinar BERT

```bash
python scripts/run_bert.py     --cfg configs/base.yaml configs/data.yaml configs/bert.yaml
```

---

## Treinar Clássico

```bash
python scripts/run_classic.py     --cfg configs/base.yaml configs/data.yaml configs/classic.yaml
```

---

## Otimização SCut / Fβ

```bash
python scripts/run_retune_scut.py
```

---

## Predição

```bash
python scripts/predict_bert_calibrated.py --text "Estou muito feliz hoje!"
```

---

## Licença

Este projeto usa licença **MIT**.

---