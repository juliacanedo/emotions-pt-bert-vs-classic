#!/bin/bash

GREEN="\e[32m"
CYAN="\e[36m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

echo -e "${GREEN}=== PIPELINE COMPLETO ===${RESET}"

echo -e "${CYAN}>>> CRIAR VENV E INSTALAR DEPENDÊNCIAS${RESET}"

# Criar ambiente virtual
python3 -m venv venv

# Ativar
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Escolha do backend CPU/GPU
USE_GPU=true    # <-- ALTERE PARA false SE QUISER CPU

if [ "$USE_GPU" = true ]; then
    echo -e "${YELLOW}>>> Instalando requirements GPU...${RESET}"
    pip install -r requirements_gpu.txt
else
    echo -e "${YELLOW}>>> Instalando requirements CPU...${RESET}"
    pip install -r requirements_cpu.txt
fi

echo -e "\n${CYAN}>>> CHECANDO GPU...${RESET}"
GPU_STATUS=$(python3 -c "import torch; print(torch.cuda.is_available())")

if [ "$GPU_STATUS" = "True" ]; then
    echo -e "${GREEN}>>> GPU DISPONÍVEL! (torch.cuda.is_available = True)${RESET}"
else
    echo -e "${RED}>>> GPU NÃO DISPONÍVEL. (torch.cuda.is_available = False)${RESET}"
fi

echo -e ""
read -p "Continuar com o pipeline? (s/n): " RESP

if [[ "$RESP" != "s" && "$RESP" != "S" && "$RESP" != "y" && "$RESP" != "Y" ]]; then
    echo -e "${RED}Pipeline cancelado pelo usuário.${RESET}"
    exit 1
fi

echo -e "${GREEN}>>> Continuando pipeline...${RESET}"
# --- Exportar PYTHONPATH ---
export PYTHONPATH=$(pwd)

echo -e "\n${CYAN}1. PREPROCESSAMENTO${RESET}"
python scripts/preprocess.py \
    --train data/raw/train.csv \
    --val data/raw/val.csv \
    --test data/raw/test.csv \
    --out_dir data/processed \
    --tokenize --max_length 128

echo -e "\n${CYAN}2. EDA DATASET RAW${RESET}"
python scripts/eda_goemotions.py eda \
    --train data/raw/train.csv \
    --val data/raw/val.csv \
    --test data/raw/test.csv \
    --out_dir reports/eda_raw

echo -e "\n${CYAN}3. EDA DATASET PROCESSADO${RESET}"
python scripts/eda_goemotions.py eda \
    --train data/processed/train_trunc128.csv \
    --val data/processed/val_trunc128.csv \
    --test data/processed/test_trunc128.csv \
    --out_dir reports/eda_processed

echo -e "\n${CYAN}4. TREINAMENTO BERT${RESET}"
python scripts/run_bert.py \
    --cfg configs/base.yaml configs/data.yaml configs/bert.yaml \
    +data.dataset.train_csv="data/processed/train_trunc128.csv" \
    +data.dataset.val_csv="data/processed/val_trunc128.csv" \
    +data.dataset.test_csv="data/processed/test_trunc128.csv"

echo -e "\n${CYAN}5. TREINAMENTO CLASSIC${RESET}"
python scripts/run_classic.py \
    --cfg configs/base.yaml configs/data.yaml configs/classic.yaml \
    +data.dataset.train_csv="data/processed/train_trunc128.csv" \
    +data.dataset.val_csv="data/processed/val_trunc128.csv" \
    +data.dataset.test_csv="data/processed/test_trunc128.csv"

echo -e "\n${CYAN}6. OTIMIZAÇÃO SCUT${RESET}"
# Em Linux não existe PowerShell por padrão → chamamos a versão Python diretamente
python scripts/run_retune_scut.py

echo -e "\n${GREEN}=== PIPELINE COMPLETO ===${RESET}"

echo -e "\n${CYAN}7. INICIALIZAR MODELO TREINADO${RESET}"
export PYTHONPATH=$(pwd)
python scripts/predict_bert_calibrated.py