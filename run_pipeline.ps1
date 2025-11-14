Write-Host "=== PIPELINE COMPLETO ===" -ForegroundColor Green

Write-Host ">>> CRIAR VENV E INSTALAR DEPENDENCIAS" -ForegroundColor Cyan

# Cria ambiente virtual
python -m venv venv

# Ativa
& .\venv\Scripts\Activate.ps1

# Atualiza pip
pip install --upgrade pip

# Escolha do backend CPU/GPU
$useGPU = $true   # <-- ALTERE PARA $false SE QUISER CPU

if ($useGPU) {
    Write-Host ">>> Instalando requirements GPU..." -ForegroundColor Yellow
    pip install -r requirements_gpu.txt
} else {
    Write-Host ">>> Instalando requirements CPU..." -ForegroundColor Yellow
    pip install -r requirements_cpu.txt
}

Write-Host "`n>>> CHECANDO GPU..." -ForegroundColor Cyan
$gpu_status = python -c "import torch; print(torch.cuda.is_available())"

if ($gpu_status -eq "True") {
    Write-Host ">>> GPU DISPONÍVEL! (torch.cuda.is_available = True)" -ForegroundColor Green
} else {
    Write-Host ">>> GPU NÃO DISPONÍVEL. (torch.cuda.is_available = False)" -ForegroundColor Red
}

# ---------------------------------------------------------
# Solicitar autorização para continuar
$resp = Read-Host "`nContinuar com o pipeline? (s/n)"

if ($resp.ToLower() -ne "s" -and $resp.ToLower() -ne "y") {
    Write-Host "Pipeline cancelado pelo usuário." -ForegroundColor Red
    exit
}
# ---------------------------------------------------------

Write-Host "`n>>> Continuando pipeline..." -ForegroundColor Green
$env:PYTHONPATH = (Get-Location).Path

Write-Host "`n1. PREPROCESSAMENTO" -ForegroundColor Cyan
python scripts/preprocess.py --train data/raw/train.csv --val data/raw/val.csv --test data/raw/test.csv --out_dir data/processed --tokenize --max_length 128

Write-Host "`n2. EDA DATASET RAW" -ForegroundColor Cyan
python scripts/eda_goemotions.py eda --train data/raw/train.csv --val data/raw/val.csv --test data/raw/test.csv --out_dir reports/eda_raw

Write-Host "`n3. EDA DATASET PROCESSADO" -ForegroundColor Cyan
python scripts/eda_goemotions.py eda --train data/processed/train_trunc128.csv --val data/processed/val_trunc128.csv --test data/processed/test_trunc128.csv --out_dir reports/eda_processed

Write-Host "`n4. TREINAMENTO BERT" -ForegroundColor Cyan
python scripts/run_bert.py --cfg configs/base.yaml configs/data.yaml configs/bert.yaml +data.dataset.train_csv="data/processed/train_trunc128.csv" +data.dataset.val_csv="data/processed/val_trunc128.csv" +data.dataset.test_csv="data/processed/test_trunc128.csv"

Write-Host "`n5. TREINAMENTO CLASSIC" -ForegroundColor Cyan
python scripts/run_classic.py --cfg configs/base.yaml configs/data.yaml configs/classic.yaml +data.dataset.train_csv="data/processed/train_trunc128.csv" +data.dataset.val_csv="data/processed/val_trunc128.csv" +data.dataset.test_csv="data/processed/test_trunc128.csv"

Write-Host "`n6. OTIMIZAÇÃO SCUT" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File scripts\run_retune_scut.ps1

Write-Host "`=== PIPELINE COMPLETO ===" -ForegroundColor Green

Write-Host "n7. INICIALIZAR MODELO TREINADO" -ForegroundColor Cyan
$env:PYTHONPATH = (Get-Location).Path
python scripts/predict_bert_calibrated.py