# ================== run_retune_scut.ps1 (grid de variações) ==================
$ErrorActionPreference = "Stop"

# --- Caminhos base robustos ---
# Pasta onde este .ps1 está (scripts/)
$scriptDir = $PSScriptRoot
# Raiz do projeto (pasta acima de scripts/)
$projRoot  = Split-Path $scriptDir -Parent

# --- Python (usa o da venv) ---
$PY = Join-Path $projRoot "venv\Scripts\python.exe"

# --- Script de retune (arquivo .py dentro de scripts/) ---
$RETUNE_SCRIPT = Join-Path $scriptDir "retune_adaptative_scut.py"

# --- Diretório base de saída (outputs/retunes/<TAG>/...) ---
$OUTBASE = Join-Path $projRoot "outputs\retunes"

# --- Modelos ---
$MODELS = @(
  @{ Tag = "bert_base_cb_loss"; Kind = "bert"},
  @{ Tag = "classic_tfidf";     Kind = "classic"}
)

# --- Grades ---
$CALIBRATIONS_BERT    = @("platt", "temperature")
$CALIBRATIONS_CLASSIC = @("isotonic")
$BETAS   = @(2.0)
$PFLOORS = @("None", 0.30)
$TOPKS   = @(0, 1)
$CVS     = @(5)
$LAMBDAS = @(0.85)       # puxa t levemente p/ 0.5
$SMOOTH_ALPHAS = @(1.0)  # smoothing padrão
$PREV_CAPS     = @("None", 3.0) # None=off, 3.0=não prever 3x mais que a prevalência

function PathJoin([string] $a, [string] $b) {
    return [System.IO.Path]::Combine($a, $b)
}

foreach ($M in $MODELS) {
  $TAG  = $M.Tag
  $KIND = $M.Kind
  $CALIBS = if ($KIND -eq "bert") { $CALIBRATIONS_BERT } else { $CALIBRATIONS_CLASSIC }

  foreach ($calib in $CALIBS) {
  foreach ($beta in $BETAS) {
  foreach ($pf in $PFLOORS) {
  foreach ($topk in $TOPKS) {
  foreach ($cv in $CVS) {
  foreach ($lam in $LAMBDAS) {
  foreach ($alpha in $SMOOTH_ALPHAS) {
  foreach ($pcap in $PREV_CAPS) {

    # ---- Construir nome da pasta de saída (com slugs) ----
    $pfSlug    = if ($pf -eq "None") { "pfNone" } else { "pf$("{0:N2}" -f [double]$pf)".Replace(',','.') }
    $lamSlug   = if ($lam -eq "None") { "lamNone" } else { "lam$("{0:N2}" -f [double]$lam)".Replace(',','.') }
    $alphaSlug = "alpha$("{0:N1}" -f [double]$alpha)".Replace(',','.')
    $pcapSlug  = if ($pcap -eq "None") { "pcapNone" } else { "pcap$("{0:N1}" -f [double]$pcap)".Replace(',','.') }
    
    $name  = "calib=$calib.beta=$beta.$pfSlug.topk=$topk.cv=$cv.$lamSlug.$alphaSlug.$pcapSlug"
    $tagOutBase = PathJoin $OUTBASE $TAG
    $OUTDIR = PathJoin $tagOutBase $name
    New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

    # ---- Montar argumentos do Python ----
    $argsList = @(
        $RETUNE_SCRIPT,
        $TAG,
        "--kind",              $KIND,
        "--export_dir",        $OUTDIR,
        "--calibration",       $calib,
        "--beta",              $beta,
        "--topk_hybrid",       $topk,
        "--cv_thresholds",     $cv
    )

    if ($pf -ne "None")   { $argsList += @("--precision_floor",       [string]$pf) }
    if ($lam -ne "None")  { $argsList += @("--regularize_lambda",     [string]$lam) }
    if ($alpha -ne 0)     { $argsList += @("--precision_smooth_alpha",[string]$alpha) }
    if ($pcap -ne "None") { $argsList += @("--prevalence_cap",        [string]$pcap) }

    # ---- Execução ----
    Write-Host ""
    Write-Host ">>> $TAG | $name" -ForegroundColor Cyan
    & $PY @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Falha no retune: $TAG / $name"
    }
    Write-Host ("OK -> {0}" -f $OUTDIR) -ForegroundColor Green

  }}}}}}}} # Fecha todos os loops
}

Write-Host "`nTodos os retunes finalizados." -ForegroundColor Green