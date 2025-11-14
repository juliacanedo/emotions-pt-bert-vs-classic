''' Esse script gera figuras de análise de interpretabilidade a partir dos arquivos CSV
    produzidos pelo script de interpretabilidade'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np
from pathlib import Path

# --- Configurações ---
BASE_PATH = Path("../outputs/interpretability")
CLASS_FILES_PATH = BASE_PATH / "ig"
CONF_FILES_PATH = BASE_PATH / "conf_pairs"
OUTPUT_DIR = Path("../outputs/interpretability_analysis")

# --- LISTA DE CLASSES ---

CLASSES_TO_PLOT = [
    "alegria", "empolgacao", "raiva", 
    "irritacao", "neutro", "aprovacao"
]

# -------------------------------------------

MAX_CONF_PAIRS_TO_PLOT = 5 # Limite máximo de pares de confusão
MIN_FREQ = 1 
TOP_K_ANCHORS = 15 
MAX_ANCHORS = 40 

plt.style.use('ggplot')

# --- Funções Auxiliares ---

def _safe_read(file_path: Path):
    if not file_path or not file_path.exists():
        return None
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return None
        return df if not df.empty else None
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return None

def read_csv_top5(file_path: Path):
    df = _safe_read(file_path)
    if df is None:
        return pd.DataFrame(columns=['token', 'value'])
    df = df.iloc[:, [0, 1]].head(5)
    df.columns = ['token', 'value'] 
    return df

def find_files(class_path: Path, conf_path: Path):
    class_files = {} 
    conf_pair_files = {}
    
    class_top_pattern = re.compile(r'^(.*?)_(top_pos|top_neg|top_import)\.csv$')
    conf_pattern = re.compile(r'^(.*?)_to_(.*?)_(top_pos|top_neg|top_import)\.csv$')

    # 1. Busca por arquivos de CLASSE
    if not class_path.exists():
        print(f"Aviso: Diretório de classes não encontrado: {class_path}")
    else:
        for file_path in class_path.glob('*.csv'):
            top_match = class_top_pattern.match(file_path.name)
            if top_match:
                class_name, file_type = top_match.groups()
                if class_name not in class_files:
                    class_files[class_name] = {}
                class_files[class_name][file_type] = file_path

    # 2. Busca por arquivos de CONFUSÃO
    if not conf_path.exists():
        print(f"Aviso: Diretório de pares de confusão não encontrado: {conf_path}")
    else:
        for file_path in conf_path.glob('*.csv'):
            conf_match = conf_pattern.match(file_path.name)
            if conf_match:
                a, b, file_type = conf_match.groups()
                pair_name = f"{a}_to_{b}"
                if pair_name not in conf_pair_files:
                    conf_pair_files[pair_name] = {}
                conf_pair_files[pair_name][file_type] = file_path
                
    return class_files, conf_pair_files

def plot_hbar(ax, data, color, title_override=None):
    if data.empty or data is None:
        ax.text(0.5, 0.5, "Dados não encontrados", ha='center', va='center', 
                transform=ax.transAxes, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
        if title_override: ax.set_title(title_override, fontsize=10)
        return
    tokens = data['token'].iloc[::-1]; values = data['value'].iloc[::-1]
    ax.barh(tokens, values, color=color, alpha=0.8)
    if title_override: ax.set_title(title_override, fontsize=12)
    ax.set_xlabel("Atribuição Média (IG)"); ax.grid(axis='x', linestyle='--', alpha=0.7)

# --- Funções de Geração (Requisitos) ---

def generate_heatmap(class_files_dict, output_path):
    print("Gerando heatmap (classes vs. tokens)...")
    classes = sorted(class_files_dict.keys())
    if not classes: return
    anchors = set()
    for class_name, files in class_files_dict.items():
        df_imp = _safe_read(files.get('top_import'))
        if df_imp is None or "token" not in df_imp.columns: continue
        sort_cols = [df_imp.columns[1]];
        if "freq" in df_imp.columns:
            df_imp = df_imp[df_imp["freq"] >= MIN_FREQ]; sort_cols.append("freq")
        df_imp = df_imp.sort_values(sort_cols, ascending=[False, False])
        anchors.update(df_imp.head(TOP_K_ANCHORS)["token"].tolist())
    anchors = sorted([a for a in anchors if a])
    if len(anchors) > MAX_ANCHORS: anchors = anchors[:MAX_ANCHORS]
    if not anchors: return
    M = np.zeros((len(classes), len(anchors)), dtype=float)
    class_token_maps = {}
    for class_name, files in class_files_dict.items():
        token_map = {}; df_pos = _safe_read(files.get('top_pos')); df_neg = _safe_read(files.get('top_neg'))
        if df_pos is not None and "token" in df_pos.columns:
            token_map.update(dict(zip(df_pos.iloc[:, 0], df_pos.iloc[:, 1])))
        if df_neg is not None and "token" in df_neg.columns:
            token_map.update(dict(zip(df_neg.iloc[:, 0], df_neg.iloc[:, 1])))
        class_token_maps[class_name] = token_map
    for i, class_name in enumerate(classes):
        mp = class_token_maps.get(class_name, {});
        for j, token in enumerate(anchors): M[i, j] = float(mp.get(token, 0.0))
    fig_width = max(10, 0.25 * len(anchors)); fig_height = max(4, 0.3 * len(classes))
    plt.figure(figsize=(fig_width, fig_height))
    vmax = np.abs(M).max();
    if vmax == 0: vmax = 1.0
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="vlag", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="Atribuição Média (IG)")
    plt.xticks(range(len(anchors)), anchors, rotation=90); plt.yticks(range(len(classes)), classes)
    plt.title(f"Atribuição Média por Token — Classes vs. Tokens Âncora")
    plt.gca().xaxis.tick_top(); plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False); plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap (classes vs. tokens) salvo em {output_path}")


def generate_class_subplots(class_files_dict, output_path): # <-- Assinatura MODIFICADA
    """
    Gera subplots para Top 5 (Import, Pos, Neg) 
    APENAS para as classes em CLASSES_TO_PLOT (definido no topo).
    """
    print("Gerando subplots das classes (Otimizado por Polaridade)...")
    
    classes_to_plot_final = [
        cls for cls in CLASSES_TO_PLOT if cls in class_files_dict
    ]
    
    num_classes = len(classes_to_plot_final)
    if num_classes == 0:
        print(f"ERRO: Nenhuma das classes em {CLASSES_TO_PLOT} foi encontrada nos arquivos.")
        print(f"Certifique-se que os nomes estão corretos e os arquivos CSV existem em {CLASS_FILES_PATH}")
        print(f"Classes que eu encontrei: {list(class_files_dict.keys())}")
        return

    print(f"Encontrados {num_classes} de {len(CLASSES_TO_PLOT)} classes solicitadas. Plotando...")

    fig, axes = plt.subplots(nrows=num_classes, ncols=3, 
                             figsize=(12, num_classes * 2.5), squeeze=False)
    
    axes[0, 0].set_title("Top 5 Importantes (|atrib|)", fontsize=14)
    axes[0, 1].set_title("Top 5 Positivos (Aumentam Classe)", fontsize=14)
    axes[0, 2].set_title("Top 5 Negativos (Diminuem Classe)", fontsize=14)

    for i, class_name in enumerate(classes_to_plot_final):
        files = class_files_dict[class_name]
        df_import = read_csv_top5(files.get('top_import'))
        df_pos = read_csv_top5(files.get('top_pos'))
        df_neg = read_csv_top5(files.get('top_neg'))
        
        plot_hbar(axes[i, 0], df_import, "gray")
        plot_hbar(axes[i, 1], df_pos, "seagreen")
        plot_hbar(axes[i, 2], df_neg, "crimson")
        
        fig.text(0.01, (num_classes - 1 - i) / num_classes + (0.5 / num_classes), 
                 class_name, va='center', ha='center', rotation='vertical', 
                 fontsize=14, weight='bold')

    fig.suptitle("Análise de Explicabilidade (IG) - Top 5 Tokens por Classe", 
                 fontsize=18)
    plt.tight_layout(rect=[0.03, 0, 1, 0.98]) 
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Subplots de classes salvos em {output_path}")


def generate_conf_pair_subplots(conf_pair_files_dict, output_path):
    print("Gerando subplots dos pares de confusão (Otimizado - Vertical)...")
    
    pairs_list = sorted(conf_pair_files_dict.keys())
    if len(pairs_list) > MAX_CONF_PAIRS_TO_PLOT:
        pairs_list = pairs_list[:MAX_CONF_PAIRS_TO_PLOT]
    
    num_pairs = len(pairs_list)
    if num_pairs == 0: 
        print("Nenhum par de confusão encontrado para plotar.")
        return
        
    fig, axes = plt.subplots(nrows=num_pairs, ncols=1, 
                             figsize=(8, num_pairs * 2.5), squeeze=False) # Vertical
    
    for i, pair_name in enumerate(pairs_list):
        files = conf_pair_files_dict[pair_name]
        df_pos = read_csv_top5(files.get('top_pos'))
        title = f"Confusão: {pair_name.replace('_to_', ' -> ')}\n(Tokens que aumentam a classe B)"
        
        # --- MUDANÇA AQUI: axes[i, 0] ---
        plot_hbar(axes[i, 0], df_pos, "darkorange", title_override=title)
    
    fig.suptitle("Análise de Confusão (IG) - Top 5 Tokens Causadores", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Subplots de confusão (Vertical) salvos em {output_path}")

# --- Função Principal ---

def main():
    """Roda todo o pipeline de análise."""
    print("Iniciando pipeline de análise de explicabilidade...")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # --- Etapa 1: Lendo arquivos de Interpretabilidade ---
    class_files, conf_pair_files = find_files(CLASS_FILES_PATH, CONF_FILES_PATH)
    
    print(f"\n--- DEBUG START ---")
    print(f"Debug: Verificando classes em: {CLASS_FILES_PATH.resolve()}")
    print(f"Debug: Verificando confusão em: {CONF_FILES_PATH.resolve()}")
    print(f"Debug: Classes (chaves) encontradas: {list(class_files.keys())}")
    print(f"Debug: Pares (chaves) encontrados: {list(conf_pair_files.keys())}")
    print(f"--- DEBUG END ---\n")
    
    if not class_files and not conf_pair_files:
        print(f"Nenhum arquivo .csv encontrado em {CLASS_FILES_PATH} ou {CONF_FILES_PATH}")
        return
    print("--- Leitura de arquivos concluída ---")

    print("\n--- Etapa 2: Gerando Gráficos ---")
    if class_files:
        generate_heatmap(class_files, OUTPUT_DIR / "01_heatmap_classes_tokens.png")
        
        generate_class_subplots(
            class_files, 
            OUTPUT_DIR / "02_subplots_classes.png"
        )
    else:
        print("Nenhum arquivo de CLASSE encontrado, pulando heatmap e subplots de classe.")
    
    if conf_pair_files:
        generate_conf_pair_subplots(conf_pair_files, OUTPUT_DIR / "03_subplots_confusao.png")
    else:
        print("Nenhum arquivo de PAR DE CONFUSÃO encontrado, pulando subplots de confusão.")
    
    print("\n--- Pipeline de análise concluído! ---")
    print(f"Resultados salvos em: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()