from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Funções para formato ONE-HOT (Labels em colunas separadas 0/1)
# ---------------------------------------------------------------------------

def infer_labels(df: pd.DataFrame, text_col: str):
    """
    Infere colunas de label (formato one-hot) a partir de um DataFrame.
    Procura por colunas que contenham apenas 0s e 1s.
    """
    cols = [c for c in df.columns if c != text_col]
    label_cols = []
    for c in cols:
        try:
            arr = df[c].dropna().astype(int).unique()
            if set(arr).issubset({0,1}):
                label_cols.append(c)
        except Exception:
            pass
    return sorted(label_cols)

def per_class_counts(df: pd.DataFrame, label_cols: list[str]):
    """
    Calcula contagens de positivos para formato one-hot.
    """
    Y = df[label_cols].values.astype(int)
    c = Y.sum(axis=0).astype(int)
    freq = (c / len(df) * 100.0)
    out = pd.DataFrame({"label": label_cols, "positives": c, "freq_pct": freq})
    out = out.sort_values("positives", ascending=False).reset_index(drop=True)
    return out

def cardinality_density(df: pd.DataFrame, label_cols: list[str]):
    """
    Calcula cardinalidade para formato one-hot.
    """
    Y = df[label_cols].values.astype(int)
    per_row = Y.sum(axis=1)
    return {
        "mean_cardinality": float(per_row.mean()),
        "median_cardinality": float(np.median(per_row)),
        "min_cardinality": int(per_row.min()),
        "max_cardinality": int(per_row.max())
    }

# ---------------------------------------------------------------------------
# Funções para formato STRING (Labels em uma única coluna, ex: "a;b;c")
# ---------------------------------------------------------------------------

def per_class_counts_from_string(df: pd.DataFrame, label_col: str):
    """
    Calcula contagens de positivos para formato de string única.
    """
    # Assegura que NaNs sejam tratados como strings vazias
    df_labels = df[label_col].fillna('').str.get_dummies(sep=';')
    
    # Remove colunas vazias que podem ser criadas por ';;' ou NaNs
    if '' in df_labels.columns:
        df_labels = df_labels.drop(columns=[''])
        
    c = df_labels.sum(axis=0).astype(int)
    freq = (c / len(df) * 100.0)
    
    out = pd.DataFrame({"label": c.index, "positives": c.values, "freq_pct": freq.values})
    out = out.sort_values("positives", ascending=False).reset_index(drop=True)
    return out

def cardinality_density_from_string(df: pd.DataFrame, label_col: str):
    """
    Calcula cardinalidade para formato de string única.
    """
    # Assegura que NaNs sejam tratados como strings vazias
    df_labels = df[label_col].fillna('').str.get_dummies(sep=';')
    
    # Remove colunas vazias
    if '' in df_labels.columns:
        df_labels = df_labels.drop(columns=[''])
        
    per_row = df_labels.sum(axis=1)
    
    return {
        "mean_cardinality": float(per_row.mean()),
        "median_cardinality": float(np.median(per_row)),
        "min_cardinality": int(per_row.min()),
        "max_cardinality": int(per_row.max())
    }

def get_all_labels_from_string(df: pd.DataFrame, label_col: str):
    """
    Extrai a lista de todos os labels únicos do formato string.
    """
    df_labels = df[label_col].fillna('').str.get_dummies(sep=';')
    if '' in df_labels.columns:
        df_labels = df_labels.drop(columns=[''])
    return sorted(df_labels.columns.tolist())

# ---------------------------------------------------------------------------
# Função de plotagem (Comum aos dois formatos)
# ---------------------------------------------------------------------------

def plot_bar_counts(counts_df: pd.DataFrame, title: str, out_png: Path):
    """
    Plota um gráfico de barras com as contagens.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(counts_df["label"], counts_df["positives"])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------------------------------------------------------------------------
# Comando principal da EDA
# ---------------------------------------------------------------------------

def eda_command(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for split, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if path is None: 
            continue
        try:
            dfs[split] = pd.read_csv(path)
        except Exception as e:
            print(f"Erro ao ler o arquivo {path}: {e}")
            continue
    
    if not dfs:
        raise ValueError("Nenhum arquivo de dados (train/val/test) foi carregado com sucesso.")

    # ---- Infere col de texto ----
    text_col = args.text_col
    if text_col is None:
        for candidate in ("texto","text"):
            if candidate in dfs.get("train", pd.DataFrame()).columns:
                text_col = candidate; break
        if text_col is None:
            text_col = list(next(iter(dfs.values())).columns)[0]
    
    print(f"[EDA] Usando coluna de texto: {text_col}")

    # ---- Detecta formato de label ----
    base_df = None
    for key in ("train", "val", "test"):
        if key in dfs and isinstance(dfs[key], pd.DataFrame) and not dfs[key].empty:
            base_df = dfs[key]
            break
    
    if base_df is None:
        raise ValueError("Nenhum split válido foi carregado (train/val/test) para inferir o formato.")

    is_string_format = False
    label_col_name = args.label_col
    all_label_cols = []

    if label_col_name in base_df.columns:
        print(f"[EDA] Formato detectado: Coluna de string única '{label_col_name}'")
        is_string_format = True
        all_label_cols = get_all_labels_from_string(base_df, label_col_name)
    else:
        print(f"[EDA] Formato detectado: One-hot (colunas binárias). (Coluna '{label_col_name}' não encontrada.)")
        all_label_cols = infer_labels(base_df, text_col)
        
    if not all_label_cols:
        raise ValueError("Não foi possível inferir/encontrar nenhuma coluna de label.")
    
    print(f"[EDA] Encontrados {len(all_label_cols)} labels.")

    # ---- Geração de Relatórios ----
    global_report = {"text_col": text_col, "labels": all_label_cols, "splits": {}}
    
    for split, df in dfs.items():
        if is_string_format:
            # Lógica para formato de STRING ÚNICA
            if label_col_name not in df.columns:
                print(f"Aviso: Pulando split '{split}' porque a coluna '{label_col_name}' não foi encontrada.")
                continue
            cnt = per_class_counts_from_string(df, label_col_name)
            card = cardinality_density_from_string(df, label_col_name)
        else:
            # Lógica para formato ONE-HOT
            cnt = per_class_counts(df, all_label_cols)
            card = cardinality_density(df, all_label_cols)
            
        global_report["splits"][split] = {"n": int(len(df)), "cardinality": card}
        cnt.to_csv(out_dir / f"{split}_class_counts.csv", index=False)
        plot_bar_counts(cnt, f"Total de Amostras no conjunto {split.upper()} por classe", out_dir / f"{split}_class_counts.png")

    # ---- Combinando Splits (se todos existirem) ----
    if len(dfs) == 3:
        df_all = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
        
        if is_string_format:
            cnt_all = per_class_counts_from_string(df_all, label_col_name)
            card_all = cardinality_density_from_string(df_all, label_col_name)
        else:
            cnt_all = per_class_counts(df_all, all_label_cols)
            card_all = cardinality_density(df_all, all_label_cols)

        global_report["splits"]["all"] = {"n": int(len(df_all)), "cardinality": card_all}
        cnt_all.to_csv(out_dir / "all_class_counts.csv", index=False)
        plot_bar_counts(cnt_all, "Total de Amostras Positivas por Classe", out_dir / "all_class_counts.png")

    (out_dir / "eda_report.json").write_text(json.dumps(global_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EDA] Relatórios salvos em: {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eda = sub.add_parser("eda", help="Basic EDA for per-class distributions")
    ap_eda.add_argument("--train", type=str, required=False)
    ap_eda.add_argument("--val",   type=str, required=False)
    ap_eda.add_argument("--test",  type=str, required=False)
    ap_eda.add_argument("--text_col", type=str, required=False, default=None)
    ap_eda.add_argument("--label_col", type=str, required=False, default="labels", help="Nome da coluna que contém labels como string única (ex: 'label1;label2')")
    ap_eda.add_argument("--out_dir", type=str, required=True)
    ap_eda.set_defaults(func=eda_command)
    
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()