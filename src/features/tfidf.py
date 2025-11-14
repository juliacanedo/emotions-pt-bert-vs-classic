from __future__ import annotations
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp

@dataclass
class TFIDFConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    word_min_df: int = 2
    word_max_df: float = 0.9
    word_max_features: int | None = 200_000
    char_ngram_range: tuple[int, int] = (3, 5)
    char_min_df: int = 2
    char_max_df: float = 0.95
    char_analyzer: str = "char_wb"
    char_max_features: int | None = 200_000

class TFIDFFeaturizer:
    def __init__(self, cfg: TFIDFConfig):
        self.cfg = cfg
        self.vec_word = TfidfVectorizer(
            ngram_range=cfg.word_ngram_range,
            min_df=cfg.word_min_df,
            max_df=cfg.word_max_df,
            max_features=cfg.word_max_features,
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )
        self.vec_char = TfidfVectorizer(
            ngram_range=cfg.char_ngram_range,
            min_df=cfg.char_min_df,
            max_df=cfg.char_max_df,
            max_features=cfg.char_max_features,
            analyzer=cfg.char_analyzer,
            strip_accents="unicode",
            lowercase=True,
        )
    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        Xw = self.vec_word.fit_transform(texts)
        Xc = self.vec_char.fit_transform(texts)
        return sp.hstack([Xw, Xc]).tocsr()
    def transform(self, texts: list[str]) -> sp.csr_matrix:
        Xw = self.vec_word.transform(texts)
        Xc = self.vec_char.transform(texts)
        return sp.hstack([Xw, Xc]).tocsr()
    def save(self, out_path_prefix: str) -> None:
        from joblib import dump
        dump(self.vec_word, f"{out_path_prefix}_word.joblib")
        dump(self.vec_char, f"{out_path_prefix}_char.joblib")
