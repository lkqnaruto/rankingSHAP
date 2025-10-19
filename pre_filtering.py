from typing import Iterable, Tuple, Set, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"          # handles hyphenated words & apostrophes: policy-maker, bank's
_PLACEHOLDER = r"__PHRASE_\d+__"
_PUNCT = r"[^\w\s]"                             # any single non-word, non-space (.,;:!?()[]{}"”’ etc.)
_NUMBER = r"\d+(?:[.,]\d+)*%?|\$\d+(?:[.,]\d+)*"
# _TOKEN_PATTERN = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_PUNCT}|{_NUMBER}")
_TOKEN_PATTERN = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_NUMBER}")


def vocab_prune_by_global_tfidf(docs, keep_top_k=20000, ngram_range=(1,1)):
    tfidf = TfidfVectorizer(ngram_range=ngram_range, 
                            stop_words='english', 
                            norm=None, 
                            tokenizer=my_tokenizer)
    X = tfidf.fit_transform(docs)            # N x V (sparse)
    vocab = np.array(tfidf.get_feature_names_out())
    # print(len(vocab))
    # Mean TF-IDF across docs as an importance proxy
    importance = np.asarray(X.mean(axis=0)).ravel()
    keep_idx = np.argsort(-importance)[:keep_top_k]
    return set(vocab[keep_idx])


def my_tokenizer(text):
    # Example: split on alphabetic words and hyphenated forms
    return re.findall(_TOKEN_PATTERN, text.lower())


def bm25_idf(df: np.ndarray, N: int, add_one: bool = False) -> np.ndarray:
    """
    Robertson–Spärck Jones IDF (used by BM25):
      idf(t) = log( (N - df_t + 0.5) / (df_t + 0.5) )
    If add_one=True, add +1 to keep non-negativity (some implementations do this).
    """
    idf = np.log((N - df + 0.5) / (df + 0.5))
    if add_one:
        idf = idf + 1.0
    return idf

def prune_vocab_by_bm25_idf(
    docs: Iterable[str],
    *,
    ngram_range: Tuple[int, int] = (1, 1),
    stop_words: Optional[str] = None,   # e.g., 'english'
    lowercase: bool = True,
    min_df: int | float = 1,            # you can still apply DF gates
    max_df: int | float = 1.0,
    add_one: bool = False,              # BM25-IDF + 1 or not
    floor_nonnegative: bool = True,     # drop tokens with negative BM25-IDF
    keep_top_k: Optional[int] = None,   # keep exactly top-k by BM25-IDF
    idf_percentile: Optional[float] = 90.0,  # or keep tokens >= this percentile
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    Returns:
      {
        'kept_vocab': Set[str],
        'bm25_idf_table': List[(token, bm25_idf_value)]  # sorted desc
      }
    """
    docs = list(docs)
    N = len(docs)

    # 1) CountVectorizer to get DF with desired tokenization
    cv = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        min_df=min_df,
        max_df=max_df,
        # tokenizer=my_tokenizer
    )
    X = cv.fit_transform(docs)                # [N x V] sparse
    vocab = np.array(cv.get_feature_names_out())
    # print(vocab)
    df = np.asarray((X > 0).sum(axis=0)).ravel()  # doc frequency per term

    # 2) BM25-IDF
    idf_vals = bm25_idf(df, N, add_one=False)

    # 3) Base keep mask
    keep_mask = np.ones_like(idf_vals, dtype=bool)

    if floor_nonnegative and not add_one:
        # Drop very common terms whose BM25-IDF < 0
        keep_mask &= (idf_vals >= 0)

    # 4) Threshold by either top-k or percentile (if provided)
    #    (Apply after negative filter so ranking is among plausible terms.)
    candidate_idx = np.where(keep_mask)[0]
    if candidate_idx.size == 0:
        return {"kept_vocab": set(), "bm25_idf_table": []}

    cand_idf = idf_vals[candidate_idx]
    if keep_top_k is not None:
        print(keep_top_k)
        order = candidate_idx[np.argsort(-cand_idf)[:keep_top_k]]
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        keep_mask[order] = True
    elif idf_percentile is not None:
        thr = np.percentile(cand_idf, idf_percentile)
        keep_mask &= (idf_vals >= thr)

    kept_tokens = set(vocab[keep_mask])

    # 5) Apply whitelist / blacklist
    if whitelist:
        kept_tokens |= set(whitelist)
    if blacklist:
        kept_tokens -= set(blacklist)

    # 6) Produce a sorted table (useful for inspection/governance)
    order_all = np.argsort(-idf_vals)
    bm25_table = [(vocab[i], float(idf_vals[i])) for i in order_all]

    return {"kept_vocab": kept_tokens, "bm25_idf_table": bm25_table}




from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk


def normalize_features(features: Set[str]) -> Set[str]:

    try:
        try:
            wordnet.ensure_loaded()
        except Exception:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        _lemmatizer = WordNetLemmatizer()
        def _best_lemma(w: str) -> str:
            forms = [
                _lemmatizer.lemmatize(w, pos='n'),
                _lemmatizer.lemmatize(w, pos='v'),
                _lemmatizer.lemmatize(w, pos='a'),
                _lemmatizer.lemmatize(w, pos='r'),
            ]
            return min(forms, key=len)
        features = {_best_lemma(t) for t in features}
        return features
    except Exception:
        def _rule_lemma(w: str) -> str:
            if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
            if w.endswith("ves") and len(w) > 4: return w[:-3] + "f"
            if w.endswith("es") and len(w) > 3:  return w[:-2]
            if w.endswith("s") and len(w) > 3:   return w[:-1]
            return w
        features = {_rule_lemma(t) for t in features}
        return features
    # print("after stemming/lemmatization:", features)







