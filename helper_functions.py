import numpy as np

# helper function, rank list with highest rank for highest value
import pandas as pd
from pyltr.data.letor import read_dataset
import re

def test_rank_list():
    scores1 = np.array([0, 1, 2, 3])
    rank1 = rank_list(scores1)
    scores2 = np.array([1, 0, 3, 2])
    rank2 = rank_list(scores2)
    scores3 = np.array([2, 1, 0, 3])
    rank3 = rank_list(scores3)
    assert np.all(rank1 == np.array([4, 3, 2, 1]))
    assert np.all(rank2 == np.array([3, 4, 1, 2]))
    assert np.all(rank3 == np.array([2, 3, 4, 1]))


def rank_list(preds):
    """
    returns ndarray containing rank(i) for documents at position i
    """
    # print(preds)
    vector = np.array([pred['_score'] for pred in preds])
    temp = vector.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks


def rank_by_original_index(preds):
    """
    Returns a dict mapping each document's original _index to its rank.
    Assumes preds are already sorted by _score descending.
    """
    return np.array([p["_index"] for p in preds])



# def rank_list(vector):
#     """
#     returns ndarray containing rank(i) for documents at position i
#     """
#     temp = vector.argsort()[::-1]
#     ranks = np.empty_like(temp)
#     ranks[temp] = np.arange(1, len(vector) + 1)

#     return ranks


def rank_based_on_column_per_query(
    data_frame,
    name_column_to_rank,
    new_column_name,
    name_query_column="query_number",
    biggest_first=True,
):
    dfs = []
    for query in set(data_frame[name_query_column].values):
        data_frame_query = data_frame[data_frame[name_query_column] == query]
        data_frame_query = data_frame_query.sample(frac=1).reset_index(
            drop=True
        )  # Shuffle rows for breaking ties randomly
        data_frame_query[new_column_name] = (
            data_frame_query[[name_column_to_rank]]
            .apply(tuple, axis=1)
            .rank(method="first", ascending=False)
            .astype(int)
        )
        dfs.append(data_frame_query)
    return pd.concat(dfs, axis=0)


def test_rank_based_on_column_per_query():
    data_frame = pd.DataFrame(
        {"query_number": [1, 1, 1, 2, 2, 2], "attribution_value": [3, 2, 1, 1, 2, 3]}
    )
    data_frame = rank_based_on_column_per_query(
        data_frame,
        name_column_to_rank="attribution_value",
        new_column_name="ranked",
        biggest_first=True,
    )
    assert list(data_frame.ranked.values) == [1, 2, 3, 3, 2, 1]


def get_data(data_file):
    with open(data_file) as evalfile:
        exx, Ey, Eqids, _ = read_dataset(evalfile)
    return (exx, Ey, Eqids)


def get_queryids_as_list(Eqids):
    qids = []
    [qids.append(x) for x in Eqids if x not in qids]
    return qids


def get_documents_per_query(Eqids):
    # determine the amount of queries in the to be analysed set and construct a countlist
    df = pd.DataFrame(data=Eqids, columns=["Eqids"])
    qid_count_list = df.groupby("Eqids")["Eqids"].count()
    return qid_count_list


# def replace_words_in_sentences(documents, 
#                                words_to_replace, 
#                                unk_token="<unk>", 
#                                case_sensitive=False,
#                                allow_simple_plural=True):
#     if not words_to_replace:
#         return documents
    
#     # Sort by length to match longer words first (avoids subword matching confusion)
#     sorted_words = sorted(words_to_replace, key=len, reverse=True)
#     # Escape special characters
#     escaped_words = []
#     for w in sorted_words:
#         e = re.escape(w)
#         if allow_simple_plural:
#             e = f"{e}(s)?"
#         escaped_words.append(e)


#     # escaped_words = [re.escape(w) for w in sorted_words]
#     # Build regex pattern: \b matches word boundaries (including punctuation)
#     pattern_str = r'\b(' + '|'.join(escaped_words) + r')\b'
#     flags = 0 if case_sensitive else re.IGNORECASE
#     pattern = re.compile(pattern_str, flags)
#     # Replace, preserving surrounding punctuation
#     return [pattern.sub(unk_token, sent) for sent in documents]



import re
from typing import Callable, Iterable, List, Set

# Example: using NLTK's PorterStemmer as a stand-in.
# In production, prefer the exact stemmer Elastic uses to avoid discrepancies.
try:
    from nltk.stem import PorterStemmer
    _default_stemmer = PorterStemmer().stem
except Exception:
    _default_stemmer = lambda w: w  # no-op fallback if NLTK not available


# --- helpers 
_WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"          # handles hyphenated words & apostrophes: policy-maker, bank's
_PLACEHOLDER = r"__PHRASE_\d+__"
_PUNCT = r"[^\w\s]"                             # any single non-word, non-space (.,;:!?()[]{}"”’ etc.)
_NUMBER = r"\d+(?:[.,]\d+)*%?|\$\d+(?:[.,]\d+)*"
_TOKEN_PATTERN = re.compile(fr"{_PLACEHOLDER}|{_WORD}|{_PUNCT}|{_NUMBER}")

# def tokenize(text: str):
#     # Returns a list of tokens: words/placeholders/punctuation
#     return TOKEN_RX.findall(text)

# _TOKEN_PATTERN = re.compile(r"\w+|\s+|[^\w\s]")  # words | spaces | punctuation

def replace_words_in_sentences(
    documents: Iterable[str],
    words_to_replace: Set[str],
    unk_token: str = "<unk>",
    case_sensitive: bool = False,
    stemmer: Callable[[str], str] = _default_stemmer,
) -> List[str]:
    """
    Replace tokens in `documents` whose STEM (via `stemmer`) is in `words_to_replace`.
    Preserves original punctuation and spacing.
    """
    if not words_to_replace:
        return list(documents)

    # Normalize the replacement set to stems (lowercased for stability).
    # If caller already provided stems, this is still fine.
    replace_stems = {stemmer(w if case_sensitive else w.lower()) for w in words_to_replace}

    out = []
    for sent in documents:
        pieces = []
        for m in _TOKEN_PATTERN.finditer(sent):
            tok = m.group(0)

            # Only stem word tokens; keep spaces/punct as-is
            if tok.isalnum() or re.match(r"^\w+$", tok):
                probe = tok if case_sensitive else tok.lower()
                stem = stemmer(probe)
                if stem in replace_stems:
                    pieces.append(unk_token)
                else:
                    pieces.append(tok)
            else:
                pieces.append(tok)
        # out.append(" ".join(pieces))
        out.append(detokenize(pieces))
    return out

def detokenize(tokens):
    out = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        # If current token is punctuation, attach to previous without space
        if re.fullmatch(_PUNCT, t):
            out[-1] += t
        else:
            # otherwise add a space then the token
            out.append(" " + t)
    return "".join(out)




if __name__ == "__main__":
    test_rank_list()
    test_rank_based_on_column_per_query()


