r"""Text analysis: normalization, tokenization, stopwords, stemming, phrase parsing.

The same :class:`Tokenizer` instance (same configuration) MUST be used at index time and
at query time, otherwise postings and query terms will not line up. The configuration is
persisted in the index metadata and checked on load.

Pipeline for a piece of text::

    NFKD normalize -> drop combining marks (accents) -> lowercase
    -> regex [^\W_]+ (Unicode letters/digits) -> drop short tokens -> drop stopwords -> Porter stem

Positions are assigned *after* stopword removal, so a phrase query ``"cancer of the lung"``
becomes the adjacent pair (cancer, lung) both at index and query time.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

_TOKEN_RE = re.compile(r"[^\W_]+")
_PHRASE_RE = re.compile(r'"([^"]+)"')

# A compact English stopword list (union of the classic SMART short list and Lucene's
# EnglishAnalyzer set). Kept small on purpose: aggressive stopping hurts scientific text.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such that the their
    then there these they this to was will with i me my we our you your he him his she her
    its them what which who whom those am been being have has had having do does did doing
    would should could can may might must shall from up down out off over under again further
    once here when where why how all any both each few more most other some nor only own same
    so than too very s t just don now about above after before below between during through
    until while against because
    """.split()
)


@dataclass(frozen=True)
class TokenizerConfig:
    """Serializable tokenizer configuration (stored in index metadata)."""

    use_stemmer: bool = True
    use_stopwords: bool = True
    min_token_len: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict."""
        return asdict(self)


@lru_cache(maxsize=1)
def _porter() -> Any:
    """Lazily construct NLTK's Porter stemmer (pure Python, no corpora needed)."""
    from nltk.stem.porter import PorterStemmer

    return PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Porter-stem a single lowercase token (memoized; vocabularies are Zipfian)."""
    stemmed: str = _porter().stem(token)
    return stemmed


def normalize(text: str) -> str:
    """Unicode-normalize (NFKD), strip accents and lowercase."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


@dataclass(frozen=True)
class ParsedQuery:
    """A query split into bag-of-words terms and quoted phrases (both analyzed)."""

    raw: str
    terms: list[str]
    phrases: list[list[str]]


class Tokenizer:
    """Configurable analyzer used for both documents and queries."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def raw_tokens(self, text: str) -> list[str]:
        """Normalized surface tokens before stopping/stemming."""
        return _TOKEN_RE.findall(normalize(text))

    def analyze_token(self, token: str) -> str | None:
        """Apply length filter, stopwords and stemming to one surface token."""
        if len(token) < self.config.min_token_len:
            return None
        if self.config.use_stopwords and token in STOPWORDS:
            return None
        return stem(token) if self.config.use_stemmer else token

    def tokenize(self, text: str) -> list[str]:
        """Full analysis chain; returns index terms in order (positions = list indices)."""
        out: list[str] = []
        for tok in self.raw_tokens(text):
            term = self.analyze_token(tok)
            if term is not None:
                out.append(term)
        return out

    def parse_query(self, query: str) -> ParsedQuery:
        """Extract ``"quoted phrases"`` and analyze the whole query as a bag of words.

        Phrase words are also part of ``terms`` (the phrase is an *additional* boost, not a
        filter), which keeps recall identical to the unquoted query.
        """
        phrases: list[list[str]] = []
        for match in _PHRASE_RE.finditer(query):
            analyzed = self.tokenize(match.group(1))
            if len(analyzed) >= 2:
                phrases.append(analyzed)
        terms = self.tokenize(query.replace('"', " "))
        return ParsedQuery(raw=query, terms=terms, phrases=phrases)

    def highlight_spans(self, text: str, terms: set[str]) -> list[tuple[int, int]]:
        """Character spans in *text* whose analyzed form is in *terms* (for UI highlighting).

        Offsets refer to the original string. NFKD normalization can change string length
        for exotic characters, so matching is done per original word with a regex over the
        original text and each word analyzed independently.
        """
        spans: list[tuple[int, int]] = []
        for m in re.finditer(r"\w+", text):
            for tok in _TOKEN_RE.findall(normalize(m.group(0))):
                term = self.analyze_token(tok)
                if term is not None and term in terms:
                    spans.append((m.start(), m.end()))
                    break
        return spans
