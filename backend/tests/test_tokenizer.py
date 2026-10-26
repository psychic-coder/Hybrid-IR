from app.core.tokenizer import Tokenizer, TokenizerConfig, normalize


def test_normalize_strips_accents_and_lowercases() -> None:
    assert normalize("Café NAÏVE Ångström") == "cafe naive angstrom"


def test_stopwords_and_stemming() -> None:
    tok = Tokenizer()
    assert tok.tokenize("The running of the Dogs") == ["run", "dog"]


def test_config_switches() -> None:
    raw = Tokenizer(TokenizerConfig(use_stemmer=False, use_stopwords=False))
    assert raw.tokenize("The running dogs") == ["the", "running", "dogs"]
    nostop = Tokenizer(TokenizerConfig(use_stemmer=True, use_stopwords=False))
    assert nostop.tokenize("the running") == ["the", "run"]


def test_min_token_len() -> None:
    tok = Tokenizer(TokenizerConfig(use_stemmer=False, use_stopwords=False, min_token_len=3))
    assert tok.tokenize("a bb ccc dddd") == ["ccc", "dddd"]


def test_regex_splits_punctuation_and_keeps_digits() -> None:
    tok = Tokenizer(TokenizerConfig(use_stemmer=False, use_stopwords=False))
    assert tok.tokenize("COVID-19, IL-6/TNF-α") == ["covid", "19", "il", "6", "tnf", "α"]


def test_parse_query_phrases() -> None:
    tok = Tokenizer()
    pq = tok.parse_query('effects of "lung cancer" screening')
    assert pq.phrases == [["lung", "cancer"]]
    assert pq.terms == ["effect", "lung", "cancer", "screen"]


def test_single_word_phrase_ignored() -> None:
    pq = Tokenizer().parse_query('"cancer" risk')
    assert pq.phrases == []
    assert pq.terms == ["cancer", "risk"]


def test_highlight_spans_match_stems() -> None:
    tok = Tokenizer()
    text = "Running dogs ran."
    spans = tok.highlight_spans(text, {"run"})
    assert [text[a:b] for a, b in spans] == ["Running"]


def test_deterministic() -> None:
    tok = Tokenizer()
    s = "Vitamin D deficiency and bone fractures in elderly populations"
    assert tok.tokenize(s) == tok.tokenize(s)
