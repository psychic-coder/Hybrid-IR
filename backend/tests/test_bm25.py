import math

from app.core.bm25 import BM25Scorer
from app.core.inverted_index import InvertedIndex, robertson_idf


def test_idf_formula() -> None:
    assert math.isclose(robertson_idf(10, 2), math.log(1 + (10 - 2 + 0.5) / 2.5))
    assert robertson_idf(10, 10) > 0  # +1 smoothing keeps IDF positive


def test_hand_computed_score(tiny_index: InvertedIndex) -> None:
    k1, b = 1.2, 0.75
    scorer = BM25Scorer(tiny_index, k1, b)
    cid = tiny_index.doc_chunks["d1"][0]
    dl = float(tiny_index.doc_lengths[cid])
    avgdl = tiny_index.avgdl
    n = tiny_index.n_chunks
    expected = 0.0
    for term in ["lung", "cancer"]:
        df = tiny_index.df(term)
        tf = tiny_index.tf(term, cid)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        expected += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert math.isclose(scorer.score_chunk({"lung": 1, "cancer": 1}, cid), expected, rel_tol=1e-12)


def test_ranking_prefers_matching_docs(tiny_index: InvertedIndex) -> None:
    scorer = BM25Scorer(tiny_index)
    res = scorer.search(tiny_index.tokenizer.parse_query("lung cancer"), 3)
    assert res.hits[0].doc_id in {"d1", "d6"}
    assert {h.doc_id for h in res.hits} <= {"d1", "d6", "d4", "d8"}


def test_unknown_terms_return_empty(tiny_index: InvertedIndex) -> None:
    scorer = BM25Scorer(tiny_index)
    assert scorer.search(tiny_index.tokenizer.parse_query("xyzzy plugh"), 5).hits == []


def test_b_zero_ignores_length(tiny_index: InvertedIndex) -> None:
    scorer = BM25Scorer(tiny_index, k1=1.2, b=0.0)
    s1 = scorer.term_score("cancer", 1, 0)
    s2 = scorer.term_score("cancer", 1, 3)
    assert math.isclose(s1, s2)


def test_upper_bound_dominates_all_postings(tiny_index: InvertedIndex) -> None:
    for k1, b in [(1.2, 0.75), (0.6, 0.25), (2.0, 0.9)]:
        scorer = BM25Scorer(tiny_index, k1, b)
        for term in tiny_index.terms:
            ub = tiny_index.upper_bound(term, k1, b)
            docs, tfs = tiny_index.postings(term)
            best = max(scorer.term_score(term, tf, d) for d, tf in zip(docs, tfs, strict=True))
            assert ub >= best - 1e-12
            assert math.isclose(ub, best, rel_tol=1e-9)  # the Pareto-front bound is exact


def test_phrase_boost(tiny_index: InvertedIndex) -> None:
    scorer = BM25Scorer(tiny_index, phrase_boost=1.0)
    plain = {
        h.doc_id: h.score for h in scorer.search(tiny_index.tokenizer.parse_query("lung cancer"), 8).hits
    }
    boosted = scorer.search(tiny_index.tokenizer.parse_query('"lung cancer"'), 8).hits
    by_id = {h.doc_id: h for h in boosted}
    assert by_id["d1"].phrase_matched
    assert by_id["d1"].score > plain["d1"]
    assert not by_id["d6"].phrase_matched
    assert math.isclose(by_id["d6"].score, plain["d6"])
