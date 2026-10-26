# Glossary

| Term | Meaning in this project |
|---|---|
| **Ablation** | An experiment that changes one parameter while holding the rest fixed, to attribute an effect to it. |
| **ANN (approximate nearest neighbor)** | Vector search that trades a small loss of recall for sub-linear query time, e.g. HNSW. |
| **avgdl** | Average chunk length in index terms; the reference length in BM25 length normalization. |
| **BEIR** | A benchmark of heterogeneous retrieval datasets with a common format (Thakur et al., 2021). |
| **Bi-encoder** | A model that embeds query and document independently; similarity is a dot product, so documents can be pre-embedded. |
| **BM25** | Okapi Best Matching 25: a probabilistic term-weighting ranking function with saturation (k1) and length normalization (b). |
| **Bootstrap (paired)** | Resampling queries with replacement to estimate the sampling distribution of a mean difference between two systems on the same queries. |
| **Candidate depth** | How many documents each retriever passes to fusion (`CANDIDATE_DEPTH`). |
| **Chunk** | A window of `CHUNK_SIZE` words of a document (plus its title); the unit that is indexed and embedded. |
| **Coordinated omission** | A load-testing flaw where the client sends less load while the server is slow, hiding tail latency. |
| **Cosine similarity** | `u·v / (‖u‖‖v‖)`; equals the dot product for L2-normalized vectors. |
| **Cross-encoder** | A transformer that reads query and passage together and outputs one relevance score; accurate but one forward pass per pair. |
| **Cold / warm cache** | Cold: caches bypassed or empty; warm: the needed entries are already cached. |
| **DAAT / TAAT** | Document-at-a-time vs term-at-a-time query processing over postings lists. |
| **DCG / nDCG** | (Normalized) Discounted Cumulative Gain: graded relevance discounted by log of rank, normalized by the ideal ordering. |
| **Delta encoding** | Storing differences between consecutive sorted integers instead of the integers, making them small. |
| **df (document frequency)** | Number of chunks containing a term. |
| **ef / efSearch** | Size of the dynamic candidate list during HNSW search; larger = higher recall, slower. |
| **efConstruction** | Candidate list size while inserting into HNSW; larger = better graph, slower build. |
| **Embedding** | A dense vector representation of text produced by a neural encoder. |
| **Fusion** | Combining ranked lists from several retrievers into one (RRF or weighted). |
| **HNSW** | Hierarchical Navigable Small World graph (Malkov & Yashunin): multi-layer proximity graph for ANN search. |
| **IDF** | Inverse document frequency; rarer terms weigh more. Here `ln(1 + (N − df + 0.5)/(df + 0.5))`. |
| **Inverted index** | Map from term to the postings list of chunks containing it. |
| **k1** | BM25 parameter controlling how quickly repeated occurrences of a term stop adding score. |
| **b** | BM25 parameter in [0, 1] controlling how strongly long chunks are penalized. |
| **Length normalization** | The `1 − b + b·dl/avgdl` factor in BM25. |
| **M** | HNSW maximum neighbors per node on upper layers (2M on layer 0). |
| **MAP** | Mean Average Precision: mean over queries of the average of precision at each relevant hit. |
| **Max-aggregation** | A document's score is the maximum of its chunks' scores. |
| **mmap** | Memory-mapping a file so the OS pages it in on demand; the postings file is read this way. |
| **mL** | HNSW level multiplier `1/ln(M)`; a node's top level is `floor(−ln(U)·mL)`. |
| **MRR** | Mean Reciprocal Rank of the first relevant result. |
| **Pareto frontier** | Configurations not dominated by any other on both quality and latency. |
| **Percentile (p50/p95/p99)** | Latency below which 50/95/99% of requests fall. |
| **Phrase query** | A quoted multi-word query matched via positions; used here as an additive boost. |
| **Positions** | Offsets of a term inside a chunk, stored in postings for phrase matching. |
| **Postings list** | Sorted list of (chunk id, tf, positions) for one term. |
| **Precision@k** | Fraction of the top k that is relevant. |
| **Qrels** | Query relevance judgments: (query, document, grade) triples. |
| **QPS** | Queries per second completed by a concurrent run. |
| **Recall@k** | Fraction of all relevant documents found in the top k. |
| **Reranking** | Re-scoring the top candidates with a more expensive model. |
| **RRF** | Reciprocal Rank Fusion: `Σ 1/(k + rank)` over lists (Cormack et al., 2009). |
| **Skip pointer** | Per-block entry (last doc id, byte offset) that lets a cursor jump over blocks it does not need. |
| **Stemming** | Reducing words to a common stem (Porter): *running → run*. |
| **Stopwords** | Very frequent function words removed before indexing. |
| **tf (term frequency)** | Occurrences of a term in a chunk. |
| **Unjudged document** | A retrieved document with no qrel; counted as non-relevant. |
| **Upper bound (WAND)** | The maximum score a term can contribute to any chunk; lets WAND skip chunks that cannot enter the top k. |
| **Varint** | Variable-length integer encoding: 7 data bits per byte, high bit = "more bytes follow". |
| **WAND** | Weak AND (Broder et al., 2003): DAAT top-k algorithm that fully scores only chunks whose summed upper bounds can beat the current threshold. |
| **Weighted fusion** | `alpha·norm(dense) + (1 − alpha)·norm(sparse)` with min-max or z-score normalization. |
| **z-score** | `(s − mean)/std` normalization of a score list. |
