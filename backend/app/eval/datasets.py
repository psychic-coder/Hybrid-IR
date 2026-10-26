"""BEIR dataset download, extraction and parsing.

Expected layout after extraction (BEIR zips contain a top-level ``<name>/`` folder)::

    <raw_dir>/corpus.jsonl       {"_id", "title", "text", ...}
    <raw_dir>/queries.jsonl      {"_id", "text", ...}
    <raw_dir>/qrels/test.tsv     query-id <TAB> corpus-id <TAB> score   (with header row)

Only the *test* split is evaluated. Queries without at least one positive judgment are
dropped (standard BEIR practice).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Document:
    """A corpus document."""

    doc_id: str
    title: str
    text: str


@dataclass
class Dataset:
    """Corpus + test queries + graded relevance judgments."""

    name: str
    corpus: dict[str, Document]
    queries: dict[str, str]
    qrels: Qrels


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file (recorded in results for reproducibility)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_dataset(
    name: str, raw_dir: Path, base_url: str, retries: int = 3, timeout: float = 60.0
) -> Path:
    """Download and extract ``<base_url>/<name>.zip`` into ``raw_dir`` (idempotent).

    Returns the path of the downloaded zip (kept next to ``raw_dir`` for hashing).
    """
    raw_dir = Path(raw_dir)
    if (raw_dir / "corpus.jsonl").exists() and (raw_dir / "qrels" / "test.tsv").exists():
        log.info("dataset %s already extracted at %s", name, raw_dir)
        zip_path = raw_dir.parent / f"{name}.zip"
        return zip_path
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir.parent / f"{name}.zip"
    url = f"{base_url.rstrip('/')}/{name}.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".zip.part")
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                log.info("downloading %s (attempt %d/%d)", url, attempt, retries)
                t0 = time.perf_counter()
                with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                    total = int(resp.headers.get("Content-Length") or 0)
                    done = 0
                    next_report = 0.0
                    while block := resp.read(1 << 20):
                        out.write(block)
                        done += len(block)
                        elapsed = time.perf_counter() - t0
                        if elapsed >= next_report:
                            pct = f"{100 * done / total:5.1f}%" if total else "?"
                            rate = done / max(elapsed, 1e-6)
                            eta = (total - done) / rate if total and rate else 0.0
                            log.info("  %s  %.1f MB  eta %.0fs", pct, done / 1e6, eta)
                            next_report = elapsed + 2.0
                tmp.replace(zip_path)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("download failed: %s", exc)
                time.sleep(2 * attempt)
        if last_err is not None:
            raise RuntimeError(f"could not download {url}: {last_err}") from last_err
    extract_root = raw_dir.parent / "_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    inner = extract_root / name
    src = inner if inner.exists() else extract_root
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.move(str(src), str(raw_dir))
    shutil.rmtree(extract_root, ignore_errors=True)
    log.info("extracted %s to %s", name, raw_dir)
    return zip_path


def iter_corpus(raw_dir: Path) -> Iterator[Document]:
    """Stream documents from corpus.jsonl in file order (deterministic)."""
    with open(Path(raw_dir) / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Document(
                doc_id=str(obj["_id"]),
                title=str(obj.get("title") or ""),
                text=str(obj.get("text") or ""),
            )


def load_corpus(raw_dir: Path) -> dict[str, Document]:
    """Whole corpus as an insertion-ordered dict."""
    return {d.doc_id: d for d in iter_corpus(raw_dir)}


def load_queries(raw_dir: Path) -> dict[str, str]:
    """All queries (id -> text)."""
    out: dict[str, str] = {}
    with open(Path(raw_dir) / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[str(obj["_id"])] = str(obj["text"])
    return out


def load_qrels(raw_dir: Path, split: str = "test") -> Qrels:
    """Graded judgments ``{query_id: {doc_id: grade}}`` for ``split``."""
    path = Path(raw_dir) / "qrels" / f"{split}.tsv"
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if i == 0 and not parts[2].lstrip("-").isdigit():
                continue  # header
            qid, did, grade = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = grade
    return qrels


def load_dataset(name: str, raw_dir: Path, split: str = "test") -> Dataset:
    """Corpus, test queries that have >=1 positive judgment, and their qrels."""
    corpus = load_corpus(raw_dir)
    all_queries = load_queries(raw_dir)
    qrels = load_qrels(raw_dir, split)
    queries = {
        qid: all_queries[qid]
        for qid in sorted(qrels)
        if qid in all_queries and any(g > 0 for g in qrels[qid].values())
    }
    qrels = {qid: qrels[qid] for qid in queries}
    return Dataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
