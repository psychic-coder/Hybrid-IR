# Troubleshooting

Start with `docker compose ps` and `docker compose logs <service> --tail=200`. All backend
logs are JSON lines; `docker compose logs api | grep '"level": "ERROR"'` finds failures fast.

| Symptom | Likely cause | Fix |
|---|---|---|
| `seed` exits 1 with "timed out waiting for services" | Qdrant or Redis did not become healthy within `SEED_WAIT_TIMEOUT_SECONDS` | `docker compose logs qdrant redis`; on slow disks raise the timeout in `.env` |
| `seed` fails downloading the dataset | No outbound HTTPS to `public.ukp.informatik.tu-darmstadt.de`, or a proxy | Download `<dataset>.zip` manually and copy it into the volume: `docker compose run --rm -v "$PWD:/in" seed sh -c 'mkdir -p /data/scifact && cp /in/scifact.zip /data/scifact/'`, then `make seed` |
| `seed` fails downloading a model with `ChunkedEncodingError` or `IncompleteRead` | The Hugging Face connection was interrupted during a large model download | Run `make seed` again; the seed job resumes cached files and retries each incomplete model download up to three times. If it still fails, allow outbound HTTPS to `huggingface.co` or download the model cache on a networked machine |
| `seed` fails downloading models (`huggingface_hub` errors) | No access to huggingface.co | Run once with network access, or pre-populate `hf_cache` (`HF_HOME` layout) from another machine |
| `api` restarts or is never healthy | `/ready` failing; see which check | `curl -s localhost:8000/ready | python -m json.tool` lists every check with a reason |
| `/ready` says `qdrant: collection … missing or empty` | Qdrant volume was reset while `index_data` survived | `make seed` (re-upserts from `embeddings.npy`; no re-embedding) |
| `/ready` says `embedder: … LocalEntryNotFoundError` | api runs offline but the cache lacks the model (seed skipped or `hf_cache` volume deleted) | `make seed` (downloads into `hf_cache`), then `docker compose restart api` |
| Search works for BM25 but hybrid returns 503 `stage_unavailable` / `dense` | Qdrant down or collection missing | as above; BM25 continues to work by design |
| `*_rerank` returns 503 with `stage: rerank` | Cross-encoder failed to load (memory, missing weights) | Check `docker compose logs api`; give Docker more memory; `make seed` |
| Requests return 504 `timeout` | CPU saturated, often by reranking 100+ candidates or a parallel `make eval` | Lower `rerank_top_k`, raise `SEARCH_TIMEOUT_SECONDS`, avoid running eval and load tests at the same time |
| Web page shows "Could not reach the API at http://localhost:8000" | Browser cannot reach the API URL baked into the bundle | Set `NEXT_PUBLIC_API_URL` to an address your browser can reach and `docker compose build web`; add the web origin to `CORS_ORIGINS` |
| Browser console shows CORS errors | Web origin not in `CORS_ORIGINS` | Add it (comma separated) and restart api |
| Metrics page says "No evaluation results yet" | Seed eval skipped or failed, or results volume new | `make eval`; check `docker compose logs seed` |
| Ablations page empty | `ABLATIONS_ON_SEED=false` | `make ablations` |
| `make readme` writes the placeholder | No `summary.json` in the results volume | `make eval` first |
| Out of memory during seed on fiqa | Embedding batch too large for available RAM | `EMBED_BATCH_SIZE=16`, close other apps, ensure Docker has ≥ 6 GB |
| `DENSE_BACKEND=custom` seed takes very long | The from-scratch HNSW is pure Python | Expected on large corpora (fiqa); use `qdrant` for big datasets |
| Changing `.env` has no effect | Built artifacts depend on it, or the web bundle | `[reseed]` variables need `make reseed`; `NEXT_PUBLIC_*` need `docker compose build web` |
| k6: "Insufficient VUs" warning | Target rate exceeds capacity | Lower `K6_*_RATE`; numbers from that run are not valid at that rate |
| k6 exits non-zero | A threshold failed (p95 or error rate) | Read the summary; raise `K6_P95_MS` if the machine is simply slower, or lower rates |
| Apple Silicon: images build slowly | amd64 emulation not involved, but torch wheels are large | First build only; layers are cached afterwards |

## Resetting

```bash
make down          # stop, keep volumes
make reseed        # rebuild artifacts and results in place
make clean         # delete ALL volumes (models, index, embeddings, results)
```
