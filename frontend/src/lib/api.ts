/**
 * Typed API client.
 *
 * Two base URLs exist because requests come from two places:
 * - Browser (client components): NEXT_PUBLIC_API_URL, inlined at build time and resolved by the
 *   user's machine, e.g. http://localhost:8000.
 * - Server components (rendered inside the `web` container): API_INTERNAL_URL, read at runtime
 *   and resolved on the Docker network, e.g. http://api:8000. "localhost" inside the web
 *   container would point at the container itself, so the browser URL cannot be reused.
 */

import type {
  AblationsResponse,
  CacheStats,
  CompareRequest,
  CompareResponse,
  ErrorInfo,
  ExplainResponse,
  ReadyResponse,
  SearchParams,
  SearchRequest,
  SearchResponse,
  SummaryResponse,
} from "./types";

const DEFAULT_TIMEOUT_MS = 30_000;

export function apiBaseUrl(): string {
  if (typeof window === "undefined") {
    return process.env.API_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  }
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
}

export class ApiError extends Error {
  readonly status: number;
  readonly info: ErrorInfo;

  constructor(status: number, info: ErrorInfo) {
    super(`${info.error}: ${info.detail}`);
    this.status = status;
    this.info = info;
  }
}

function isErrorInfo(value: unknown): value is ErrorInfo {
  return (
    typeof value === "object" &&
    value !== null &&
    "error" in value &&
    "detail" in value &&
    typeof (value as { error: unknown }).error === "string"
  );
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      cache: "no-store",
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    throw new ApiError(0, {
      error: aborted ? "timeout" : "network_error",
      detail: aborted
        ? `The API did not answer within ${timeoutMs / 1000}s.`
        : `Could not reach the API at ${apiBaseUrl()}. Is the api service running?`,
      stage: null,
      request_id: null,
    });
  } finally {
    clearTimeout(timer);
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(
      res.status,
      isErrorInfo(body)
        ? body
        : { error: "http_error", detail: `HTTP ${res.status}`, stage: null, request_id: null },
    );
  }
  return body as T;
}

function paramsToQuery(params: SearchParams): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined) qs.set(k, String(v));
  }
  return qs.toString();
}

export const api = {
  search: (body: SearchRequest) =>
    request<SearchResponse>("/search", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: CompareRequest) =>
    request<CompareResponse>("/compare", { method: "POST", body: JSON.stringify(body) }, 60_000),
  explain: (query: string, docId: string, params: SearchParams = {}) => {
    const extra = paramsToQuery(params);
    const qs = new URLSearchParams({ query, doc_id: docId });
    return request<ExplainResponse>(`/explain?${qs.toString()}${extra ? `&${extra}` : ""}`, {}, 60_000);
  },
  summary: () => request<SummaryResponse>("/metrics/summary"),
  ablations: () => request<AblationsResponse>("/metrics/ablations"),
  cache: () => request<CacheStats>("/metrics/cache"),
  ready: () => request<ReadyResponse>("/ready"),
};
