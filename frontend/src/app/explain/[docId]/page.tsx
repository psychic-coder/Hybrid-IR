import Link from "next/link";
import { EmptyState, FailureNotice } from "@/components/ErrorBoundary";
import { ExplainPanel } from "@/components/ExplainPanel";
import { ApiError, api } from "@/lib/api";
import type { ExplainResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

interface PageProps {
  params: { docId: string };
  searchParams: { [key: string]: string | string[] | undefined };
}

function first(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

/** Server component: fetched through API_INTERNAL_URL inside the Docker network. */
export default async function ExplainPage({ params, searchParams }: PageProps) {
  const docId = decodeURIComponent(params.docId);
  const query = first(searchParams.q)?.trim() ?? "";
  if (!query) {
    return (
      <EmptyState title="This page explains one document for one query">
        Open it from a search result, or add <code>?q=your+query</code> to the address.
      </EmptyState>
    );
  }
  let data: ExplainResponse | null = null;
  let failure: { title: string; detail: string } | null = null;
  try {
    data = await api.explain(query, docId);
  } catch (e) {
    failure =
      e instanceof ApiError
        ? { title: e.status === 404 ? `No document with id ${docId}` : e.info.error.replace(/_/g, " "), detail: e.info.detail }
        : { title: "Could not load the explanation", detail: String(e) };
  }
  return (
    <div className="space-y-5">
      <nav className="text-sm text-muted">
        <Link href={`/compare?q=${encodeURIComponent(query)}`} className="hover:text-accent hover:underline">
          Back to the comparison for this query
        </Link>
      </nav>
      {failure && <FailureNotice title={failure.title} detail={failure.detail} />}
      {data && <ExplainPanel data={data} />}
    </div>
  );
}
