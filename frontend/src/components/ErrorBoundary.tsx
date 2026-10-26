"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";
import { ApiError } from "@/lib/api";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

/** Catches render errors in client components and shows what went wrong. */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("render error", error, info.componentStack);
  }

  override render(): ReactNode {
    if (this.state.error) {
      return this.props.fallback ?? <ErrorNotice error={this.state.error} onRetry={() => this.setState({ error: null })} />;
    }
    return this.props.children;
  }
}

/** Shared error display for API and render failures. */
export function ErrorNotice({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const isApi = error instanceof ApiError;
  const title = isApi
    ? error.info.stage
      ? `The ${error.info.stage} stage is unavailable`
      : error.info.error.replace(/_/g, " ")
    : "Something broke while rendering this page";
  const detail = isApi ? error.info.detail : error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="rounded-md border border-bad/40 bg-bad/5 p-4 text-sm">
      <p className="font-medium text-bad">{title}</p>
      <p className="mt-1 text-ink">{detail}</p>
      {isApi && error.info.stage && (
        <p className="mt-1 text-muted">Strategies that do not use this stage still work.</p>
      )}
      {isApi && error.info.request_id && <p className="mt-1 text-xs text-muted">Request {error.info.request_id}</p>}
      {onRetry && (
        <button type="button" className="btn-quiet mt-3" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

/** String-only variant usable from server components (props must be serializable). */
export function FailureNotice({ title, detail }: { title: string; detail: string }) {
  return (
    <div role="alert" className="rounded-md border border-bad/40 bg-bad/5 p-4 text-sm">
      <p className="font-medium text-bad">{title}</p>
      <p className="mt-1 text-ink">{detail}</p>
    </div>
  );
}

/** Shown when the API answered but there is nothing to display yet. */
export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-line p-6 text-sm">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 text-muted">{children}</div>}
    </div>
  );
}
