/** Loading placeholders that match the shape of the content they stand in for. */

function Bar({ className }: { className: string }) {
  return <div className={`animate-pulse rounded bg-line/70 ${className}`} />;
}

export function ResultSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <ul aria-busy="true" aria-label="Loading results" className="space-y-3">
      {Array.from({ length: rows }, (_, i) => (
        <li key={i} className="panel space-y-2 p-3">
          <Bar className="h-4 w-2/3" />
          <Bar className="h-3 w-full" />
          <Bar className="h-3 w-5/6" />
        </li>
      ))}
    </ul>
  );
}

export function TableSkeleton({ rows = 6, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading table" className="panel space-y-2 p-4">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }, (_, c) => (
            <Bar key={c} className="h-4 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

export function ChartSkeleton({ height = 280 }: { height?: number }) {
  return <div aria-busy="true" aria-label="Loading chart" className="panel animate-pulse" style={{ height }} />;
}
