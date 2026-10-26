import Link from "next/link";

const README_URL = "https://github.com/your-org/hybridir#readme";

function Box({ x, y, w, h, title, sub, stage }: { x: number; y: number; w: number; h: number; title: string; sub: string; stage?: string }) {
  const stroke = stage ? `rgb(var(--${stage}))` : "rgb(var(--line))";
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill="rgb(var(--panel))" stroke={stroke} strokeWidth={stage ? 2 : 1.5} />
      <text x={x + w / 2} y={y + h / 2 - 4} textAnchor="middle" fontSize={14} fontWeight={600} fill="rgb(var(--ink))">
        {title}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={11} fill="rgb(var(--muted))">
        {sub}
      </text>
    </g>
  );
}

function Arrow({ d }: { d: string }) {
  return <path d={d} fill="none" stroke="rgb(var(--muted))" strokeWidth={1.5} markerEnd="url(#arrow)" />;
}

/** Hand-drawn SVG of the query path; full Mermaid diagrams live in the README. */
function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 900 360" className="h-auto w-full" role="img" aria-labelledby="arch-title">
      <title id="arch-title">Query path: tokenizer and embedder feed BM25 and HNSW, fused, then reranked</title>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="rgb(var(--muted))" />
        </marker>
      </defs>
      <Box x={10} y={150} w={120} h={60} title="Query" sub="web :3000 → api :8000" />
      <Box x={180} y={60} w={160} h={60} title="Tokenizer" sub="normalize, stop, stem" stage="bm25" />
      <Box x={180} y={240} w={160} h={60} title="bge-small" sub="query embedding" stage="dense" />
      <Box x={390} y={60} w={170} h={60} title="BM25 + WAND" sub="mmap inverted index" stage="bm25" />
      <Box x={390} y={240} w={170} h={60} title="HNSW" sub="Qdrant or custom" stage="dense" />
      <Box x={610} y={150} w={120} h={60} title="Fusion" sub="RRF or weighted" stage="fusion" />
      <Box x={770} y={150} w={120} h={60} title="Rerank" sub="MiniLM cross-encoder" stage="rerank" />
      <Box x={390} y={150} w={170} h={50} title="Redis" sub="embedding + result cache" />
      <Arrow d="M130,170 C155,170 155,90 178,90" />
      <Arrow d="M130,190 C155,190 155,270 178,270" />
      <Arrow d="M340,90 L388,90" />
      <Arrow d="M340,270 L388,270" />
      <Arrow d="M560,90 C590,90 585,170 608,170" />
      <Arrow d="M560,270 C590,270 585,190 608,190" />
      <Arrow d="M730,180 L768,180" />
      <path d="M260,240 L260,200 L388,180" fill="none" stroke="rgb(var(--line))" strokeDasharray="4 3" strokeWidth={1.5} />
      <text x={450} y={340} textAnchor="middle" fontSize={12} fill="rgb(var(--muted))">
        Every stage is timed per request and every strategy is scored against BEIR relevance judgments.
      </text>
    </svg>
  );
}

export default function AboutPage() {
  return (
    <div className="space-y-6">
      <div className="max-w-prose space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">About HybridIR</h1>
        <p>
          HybridIR combines a hand-built BM25 inverted index with dense HNSW search, fuses the two candidate lists, and
          optionally reranks the top candidates with a cross-encoder. The evaluation harness scores every stage against
          BEIR relevance judgments and records its latency, so you can see which strategy wins and what it costs.
        </p>
      </div>
      <section className="panel p-4">
        <ArchitectureDiagram />
      </section>
      <section className="max-w-prose space-y-2 text-sm">
        <h2 className="text-base font-semibold">Where to go next</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            <Link className="text-accent hover:underline" href="/compare">
              Compare
            </Link>{" "}
            shows how fusion and reranking reorder BM25&rsquo;s results for one query.
          </li>
          <li>
            <Link className="text-accent hover:underline" href="/metrics">
              Metrics
            </Link>{" "}
            holds the quality and latency of every strategy, with significance tests against BM25.
          </li>
          <li>
            The project README is a full reference: IR and evaluation primers, index internals, API, configuration and
            troubleshooting.{" "}
            <a className="text-accent hover:underline" href={process.env.NEXT_PUBLIC_README_URL ?? README_URL}>
              Read the README
            </a>
            .
          </li>
        </ul>
      </section>
    </div>
  );
}
