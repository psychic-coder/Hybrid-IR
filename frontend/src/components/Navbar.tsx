"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { STAGE_LABELS } from "@/lib/format";
import type { Stage } from "@/lib/types";

const LINKS = [
  { href: "/", label: "Search" },
  { href: "/compare", label: "Compare" },
  { href: "/metrics", label: "Metrics" },
  { href: "/ablations", label: "Ablations" },
  { href: "/about", label: "About" },
];

const STAGE_DOT: Record<Stage, string> = {
  bm25: "bg-bm25",
  dense: "bg-dense",
  fusion: "bg-fusion",
  rerank: "bg-rerank",
};

export function Navbar() {
  const pathname = usePathname();
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("hybridir-theme", next ? "dark" : "light");
    } catch {
      /* storage unavailable: theme still applies for this page view */
    }
  }

  return (
    <header className="border-b border-line bg-panel">
      <div className="mx-auto flex max-w-page flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
        <Link href="/" className="text-lg font-semibold tracking-tight">
          Hybrid<span className="font-normal text-muted">IR</span>
        </Link>
        <nav aria-label="Main" className="flex flex-wrap gap-1">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`rounded px-2.5 py-1 text-sm ${
                  active ? "bg-accent/10 font-medium text-accent" : "text-muted hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-4">
          <ul className="hidden items-center gap-3 text-xs text-muted md:flex" aria-label="Stage colors">
            {(Object.keys(STAGE_LABELS) as Stage[]).map((s) => (
              <li key={s} className="flex items-center gap-1.5">
                <span className={`inline-block h-2.5 w-2.5 rounded-sm ${STAGE_DOT[s]}`} aria-hidden />
                {STAGE_LABELS[s]}
              </li>
            ))}
          </ul>
          <button type="button" onClick={toggleTheme} className="btn-quiet" aria-label="Toggle dark mode">
            {dark ? "Light" : "Dark"}
          </button>
        </div>
      </div>
    </header>
  );
}
