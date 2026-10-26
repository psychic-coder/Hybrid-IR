"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

interface Props {
  initial?: string;
  busy?: boolean;
  onSubmit: (query: string) => void;
  placeholder?: string;
}

/** Query input. Press "/" anywhere on the page to focus it; quote words for a phrase boost. */
export function SearchBar({ initial = "", busy = false, onSubmit, placeholder }: Props) {
  const [value, setValue] = useState(initial);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => setValue(initial), [initial]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (e.key === "/" && !typing) {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = value.trim();
    if (q) onSubmit(q);
  }

  return (
    <form onSubmit={submit} role="search" className="flex gap-2">
      <label htmlFor="q" className="sr-only">
        Query
      </label>
      <div className="relative flex-1">
        <input
          id="q"
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder ?? 'Ask the corpus, e.g. vitamin D and bone density or "lung cancer" risk'}
          maxLength={512}
          autoComplete="off"
          className="w-full rounded-md border border-line bg-panel px-3 py-2.5 pr-10 text-base outline-none focus:border-accent"
        />
        <kbd
          className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-line px-1.5 text-xs text-muted"
          aria-hidden
        >
          /
        </kbd>
      </div>
      <button type="submit" className="btn-primary px-5" disabled={busy || !value.trim()}>
        {busy ? "Searching…" : "Search"}
      </button>
    </form>
  );
}
