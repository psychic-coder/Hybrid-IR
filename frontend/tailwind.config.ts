import type { Config } from "tailwindcss";

/**
 * Colors are CSS variables (see src/app/globals.css) so light/dark themes swap in one place.
 * The four stage colors (bm25, dense, fusion, rerank) are the visual system of the app:
 * every score chip, latency segment and chart series for a stage uses its color.
 */
const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: v("paper"),
        panel: v("panel"),
        ink: v("ink"),
        muted: v("muted"),
        line: v("line"),
        accent: v("accent"),
        good: v("good"),
        bad: v("bad"),
        bm25: v("bm25"),
        dense: v("dense"),
        fusion: v("fusion"),
        rerank: v("rerank"),
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      maxWidth: { page: "1200px" },
    },
  },
  plugins: [],
};

export default config;
