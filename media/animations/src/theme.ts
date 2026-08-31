/** Shared look-and-feel for every video so the channel feels cohesive. */
export const theme = {
  bg: '#0b1020',
  bgPanel: '#141b33',
  text: '#f2f5ff',
  textDim: '#9aa5c4',
  // Aleatoric = the world's own variability (cool cyan)
  aleatoric: '#4cc9f0',
  // Epistemic = our ignorance, reducible (warm amber)
  epistemic: '#ffb703',
  accent: '#f72585',
  good: '#80ed99',
  grid: 'rgba(154, 165, 196, 0.18)',
  fontDisplay:
    "'Avenir Next', 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif",
  fontMono: "'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace",
  /** Channel handle shown as a small watermark. */
  handle: '@mindthemath',
} as const;

/** Vertical (Shorts) canvas. */
export const SHORT = {width: 1080, height: 1920, fps: 30} as const;
