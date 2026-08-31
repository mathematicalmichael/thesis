import React from 'react';
import {theme} from '../theme';

export const gaussianPdf = (x: number, mu: number, sigma: number): number =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

/**
 * SVG path for a gaussian bump inside a [0,width]x[0,height] box.
 * The curve's peak reaches `height * peak` (0..1), baseline sits at y=height.
 */
export const gaussianPath = (
  width: number,
  height: number,
  mu: number, // in [0, 1] of width
  sigma: number, // in fraction of width
  peak = 0.9,
  samples = 80,
): string => {
  const maxPdf = gaussianPdf(0, 0, sigma);
  const pts: string[] = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const y =
      height - (gaussianPdf(t, mu, sigma) / maxPdf) * height * peak;
    pts.push(`${(t * width).toFixed(1)},${y.toFixed(1)}`);
  }
  return `M0,${height} L${pts.join(' L')} L${width},${height} Z`;
};

/** A filled gaussian curve with a stroked outline. */
export const GaussianCurve: React.FC<{
  width: number;
  height: number;
  mu: number;
  sigma: number;
  color: string;
  peak?: number;
  opacity?: number;
}> = ({width, height, mu, sigma, color, peak = 0.9, opacity = 1}) => (
  <svg
    width={width}
    height={height}
    viewBox={`0 0 ${width} ${height}`}
    style={{overflow: 'visible', opacity}}
  >
    <path d={gaussianPath(width, height, mu, sigma, peak)} fill={color} fillOpacity={0.22} />
    <path
      d={gaussianPath(width, height, mu, sigma, peak)}
      fill="none"
      stroke={color}
      strokeWidth={7}
      strokeLinejoin="round"
    />
  </svg>
);

/** Horizontal axis line with optional tick label. */
export const AxisLine: React.FC<{width: number; label?: string}> = ({
  width,
  label,
}) => (
  <div style={{width}}>
    <div style={{height: 5, background: theme.textDim, borderRadius: 3}} />
    {label ? (
      <div
        style={{
          marginTop: 14,
          textAlign: 'center',
          fontFamily: theme.fontMono,
          fontSize: 36,
          color: theme.textDim,
        }}
      >
        {label}
      </div>
    ) : null}
  </div>
);

/**
 * Deterministic normal-ish sample stream (Box-Muller over remotion-style
 * seeded uniforms) so renders are reproducible.
 */
export const normalSamples = (
  count: number,
  seed: string,
  mu = 0,
  sigma = 1,
): number[] => {
  // xorshift-based hash of the seed for reproducibility without deps
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 16777619);
  }
  const rand = () => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return ((h >>> 0) % 100000) / 100000;
  };
  const out: number[] = [];
  while (out.length < count) {
    const u = Math.max(rand(), 1e-6);
    const v = Math.max(rand(), 1e-6);
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    out.push(mu + sigma * z);
  }
  return out;
};
