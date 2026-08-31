import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {theme} from '../theme';

/** Solid background + subtle dot grid so scenes never feel empty. */
export const Backdrop: React.FC<{children?: React.ReactNode}> = ({
  children,
}) => (
  <AbsoluteFill
    style={{
      background: `radial-gradient(1200px 900px at 50% 20%, #131b38 0%, ${theme.bg} 60%)`,
    }}
  >
    <AbsoluteFill
      style={{
        backgroundImage: `radial-gradient(${theme.grid} 3px, transparent 3px)`,
        backgroundSize: '72px 72px',
      }}
    />
    {children}
  </AbsoluteFill>
);

/** Thin progress bar along the top: signals length, keeps viewers to the end. */
export const ProgressBar: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        height: 12,
        width: `${(frame / durationInFrames) * 100}%`,
        background: `linear-gradient(90deg, ${theme.aleatoric}, ${theme.accent})`,
      }}
    />
  );
};

export const Watermark: React.FC = () => (
  <div
    style={{
      position: 'absolute',
      top: 48,
      right: 60,
      fontFamily: theme.fontMono,
      fontSize: 34,
      color: theme.textDim,
      opacity: 0.85,
    }}
  >
    {theme.handle}
  </div>
);

/** Big springy headline used for hooks and section titles. */
export const Headline: React.FC<{
  children: React.ReactNode;
  at?: number;
  size?: number;
  color?: string;
  top?: number;
}> = ({children, at = 0, size = 92, color = theme.text, top = 200}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - at, fps, config: {damping: 14, mass: 0.8}});
  return (
    <div
      style={{
        position: 'absolute',
        top,
        left: 70,
        right: 70,
        textAlign: 'center',
        fontFamily: theme.fontDisplay,
        fontWeight: 800,
        fontSize: size,
        lineHeight: 1.12,
        color,
        opacity: s,
        transform: `scale(${0.85 + 0.15 * s})`,
        textShadow: '0 4px 24px rgba(0,0,0,0.5)',
      }}
    >
      {children}
    </div>
  );
};

/** Fade a scene's content in/out inside a Sequence. */
export const FadeScene: React.FC<{
  children: React.ReactNode;
  durationInFrames: number;
  fade?: number;
}> = ({children, durationInFrames, fade = 10}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, fade, durationInFrames - fade, durationInFrames],
    [0, 1, 1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  return <AbsoluteFill style={{opacity}}>{children}</AbsoluteFill>;
};

/** Closing CTA card shared by all shorts. */
export const EndCard: React.FC<{at?: number}> = ({at = 0}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - at, fps, config: {damping: 15}});
  return (
    <div
      style={{
        position: 'absolute',
        top: 640,
        left: 80,
        right: 80,
        textAlign: 'center',
        opacity: s,
        transform: `translateY(${(1 - s) * 60}px)`,
        fontFamily: theme.fontDisplay,
      }}
    >
      <div style={{fontSize: 76, fontWeight: 800, color: theme.text}}>
        Want the full story?
      </div>
      <div
        style={{
          marginTop: 40,
          fontSize: 52,
          fontWeight: 600,
          color: theme.textDim,
          lineHeight: 1.35,
        }}
      >
        The dissertation — and all the code to reproduce it — is free and open
        source.
      </div>
      <div
        style={{
          marginTop: 56,
          display: 'inline-block',
          fontSize: 48,
          fontWeight: 700,
          color: theme.bg,
          background: theme.epistemic,
          borderRadius: 999,
          padding: '26px 54px',
        }}
      >
        link in description ↓
      </div>
    </div>
  );
};
