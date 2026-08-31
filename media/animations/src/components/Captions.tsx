import React from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {theme} from '../theme';
import type {VoiceoverLine} from '../types';

/**
 * Burned-in captions driven by the same VoiceoverLine[] that generates the
 * voiceover script. Renders the line whose [from, to) window contains the
 * current frame, with a springy entrance and a gentle exit.
 *
 * Positioned in the Shorts-safe zone: well above the bottom UI (like/share
 * buttons, video title) which YouTube overlays on the lower ~25% of the frame.
 */
export const Captions: React.FC<{lines: VoiceoverLine[]}> = ({lines}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const line = lines.find((l) => frame >= l.from && frame < l.to);
  if (!line) {
    return null;
  }
  const enter = spring({
    frame: frame - line.from,
    fps,
    config: {damping: 200, stiffness: 120},
  });
  const exit = interpolate(frame, [line.to - 8, line.to], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        position: 'absolute',
        left: 60,
        right: 60,
        bottom: 420,
        display: 'flex',
        justifyContent: 'center',
        opacity: enter * exit,
        transform: `translateY(${(1 - enter) * 40}px)`,
      }}
    >
      <div
        style={{
          fontFamily: theme.fontDisplay,
          fontWeight: 700,
          fontSize: 58,
          lineHeight: 1.25,
          textAlign: 'center',
          color: theme.text,
          background: 'rgba(11, 16, 32, 0.75)',
          borderRadius: 24,
          padding: '24px 36px',
          textShadow: '0 2px 12px rgba(0,0,0,0.6)',
        }}
      >
        {line.caption}
      </div>
    </div>
  );
};
