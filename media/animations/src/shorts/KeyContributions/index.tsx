import React from 'react';
import {
  AbsoluteFill,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {Captions} from '../../components/Captions';
import {
  Backdrop,
  EndCard,
  FadeScene,
  Headline,
  ProgressBar,
  Watermark,
} from '../../components/Layout';
import {AxisLine, GaussianCurve} from '../../components/charts';
import {theme} from '../../theme';
import {KEY_CONTRIBUTIONS} from './script';

const {script} = KEY_CONTRIBUTIONS;

/** Numbered contribution badge. */
const Badge: React.FC<{n: number; label: string; color: string}> = ({
  n,
  label,
  color,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - 4, fps, config: {damping: 13}});
  return (
    <div
      style={{
        position: 'absolute',
        top: 170,
        left: 0,
        right: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 30,
        opacity: s,
        transform: `scale(${0.8 + 0.2 * s})`,
      }}
    >
      <div
        style={{
          width: 110,
          height: 110,
          borderRadius: 55,
          background: color,
          color: theme.bg,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: theme.fontDisplay,
          fontWeight: 800,
          fontSize: 64,
          flexShrink: 0,
        }}
      >
        {n}
      </div>
      <div
        style={{
          fontFamily: theme.fontDisplay,
          fontWeight: 800,
          fontSize: 58,
          color: theme.text,
          maxWidth: 760,
          lineHeight: 1.1,
        }}
      >
        {label}
      </div>
    </div>
  );
};

/**
 * Contribution 1: the push-forward / pull-back loop.
 * Top box: parameter space. Bottom box: data space. The updated distribution's
 * push-forward morphs until it exactly overlays the observed one.
 */
const DciDiagram: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const boxW = 820;
  const boxH = 380;
  // Phases within this sequence (470 frames total)
  const pushDown = spring({frame: frame - 40, fps, config: {damping: 30}});
  const mismatch = interpolate(frame, [120, 150], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const pullUp = spring({frame: frame - 210, fps, config: {damping: 30}});
  // predicted distribution slides/morphs onto the observed one
  const match = interpolate(frame, [280, 360], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const check = spring({frame: frame - 380, fps, config: {damping: 12}});
  const predictedMu = interpolate(match, [0, 1], [0.38, 0.6]);
  const predictedSigma = interpolate(match, [0, 1], [0.16, 0.09]);
  const updatedMu = interpolate(pullUp, [0, 1], [0.5, 0.58]);

  const box = (
    top: number,
    label: string,
    child: React.ReactNode,
  ): React.ReactNode => (
    <div
      style={{
        position: 'absolute',
        top,
        left: (1080 - boxW) / 2,
        width: boxW,
        height: boxH,
        borderRadius: 28,
        background: theme.bgPanel,
        border: `3px solid ${theme.grid}`,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 18,
          left: 28,
          fontFamily: theme.fontMono,
          fontSize: 34,
          color: theme.textDim,
        }}
      >
        {label}
      </div>
      {child}
    </div>
  );

  return (
    <>
      <Badge n={1} label="Data-Consistent Inversion" color={theme.aleatoric} />
      {box(
        330,
        'parameter space (hidden)',
        <div style={{position: 'absolute', bottom: 24, left: 60}}>
          {/* initial → updated */}
          <div style={{position: 'absolute', bottom: 0, opacity: 1 - pullUp * 0.7}}>
            <GaussianCurve width={700} height={250} mu={0.5} sigma={0.17} color={theme.textDim} />
          </div>
          <div style={{position: 'absolute', bottom: 0, opacity: pullUp}}>
            <GaussianCurve width={700} height={250} mu={updatedMu} sigma={0.09} color={theme.aleatoric} />
          </div>
        </div>,
      )}
      {box(
        880,
        'data space (observed)',
        <div style={{position: 'absolute', bottom: 24, left: 60}}>
          {/* observed distribution */}
          <div style={{position: 'absolute', bottom: 0}}>
            <GaussianCurve width={700} height={250} mu={0.6} sigma={0.09} color={theme.good} />
          </div>
          {/* predicted (push-forward) distribution */}
          <div style={{position: 'absolute', bottom: 0, opacity: pushDown * (1 - match * 0.15)}}>
            <GaussianCurve
              width={700}
              height={250}
              mu={predictedMu}
              sigma={predictedSigma}
              color={theme.accent}
            />
          </div>
          {/* mismatch callout */}
          <div
            style={{
              position: 'absolute',
              bottom: 220,
              left: 120,
              fontFamily: theme.fontDisplay,
              fontWeight: 700,
              fontSize: 40,
              color: theme.accent,
              opacity: mismatch * (1 - match),
            }}
          >
            prediction ≠ observation
          </div>
        </div>,
      )}
      {/* arrows between spaces */}
      <FlowArrow top={735} down opacity={pushDown} label="model (push-forward)" />
      <FlowArrow top={735} down={false} opacity={pullUp} label="update (pull-back)" />
      {/* success check */}
      <div
        style={{
          position: 'absolute',
          top: 820,
          right: 90,
          fontSize: 110,
          opacity: check,
          transform: `scale(${check})`,
        }}
      >
        ✅
      </div>
    </>
  );
};

const FlowArrow: React.FC<{
  top: number;
  down: boolean;
  opacity: number;
  label: string;
}> = ({top, down, opacity, label}) => (
  <div
    style={{
      position: 'absolute',
      top,
      left: down ? 200 : 620,
      display: 'flex',
      alignItems: 'center',
      gap: 20,
      opacity,
    }}
  >
    <div style={{fontSize: 90, color: down ? theme.accent : theme.aleatoric}}>
      {down ? '⬇' : '⬆'}
    </div>
    <div
      style={{
        fontFamily: theme.fontMono,
        fontSize: 34,
        color: theme.textDim,
        maxWidth: 300,
      }}
    >
      {label}
    </div>
  </div>
);

/**
 * Contribution 2: MAP drifts as prior confidence is scaled; MUD stays put on
 * the truth's contour.
 */
const MudVsMap: React.FC = () => {
  const frame = useCurrentFrame();
  const W = 860;
  const H = 520;
  const truth = 0.62;
  // prior confidence oscillates: sigma of the prior sweeps back and forth
  const t = Math.sin(frame / 45) * 0.5 + 0.5; // 0..1
  const priorSigma = 0.08 + t * 0.14;
  // MAP gets pulled toward the prior mean when the prior is confident
  const priorMean = 0.3;
  const pull = 0.08 / priorSigma; // stronger pull for tighter prior
  const mapX = truth + (priorMean - truth) * Math.min(1, pull * 0.55);
  return (
    <>
      <Badge n={2} label="Parameter identification: the MUD point" color={theme.epistemic} />
      <div
        style={{
          position: 'absolute',
          top: 430,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontFamily: theme.fontDisplay,
          fontSize: 42,
          color: theme.textDim,
        }}
      >
        MAP drifts with your prior — MUD stays on target
      </div>
      <div
        style={{
          position: 'absolute',
          top: 560,
          left: (1080 - W) / 2,
          width: W,
          height: H + 220,
        }}
      >
        {/* prior belief (breathing) */}
        <GaussianCurve width={W} height={H} mu={priorMean} sigma={priorSigma} color={theme.textDim} opacity={0.75} />
        {/* truth line */}
        <div
          style={{
            position: 'absolute',
            left: truth * W - 4,
            top: 0,
            width: 8,
            height: H,
            background: theme.good,
            borderRadius: 4,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: truth * W - 60,
            top: -60,
            fontFamily: theme.fontMono,
            fontSize: 34,
            color: theme.good,
          }}
        >
          truth
        </div>
        {/* MAP dot: slides with prior confidence */}
        <Dot x={mapX * W} y={H + 60} color={theme.accent} label="MAP (Bayesian)" />
        {/* MUD dot: fixed at the truth */}
        <Dot
          x={truth * W}
          y={H + 150}
          color={theme.epistemic}
          label="MUD (data-consistent)"
          labelSide="left"
        />
        <div style={{position: 'absolute', top: H + 230}}>
          <AxisLine width={W} />
        </div>
      </div>
    </>
  );
};

const Dot: React.FC<{
  x: number;
  y: number;
  color: string;
  label: string;
  labelSide?: 'left' | 'right';
}> = ({x, y, color, label, labelSide = 'right'}) => (
  <div style={{position: 'absolute', left: x - 18, top: y}}>
    <div
      style={{
        width: 36,
        height: 36,
        borderRadius: 18,
        background: color,
        boxShadow: `0 0 30px ${color}`,
      }}
    />
    <div
      style={{
        position: 'absolute',
        ...(labelSide === 'right' ? {left: 50} : {right: 50}),
        top: -4,
        whiteSpace: 'nowrap',
        fontFamily: theme.fontMono,
        fontSize: 36,
        color,
      }}
    >
      {label}
    </div>
  </div>
);

/** Contribution 3: a fake terminal typing out the reproducibility story. */
const SoftwareTerminal: React.FC = () => {
  const frame = useCurrentFrame();
  const lines: Array<{text: string; at: number; out?: boolean}> = [
    {text: '$ pip install mud bet', at: 20},
    {text: '✓ installed', at: 70, out: true},
    {text: '$ ./docker_make', at: 110},
    {text: '✓ dissertation.pdf rebuilt from source', at: 180, out: true},
    {text: '$ make figures', at: 220},
    {text: '✓ every figure regenerated', at: 290, out: true},
  ];
  return (
    <>
      <Badge n={3} label="Open-source scientific software" color={theme.good} />
      <div
        style={{
          position: 'absolute',
          top: 480,
          left: 70,
          right: 70,
          borderRadius: 28,
          background: '#0a0e1a',
          border: `3px solid ${theme.grid}`,
          padding: '90px 50px 60px',
          fontFamily: theme.fontMono,
          fontSize: 42,
          lineHeight: 2.0,
          minHeight: 700,
        }}
      >
        {/* window chrome */}
        <div style={{position: 'absolute', top: 28, left: 40, display: 'flex', gap: 18}}>
          {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
            <div key={c} style={{width: 28, height: 28, borderRadius: 14, background: c}} />
          ))}
        </div>
        {lines
          .filter((l) => frame >= l.at)
          .map((l, i) => {
            const chars = l.out
              ? l.text.length
              : Math.min(l.text.length, Math.floor((frame - l.at) / 2));
            return (
              <div key={i} style={{color: l.out ? theme.good : theme.text}}>
                {l.text.slice(0, chars)}
                {!l.out && chars < l.text.length ? (
                  <span style={{opacity: frame % 20 < 10 ? 1 : 0}}>▋</span>
                ) : null}
              </div>
            );
          })}
      </div>
    </>
  );
};

export const KeyContributions: React.FC = () => {
  const s = script;
  return (
    <AbsoluteFill style={{backgroundColor: theme.bg}}>
      <Backdrop />
      <Sequence durationInFrames={s[0].to}>
        <FadeScene durationInFrames={s[0].to}>
          <Headline at={5} top={600} size={96}>
            my PhD,
            <br />
            in <span style={{color: theme.accent}}>3 ideas</span>
          </Headline>
        </FadeScene>
      </Sequence>
      <Sequence from={s[1].from} durationInFrames={s[1].to - s[1].from}>
        <FadeScene durationInFrames={s[1].to - s[1].from}>
          <DciDiagram />
        </FadeScene>
      </Sequence>
      <Sequence from={s[2].from} durationInFrames={s[2].to - s[2].from}>
        <FadeScene durationInFrames={s[2].to - s[2].from}>
          <MudVsMap />
        </FadeScene>
      </Sequence>
      <Sequence from={s[3].from} durationInFrames={s[3].to - s[3].from}>
        <FadeScene durationInFrames={s[3].to - s[3].from}>
          <SoftwareTerminal />
        </FadeScene>
      </Sequence>
      <Sequence from={s[4].from} durationInFrames={s[4].to - s[4].from}>
        <FadeScene durationInFrames={s[4].to - s[4].from}>
          <EndCard at={5} />
        </FadeScene>
      </Sequence>
      <Captions lines={s} />
      <ProgressBar />
      <Watermark />
    </AbsoluteFill>
  );
};
