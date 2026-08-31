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
import {AxisLine, normalSamples} from '../../components/charts';
import {theme} from '../../theme';
import {MORE_DATA_LESS_UNCERTAINTY} from './script';

const {script} = MORE_DATA_LESS_UNCERTAINTY;

const TRUTH = 0.55;
const NOISE_SIGMA = 0.12;
const SAMPLES = normalSamples(100, 'repeated-trials', 0, NOISE_SIGMA);

const runningMean = (n: number): number => {
  const slice = SAMPLES.slice(0, Math.max(1, n));
  return TRUTH + slice.reduce((a, b) => a + b, 0) / slice.length;
};

/** The central experiment panel: ticks, running mean, shrinking band. */
const Experiment: React.FC<{
  n: number;
  showBand: boolean;
  showTruth: boolean;
}> = ({n, showBand, showTruth}) => {
  const W = 900;
  const H = 700;
  const mean = runningMean(n);
  const band = (NOISE_SIGMA / Math.sqrt(Math.max(1, n))) * 2;
  return (
    <div
      style={{
        position: 'absolute',
        top: 560,
        left: (1080 - W) / 2,
        width: W,
        height: H,
      }}
    >
      {/* uncertainty band around the running mean */}
      {showBand ? (
        <div
          style={{
            position: 'absolute',
            left: (mean - band) * W,
            width: band * 2 * W,
            top: 0,
            height: H - 120,
            background: theme.epistemic,
            opacity: 0.18,
            borderRadius: 16,
          }}
        />
      ) : null}
      {/* true value */}
      {showTruth ? (
        <>
          <div
            style={{
              position: 'absolute',
              left: TRUTH * W - 4,
              top: 0,
              width: 8,
              height: H - 120,
              background: theme.good,
              borderRadius: 4,
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: TRUTH * W + 18,
              top: 0,
              fontFamily: theme.fontMono,
              fontSize: 36,
              color: theme.good,
            }}
          >
            truth
          </div>
        </>
      ) : null}
      {/* running mean */}
      {n > 0 ? (
        <>
          <div
            style={{
              position: 'absolute',
              left: mean * W - 4,
              top: 60,
              width: 8,
              height: H - 180,
              background: theme.epistemic,
              borderRadius: 4,
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: mean * W - 340,
              top: 66,
              width: 300,
              textAlign: 'right',
              fontFamily: theme.fontMono,
              fontSize: 36,
              color: theme.epistemic,
            }}
          >
            average →
          </div>
        </>
      ) : null}
      {/* measurement ticks: newest drops in from above */}
      {SAMPLES.slice(0, n).map((e, i) => {
        const isNewest = i === n - 1;
        const row = i % 12;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: (TRUTH + e) * W - 6,
              top: H - 520 + row * 32,
              width: 12,
              height: 26,
              borderRadius: 5,
              background: theme.aleatoric,
              opacity: isNewest ? 1 : 0.55,
              transform: isNewest ? 'scale(1.5)' : 'none',
            }}
          />
        );
      })}
      <div style={{position: 'absolute', top: H - 100}}>
        <AxisLine width={W} label="measured value" />
      </div>
      {/* N counter */}
      <div
        style={{
          position: 'absolute',
          top: -140,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontFamily: theme.fontMono,
          fontSize: 64,
          fontWeight: 700,
          color: theme.text,
        }}
      >
        N = {n}
      </div>
    </div>
  );
};

/** 1/sqrt(N) decay curve with the "4x data → half error" annotation. */
const RateCurve: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const W = 860;
  const H = 560;
  const progress = interpolate(frame, [10, 90], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const pts: string[] = [];
  const maxN = 100;
  const shownN = Math.max(2, Math.floor(progress * maxN));
  for (let i = 1; i <= shownN; i++) {
    const x = (i / maxN) * W;
    const y = H - (1 / Math.sqrt(i)) * (H - 40);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const anno = spring({frame: frame - 100, fps, config: {damping: 15}});
  const yAt = (nn: number) => H - (1 / Math.sqrt(nn)) * (H - 40);
  return (
    <>
      <Headline at={0} top={220} size={80}>
        error ∝ <span style={{color: theme.epistemic}}>1 / √N</span>
      </Headline>
      <div
        style={{
          position: 'absolute',
          top: 560,
          left: (1080 - W) / 2,
          width: W,
          height: H + 80,
        }}
      >
        <svg width={W} height={H} style={{overflow: 'visible'}}>
          <polyline
            points={pts.join(' ')}
            fill="none"
            stroke={theme.epistemic}
            strokeWidth={9}
            strokeLinecap="round"
          />
          {/* markers at N=25 and N=100: error halves */}
          <g opacity={anno}>
            <circle cx={(25 / 100) * W} cy={yAt(25)} r={16} fill={theme.accent} />
            <circle cx={W} cy={yAt(100)} r={16} fill={theme.accent} />
          </g>
        </svg>
        <div style={{position: 'absolute', top: H + 8}}>
          <AxisLine width={W} label="number of trials N" />
        </div>
        <div
          style={{
            position: 'absolute',
            top: yAt(25) - 110,
            left: (25 / 100) * W - 60,
            whiteSpace: 'nowrap',
            fontFamily: theme.fontDisplay,
            fontWeight: 700,
            fontSize: 42,
            color: theme.accent,
            opacity: anno,
            transform: `translateY(${(1 - anno) * 30}px)`,
          }}
        >
          4× the data → ½ the error
        </div>
      </div>
    </>
  );
};

const SingleMeasurement: React.FC = () => {
  const frame = useCurrentFrame();
  const appear = interpolate(frame, [20, 45], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <>
      <Headline at={0} top={230} size={72}>
        one experiment,
        <br />
        one noisy answer
      </Headline>
      <AbsoluteFill style={{opacity: appear}}>
        <Experiment n={1} showBand={false} showTruth />
      </AbsoluteFill>
    </>
  );
};

const Accumulate: React.FC = () => {
  const frame = useCurrentFrame();
  const dur = script[2].to - script[2].from;
  const n = Math.max(
    1,
    Math.min(SAMPLES.length, Math.ceil((frame / dur) * SAMPLES.length)),
  );
  return (
    <>
      <Headline at={0} top={250} size={72}>
        …run it again. and again.
      </Headline>
      <Experiment n={n} showBand showTruth />
    </>
  );
};

export const MoreDataLessUncertainty: React.FC = () => {
  const s = script;
  return (
    <AbsoluteFill style={{backgroundColor: theme.bg}}>
      <Backdrop />
      <Sequence durationInFrames={s[0].to}>
        <FadeScene durationInFrames={s[0].to}>
          <Headline at={5} top={640} size={100}>
            more data =
            <br />
            <span style={{color: theme.epistemic}}>less uncertainty?</span>
          </Headline>
        </FadeScene>
      </Sequence>
      <Sequence from={s[1].from} durationInFrames={s[1].to - s[1].from}>
        <FadeScene durationInFrames={s[1].to - s[1].from}>
          <SingleMeasurement />
        </FadeScene>
      </Sequence>
      <Sequence from={s[2].from} durationInFrames={s[2].to - s[2].from}>
        <FadeScene durationInFrames={s[2].to - s[2].from}>
          <Accumulate />
        </FadeScene>
      </Sequence>
      <Sequence from={s[3].from} durationInFrames={s[3].to - s[3].from}>
        <FadeScene durationInFrames={s[3].to - s[3].from}>
          <RateCurve />
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
