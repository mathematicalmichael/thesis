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
import {
  AxisLine,
  GaussianCurve,
  normalSamples,
} from '../../components/charts';
import {theme} from '../../theme';
import {ALEATORIC_VS_EPISTEMIC} from './script';

const {script} = ALEATORIC_VS_EPISTEMIC;

/** Hook: two pulsing curves teasing the dichotomy. */
const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const pulse = 1 + 0.06 * Math.sin(frame / 6);
  return (
    <>
      <Headline at={5} top={330} size={104}>
        two kinds of
        <br />
        <span style={{color: theme.accent}}>uncertainty</span>
      </Headline>
      <div
        style={{
          position: 'absolute',
          top: 800,
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'center',
          gap: 60,
        }}
      >
        <div style={{transform: `scale(${pulse})`}}>
          <GaussianCurve width={380} height={300} mu={0.5} sigma={0.16} color={theme.aleatoric} />
        </div>
        <div style={{transform: `scale(${2 - pulse})`}}>
          <GaussianCurve width={380} height={300} mu={0.5} sigma={0.07} color={theme.epistemic} />
        </div>
      </div>
    </>
  );
};

/**
 * Aleatoric scene: parts drop from a "factory line" into a histogram whose
 * gaussian envelope becomes sharper in *outline* but keeps the same width.
 */
const Aleatoric: React.FC<{
  emphasizeIrreducible?: boolean;
  braceAt?: number;
}> = ({emphasizeIrreducible, braceAt = 0}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const W = 880;
  const H = 520;
  const samples = normalSamples(120, 'factory', 0.5, 0.14);
  const shown = Math.min(samples.length, Math.floor(frame / 2));
  const curveIn = spring({frame: frame - 30, fps, config: {damping: 30}});
  return (
    <>
      <Headline at={0} top={190} size={78} color={theme.aleatoric}>
        Aleatoric
      </Headline>
      <div
        style={{
          position: 'absolute',
          top: 320,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontFamily: theme.fontDisplay,
          fontSize: 44,
          color: theme.textDim,
        }}
      >
        every manufactured part is a little different
      </div>
      <div
        style={{
          position: 'absolute',
          top: 470,
          left: (1080 - W) / 2,
          width: W,
          height: H + 80,
        }}
      >
        {/* falling sample dots */}
        {samples.slice(0, shown).map((x, i) => {
          const age = frame - i * 2;
          const drop = interpolate(age, [0, 25], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
          const jitter = ((i * 37) % 10) / 10;
          const yEnd = H - 24 - jitter * 140;
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: x * W - 9,
                top: -60 + drop * (yEnd + 60),
                width: 18,
                height: 18,
                borderRadius: 9,
                background: theme.aleatoric,
                opacity: 0.35 + 0.4 * drop,
              }}
            />
          );
        })}
        <div style={{position: 'absolute', top: 0, opacity: curveIn}}>
          <GaussianCurve width={W} height={H} mu={0.5} sigma={0.14} color={theme.aleatoric} />
        </div>
        <div style={{position: 'absolute', top: H + 16}}>
          <AxisLine width={W} label="component property" />
        </div>
        {emphasizeIrreducible ? (
          <WidthBrace
            width={W}
            sigma={0.14}
            at={braceAt}
            label="this spread is REAL — it stays"
          />
        ) : null}
      </div>
    </>
  );
};

/** Double-headed arrow under a curve marking its (constant) width. */
const WidthBrace: React.FC<{
  width: number;
  sigma: number;
  label: string;
  at?: number;
}> = ({width, sigma, label, at = 0}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - at, fps, config: {damping: 16}});
  const half = sigma * 2 * width;
  return (
    <div
      style={{
        position: 'absolute',
        top: 240,
        left: width / 2 - half,
        width: half * 2,
        opacity: s,
        transform: `scale(${s})`,
      }}
    >
      <div
        style={{
          height: 6,
          background: theme.accent,
          borderRadius: 3,
          position: 'relative',
        }}
      >
        <Arrowhead side="left" />
        <Arrowhead side="right" />
      </div>
      <div
        style={{
          marginTop: 22,
          marginLeft: half - 350,
          width: 700,
          textAlign: 'center',
          fontFamily: theme.fontDisplay,
          fontWeight: 700,
          fontSize: 38,
          color: theme.accent,
          whiteSpace: 'nowrap',
        }}
      >
        {label}
      </div>
    </div>
  );
};

const Arrowhead: React.FC<{side: 'left' | 'right'}> = ({side}) => (
  <div
    style={{
      position: 'absolute',
      [side]: -6,
      top: -10,
      width: 0,
      height: 0,
      borderTop: '13px solid transparent',
      borderBottom: '13px solid transparent',
      ...(side === 'left'
        ? {borderRight: `26px solid ${theme.accent}`}
        : {borderLeft: `26px solid ${theme.accent}`}),
    }}
  />
);

/**
 * Epistemic scene: one hidden true value, noisy measurements accumulate,
 * and the uncertainty band shrinks like 1/sqrt(N).
 */
const Epistemic: React.FC<{shrink?: boolean}> = ({shrink}) => {
  const frame = useCurrentFrame();
  const W = 880;
  const H = 520;
  const truth = 0.52;
  const noise = normalSamples(60, 'measurement', 0, 0.11);
  const n = Math.max(1, Math.min(noise.length, Math.floor(frame / 6)));
  const sigma0 = 0.13;
  const sigmaN = shrink ? sigma0 / Math.sqrt(n) : sigma0;
  return (
    <>
      <Headline at={0} top={190} size={78} color={theme.epistemic}>
        Epistemic
      </Headline>
      <div
        style={{
          position: 'absolute',
          top: 320,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontFamily: theme.fontDisplay,
          fontSize: 44,
          color: theme.textDim,
        }}
      >
        one true value, measured with noise
      </div>
      <div
        style={{
          position: 'absolute',
          top: 470,
          left: (1080 - W) / 2,
          width: W,
          height: H + 120,
        }}
      >
        {/* belief distribution around the truth */}
        <GaussianCurve
          width={W}
          height={H}
          mu={truth}
          sigma={Math.max(sigmaN, 0.018)}
          color={theme.epistemic}
        />
        {/* hidden true value */}
        <div
          style={{
            position: 'absolute',
            left: truth * W - 4,
            top: 0,
            width: 8,
            height: H,
            background: theme.good,
            borderRadius: 4,
            opacity: 0.9,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: truth * W + 20,
            top: 6,
            fontFamily: theme.fontMono,
            fontSize: 36,
            color: theme.good,
          }}
        >
          true value
        </div>
        {/* measurement ticks along the axis */}
        {noise.slice(0, n).map((e, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: (truth + e) * W - 5,
              top: H + 14,
              width: 10,
              height: 34,
              borderRadius: 4,
              background: theme.epistemic,
              opacity: 0.8,
            }}
          />
        ))}
        <div style={{position: 'absolute', top: H + 64}}>
          <AxisLine width={W} label="measurements" />
        </div>
        {shrink ? (
          <div
            style={{
              position: 'absolute',
              top: -80,
              right: 0,
              fontFamily: theme.fontMono,
              fontSize: 44,
              color: theme.text,
            }}
          >
            N = {n}
          </div>
        ) : null}
      </div>
    </>
  );
};

/** Side-by-side recap: one spread stays, the other collapses. */
const Compare: React.FC = () => {
  const frame = useCurrentFrame();
  const n = Math.max(1, Math.floor(frame / 5));
  const sigmaEpi = Math.max(0.13 / Math.sqrt(n), 0.02);
  const panel = (
    title: string,
    color: string,
    sigma: number,
    note: string,
  ) => (
    <div style={{textAlign: 'center'}}>
      <div
        style={{
          fontFamily: theme.fontDisplay,
          fontWeight: 800,
          fontSize: 54,
          color,
          marginBottom: 20,
        }}
      >
        {title}
      </div>
      <GaussianCurve width={800} height={260} mu={0.5} sigma={sigma} color={color} />
      <AxisLine width={800} />
      <div
        style={{
          marginTop: 14,
          fontFamily: theme.fontDisplay,
          fontSize: 36,
          color: theme.textDim,
        }}
      >
        {note}
      </div>
    </div>
  );
  return (
    <>
      <Headline at={0} top={130} size={66}>
        more data, same experiment…
      </Headline>
      <div
        style={{
          position: 'absolute',
          top: 330,
          left: 0,
          right: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 50,
        }}
      >
        {panel('aleatoric', theme.aleatoric, 0.14, 'the spread stays — it is the answer')}
        {panel('epistemic', theme.epistemic, sigmaEpi, 'the spread collapses onto the truth')}
      </div>
    </>
  );
};

export const AleatoricVsEpistemic: React.FC = () => {
  const s = script;
  return (
    <AbsoluteFill style={{backgroundColor: theme.bg}}>
      <Backdrop />
      <Sequence durationInFrames={s[0].to}>
        <FadeScene durationInFrames={s[0].to}>
          <Hook />
        </FadeScene>
      </Sequence>
      <Sequence from={s[1].from} durationInFrames={s[2].to - s[1].from}>
        <FadeScene durationInFrames={s[2].to - s[1].from}>
          <AleatoricSection />
        </FadeScene>
      </Sequence>
      <Sequence from={s[3].from} durationInFrames={s[4].to - s[3].from}>
        <FadeScene durationInFrames={s[4].to - s[3].from}>
          <EpistemicSection />
        </FadeScene>
      </Sequence>
      <Sequence from={s[5].from} durationInFrames={s[5].to - s[5].from}>
        <FadeScene durationInFrames={s[5].to - s[5].from}>
          <Compare />
        </FadeScene>
      </Sequence>
      <Sequence from={s[6].from} durationInFrames={s[6].to - s[6].from}>
        <FadeScene durationInFrames={s[6].to - s[6].from}>
          <EndCard at={5} />
        </FadeScene>
      </Sequence>
      <Captions lines={s} />
      <ProgressBar />
      <Watermark />
    </AbsoluteFill>
  );
};

/** Aleatoric beat spans two caption windows; flip emphasis at the second. */
const AleatoricSection: React.FC = () => {
  const frame = useCurrentFrame();
  const emphasizeAt = script[2].from - script[1].from;
  return (
    <Aleatoric
      emphasizeIrreducible={frame >= emphasizeAt}
      braceAt={emphasizeAt + 5}
    />
  );
};

const EpistemicSection: React.FC = () => {
  const frame = useCurrentFrame();
  const shrinkAt = script[4].from - script[3].from;
  return <Epistemic shrink={frame >= shrinkAt} />;
};
