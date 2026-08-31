import {SHORT} from '../../theme.ts';
import type {ShortMeta} from '../../types.ts';

/**
 * Single source of truth for Short #1.
 * Frame windows drive the on-screen captions AND the exported voiceover
 * script (`npm run scripts:export`). 30 fps.
 */
export const ALEATORIC_VS_EPISTEMIC: ShortMeta = {
  id: 'AleatoricVsEpistemic',
  title: 'Two Kinds of Uncertainty (Aleatoric vs. Epistemic)',
  logline:
    'Why "more data" sometimes shrinks your uncertainty — and sometimes cannot, ever.',
  ...SHORT,
  durationInFrames: 1650,
  script: [
    {
      from: 0,
      to: 140,
      caption: 'Not all uncertainty is the same.',
      voiceover:
        "Here's something most people get wrong about uncertainty: there are two completely different kinds — and they behave in opposite ways.",
    },
    {
      from: 140,
      to: 420,
      caption: 'Aleatoric = randomness in the world itself',
      voiceover:
        'The first kind is aleatoric uncertainty — genuine randomness in the world. Picture a factory making electronic components: tiny impurities in the raw materials mean no two parts ever come off the line exactly alike.',
    },
    {
      from: 420,
      to: 650,
      caption: 'More data maps it… but never shrinks it.',
      voiceover:
        'Sample more and more parts and a pattern emerges — a distribution of variability. More data sharpens your picture of that spread, but the spread itself never shrinks. This uncertainty is irreducible.',
    },
    {
      from: 650,
      to: 900,
      caption: 'Epistemic = uncertainty from ignorance',
      voiceover:
        "The second kind is epistemic uncertainty — uncertainty that exists only because we don't know enough. Now take a single part off that line. It has one true value. We just can't measure it perfectly.",
    },
    {
      from: 900,
      to: 1150,
      caption: 'More data destroys it.',
      voiceover:
        'Every measurement is noisy — but each new one narrows in on the truth. This uncertainty is reducible: more data literally destroys it.',
    },
    {
      from: 1150,
      to: 1420,
      caption: 'Two kinds → two different math problems',
      voiceover:
        "And here's why this matters: describing the variability of a whole population, and pinning down one true value, are fundamentally different inverse problems. My PhD research built the mathematics that treats each one honestly.",
    },
    {
      from: 1420,
      to: 1650,
      caption: 'Full dissertation — free & open source',
      voiceover:
        "I'm turning my dissertation into animations like this one. Follow along — and if you want the deep end, the full dissertation and every line of code behind it are linked below.",
    },
  ],
};
