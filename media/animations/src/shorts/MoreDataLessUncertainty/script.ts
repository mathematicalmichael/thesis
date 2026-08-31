import {SHORT} from '../../theme.ts';
import type {ShortMeta} from '../../types.ts';

/** Single source of truth for Short #2. 30 fps. */
export const MORE_DATA_LESS_UNCERTAINTY: ShortMeta = {
  id: 'MoreDataLessUncertainty',
  title: 'What More Data Actually Buys You',
  logline:
    'Repeated observations of the same experiment, the law of 1/√N, and why the last digit of precision is the most expensive.',
  ...SHORT,
  durationInFrames: 1350,
  script: [
    {
      from: 0,
      to: 130,
      caption: 'What does more data actually buy you?',
      voiceover:
        'You run an experiment once. Then again. Then a hundred more times. What does all that repetition actually buy you?',
    },
    {
      from: 130,
      to: 400,
      caption: 'One measurement can lie.',
      voiceover:
        "Here's a single noisy measurement of a hidden true value. On its own it can land far from the truth — and you'd never know.",
    },
    {
      from: 400,
      to: 800,
      caption: 'Repetition reveals the truth.',
      voiceover:
        'But repeat the same experiment and watch what happens. Individual measurements scatter — yet their average marches steadily toward the truth, and the uncertainty band around it tightens.',
    },
    {
      from: 800,
      to: 1100,
      caption: 'Uncertainty shrinks like 1 / √N',
      voiceover:
        "There's a famous rate hiding in this picture: uncertainty shrinks like one over the square root of N. Want half the error? You need four times the data. That square root is why the last digit of precision is always the most expensive one.",
    },
    {
      from: 1100,
      to: 1350,
      caption: 'This idea seeds my whole dissertation.',
      voiceover:
        'This simple idea — aggregating repeated observations to squeeze out epistemic uncertainty — is the seed of my entire dissertation. The full document and all the code to reproduce it are linked below.',
    },
  ],
};
