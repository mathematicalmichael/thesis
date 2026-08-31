import {SHORT} from '../../theme.ts';
import type {ShortMeta} from '../../types.ts';

/** Single source of truth for Short #3. 30 fps. */
export const KEY_CONTRIBUTIONS: ShortMeta = {
  id: 'KeyContributions',
  title: 'What My PhD Actually Contributed',
  logline:
    'Three contributions of the dissertation: data-consistent inversion, the MUD point for parameter identification, and open-source scientific software.',
  ...SHORT,
  durationInFrames: 1800,
  script: [
    {
      from: 0,
      to: 150,
      caption: 'Years of a PhD → 3 contributions',
      voiceover:
        'I spent years of my life on one question: how do you turn noisy data into honest knowledge about the things you cannot observe directly? Here are the three things my dissertation actually contributed.',
    },
    {
      from: 150,
      to: 620,
      caption: '1 · Data-Consistent Inversion',
      voiceover:
        "First: data-consistent inversion. Your model pushes a distribution of parameters forward into predictions. Data-consistent inversion pulls the observed distribution back onto the parameters — using measure theory, not Bayes' rule. The guarantee: push the answer back through the model and you reproduce exactly the distribution you observed. Bayesian posteriors don't promise that. Data-consistent solutions do — by construction.",
    },
    {
      from: 620,
      to: 1120,
      caption: '2 · One truth, noisy data → the MUD point',
      voiceover:
        'Second — and this is the core of the thesis — I extended that framework to the classic problem where a single true parameter hides behind noisy data. By pulling back distributions of residuals, you get the maximal updated density point: the MUD point. It matches the accuracy of least squares, but unlike the Bayesian MAP point, it is not dragged around by how strongly you state your prior beliefs. No hyper-parameter tuning required.',
    },
    {
      from: 1120,
      to: 1500,
      caption: '3 · Open-source software for UQ',
      voiceover:
        "Third: a beautiful method nobody can run is a method nobody will use. So a serious chunk of the dissertation is software: open-source Python packages, fully reproducible experiments — even the document itself rebuilds from source with one command in Docker.",
    },
    {
      from: 1500,
      to: 1800,
      caption: 'Rigorous math · honest uncertainty · runnable code',
      voiceover:
        'Rigorous mathematics, honest uncertainty, and code you can run tonight. The full dissertation and every experiment in it are linked below — and this is the first of a whole series of animations unpacking it.',
    },
  ],
};
