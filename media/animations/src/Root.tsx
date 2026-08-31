import React from 'react';
import {Composition} from 'remotion';
import {AleatoricVsEpistemic} from './shorts/AleatoricVsEpistemic';
import {ALEATORIC_VS_EPISTEMIC} from './shorts/AleatoricVsEpistemic/script';
import {KeyContributions} from './shorts/KeyContributions';
import {KEY_CONTRIBUTIONS} from './shorts/KeyContributions/script';
import {MoreDataLessUncertainty} from './shorts/MoreDataLessUncertainty';
import {MORE_DATA_LESS_UNCERTAINTY} from './shorts/MoreDataLessUncertainty/script';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id={ALEATORIC_VS_EPISTEMIC.id}
        component={AleatoricVsEpistemic}
        durationInFrames={ALEATORIC_VS_EPISTEMIC.durationInFrames}
        fps={ALEATORIC_VS_EPISTEMIC.fps}
        width={ALEATORIC_VS_EPISTEMIC.width}
        height={ALEATORIC_VS_EPISTEMIC.height}
      />
      <Composition
        id={MORE_DATA_LESS_UNCERTAINTY.id}
        component={MoreDataLessUncertainty}
        durationInFrames={MORE_DATA_LESS_UNCERTAINTY.durationInFrames}
        fps={MORE_DATA_LESS_UNCERTAINTY.fps}
        width={MORE_DATA_LESS_UNCERTAINTY.width}
        height={MORE_DATA_LESS_UNCERTAINTY.height}
      />
      <Composition
        id={KEY_CONTRIBUTIONS.id}
        component={KeyContributions}
        durationInFrames={KEY_CONTRIBUTIONS.durationInFrames}
        fps={KEY_CONTRIBUTIONS.fps}
        width={KEY_CONTRIBUTIONS.width}
        height={KEY_CONTRIBUTIONS.height}
      />
    </>
  );
};
