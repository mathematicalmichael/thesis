/**
 * A single voiceover beat. `from`/`to` are frame numbers within the
 * composition; they drive both the burned-in caption and the timestamped
 * voiceover script exported by `npm run scripts:export`.
 */
export type VoiceoverLine = {
  /** Short on-screen caption (burned in). Keep it punchy. */
  caption: string;
  /** The full sentence(s) to speak over this beat. */
  voiceover: string;
  from: number;
  to: number;
};

export type ShortMeta = {
  /** Remotion composition id. */
  id: string;
  /** Human title (also used as the script's heading). */
  title: string;
  fps: number;
  durationInFrames: number;
  width: number;
  height: number;
  /** One-line description for the video description / script header. */
  logline: string;
  script: VoiceoverLine[];
};
