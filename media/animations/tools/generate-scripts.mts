/**
 * Generates timestamped voiceover scripts (markdown) from the same
 * VoiceoverLine[] data that drives the burned-in captions, so the script you
 * read aloud can never drift out of sync with what's on screen.
 *
 * Usage: npm run scripts:export
 * (runs via `node --experimental-strip-types`, no build step needed)
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {ALEATORIC_VS_EPISTEMIC} from '../src/shorts/AleatoricVsEpistemic/script.ts';
import {KEY_CONTRIBUTIONS} from '../src/shorts/KeyContributions/script.ts';
import {MORE_DATA_LESS_UNCERTAINTY} from '../src/shorts/MoreDataLessUncertainty/script.ts';
import type {ShortMeta} from '../src/types.ts';

const SHORTS: ShortMeta[] = [
  ALEATORIC_VS_EPISTEMIC,
  MORE_DATA_LESS_UNCERTAINTY,
  KEY_CONTRIBUTIONS,
];

const mmss = (frames: number, fps: number): string => {
  const total = frames / fps;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
};

const renderScript = (short: ShortMeta): string => {
  const dur = mmss(short.durationInFrames, short.fps);
  const lines: string[] = [
    `# ${short.title}`,
    '',
    `> ${short.logline}`,
    '',
    `- **Composition:** \`${short.id}\``,
    `- **Format:** ${short.width}×${short.height} @ ${short.fps} fps (YouTube Short)`,
    `- **Runtime:** ${dur}`,
    '',
    '## Voiceover script',
    '',
    'Read each block during its time window. The **caption** column is what',
    'appears on screen (burned in); speak the **voiceover** text naturally —',
    'the windows include breathing room.',
    '',
    '| time | caption (on screen) | voiceover (read this) |',
    '| --- | --- | --- |',
  ];
  for (const l of short.script) {
    const t = `${mmss(l.from, short.fps)}–${mmss(l.to, short.fps)}`;
    lines.push(`| ${t} | ${l.caption} | ${l.voiceover} |`);
  }
  lines.push(
    '',
    '## Plain read-through',
    '',
    ...short.script.map((l) => `${l.voiceover}`),
    '',
    '---',
    '',
    '_This file is generated from the composition source. Do not edit by hand:_',
    '_run `npm run scripts:export` after changing `script.ts`._',
    '',
  );
  return lines.join('\n');
};

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, '..', 'scripts');
fs.mkdirSync(outDir, {recursive: true});
for (const short of SHORTS) {
  const kebab = short.id
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .toLowerCase();
  const file = path.join(outDir, `${kebab}.md`);
  fs.writeFileSync(file, renderScript(short));
  console.log(`wrote ${path.relative(process.cwd(), file)}`);
}
