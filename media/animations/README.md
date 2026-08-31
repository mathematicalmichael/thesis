# Dissertation Animations

React-based animated videos (via [Remotion](https://www.remotion.dev/)) for
disseminating the dissertation's research on YouTube — starting with Shorts
(vertical, ≤60s), with long-form videos to follow.

The three initial shorts are drawn straight from the dissertation:

| Composition | Topic | Source material |
| --- | --- | --- |
| `AleatoricVsEpistemic` | The two kinds of uncertainty and why they demand different math | Ch. 1 intro (`intro/intro.tex`) |
| `MoreDataLessUncertainty` | Repeated observations of the same experiment shrink epistemic uncertainty like 1/√N | Ch. 1 + repeated-observation experiments (`figures/MUD_pt_Repeated_Obs.ipynb`) |
| `KeyContributions` | The thesis' three contributions: DCI, the MUD point, open-source software | Abstract + Ch. 5/6 |

## Quick start

```sh
cd media/animations
npm install
npm run studio        # interactive preview at http://localhost:3000
```

## Rendering

```sh
npm run render:aleatoric      # -> out/aleatoric-vs-epistemic.mp4
npm run render:more-data      # -> out/more-data-less-uncertainty.mp4
npm run render:contributions  # -> out/key-contributions.mp4
npm run render:all
```

All compositions are 1080×1920 @ 30fps (YouTube Shorts / vertical). Remotion
bundles its own ffmpeg; the only external requirement is a Chromium-family
browser. `remotion.config.ts` auto-detects common install locations (or set
`REMOTION_BROWSER_EXECUTABLE`), and uses Chrome's *new* headless mode, so any
recent Chromium works.

## Voiceover scripts + captions (single source of truth)

Each short's narration lives in `src/shorts/<Name>/script.ts` as a list of
`{caption, voiceover, from, to}` beats (frame-based timing):

- the **caption** is burned into the video by `src/components/Captions.tsx`
- the **voiceover** text is exported as a timestamped markdown script you can
  read while recording:

```sh
npm run scripts:export   # regenerates scripts/*.md
```

Change the timing or wording in `script.ts` and both the video and the
markdown stay in sync — the generated files in `scripts/` should never be
edited by hand.

## Recording workflow (suggested)

1. `npm run scripts:export` and open the short's script in `scripts/`.
2. Record the voiceover following the timestamps (each window has breathing
   room; land anywhere inside it).
3. `npm run render:<short>` to produce the MP4.
4. Mux the audio in any editor, or via ffmpeg:
   `ffmpeg -i out/<short>.mp4 -i voiceover.wav -c:v copy -map 0:v -map 1:a -shortest final.mp4`
5. Upload as a Short (title/description in the script header; link the
   dissertation PDF release and this repo).

## Project layout

```
media/animations/
├── remotion.config.ts        # render settings + browser detection
├── src/
│   ├── index.ts              # Remotion entry (registerRoot)
│   ├── Root.tsx              # composition registry
│   ├── theme.ts              # shared palette / typography / canvas size
│   ├── types.ts              # VoiceoverLine / ShortMeta
│   ├── components/           # captions, layout, chart primitives
│   └── shorts/<Name>/        # one folder per video: script.ts + index.tsx
├── tools/generate-scripts.mts# markdown script generator
├── scripts/                  # generated voiceover scripts (do not edit)
└── out/                      # rendered videos (gitignored)
```

## Adding a new short

1. Create `src/shorts/MyShort/script.ts` exporting a `ShortMeta` (copy an
   existing one; keep `...SHORT` for the vertical format).
2. Build the visuals in `src/shorts/MyShort/index.tsx`, ending with
   `<Captions lines={script} />`, `<ProgressBar />`, `<Watermark />`.
3. Register it in `src/Root.tsx` and add it to `SHORTS` in
   `tools/generate-scripts.mts`.
4. Add a `render:*` npm script.

Design conventions: captions sit above YouTube's bottom UI overlay
(`bottom: 420`), keep content inside ~60px side margins, use `theme.ts`
colors (cyan = aleatoric, amber = epistemic, green = truth/success).

## Relationship to the rest of the repo

Nothing here touches the LaTeX build. The dissertation PDF ships from GitHub
Releases, and all results are reproducible via
`docker.io/mathematicalmichael/latex-thesis` (see the repo root README) —
which is also the long-term plan for regenerating publication figures to
feature in long-form videos.
