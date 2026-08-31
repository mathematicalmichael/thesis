import fs from 'node:fs';
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
// Modern Chromium builds only ship the new headless mode; the default
// 'headless-shell' mode tries the removed --headless=old flag.
Config.setChromeMode('chrome-for-testing');

// Prefer a pre-installed Chromium (e.g. Playwright's) over downloading one.
const candidates = [
  process.env.REMOTION_BROWSER_EXECUTABLE,
  '/opt/pw-browsers/chromium',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter((p): p is string => Boolean(p));

for (const candidate of candidates) {
  if (fs.existsSync(candidate)) {
    Config.setBrowserExecutable(candidate);
    break;
  }
}
