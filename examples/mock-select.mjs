// Practice run with no network and no model: real selection code, fake loader.
//   npm run example:mock                  (English)
//   npm run example:mock -- --lang ko     (Korean)
import { resolveLang } from '../dist/index.js';
import { runMockDemo } from '../dist/testing/index.js';

const args = process.argv.slice(2);
const i = args.lastIndexOf('--lang');
const lang = resolveLang(i >= 0 ? args[i + 1] : undefined, process.env);
await runMockDemo({ lang });
