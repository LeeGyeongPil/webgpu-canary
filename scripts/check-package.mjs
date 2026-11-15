// Checks the built package shape: exports have JS + d.ts, no runtime dependencies,
// the core entry does not import react or transformers, and `npm pack` would ship the right files.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const problems = [];

for (const sub of ['.', './react', './transformers', './testing']) {
  const entry = pkg.exports?.[sub];
  if (!entry?.import || !entry?.types) { problems.push(`exports["${sub}"] needs import and types`); continue; }
  for (const f of [entry.import, entry.types]) {
    if (!fs.existsSync(path.join(root, f))) problems.push(`missing ${f} (run npm run build)`);
  }
}

if (Object.keys(pkg.dependencies ?? {}).length > 0) problems.push('dependencies must be empty (use optional peerDependencies)');
for (const peer of ['react', '@huggingface/transformers']) {
  if (!pkg.peerDependencies?.[peer] || !pkg.peerDependenciesMeta?.[peer]?.optional) problems.push(`${peer} must be an optional peer dependency`);
}

// Follow relative imports from the core entry and make sure none reaches react or transformers.
const seen = new Set();
function scan(file) {
  if (seen.has(file) || !fs.existsSync(file)) return;
  seen.add(file);
  const text = fs.readFileSync(file, 'utf8');
  for (const m of text.matchAll(/(?:from\s+|import\s*\()\s*['"]([^'"]+)['"]/g)) {
    const spec = m[1];
    if (spec === 'react' || spec.startsWith('react/') || spec.startsWith('@huggingface/')) problems.push(`core entry imports ${spec} (via ${path.relative(root, file)})`);
    else if (spec.startsWith('.')) scan(path.resolve(path.dirname(file), spec));
  }
}
scan(path.join(root, pkg.exports['.'].import));

const raw = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
const files = JSON.parse(raw)[0].files.map((f) => f.path);
for (const need of ['dist/index.js', 'dist/index.d.ts', 'fixtures/minilm.json', 'fixtures/smollm2.json', 'LICENSE', 'package.json']) {
  if (!files.includes(need)) problems.push(`npm pack does not include ${need}`);
}
// README files are written at release time; once they exist they must be packed.
for (const readme of ['README.md', 'README.ko.md']) {
  if (!pkg.files.includes(readme)) problems.push(`package.json "files" must list ${readme}`);
  if (fs.existsSync(path.join(root, readme)) && !files.includes(readme)) problems.push(`npm pack does not include ${readme}`);
  if (!fs.existsSync(path.join(root, readme))) console.log(`note: ${readme} does not exist yet (written at release)`);
}

if (problems.length > 0) {
  console.error('package problems:\n- ' + problems.join('\n- '));
  process.exit(1);
}
console.log('package ok');
