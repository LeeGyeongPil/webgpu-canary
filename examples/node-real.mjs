// Real mode: transformers.js on Node (cpu), model Xenova/all-MiniLM-L6-v2.
// Candidates: webgpu/fp16 (fails where there is no GPU adapter, e.g. in a sandbox), cpu/q4, cpu/q8.
//   WEBGPU_CANARY_REAL=1 NODE_USE_ENV_PROXY=1 npm run example:node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { selectRuntime, formatAttempts, resolveLang, t, candidateName } from '../dist/index.js';
import { transformersLoader, transformersVersion } from '../dist/transformers/index.js';
import { fixtures } from '../dist/testing/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const i = args.lastIndexOf('--lang');
const lang = resolveLang(i >= 0 ? args[i + 1] : undefined, process.env);

if (process.env.WEBGPU_CANARY_REAL !== '1') {
  console.log(t(lang, 'real.needFlag'));
  process.exit(0);
}

const f = fixtures.minilm;
const baseline = {
  schema: 1, task: 'embeddings', modelId: f.modelId, revision: f.revision, device: 'cpu', dtype: 'fp32',
  input: f.input, vector: [...f.vectors.fp32], libVersion: f.libVersion, createdAt: f.recordedAt,
};
const cacheDir = path.join(root, '.cache');
const loader = transformersLoader({ cacheDir });
const outFile = path.join(root, 'out', 'report.json');

console.log(t(lang, 'real.banner', { model: f.modelId }));
try {
  const { runner, candidate, report } = await selectRuntime({
    modelId: f.modelId,
    revision: f.revision,
    task: 'embeddings',
    candidates: [
      { device: 'webgpu', dtype: 'fp16' },
      { device: 'cpu', dtype: 'q4' },
      { device: 'cpu', dtype: 'q8' },
    ],
    baseline,
    loader,
    libVersion: await transformersVersion({ cacheDir }),
    // Node has no navigator.gpu, but onnxruntime-node ships a WebGPU provider: let it try and record the outcome.
    env: { hasWebGpu: true, shaderF16: true, adapter: { vendor: '', architecture: '', device: '', description: 'node' }, maxBufferSize: 0, userAgent: `node/${process.versions.node}` },
    timeoutMs: 900_000,
  });
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, JSON.stringify(report, null, 2) + '\n');
  formatAttempts(report.attempts, lang).forEach((l) => console.log(l));
  console.log(t(lang, 'real.chosen', { candidate: candidateName(candidate) }));
  console.log(t(lang, 'real.wrote', { path: path.relative(root, outFile) }));
  await runner.dispose();
} catch (error) {
  const report = error?.report;
  if (report) {
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    fs.writeFileSync(outFile, JSON.stringify(report, null, 2) + '\n');
    formatAttempts(report.attempts, lang).forEach((l) => console.log(l));
    console.log(t(lang, 'real.failed', { path: path.relative(root, outFile) }));
  } else {
    console.error(error);
  }
  process.exit(1);
}
