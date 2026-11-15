// Real mode: needs the model download (about 100 MB). Run with
//   WEBGPU_CANARY_REAL=1 NODE_USE_ENV_PROXY=1 npm test
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { selectRuntime } from '../src/select.js';
import { fixtures } from '../src/testing/fixtures-data.js';
import { transformersLoader, transformersVersion } from '../src/transformers/index.js';
import { embeddingBaseline, makeEnv } from './helpers.js';

describe.skipIf(process.env.WEBGPU_CANARY_REAL !== '1')('real transformers.js on Node cpu', () => {
  it('selects cpu/q4 for MiniLM and the cosine matches the recorded fixture', async () => {
    const cacheDir = path.resolve('.cache');
    const f = fixtures.minilm;
    const { candidate, report, runner } = await selectRuntime({
      modelId: f.modelId,
      revision: f.revision,
      task: 'embeddings',
      baseline: embeddingBaseline(),
      loader: transformersLoader({ cacheDir }),
      libVersion: await transformersVersion({ cacheDir }),
      env: makeEnv(),
      candidates: [{ device: 'cpu', dtype: 'q4' }],
      timeoutMs: 600_000,
    });
    await runner.dispose();
    expect(candidate).toEqual({ device: 'cpu', dtype: 'q4' });
    expect(report.attempts[0]!.metric).toBeCloseTo(0.9729, 3);
  }, 700_000);
});
