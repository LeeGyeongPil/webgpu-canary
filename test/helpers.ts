import { fixtures } from '../src/testing/fixtures-data.js';
import type { Baseline, EnvInfo } from '../src/types.js';

export const MODEL = fixtures.minilm.modelId;
export const REVISION = fixtures.minilm.revision;
export const V = fixtures.minilm.vectors;

export function makeEnv(over: Partial<EnvInfo> = {}): EnvInfo {
  return {
    hasWebGpu: true,
    shaderF16: true,
    adapter: { vendor: 'apple', architecture: 'metal-3', device: '', description: 'Apple M2' },
    maxBufferSize: 4294967296,
    userAgent: 'Mozilla/5.0 (Macintosh) Chrome/131.0.6778.86 Safari/537.36',
    ...over,
  };
}

export function embeddingBaseline(over: Partial<Baseline> = {}): Baseline {
  return {
    schema: 1,
    task: 'embeddings',
    modelId: MODEL,
    revision: REVISION,
    device: 'cpu',
    dtype: 'fp32',
    input: fixtures.minilm.input,
    vector: [...V.fp32],
    libVersion: '4.3.2',
    createdAt: '2026-11-16',
    ...over,
  };
}

export function generationBaseline(tokens: number[] = [...fixtures.smollm2.tokens.fp32]): Baseline {
  return {
    schema: 1,
    task: 'text-generation',
    modelId: fixtures.smollm2.modelId,
    revision: fixtures.smollm2.revision,
    device: 'cpu',
    dtype: 'fp32',
    input: fixtures.smollm2.input,
    tokens,
    maxNewTokens: 12,
    libVersion: '4.3.2',
    createdAt: '2026-11-16',
  };
}
