import { formatAttempts, candidateName } from '../format.js';
import { t, type Lang } from '../i18n/messages.js';
import { judgeEmbedding } from '../judge.js';
import { selectRuntime } from '../select.js';
import { memoryStore } from '../store.js';
import type { Baseline, Candidate, EnvInfo } from '../types.js';
import { fixtures } from './fixtures-data.js';
import { fakeLoader, type Behavior } from './index.js';

export interface MockDemoOptions {
  lang: Lang;
  /** Receives each output line. Default: console.log. */
  write?: (line: string) => void;
  /** Which injected GPU failures to switch on. All on by default. */
  faults?: { error?: boolean; nan?: boolean; drift?: boolean };
}

/** A browser with WebGPU and shader-f16, so the webgpu candidates really get tried. */
export const MOCK_ENV: EnvInfo = {
  hasWebGpu: true,
  shaderF16: true,
  adapter: { vendor: 'mock', architecture: 'mock', device: '', description: 'mock adapter' },
  maxBufferSize: 4294967296,
  userAgent: 'Mozilla/5.0 Chrome/131.0.0.0',
};

export const MOCK_CANDIDATES: Candidate[] = [
  { device: 'webgpu', dtype: 'q4f16' },
  { device: 'webgpu', dtype: 'fp16' },
  { device: 'webgpu', dtype: 'q8' },
  { device: 'wasm', dtype: 'q4' },
  { device: 'wasm', dtype: 'q8' },
];

/** The baseline of the recorded MiniLM run (cpu fp32). */
export function mockBaseline(): Baseline {
  const f = fixtures.minilm;
  return {
    schema: 1,
    task: 'embeddings',
    modelId: f.modelId,
    revision: f.revision,
    device: 'cpu',
    dtype: 'fp32',
    input: f.input,
    vector: [...f.vectors.fp32],
    libVersion: f.libVersion,
    createdAt: f.recordedAt,
  };
}

/** The plan for {@link fakeLoader}: recorded healthy vectors for wasm, injected failures for webgpu. */
export function mockPlan(faults: MockDemoOptions['faults'] = {}, lang: Lang = 'en'): Record<string, Behavior> {
  const on = { error: true, nan: true, drift: true, ...faults };
  const v = fixtures.minilm.vectors;
  return {
    'webgpu/q4f16': on.error
      ? { kind: 'error', message: lang === 'ko' ? '[주입] WebGPU 어댑터를 얻지 못했습니다' : '[injected] Failed to get a WebGPU adapter' }
      : { kind: 'vector', vector: [...v.q4f16] },
    'webgpu/fp16': on.nan ? { kind: 'nan', dim: v.fp16.length } : { kind: 'vector', vector: [...v.fp16] },
    'webgpu/q8': on.drift ? { kind: 'drift', vector: [...v.fp32], cosine: 0.41 } : { kind: 'vector', vector: [...v.q8] },
    'wasm/q4': { kind: 'vector', vector: [...v.q4] },
    'wasm/q8': { kind: 'vector', vector: [...v.q8] },
  };
}

/**
 * Practice run. The selection, judging and cache code is the real library; the loader is fake:
 * wasm results are values recorded from real Node cpu runs, webgpu failures are injected.
 */
export async function runMockDemo(o: MockDemoOptions) {
  const write = o.write ?? ((line: string) => console.log(line));
  const { lang } = o;
  const loader = fakeLoader(mockPlan(o.faults, o.lang));
  const store = memoryStore();
  const baseline = mockBaseline();
  const options = {
    modelId: baseline.modelId,
    revision: baseline.revision,
    task: 'embeddings' as const,
    candidates: MOCK_CANDIDATES,
    baseline,
    loader,
    store,
    env: MOCK_ENV,
    timeoutMs: 5000,
  };

  write(t(lang, 'mock.banner'));
  write(t(lang, 'mock.model', { model: baseline.modelId, revision: baseline.revision.slice(0, 8), task: baseline.task }));
  write('');
  const first = await selectRuntime(options);
  formatAttempts(first.report.attempts, lang).forEach(write);
  write('');
  const chosen = candidateName(first.candidate);
  write(t(lang, 'mock.chosen', { candidate: chosen }));

  // Same data, one fixed threshold: would a naive check have rejected the healthy choice?
  const single = 0.99;
  const naive = judgeEmbedding(fixtures.minilm.vectors.q4, baseline.vector as number[], 'q4', {
    q4: { minCosine: single, minPrefix: 3 },
  });
  if (naive.status === 'mismatch') {
    write(t(lang, 'mock.single', { threshold: single, candidate: chosen, cosine: (naive.metric as number).toFixed(4) }));
  }

  const loadsBefore = loader.stats.loads.length;
  const second = await selectRuntime(options);
  write(t(lang, 'mock.cache', { loads: loader.stats.loads.length - loadsBefore, fromCache: String(second.report.fromCache) }));
  write('');
  write(t(lang, 'mock.footer'));
  write(t(lang, 'mock.runtime'));
  await first.runner.dispose();
  await second.runner.dispose();
  return { chosen, report: first.report, second: second.report };
}
