import { describe, expect, it } from 'vitest';
import { CanaryConfigError, NoCandidatePassedError } from '../src/errors.js';
import { selectRuntime } from '../src/select.js';
import { memoryStore } from '../src/store.js';
import { fakeLoader } from '../src/testing/index.js';
import { fixtures } from '../src/testing/fixtures-data.js';
import type { Candidate } from '../src/types.js';
import { MODEL, REVISION, V, embeddingBaseline, generationBaseline, makeEnv } from './helpers.js';

const base = {
  modelId: MODEL,
  revision: REVISION,
  task: 'embeddings' as const,
  baseline: embeddingBaseline(),
  env: makeEnv(),
  graceMs: 500,
};

const c = (device: string, dtype: string): Candidate => ({ device, dtype });

describe('sequential fallback', () => {
  it('walks error -> numeric -> timeout -> pass and picks wasm/q8', async () => {
    const loader = fakeLoader({
      'webgpu/q4f16': { kind: 'error', message: 'Failed to get a WebGPU adapter ' + 'x'.repeat(500) },
      'webgpu/fp16': { kind: 'nan', dim: 384 },
      // loads far too slowly: a real ONNX session load cannot be cancelled either
      'webgpu/q4': { kind: 'delay', ms: 120, then: { kind: 'vector', vector: [...V.q4] } },
      'wasm/q8': { kind: 'vector', vector: [...V.q8] },
    });
    const seen: string[] = [];
    const { candidate, report, runner } = await selectRuntime({
      ...base,
      loader,
      timeoutMs: 40,
      candidates: [c('webgpu', 'q4f16'), c('webgpu', 'fp16'), c('webgpu', 'q4'), c('wasm', 'q8')],
      onAttempt: (a) => seen.push(a.status),
    });
    expect(candidate).toEqual(c('wasm', 'q8'));
    expect(report.chosen).toEqual(c('wasm', 'q8'));
    expect(report.attempts.map((a) => a.status)).toEqual(['error', 'numeric', 'timeout', 'pass']);
    expect(seen).toEqual(['error', 'numeric', 'timeout', 'pass']);
    // failed runners are disposed exactly once (the late-finishing slow load too); the winner is not
    expect(loader.stats.disposed['webgpu/fp16']).toBe(1);
    expect(loader.stats.disposed['webgpu/q4']).toBe(1);
    expect(loader.stats.disposed['wasm/q8']).toBeUndefined();
    // loads never overlap, even with the abandoned slow one
    expect(loader.stats.maxConcurrent).toBe(1);
    for (const a of report.attempts) expect(a.reason.length).toBeLessThanOrEqual(200);
    expect(report.attempts[0]!.reason.length).toBeGreaterThan(100);
    await runner.dispose();
  });

  it('a run that hangs after loading also times out and disposes the runner once', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'vector', vector: [...V.q8] } });
    const slow = {
      async load(...args: Parameters<typeof loader.load>) {
        const r = await loader.load(...args);
        return { run: () => new Promise<never>(() => {}), dispose: () => r.dispose() };
      },
    };
    await expect(
      selectRuntime({ ...base, loader: slow, timeoutMs: 30, candidates: [c('wasm', 'q8')] }),
    ).rejects.toBeInstanceOf(NoCandidatePassedError);
    expect(loader.stats.disposed['wasm/q8']).toBe(1);
  });

  it('throws NoCandidatePassedError with the report when everything fails', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'error', message: 'boom' }, 'wasm/q4': { kind: 'zero', dim: 384 } });
    const err = await selectRuntime({ ...base, loader, candidates: [c('wasm', 'q8'), c('wasm', 'q4')] }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NoCandidatePassedError);
    const report = (err as NoCandidatePassedError).report;
    expect(report.chosen).toBeNull();
    expect(report.attempts.map((a) => a.status)).toEqual(['error', 'numeric']);
  });

  it('judges generation with the same loop', async () => {
    const f = fixtures.smollm2;
    const loader = fakeLoader({
      'webgpu/fp16': { kind: 'repeat', token: 262200, count: 12 },
      'wasm/q4': { kind: 'tokens', tokens: [...f.tokens.q4] },
    });
    const { report } = await selectRuntime({
      ...base,
      modelId: f.modelId,
      revision: f.revision,
      task: 'text-generation',
      baseline: generationBaseline(),
      loader,
      candidates: [c('webgpu', 'fp16'), c('wasm', 'q4')],
    });
    expect(report.attempts.map((a) => a.status)).toEqual(['numeric', 'pass']);
  });

  it('rejects unusable options before loading anything', async () => {
    const loader = fakeLoader({});
    const run = (over: object) => selectRuntime({ ...base, loader, candidates: [c('wasm', 'q8')], ...over });
    await expect(run({ candidates: [] })).rejects.toThrow(CanaryConfigError);
    await expect(run({ task: 'text-generation' })).rejects.toThrow(/baseline is for task "embeddings"/);
    await expect(run({ revision: 'other' })).rejects.toThrow(/Regenerate the baseline/);
    await expect(run({ candidates: [c('wasm', 'q2')] })).rejects.toThrow(/thresholds/);
    await expect(run({ timeoutMs: 0 })).rejects.toThrow(/timeoutMs/);
    expect(loader.stats.loads).toEqual([]);
  });

  it('stops when the signal is aborted and disposes what was loaded', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'delay', ms: 60, then: { kind: 'vector', vector: [...V.q8] } } });
    const ac = new AbortController();
    const p = selectRuntime({ ...base, loader, signal: ac.signal, candidates: [c('wasm', 'q8')] });
    setTimeout(() => ac.abort(), 10);
    await expect(p).rejects.toThrow(/aborted/);
    await new Promise((r) => setTimeout(r, 120));
    expect(loader.stats.disposed['wasm/q8']).toBe(1);
  });

  it('keeps going when the store throws', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'vector', vector: [...V.q8] } });
    const store = memoryStore();
    store.set = () => {
      throw new Error('quota');
    };
    const { candidate } = await selectRuntime({ ...base, loader, store, candidates: [c('wasm', 'q8')] });
    expect(candidate.dtype).toBe('q8');
  });
});
