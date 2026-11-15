// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useCanaryRuntime } from '../src/react/index.js';
import { fakeLoader } from '../src/testing/index.js';
import type { Candidate } from '../src/types.js';
import { MODEL, REVISION, V, embeddingBaseline, makeEnv } from './helpers.js';

afterEach(cleanup);

const c = (device: string, dtype: string): Candidate => ({ device, dtype });
const options = (loader: ReturnType<typeof fakeLoader>, candidates: Candidate[], extra: object = {}) => ({
  modelId: MODEL,
  revision: REVISION,
  task: 'embeddings' as const,
  baseline: embeddingBaseline(),
  env: makeEnv(),
  graceMs: 20,
  loader,
  candidates,
  ...extra,
});

describe('useCanaryRuntime', () => {
  it('goes idle -> probing -> ready and fills attempts while probing', async () => {
    const loader = fakeLoader({
      'webgpu/fp16': { kind: 'delay', ms: 30, then: { kind: 'nan', dim: 384 } },
      'wasm/q8': { kind: 'delay', ms: 30, then: { kind: 'vector', vector: [...V.q8] } },
    });
    const opts = options(loader, [c('webgpu', 'fp16'), c('wasm', 'q8')]);
    const { result } = renderHook(() => useCanaryRuntime(opts));
    // the first render happens before the effect has started
    expect(['idle', 'probing']).toContain(result.current.status);
    await waitFor(() => expect(result.current.status).toBe('probing'));
    await waitFor(() => expect(result.current.attempts.map((a) => a.status)).toEqual(['numeric']));
    expect(result.current.status).toBe('probing'); // second candidate still loading: attempts update live
    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(result.current.candidate).toEqual(c('wasm', 'q8'));
    expect(result.current.attempts.map((a) => a.status)).toEqual(['numeric', 'pass']);
    expect(result.current.report?.chosen).toEqual(c('wasm', 'q8'));
    expect(result.current.runner).toBeDefined();
  });

  it('becomes failed with the report when every candidate fails', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'error', message: 'boom' } });
    const { result } = renderHook(() => useCanaryRuntime(options(loader, [c('wasm', 'q8')])));
    await waitFor(() => expect(result.current.status).toBe('failed'));
    expect(result.current.report?.chosen).toBeNull();
    expect(result.current.report?.attempts[0]?.status).toBe('error');
    expect(result.current.error?.name).toBe('NoCandidatePassedError');
  });

  it('stays idle when disabled', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'vector', vector: [...V.q8] } });
    const { result } = renderHook(() => useCanaryRuntime(options(loader, [c('wasm', 'q8')], { enabled: false })));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 30));
    });
    expect(result.current.status).toBe('idle');
    expect(loader.stats.loads).toEqual([]);
  });

  it('on unmount during probing disposes the loaded runner and sets no state afterwards', async () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const loader = fakeLoader({ 'wasm/q8': { kind: 'delay', ms: 60, then: { kind: 'vector', vector: [...V.q8] } } });
    const { result, unmount } = renderHook(() => useCanaryRuntime(options(loader, [c('wasm', 'q8')])));
    await waitFor(() => expect(loader.stats.loads).toEqual(['wasm/q8'])); // load started
    unmount();
    await new Promise((r) => setTimeout(r, 150)); // the slow load finishes after unmount
    expect(loader.stats.disposed['wasm/q8']).toBe(1);
    expect(result.current.status).toBe('probing'); // never moved to ready
    expect(errors).not.toHaveBeenCalled(); // no "state update on unmounted component" warnings
    errors.mockRestore();
  });

  it('disposes a ready runner on unmount', async () => {
    const loader = fakeLoader({ 'wasm/q8': { kind: 'vector', vector: [...V.q8] } });
    const { result, unmount } = renderHook(() => useCanaryRuntime(options(loader, [c('wasm', 'q8')])));
    await waitFor(() => expect(result.current.status).toBe('ready'));
    unmount();
    await new Promise((r) => setTimeout(r, 10));
    expect(loader.stats.disposed['wasm/q8']).toBe(1);
  });
});
