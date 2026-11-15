import { describe, expect, it } from 'vitest';
import { cacheKey, uaMajor } from '../src/cache-key.js';
import { selectRuntime } from '../src/select.js';
import { localStorageStore, memoryStore, type StorageLike } from '../src/store.js';
import { fakeLoader, type Behavior } from '../src/testing/index.js';
import type { Candidate } from '../src/types.js';
import { MODEL, REVISION, V, embeddingBaseline, makeEnv } from './helpers.js';

const c = (device: string, dtype: string): Candidate => ({ device, dtype });
const good: Behavior = { kind: 'vector', vector: [...V.q8] };
const common = { modelId: MODEL, revision: REVISION, task: 'embeddings' as const, baseline: embeddingBaseline(), graceMs: 50 };

describe('environment filtering', () => {
  it('skips webgpu candidates without loading when navigator.gpu is missing', async () => {
    const loader = fakeLoader({ 'wasm/q8': good });
    const { report } = await selectRuntime({
      ...common,
      loader,
      env: makeEnv({ hasWebGpu: false, shaderF16: false }),
      candidates: [c('webgpu', 'fp16'), c('webgpu', 'q4'), c('wasm', 'q8')],
    });
    expect(report.attempts.map((a) => a.status)).toEqual(['skipped', 'skipped', 'pass']);
    expect(report.attempts.slice(0, 2).map((a) => a.reason)).toEqual(['no-webgpu', 'no-webgpu']);
    expect(loader.stats.loads).toEqual(['wasm/q8']);
  });

  it('skips fp16 and q4f16 on webgpu without shader-f16', async () => {
    const loader = fakeLoader({ 'webgpu/q4': { kind: 'vector', vector: [...V.q4] } });
    const { report } = await selectRuntime({
      ...common,
      loader,
      env: makeEnv({ shaderF16: false }),
      candidates: [c('webgpu', 'fp16'), c('webgpu', 'q4f16'), c('webgpu', 'q4')],
    });
    expect(report.attempts.map((a) => a.reason).slice(0, 2)).toEqual(['no-shader-f16', 'no-shader-f16']);
    expect(report.chosen).toEqual(c('webgpu', 'q4'));
    expect(loader.stats.loads).toEqual(['webgpu/q4']);
  });

  it('skips dtypes missing from availableDtypes', async () => {
    const loader = fakeLoader({ 'wasm/q8': good });
    const { report } = await selectRuntime({
      ...common,
      loader,
      env: makeEnv(),
      availableDtypes: ['fp32', 'q8'],
      candidates: [c('wasm', 'q4'), c('wasm', 'q8')],
    });
    expect(report.attempts[0]).toMatchObject({ status: 'skipped', reason: 'dtype-not-available:q4' });
    expect(loader.stats.loads).toEqual(['wasm/q8']);
  });
});

describe('decision cache', () => {
  it('a second call with the same key loads once and reports fromCache', async () => {
    const loader = fakeLoader({ 'webgpu/fp16': { kind: 'nan' }, 'wasm/q8': good });
    const store = memoryStore();
    const opts = { ...common, loader, store, env: makeEnv(), candidates: [c('webgpu', 'fp16'), c('wasm', 'q8')] };
    const first = await selectRuntime(opts);
    expect(first.report.fromCache).toBe(false);
    expect(loader.stats.loads).toEqual(['webgpu/fp16', 'wasm/q8']);
    const second = await selectRuntime(opts);
    expect(second.report.fromCache).toBe(true);
    expect(loader.stats.loads).toEqual(['webgpu/fp16', 'wasm/q8', 'wasm/q8']);
    expect(second.candidate).toEqual(c('wasm', 'q8'));
  });

  it('forgets a stored decision that fails the check run and searches again', async () => {
    const plan: Record<string, Behavior> = { 'webgpu/q8': good, 'wasm/q8': good };
    const loader = fakeLoader(plan);
    const store = memoryStore();
    const opts = { ...common, loader, store, env: makeEnv(), candidates: [c('webgpu', 'q8'), c('wasm', 'q8')] };
    expect((await selectRuntime(opts)).candidate).toEqual(c('webgpu', 'q8'));
    plan['webgpu/q8'] = { kind: 'nan' }; // e.g. a driver update broke it
    const again = await selectRuntime(opts);
    expect(again.report.fromCache).toBe(false);
    expect(again.candidate).toEqual(c('wasm', 'q8'));
    expect(again.report.attempts.map((a) => a.status)).toEqual(['numeric', 'pass']);
    // the new decision replaced the old one
    expect((await selectRuntime(opts)).candidate).toEqual(c('wasm', 'q8'));
  });

  it('ignores a stored candidate that is no longer in the candidate list', async () => {
    const loader = fakeLoader({ 'webgpu/q8': good, 'wasm/q8': good });
    const store = memoryStore();
    const env = makeEnv();
    await selectRuntime({ ...common, loader, store, env, candidates: [c('webgpu', 'q8')] });
    const next = await selectRuntime({ ...common, loader, store, env, candidates: [c('wasm', 'q8')] });
    expect(next.report.fromCache).toBe(false);
    expect(next.candidate).toEqual(c('wasm', 'q8'));
  });

  it('a new revision, library version or browser major version means a new key', async () => {
    const loader = fakeLoader({ 'wasm/q8': good });
    const store = memoryStore();
    const opts = { ...common, loader, store, env: makeEnv(), candidates: [c('wasm', 'q8')], libVersion: '4.3.2' };
    await selectRuntime(opts);
    const sameKey = await selectRuntime(opts);
    expect(sameKey.report.fromCache).toBe(true);
    const newRevision = await selectRuntime({ ...opts, revision: 'abc', baseline: embeddingBaseline({ revision: 'abc' }) });
    expect(newRevision.report.fromCache).toBe(false);
    const newLib = await selectRuntime({ ...opts, libVersion: '4.4.0' });
    expect(newLib.report.fromCache).toBe(false);
    const newUa = await selectRuntime({ ...opts, env: makeEnv({ userAgent: 'Mozilla/5.0 Chrome/132.0.0.0' }) });
    expect(newUa.report.fromCache).toBe(false);
  });
});

describe('cacheKey', () => {
  const parts = { env: makeEnv(), modelId: MODEL, revision: REVISION, task: 'embeddings' as const, libVersion: '4.3.2' };

  it('is stable for empty adapter info and for the same input', () => {
    const empty = makeEnv({ adapter: { vendor: '', architecture: '', device: '', description: '' } });
    expect(cacheKey({ ...parts, env: empty })).toBe(cacheKey({ ...parts, env: structuredClone(empty) }));
  });

  it('changes with revision, libVersion, model, task, adapter and UA major, not UA minor', () => {
    const k = cacheKey(parts);
    expect(cacheKey({ ...parts, revision: 'x' })).not.toBe(k);
    expect(cacheKey({ ...parts, libVersion: '4.3.3' })).not.toBe(k);
    expect(cacheKey({ ...parts, modelId: 'other/model' })).not.toBe(k);
    expect(cacheKey({ ...parts, task: 'text-generation' })).not.toBe(k);
    expect(cacheKey({ ...parts, env: makeEnv({ adapter: { vendor: 'nvidia', architecture: '', device: '', description: '' } }) })).not.toBe(k);
    expect(cacheKey({ ...parts, env: makeEnv({ userAgent: 'Mozilla/5.0 Chrome/132.0.1.0' }) })).not.toBe(k);
    expect(cacheKey({ ...parts, env: makeEnv({ userAgent: 'Mozilla/5.0 Chrome/131.0.9999.1' }) })).toBe(k);
  });

  it('parses UA major versions', () => {
    expect(uaMajor('Mozilla/5.0 Chrome/131.0.6778.86 Safari/537.36')).toBe('chrome/131');
    expect(uaMajor('Mozilla/5.0 Chrome/131.0 Edg/130.0.1')).toBe('edg/130');
    expect(uaMajor('Mozilla/5.0 Version/26.1 Safari/605')).toBe('version/26');
    expect(uaMajor('curl/8')).toBe('');
  });
});

const fakeStorage = (): StorageLike & { data: Map<string, string> } => {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (k) => data.get(k) ?? null,
    setItem: (k, v) => void data.set(k, v),
    removeItem: (k) => void data.delete(k),
    get length() {
      return data.size;
    },
    key: (i) => [...data.keys()][i] ?? null,
  };
};

describe('stores', () => {
  const decision = { candidate: c('wasm', 'q8'), savedAt: '2026-11-16' };

  it('localStorageStore round-trips, clears only its own prefix and survives broken JSON', () => {
    const storage = fakeStorage();
    const store = localStorageStore('wc:', storage);
    store.set('k', decision);
    storage.setItem('other', 'keep');
    expect(store.get('k')).toEqual(decision);
    storage.setItem('wc:broken', '{not json');
    expect(store.get('broken')).toBeUndefined();
    expect(storage.data.has('wc:broken')).toBe(false);
    storage.setItem('wc:odd', JSON.stringify({ hello: 1 }));
    expect(store.get('odd')).toBeUndefined();
    store.clear();
    expect(storage.data.has('wc:k')).toBe(false);
    expect(storage.data.get('other')).toBe('keep');
  });

  it('localStorageStore never throws when storage fails', () => {
    const denied = (): never => {
      throw new Error('denied');
    };
    const broken: StorageLike = { getItem: denied, setItem: denied, removeItem: denied, length: 0, key: () => null };
    const store = localStorageStore('wc:', broken);
    expect(() => store.set('k', decision)).not.toThrow();
    expect(store.get('k')).toBeUndefined();
    expect(() => store.clear()).not.toThrow();
  });
});
