import { cacheKey } from './cache-key.js';
import { detectEnv } from './env.js';
import { CanaryAbortedError, CanaryConfigError, NoCandidatePassedError, shortReason } from './errors.js';
import { judgeEmbedding, judgeGeneration, type Verdict } from './judge.js';
import { memoryStore } from './store.js';
import { resolveThreshold } from './thresholds.js';
import type {
  Attempt,
  Baseline,
  Candidate,
  DecisionStore,
  DtypeThreshold,
  EnvInfo,
  Loader,
  Report,
  Runner,
  Task,
} from './types.js';

export interface SelectOptions {
  modelId: string;
  /** Pin the exact model revision (commit sha). It is part of the cache key. */
  revision: string;
  task: Task;
  /** Candidates in order of preference. The first one that passes wins. */
  candidates: Candidate[];
  baseline: Baseline;
  loader: Loader;
  /** Where the decision is remembered. Default: in-memory. */
  store?: DecisionStore;
  /** Environment, or a function returning it. Default: {@link detectEnv}. */
  env?: EnvInfo | (() => Promise<EnvInfo>);
  /** dtypes that exist for the model (e.g. from `ModelRegistry.get_available_dtypes`). */
  availableDtypes?: string[];
  /** Per-dtype threshold overrides. */
  thresholds?: Partial<Record<string, Partial<DtypeThreshold>>>;
  /** Limit for load + check run per candidate. Default 120000. */
  timeoutMs?: number;
  /** After a timeout, how long to wait for the abandoned load to settle before the next candidate. Default 5000. */
  graceMs?: number;
  /** Library version for the cache key. Default: the baseline's. */
  libVersion?: string;
  onAttempt?: (attempt: Attempt) => void;
  signal?: AbortSignal;
}

export interface Selection {
  runner: Runner;
  candidate: Candidate;
  report: Report;
}

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_GRACE_MS = 5_000;

const F16_DTYPES = new Set(['fp16', 'q4f16']);

function dtypeNames(c: Candidate): string[] {
  return typeof c.dtype === 'string' ? [c.dtype] : Object.values(c.dtype);
}

function sameCandidate(a: Candidate, b: Candidate): boolean {
  return JSON.stringify([a.device, a.dtype]) === JSON.stringify([b.device, b.dtype]);
}

/** Why a candidate cannot work in this environment, or undefined if it may be tried. */
function skipReason(c: Candidate, env: EnvInfo, available: string[] | undefined): string | undefined {
  if (c.device.startsWith('webgpu')) {
    if (!env.hasWebGpu) return 'no-webgpu';
    if (!env.shaderF16 && dtypeNames(c).some((d) => F16_DTYPES.has(d))) return 'no-shader-f16';
  }
  if (available) {
    const missing = dtypeNames(c).find((d) => !available.includes(d));
    if (missing) return `dtype-not-available:${missing}`;
  }
  return undefined;
}

function validate(o: SelectOptions): void {
  if (o.candidates.length === 0) {
    throw new CanaryConfigError('candidates is empty. Pass at least one { device, dtype }, e.g. { device: "wasm", dtype: "q8" }.');
  }
  const b = o.baseline;
  if (b.schema !== 1) throw new CanaryConfigError(`Unsupported baseline schema ${String(b.schema)}. Regenerate it with "webgpu-canary baseline".`);
  if (b.task !== o.task) {
    throw new CanaryConfigError(`The baseline is for task "${b.task}" but task "${o.task}" was requested. Generate a baseline for "${o.task}".`);
  }
  if (b.modelId !== o.modelId) {
    throw new CanaryConfigError(`The baseline was made for "${b.modelId}", not "${o.modelId}". Regenerate it for this model.`);
  }
  if (b.revision !== o.revision) {
    throw new CanaryConfigError(`The baseline revision (${b.revision}) differs from revision ${o.revision}. Regenerate the baseline for this revision.`);
  }
  const data = o.task === 'embeddings' ? b.vector : b.tokens;
  if (!data || data.length === 0 || data.some((x) => !Number.isFinite(x))) {
    throw new CanaryConfigError(`The baseline has no usable ${o.task === 'embeddings' ? 'vector' : 'tokens'}. Regenerate it with "webgpu-canary baseline".`);
  }
  if (!(o.timeoutMs === undefined || o.timeoutMs > 0)) throw new CanaryConfigError('timeoutMs must be a positive number of milliseconds.');
  for (const c of o.candidates) resolveThreshold(c.dtype, o.thresholds); // throws for dtypes without a threshold
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function judge(o: SelectOptions, out: { vector?: ArrayLike<number>; tokens?: number[] }, c: Candidate): Verdict {
  if (o.task === 'embeddings') {
    if (!out.vector) return { status: 'mismatch', reason: 'The runner returned no vector.', reasonCode: 'no-vector' };
    return judgeEmbedding(out.vector, o.baseline.vector as number[], c.dtype, o.thresholds);
  }
  if (!out.tokens) return { status: 'mismatch', reason: 'The runner returned no tokens.', reasonCode: 'no-tokens' };
  return judgeGeneration(out.tokens, o.baseline.tokens as number[], c.dtype, o.thresholds);
}

interface Outcome {
  attempt: Attempt;
  runner?: Runner;
}

/** Loads one candidate, runs the check input, judges it. Disposes the runner unless it passed. */
async function tryCandidate(o: SelectOptions, c: Candidate): Promise<Outcome> {
  if (o.signal?.aborted) throw new CanaryAbortedError();
  const timeoutMs = o.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const ac = new AbortController();
  let runner: Runner | undefined;
  let disposed = false;
  let stop: 'timeout' | 'abort' | undefined;
  const dispose = async (): Promise<void> => {
    if (!runner || disposed) return;
    disposed = true;
    try {
      await runner.dispose();
    } catch {
      /* a failing dispose must not hide the verdict */
    }
  };

  const t0 = Date.now();
  let loadMs: number | undefined;
  let runMs: number | undefined;

  const loadP = o.loader.load(o.modelId, o.revision, c, o.task, ac.signal).then(async (r) => {
    runner = r;
    // The load finished after we gave up on it: free the model right away.
    if (stop) await dispose();
    return r;
  });
  const work = (async () => {
    const r = await loadP;
    loadMs = Date.now() - t0;
    if (stop) throw new Error('abandoned');
    const t1 = Date.now();
    const out = await r.run(o.baseline.input, { maxNewTokens: o.baseline.maxNewTokens });
    runMs = Date.now() - t1;
    return out;
  })();
  const settled = work.then(
    () => undefined,
    () => undefined,
  );

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutP = new Promise<'timeout'>((res) => {
    timer = setTimeout(() => {
      stop = 'timeout';
      ac.abort();
      res('timeout');
    }, timeoutMs);
  });
  let onAbort: (() => void) | undefined;
  const abortP = new Promise<'abort'>((res) => {
    onAbort = () => {
      stop = 'abort';
      ac.abort();
      res('abort');
    };
    o.signal?.addEventListener('abort', onAbort, { once: true });
  });

  try {
    const result = await Promise.race([
      work.then(
        (out) => ({ out }),
        (error: unknown) => ({ error }),
      ),
      timeoutP,
      abortP,
    ]);
    if (result === 'abort') {
      await dispose();
      throw new CanaryAbortedError();
    }
    if (result === 'timeout') {
      await dispose();
      // The abandoned load cannot be cancelled in transformers.js; give it time to finish so loads never overlap.
      await Promise.race([settled, sleep(o.graceMs ?? DEFAULT_GRACE_MS)]);
      return { attempt: { candidate: c, status: 'timeout', reason: `No result within ${timeoutMs} ms.`, reasonCode: 'timeout', reasonParams: { ms: timeoutMs }, loadMs, runMs } };
    }
    if ('error' in result) {
      await dispose();
      return { attempt: { candidate: c, status: 'error', reason: shortReason(result.error), loadMs, runMs } };
    }
    const v = judge(o, result.out, c);
    const attempt: Attempt = {
      candidate: c,
      status: v.status,
      reason: v.reason.length > 200 ? v.reason.slice(0, 199) + '…' : v.reason,
      reasonCode: v.reasonCode,
      reasonParams: v.reasonParams,
      metric: v.metric,
      threshold: v.threshold,
      loadMs,
      runMs,
    };
    if (v.status === 'pass') return { attempt, runner: runner as Runner };
    await dispose();
    return { attempt };
  } finally {
    clearTimeout(timer);
    if (onAbort) o.signal?.removeEventListener('abort', onAbort);
  }
}

/**
 * Finds the first (device, dtype) candidate whose real output agrees with the baseline.
 *
 * Candidates are tried one at a time, in order. A candidate is skipped without loading when the
 * environment cannot support it, and judged `error`, `timeout`, `numeric` or `mismatch` otherwise.
 * The decision is stored under a key made of adapter, browser version, model, revision and library
 * version; later calls re-verify the stored candidate with one check run and fall back to the full
 * search if it no longer passes.
 *
 * @throws CanaryConfigError when options cannot work (before loading anything).
 * @throws NoCandidatePassedError when every candidate failed; `error.report` has the evidence.
 * @throws CanaryAbortedError when `signal` was aborted.
 */
export async function selectRuntime(o: SelectOptions): Promise<Selection> {
  validate(o);
  const store = o.store ?? memoryStore();
  const env = typeof o.env === 'function' ? await o.env() : (o.env ?? (await detectEnv()));
  const key = cacheKey({ env, modelId: o.modelId, revision: o.revision, task: o.task, libVersion: o.libVersion ?? o.baseline.libVersion });
  const attempts: Attempt[] = [];
  const record = (a: Attempt): void => {
    attempts.push(a);
    try {
      o.onAttempt?.(a);
    } catch {
      /* listener errors must not break the search */
    }
  };
  const makeReport = (chosen: Candidate | null, fromCache: boolean): Report => ({
    key,
    chosen,
    fromCache,
    attempts,
    env,
    modelId: o.modelId,
    revision: o.revision,
    task: o.task,
    createdAt: new Date().toISOString(),
  });

  // 1. A remembered decision is verified with one run instead of searching again.
  let stored: Candidate | undefined;
  try {
    stored = (await store.get(key))?.candidate;
  } catch {
    stored = undefined;
  }
  if (stored) {
    const usable = o.candidates.some((c) => sameCandidate(c, stored as Candidate)) && !skipReason(stored, env, o.availableDtypes);
    if (usable) {
      const outcome = await tryCandidate(o, stored);
      record(outcome.attempt);
      if (outcome.runner) return { runner: outcome.runner, candidate: stored, report: makeReport(stored, true) };
    }
    await Promise.resolve(store.delete(key)).catch(() => undefined);
  }

  // 2. Full search, one candidate at a time.
  for (const c of o.candidates) {
    if (stored && sameCandidate(c, stored)) continue; // already failed above
    const why = skipReason(c, env, o.availableDtypes);
    if (why) {
      record({ candidate: c, status: 'skipped', reason: why });
      continue;
    }
    const outcome = await tryCandidate(o, c);
    record(outcome.attempt);
    if (outcome.runner) {
      try {
        await store.set(key, { candidate: c, savedAt: new Date().toISOString() });
      } catch {
        /* an unwritable store only means the decision is not remembered */
      }
      return { runner: outcome.runner, candidate: c, report: makeReport(c, false) };
    }
  }
  throw new NoCandidatePassedError(makeReport(null, false));
}
