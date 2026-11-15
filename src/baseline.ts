import { CanaryConfigError } from './errors.js';
import { DEFAULT_INPUTS, DEFAULT_MAX_NEW_TOKENS } from './defaults.js';
import type { Baseline, Loader, Task } from './types.js';

export interface MakeBaselineOptions {
  modelId: string;
  revision: string;
  task: Task;
  loader: Loader;
  /** Version of the library that produced the output. */
  libVersion: string;
  input?: string;
  maxNewTokens?: number;
  signal?: AbortSignal;
}

/**
 * Runs the model on cpu with fp32 and returns the reference output as a {@link Baseline}.
 * @throws CanaryConfigError when the model returns nothing usable (empty or non-finite output).
 */
export async function makeBaseline(o: MakeBaselineOptions): Promise<Baseline> {
  const input = o.input ?? DEFAULT_INPUTS[o.task];
  const maxNewTokens = o.task === 'text-generation' ? (o.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS) : undefined;
  const runner = await o.loader.load(o.modelId, o.revision, { device: 'cpu', dtype: 'fp32' }, o.task, o.signal ?? new AbortController().signal);
  try {
    const out = await runner.run(input, { maxNewTokens });
    const base: Baseline = {
      schema: 1,
      task: o.task,
      modelId: o.modelId,
      revision: o.revision,
      device: 'cpu',
      dtype: 'fp32',
      input,
      libVersion: o.libVersion,
      createdAt: new Date().toISOString(),
    };
    if (o.task === 'embeddings') {
      const vector = Array.from(out.vector ?? []);
      if (vector.length === 0 || vector.some((x) => !Number.isFinite(x))) {
        throw new CanaryConfigError('The model returned an empty or non-finite embedding; it cannot be a baseline.');
      }
      base.vector = vector;
    } else {
      const tokens = out.tokens ?? [];
      if (tokens.length === 0) throw new CanaryConfigError('The model generated no tokens; it cannot be a baseline.');
      base.tokens = tokens;
      base.maxNewTokens = maxNewTokens;
    }
    return base;
  } finally {
    await runner.dispose();
  }
}
