import type { Candidate, Loader, RunOptions, RunOutput, Runner, Task } from '../types.js';
import { cosine } from '../judge.js';

export { fixtures } from './fixtures-data.js';
export { runMockDemo } from './mock-demo.js';

/** How a fake candidate behaves. */
export type Behavior =
  /** Returns this embedding. */
  | { kind: 'vector'; vector: number[] }
  /** Returns these generated token ids. */
  | { kind: 'tokens'; tokens: number[] }
  /** `load()` rejects with this message (like a WebGPU adapter or shader failure). */
  | { kind: 'error'; message: string }
  /** Returns an embedding full of NaN (fp16 overflow). */
  | { kind: 'nan'; dim?: number }
  /** Returns the zero vector. */
  | { kind: 'zero'; dim?: number }
  /** Returns the same token `count` times (the silent "repeat forever" failure). */
  | { kind: 'repeat'; token: number; count: number }
  /** Returns `vector` rotated so its cosine to the original is `cosine`. */
  | { kind: 'drift'; vector: number[]; cosine: number }
  /** Takes `ms` to load (it does not honour the abort signal, like a real ONNX session load), then behaves as `then`. */
  | { kind: 'delay'; ms: number; then: Behavior };

export interface FakeStats {
  /** Candidate names in load order. */
  loads: string[];
  /** Number of `dispose()` calls per candidate name. */
  disposed: Record<string, number>;
  /** Number of `load()` calls running at the same time, at most. */
  maxConcurrent: number;
}

export interface FakeLoader extends Loader {
  stats: FakeStats;
}

function name(c: Candidate): string {
  return `${c.device}/${typeof c.dtype === 'string' ? c.dtype : Object.values(c.dtype).join('+')}`;
}

/** A vector whose cosine to `base` is `target` (deterministic). */
function drifted(base: number[], target: number): number[] {
  const n = base.length;
  const norm = Math.sqrt(base.reduce((s, x) => s + x * x, 0));
  const u = base.map((x) => x / norm);
  // Take a shifted copy and remove its component along u to get an orthogonal direction.
  const shifted = u.map((_, i) => u[(i + 1) % n] as number);
  const dot = shifted.reduce((s, x, i) => s + x * (u[i] as number), 0);
  const w = shifted.map((x, i) => x - dot * (u[i] as number));
  const wn = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
  const k = Math.sqrt(1 - target * target);
  return u.map((x, i) => target * x + (k * (w[i] as number)) / wn);
}

function produce(b: Behavior, task: Task): RunOutput {
  switch (b.kind) {
    case 'vector':
      return { vector: b.vector };
    case 'tokens':
      return { tokens: b.tokens };
    case 'nan':
      return { vector: new Array<number>(b.dim ?? 384).fill(NaN) };
    case 'zero':
      return { vector: new Array<number>(b.dim ?? 384).fill(0) };
    case 'repeat':
      return { tokens: new Array<number>(b.count).fill(b.token) };
    case 'drift':
      return { vector: drifted(b.vector, b.cosine) };
    case 'delay':
      return produce(b.then, task);
    case 'error':
      throw new Error(b.message);
  }
}

/**
 * A loader that never touches a model. `plan` maps `"device/dtype"` (for example `"webgpu/fp16"`)
 * to a {@link Behavior}. Candidates missing from the plan fail to load. `stats` records loads,
 * dispose calls and the maximum number of overlapping loads.
 */
export function fakeLoader(plan: Record<string, Behavior>): FakeLoader {
  const stats: FakeStats = { loads: [], disposed: {}, maxConcurrent: 0 };
  let active = 0;
  return {
    stats,
    async load(_modelId, _revision, c, task): Promise<Runner> {
      const key = name(c);
      stats.loads.push(key);
      active++;
      stats.maxConcurrent = Math.max(stats.maxConcurrent, active);
      try {
        let b = plan[key];
        if (!b) throw new Error(`fakeLoader: no behavior planned for ${key}`);
        if (b.kind === 'delay') {
          await new Promise((r) => setTimeout(r, (b as { ms: number }).ms));
          b = (b as { then: Behavior }).then;
        }
        if (b.kind === 'error') throw new Error(b.message);
        const behavior = b;
        return {
          async run(_input: string, _opts: RunOptions): Promise<RunOutput> {
            return produce(behavior, task);
          },
          dispose() {
            stats.disposed[key] = (stats.disposed[key] ?? 0) + 1;
          },
        };
      } finally {
        active--;
      }
    },
  };
}

export { cosine };
