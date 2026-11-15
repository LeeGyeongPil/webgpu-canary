import { resolveThreshold } from './thresholds.js';
import type { Dtype, DtypeThreshold, Status } from './types.js';

export interface Verdict {
  status: Extract<Status, 'pass' | 'numeric' | 'mismatch'>;
  reason: string;
  /** Stable code for translating the reason, see `reason.*` in the message table. */
  reasonCode: string;
  reasonParams?: Record<string, string | number>;
  metric?: number;
  threshold?: number;
}

type Overrides = Partial<Record<string, Partial<DtypeThreshold>>>;

/** Cosine similarity of two equal-length vectors. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / Math.sqrt(na * nb);
}

/** Longest run of one repeated value. */
export function maxRun(tokens: readonly number[]): number {
  let best = tokens.length > 0 ? 1 : 0;
  let cur = 1;
  for (let i = 1; i < tokens.length; i++) {
    cur = tokens[i] === tokens[i - 1] ? cur + 1 : 1;
    if (cur > best) best = cur;
  }
  return best;
}

/** Number of leading tokens that are equal. */
export function commonPrefix(a: readonly number[], b: readonly number[]): number {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

/**
 * Judges an embedding against the baseline with the dtype's cosine threshold.
 * NaN, Infinity and the zero vector are `numeric`; low similarity is `mismatch`.
 */
export function judgeEmbedding(
  vec: ArrayLike<number>,
  base: readonly number[],
  dtype: Dtype | Record<string, Dtype>,
  overrides: Overrides = {},
): Verdict {
  const { minCosine } = resolveThreshold(dtype, overrides);
  let allZero = true;
  for (let i = 0; i < vec.length; i++) {
    const x = vec[i] as number;
    if (!Number.isFinite(x)) {
      return { status: 'numeric', reason: `Non-finite value at index ${i} (${x}).`, reasonCode: 'non-finite', reasonParams: { index: i, value: String(x) }, threshold: minCosine };
    }
    if (x !== 0) allZero = false;
  }
  if (vec.length === 0 || allZero) {
    return { status: 'numeric', reason: 'Empty or all-zero embedding.', reasonCode: 'empty', threshold: minCosine };
  }
  if (vec.length !== base.length) {
    return {
      status: 'mismatch',
      reason: `Vector length ${vec.length} differs from the baseline length ${base.length}.`,
      reasonCode: 'length', reasonParams: { got: vec.length, expected: base.length },
      threshold: minCosine,
    };
  }
  const metric = cosine(vec, base);
  if (metric < minCosine) {
    return { status: 'mismatch', reason: `Cosine ${metric.toFixed(4)} is below ${minCosine}.`, reasonCode: 'cosine-below', reasonParams: { value: metric.toFixed(4), min: minCosine }, metric, threshold: minCosine };
  }
  return { status: 'pass', reason: `Cosine ${metric.toFixed(4)} >= ${minCosine}.`, reasonCode: 'cosine-ok', reasonParams: { value: metric.toFixed(4), min: minCosine }, metric, threshold: minCosine };
}

/**
 * Judges greedy-generated token ids against the baseline tokens.
 * A run of one token at least max(4, longest baseline run + 1) long is `numeric` (the silent
 * "same token forever" failure). Otherwise the leading tokens must match; the required count is
 * capped by the baseline length so a legitimately short answer (ended by EOS) passes.
 */
export function judgeGeneration(
  tokens: readonly number[],
  base: readonly number[],
  dtype: Dtype | Record<string, Dtype>,
  overrides: Overrides = {},
): Verdict {
  const { minPrefix } = resolveThreshold(dtype, overrides);
  const required = Math.min(minPrefix, base.length);
  if (tokens.some((t) => !Number.isInteger(t) || t < 0)) {
    return { status: 'numeric', reason: 'Token ids contain non-integer or negative values.', reasonCode: 'bad-tokens', threshold: required };
  }
  const run = maxRun(tokens);
  const runLimit = Math.max(4, maxRun(base) + 1);
  if (run >= runLimit) {
    return { status: 'numeric', reason: `One token repeated ${run} times in a row (limit ${runLimit}).`, reasonCode: 'repeat', reasonParams: { run, limit: runLimit }, threshold: required };
  }
  const metric = commonPrefix(tokens, base);
  if (metric < required) {
    return {
      status: 'mismatch',
      reason: `Only ${metric} leading tokens match the baseline, ${required} required.`,
      reasonCode: 'prefix-short', reasonParams: { got: metric, required },
      metric,
      threshold: required,
    };
  }
  return { status: 'pass', reason: `${metric} leading tokens match (>= ${required}).`, reasonCode: 'prefix-ok', reasonParams: { got: metric, required }, metric, threshold: required };
}
