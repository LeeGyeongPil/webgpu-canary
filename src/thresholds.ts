import { CanaryConfigError } from './errors.js';
import type { Dtype, DtypeThreshold, Thresholds } from './types.js';

/**
 * Defaults per dtype. They are based on cpu measurements (MiniLM cosine vs fp32:
 * q8 0.9906, fp16 0.999999, q4 0.9729, q4f16 0.9729). fp16 keeps headroom for GPU math.
 * Override any of them through `thresholds`.
 */
export const DEFAULT_THRESHOLDS: Thresholds = {
  fp32: { minCosine: 0.999, minPrefix: 4 },
  fp16: { minCosine: 0.995, minPrefix: 4 },
  q8: { minCosine: 0.98, minPrefix: 3 },
  int8: { minCosine: 0.98, minPrefix: 3 },
  uint8: { minCosine: 0.98, minPrefix: 3 },
  q4: { minCosine: 0.95, minPrefix: 3 },
  bnb4: { minCosine: 0.95, minPrefix: 3 },
  q4f16: { minCosine: 0.95, minPrefix: 3 },
};

function resolveOne(dtype: string, overrides: Partial<Record<string, Partial<DtypeThreshold>>>): DtypeThreshold {
  const merged = { ...DEFAULT_THRESHOLDS[dtype], ...overrides[dtype] };
  if (typeof merged.minCosine !== 'number' || typeof merged.minPrefix !== 'number') {
    throw new CanaryConfigError(
      `No default threshold for dtype "${dtype}". Pass thresholds: { ${dtype}: { minCosine: 0.9, minPrefix: 2 } } ` +
        'after measuring how far this dtype normally drifts from the baseline.',
    );
  }
  return { minCosine: merged.minCosine, minPrefix: merged.minPrefix };
}

/**
 * Threshold for a dtype. For a per-module dtype object the loosest (lowest-precision) value wins.
 * @throws CanaryConfigError when the dtype has no default and no override.
 */
export function resolveThreshold(
  dtype: Dtype | Record<string, Dtype>,
  overrides: Partial<Record<string, Partial<DtypeThreshold>>> = {},
): DtypeThreshold {
  const names = typeof dtype === 'string' ? [dtype] : Object.values(dtype);
  if (names.length === 0) throw new CanaryConfigError('The dtype object is empty. Use a dtype string such as "q4".');
  const all = names.map((n) => resolveOne(n, overrides));
  return {
    minCosine: Math.min(...all.map((a) => a.minCosine)),
    minPrefix: Math.min(...all.map((a) => a.minPrefix)),
  };
}
