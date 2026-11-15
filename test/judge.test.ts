import { describe, expect, it } from 'vitest';
import { cosine, judgeEmbedding, judgeGeneration } from '../src/judge.js';
import { CanaryConfigError } from '../src/errors.js';
import { fixtures } from '../src/testing/fixtures-data.js';
import { V } from './helpers.js';

const base = [...V.fp32];

/** A vector with the requested cosine to the baseline (built independently of the library's helper). */
const drifted = (target: number): number[] => {
  const n = base.length;
  const norm = Math.hypot(...base);
  const u = base.map((x) => x / norm);
  const shifted = u.map((_, i) => u[(i + 1) % n]!);
  const dot = shifted.reduce((s, x, i) => s + x * u[i]!, 0);
  const w = shifted.map((x, i) => x - dot * u[i]!);
  const wn = Math.hypot(...w);
  const k = Math.sqrt(1 - target * target);
  return u.map((x, i) => target * x + (k * w[i]!) / wn);
};

describe('embedding judge uses per-dtype thresholds', () => {
  it('recorded cosines match the measured values', () => {
    expect(cosine(V.q4, base)).toBeCloseTo(0.9729, 3);
    expect(cosine(V.q8, base)).toBeCloseTo(0.9906, 3);
    expect(cosine(V.fp16, base)).toBeGreaterThan(0.99999);
  });

  it('a real q4 vector passes as q4', () => {
    const v = judgeEmbedding(V.q4, base, 'q4');
    expect(v.status).toBe('pass');
    expect(v.threshold).toBe(0.95);
  });

  it('the same q4 vector is a mismatch when judged as fp16', () => {
    expect(judgeEmbedding(V.q4, base, 'fp16').status).toBe('mismatch');
  });

  it('a single 0.99 threshold would reject the healthy q4 vector', () => {
    const v = judgeEmbedding(V.q4, base, 'q4', { q4: { minCosine: 0.99, minPrefix: 3 } });
    expect(v.status).toBe('mismatch');
  });

  it('real q8 and fp16 vectors pass', () => {
    expect(judgeEmbedding(V.q8, base, 'q8').status).toBe('pass');
    expect(judgeEmbedding(V.fp16, base, 'fp16').status).toBe('pass');
    expect(judgeEmbedding(V.q4f16, base, 'q4f16').status).toBe('pass');
  });

  it('NaN, Infinity and zero vectors are numeric', () => {
    expect(judgeEmbedding(new Array(384).fill(NaN), base, 'fp16').status).toBe('numeric');
    expect(judgeEmbedding(base.map((x, i) => (i === 3 ? Infinity : x)), base, 'fp16').status).toBe('numeric');
    expect(judgeEmbedding(new Array(384).fill(0), base, 'fp16').status).toBe('numeric');
    expect(judgeEmbedding([], base, 'fp16').status).toBe('numeric');
  });

  it('a drifted vector (cosine 0.41) and a wrong length are mismatches', () => {
    const drift = judgeEmbedding(drifted(0.41), base, 'q8');
    expect(drift.status).toBe('mismatch');
    expect(drift.metric).toBeCloseTo(0.41, 2);
    expect(judgeEmbedding([1, 2, 3], base, 'q8').status).toBe('mismatch');
  });

  it('a dtype without defaults throws CanaryConfigError that says how to fix it', () => {
    expect(() => judgeEmbedding(V.q4, base, 'q2')).toThrow(CanaryConfigError);
    expect(() => judgeEmbedding(V.q4, base, 'q2')).toThrow(/thresholds/);
    // ...and works once a threshold is given
    expect(judgeEmbedding(V.q4, base, 'q2', { q2: { minCosine: 0.9, minPrefix: 2 } }).status).toBe('pass');
  });

  it('per-module dtype objects use the loosest threshold', () => {
    const v = judgeEmbedding(V.q4, base, { encoder: 'fp16', decoder: 'q4' });
    expect(v.threshold).toBe(0.95);
  });
});

describe('generation judge', () => {
  const tokens = [...fixtures.smollm2.tokens.fp32];

  it('the same token 12 times is numeric', () => {
    const v = judgeGeneration(new Array(12).fill(262200), tokens, 'fp16');
    expect(v.status).toBe('numeric');
    expect(v.reason).toMatch(/12 times/);
  });

  it('real SmolLM2 greedy tokens pass for every recorded dtype', () => {
    for (const dtype of ['fp32', 'fp16', 'q4', 'q4f16'] as const) {
      expect(judgeGeneration(fixtures.smollm2.tokens[dtype], tokens, dtype).status).toBe('pass');
    }
  });

  it('only the first 2 tokens matching is a mismatch', () => {
    const v = judgeGeneration([tokens[0]!, tokens[1]!, 1, 2, 3, 4, 5, 6], tokens, 'q4');
    expect(v.status).toBe('mismatch');
    expect(v.metric).toBe(2);
  });

  it('a short, correct answer ended by EOS passes (not confused with repetition)', () => {
    expect(judgeGeneration([9, 2], [9, 2], 'q4').status).toBe('pass');
    expect(judgeGeneration([9, 2], [9, 2], 'fp16').status).toBe('pass');
    // but a short answer that differs is still a mismatch
    expect(judgeGeneration([9, 3], [9, 2], 'q4').status).toBe('mismatch');
  });

  it('a baseline that itself repeats a token is allowed one more repeat', () => {
    const repeating = [7, 7, 7, 5, 2];
    expect(judgeGeneration([7, 7, 7, 5, 2], repeating, 'q4').status).toBe('pass');
    expect(judgeGeneration([7, 7, 7, 7, 7], repeating, 'q4').status).toBe('numeric');
  });

  it('invalid token ids are numeric', () => {
    expect(judgeGeneration([1, -1, 3], [1, 2, 3], 'q4').status).toBe('numeric');
    expect(judgeGeneration([1, 2.5, 3], [1, 2, 3], 'q4').status).toBe('numeric');
  });
});
