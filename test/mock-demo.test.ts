import { describe, expect, it } from 'vitest';
import { runMockDemo } from '../src/testing/mock-demo.js';

describe('mock demo (practice mode)', () => {
  it('shows error, numeric, mismatch and pass, picks wasm/q4 and says it is a mock', async () => {
    const lines: string[] = [];
    const t0 = Date.now();
    const result = await runMockDemo({ lang: 'en', write: (l) => lines.push(l) });
    expect(Date.now() - t0).toBeLessThan(5000);
    const text = lines.join('\n');
    expect(result.chosen).toBe('wasm/q4');
    expect(text).toMatch(/MOCK MODE \(practice\)/);
    expect(text).toMatch(/\[injected\] Failed to get a WebGPU adapter/);
    for (const status of ['error', 'numeric', 'mismatch', 'pass']) expect(text).toContain(status);
    expect(text).toMatch(/^chosen: wasm\/q4$/m);
    expect(text).toMatch(/would reject wasm\/q4/);
    expect(text).toMatch(/second call with the same key: 1 model load, fromCache: true/);
  });

  it('with no failures injected the first webgpu candidate wins', async () => {
    const result = await runMockDemo({ lang: 'en', write: () => undefined, faults: { error: false, nan: false, drift: false } });
    expect(result.chosen).toBe('webgpu/q4f16');
  });
});
