import { describe, expect, it } from 'vitest';
import { main, type CliIo } from '../src/cli/main.js';
import { fakeLoader } from '../src/testing/index.js';
import { fixtures } from '../src/testing/fixtures-data.js';

const REV = 'a'.repeat(40);

const run = async (argv: string[], over: Partial<CliIo> = {}) => {
  const out: string[] = [];
  const err: string[] = [];
  const files: Record<string, string> = {};
  const code = await main(argv, {
    stdout: (s) => out.push(s),
    stderr: (s) => err.push(s),
    env: { LANG: 'en_US.UTF-8' },
    writeFile: (p, text) => void (files[p] = text),
    makeLoader: async () => ({
      loader: fakeLoader({
        'cpu/fp32': { kind: 'vector', vector: [...fixtures.minilm.vectors.fp32] },
      }),
      libVersion: '4.3.2',
    }),
    ...over,
  });
  return { code, out: out.join(''), err: err.join(''), files };
};

describe('baseline CLI', () => {
  it('writes an embeddings baseline in schema 1', async () => {
    const r = await run(['baseline', 'Xenova/all-MiniLM-L6-v2', '--task', 'embeddings', '--revision', REV, '--out', 'b.json']);
    expect(r.code).toBe(0);
    const b = JSON.parse(r.files['b.json'] as string);
    expect(b).toMatchObject({ schema: 1, task: 'embeddings', modelId: 'Xenova/all-MiniLM-L6-v2', revision: REV, device: 'cpu', dtype: 'fp32', libVersion: '4.3.2' });
    expect(b.vector).toHaveLength(384);
    expect(b.vector.every(Number.isFinite)).toBe(true);
    expect(b.tokens).toBeUndefined();
    expect(r.out).toMatch(/Wrote b\.json/);
  });

  it('writes a generation baseline with finite integer tokens', async () => {
    const r = await run(['baseline', 'HuggingFaceTB/SmolLM2-135M-Instruct', '--task', 'text-generation', '--revision', REV, '--max-new-tokens', '5'], {
      makeLoader: async () => ({
        loader: fakeLoader({ 'cpu/fp32': { kind: 'tokens', tokens: [...fixtures.smollm2.tokens.fp32] } }),
        libVersion: '4.3.2',
      }),
    });
    expect(r.code).toBe(0);
    const b = JSON.parse(r.out);
    expect(b).toMatchObject({ schema: 1, task: 'text-generation', device: 'cpu', dtype: 'fp32', maxNewTokens: 5 });
    expect(b.tokens.every(Number.isInteger)).toBe(true);
    expect(b.vector).toBeUndefined();
  });

  it('exits 2 with a message when --task is missing, unknown, or --revision is missing', async () => {
    const noTask = await run(['baseline', 'm', '--revision', REV]);
    expect(noTask.code).toBe(2);
    expect(noTask.err).toMatch(/Missing --task/);
    const badTask = await run(['baseline', 'm', '--task', 'classify', '--revision', REV]);
    expect(badTask.code).toBe(2);
    expect(badTask.err).toMatch(/Unknown task "classify"/);
    const noRev = await run(['baseline', 'm', '--task', 'embeddings']);
    expect(noRev.code).toBe(2);
    expect(noRev.err).toMatch(/Missing --revision/);
    expect(noRev.err).toMatch(/--help/);
  });

  it('exits 2 for other wrong usage', async () => {
    expect((await run([])).code).toBe(2);
    expect((await run(['bogus'])).code).toBe(2);
    expect((await run(['baseline', '--task', 'embeddings', '--revision', REV])).err).toMatch(/Missing <modelId>/);
    expect((await run(['baseline', 'm', '--task', 'embeddings', '--revision', REV, '--max-new-tokens', 'abc'])).code).toBe(2);
    expect((await run(['baseline', 'm', '--nope'])).code).toBe(2);
  });

  it('exits 1 when the model fails to load, or returns non-finite values', async () => {
    const load = await run(['baseline', 'm', '--task', 'embeddings', '--revision', REV], {
      makeLoader: async () => ({ loader: fakeLoader({}), libVersion: '4.3.2' }),
    });
    expect(load.code).toBe(1);
    expect(load.err).toMatch(/Could not make the baseline/);
    const nan = await run(['baseline', 'm', '--task', 'embeddings', '--revision', REV], {
      makeLoader: async () => ({ loader: fakeLoader({ 'cpu/fp32': { kind: 'nan' } }), libVersion: '4.3.2' }),
    });
    expect(nan.code).toBe(1);
  });

  it('exits 1 when the output file cannot be written', async () => {
    const r = await run(['baseline', 'm', '--task', 'embeddings', '--revision', REV, '--out', '/nope/x.json'], {
      writeFile: () => {
        throw new Error('EACCES');
      },
    });
    expect(r.code).toBe(1);
    expect(r.err).toMatch(/Could not write/);
  });
});

describe('revision pinning warning', () => {
  it('warns (in the chosen language) when --revision is not a commit sha, and stays quiet for a sha', async () => {
    const branch = await run(['baseline', 'm', '--task', 'embeddings', '--revision', 'main', '--lang', 'ko']);
    expect(branch.err).toMatch(/커밋 sha/);
    const sha = await run(['baseline', 'm', '--task', 'embeddings', '--revision', 'a'.repeat(40), '--lang', 'ko']);
    expect(sha.err).not.toMatch(/커밋 sha/);
  });
});
