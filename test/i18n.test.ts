import { describe, expect, it } from 'vitest';
import { main } from '../src/cli/main.js';
import { messages, resolveLang, t, type MessageKey } from '../src/i18n/messages.js';
import { runMockDemo } from '../src/testing/mock-demo.js';
import { LANG_STORAGE_KEY, initialLang, saveLang } from '../demo/src/i18n.js';

describe('message tables', () => {
  it('en and ko have exactly the same keys', () => {
    expect(Object.keys(messages.ko).sort()).toEqual(Object.keys(messages.en).sort());
  });

  it('the CLI keys and the demo page keys exist in both languages and are non-empty', () => {
    const keys = Object.keys(messages.en) as MessageKey[];
    expect(keys.some((k) => k.startsWith('cli.'))).toBe(true);
    expect(keys.some((k) => k.startsWith('demo.'))).toBe(true);
    for (const k of keys) {
      expect(messages.en[k].length).toBeGreaterThan(0);
      expect(messages.ko[k].length).toBeGreaterThan(0);
    }
  });

  it('placeholders are the same in both languages', () => {
    const names = (s: string) => [...new Set([...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]))].sort();
    for (const k of Object.keys(messages.en) as MessageKey[]) {
      expect(names(messages.ko[k]), k).toEqual(names(messages.en[k]));
    }
  });

  it('t() fills placeholders', () => {
    expect(t('en', 'mock.chosen', { candidate: 'wasm/q4' })).toBe('chosen: wasm/q4');
    expect(t('ko', 'mock.chosen', { candidate: 'wasm/q4' })).toBe('선택: wasm/q4');
  });
});

describe('language priority', () => {
  it('flag > WEBGPU_CANARY_LANG > LC_ALL > LC_MESSAGES > LANG > English', () => {
    expect(resolveLang('en', { WEBGPU_CANARY_LANG: 'ko', LANG: 'ko_KR.UTF-8' })).toBe('en');
    expect(resolveLang(undefined, { WEBGPU_CANARY_LANG: 'ko', LC_ALL: 'en_US.UTF-8' })).toBe('ko');
    expect(resolveLang(undefined, { WEBGPU_CANARY_LANG: 'en', LANG: 'ko_KR.UTF-8' })).toBe('en');
    expect(resolveLang(undefined, { LC_ALL: 'ko_KR.UTF-8', LANG: 'en_US.UTF-8' })).toBe('ko');
    expect(resolveLang(undefined, { LC_MESSAGES: 'ko_KR.UTF-8', LANG: 'en_US.UTF-8' })).toBe('ko');
    expect(resolveLang(undefined, { LANG: 'ko_KR.UTF-8' })).toBe('ko');
    expect(resolveLang(undefined, { LANG: 'fr_FR.UTF-8' })).toBe('en');
    expect(resolveLang(undefined, {})).toBe('en');
  });
});

describe('CLI language', () => {
  const help = async (argv: string[], env: Record<string, string>) => {
    const out: string[] = [];
    await main(argv, { stdout: (s) => out.push(s), stderr: () => undefined, env });
    return out.join('');
  };

  it('--lang switches the help text', async () => {
    expect(await help(['--help', '--lang', 'ko'], { LANG: 'en_US.UTF-8' })).toMatch(/사용법/);
    expect(await help(['--help', '--lang', 'en'], { LANG: 'ko_KR.UTF-8' })).toMatch(/Usage/);
    expect(await help(['--help'], { LANG: 'ko_KR.UTF-8' })).toMatch(/사용법/);
    expect(await help(['--help'], { WEBGPU_CANARY_LANG: 'en', LANG: 'ko_KR.UTF-8' })).toMatch(/Usage/);
  });

  it('rejects an unknown language with exit code 2', async () => {
    const err: string[] = [];
    const code = await main(['--help', '--lang', 'fr'], { stdout: () => undefined, stderr: (s) => err.push(s), env: {} });
    expect(code).toBe(2);
    expect(err.join('')).toMatch(/Unknown language "fr"/);
  });

  it('error messages follow the language', async () => {
    const err: string[] = [];
    await main(['baseline', 'm', '--revision', 'x', '--lang', 'ko'], { stdout: () => undefined, stderr: (s) => err.push(s), env: {} });
    expect(err.join('')).toMatch(/--task 가 없습니다/);
  });
});

describe('mock example language', () => {
  const capture = async (lang: 'en' | 'ko') => {
    const lines: string[] = [];
    await runMockDemo({ lang, write: (l) => lines.push(l) });
    return lines.join('\n');
  };

  it('prints English and Korean output', async () => {
    const en = await capture('en');
    const ko = await capture('ko');
    expect(en).toMatch(/MOCK MODE \(practice\)/);
    expect(en).toMatch(/chosen: wasm\/q4/);
    expect(ko).toMatch(/목\(mock\) 모드\(연습용\)/);
    expect(ko).toMatch(/선택: wasm\/q4/);
    expect(ko).not.toMatch(/MOCK MODE/);
  });
});

describe('demo language choice', () => {
  const memory = () => {
    const data = new Map<string, string>();
    return { data, getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) };
  };

  it('is stored in localStorage and wins over the browser language afterwards', () => {
    const storage = memory();
    expect(initialLang(storage, '', 'ko-KR')).toBe('ko'); // first visit: browser language
    expect(initialLang(storage, '', 'en-US')).toBe('en');
    saveLang(storage, 'ko');
    expect(storage.data.get(LANG_STORAGE_KEY)).toBe('ko');
    expect(initialLang(storage, '', 'en-US')).toBe('ko'); // saved choice beats the browser
  });

  it('?lang= in the address wins (for screenshots), junk is ignored', () => {
    const storage = memory();
    saveLang(storage, 'ko');
    expect(initialLang(storage, '?lang=en', 'ko-KR')).toBe('en');
    expect(initialLang(storage, '?lang=xx', 'en-US')).toBe('ko');
  });

  it('survives blocked storage', () => {
    const blocked = {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    };
    expect(initialLang(blocked, '', 'ko')).toBe('ko');
    expect(() => saveLang(blocked, 'en')).not.toThrow();
  });
});

describe('translated reasons', () => {
  it('the ko mock output has no English reason sentences', async () => {
    const lines: string[] = [];
    await runMockDemo({ lang: 'ko', write: (l) => lines.push(l) });
    const text = lines.join('\n');
    expect(text).not.toMatch(/Non-finite|is below|would reject|Failed to get|No result within|\[injected\]/);
    expect(text).toContain('유한하지 않은 값');
    expect(text).toContain('미만입니다');
  });

  it('the en mock output keeps the English reasons', async () => {
    const lines: string[] = [];
    await runMockDemo({ lang: 'en', write: (l) => lines.push(l) });
    expect(lines.join('\n')).toMatch(/Non-finite value at index 0/);
  });

  it('every reason code the judge emits has a message in both languages', () => {
    for (const code of ['non-finite', 'empty', 'length', 'cosine-below', 'cosine-ok', 'bad-tokens', 'repeat', 'prefix-short', 'prefix-ok', 'no-vector', 'no-tokens', 'timeout']) {
      expect(`reason.${code}` in messages.ko && `reason.${code}` in messages.en, code).toBe(true);
    }
  });
});
