import type { DecisionStore, StoredDecision } from './types.js';

/** Keeps decisions for the lifetime of the page (or process). */
export function memoryStore(): DecisionStore {
  const map = new Map<string, StoredDecision>();
  return {
    get: (key) => map.get(key),
    set: (key, value) => void map.set(key, value),
    delete: (key) => void map.delete(key),
    clear: () => map.clear(),
  };
}

/** The part of the Web Storage API that is used. */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  readonly length: number;
  key(index: number): string | null;
}

function isDecision(v: unknown): v is StoredDecision {
  if (typeof v !== 'object' || v === null) return false;
  const c = (v as StoredDecision).candidate;
  return typeof c === 'object' && c !== null && typeof c.device === 'string' && c.dtype !== undefined;
}

/**
 * Persists decisions in `localStorage` (or any Web Storage object). Broken JSON or entries of an
 * unknown shape are ignored and removed. Storage failures (private mode, quota) never throw.
 */
export function localStorageStore(prefix = 'webgpu-canary:', storage?: StorageLike): DecisionStore {
  const area = (): StorageLike | undefined => storage ?? (globalThis as { localStorage?: StorageLike }).localStorage;
  return {
    get(key) {
      try {
        const raw = area()?.getItem(prefix + key);
        if (raw == null) return undefined;
        const parsed: unknown = JSON.parse(raw);
        if (isDecision(parsed)) return parsed;
        area()?.removeItem(prefix + key);
      } catch {
        try {
          area()?.removeItem(prefix + key);
        } catch {
          /* ignore */
        }
      }
      return undefined;
    },
    set(key, value) {
      try {
        area()?.setItem(prefix + key, JSON.stringify(value));
      } catch {
        /* storage unavailable: the decision just is not remembered */
      }
    },
    delete(key) {
      try {
        area()?.removeItem(prefix + key);
      } catch {
        /* ignore */
      }
    },
    clear() {
      try {
        const s = area();
        if (!s) return;
        const doomed: string[] = [];
        for (let i = 0; i < s.length; i++) {
          const k = s.key(i);
          if (k?.startsWith(prefix)) doomed.push(k);
        }
        doomed.forEach((k) => s.removeItem(k));
      } catch {
        /* ignore */
      }
    },
  };
}
