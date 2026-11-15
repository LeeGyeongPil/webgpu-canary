import { isLang, type Lang } from '../../src/i18n/messages.js';

/** localStorage key that holds the chosen language ("en" or "ko"). */
export const LANG_STORAGE_KEY = 'webgpu-canary-demo:lang';

interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/**
 * Language at page start: `?lang=ko` in the address (handy for screenshots), else the saved choice,
 * else the browser language (only the very first visit gets here).
 */
export function initialLang(storage: StorageLike | undefined, search: string, navigatorLanguage: string | undefined): Lang {
  const fromQuery = new URLSearchParams(search).get('lang');
  if (fromQuery && isLang(fromQuery)) return fromQuery;
  try {
    const saved = storage?.getItem(LANG_STORAGE_KEY);
    if (saved && isLang(saved)) return saved;
  } catch {
    /* storage blocked: fall through to the browser language */
  }
  return navigatorLanguage?.toLowerCase().startsWith('ko') ? 'ko' : 'en';
}

/** Remembers the choice. Never throws. */
export function saveLang(storage: StorageLike | undefined, lang: Lang): void {
  try {
    storage?.setItem(LANG_STORAGE_KEY, lang);
  } catch {
    /* ignore */
  }
}
