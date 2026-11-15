import { messages, t, type Lang, type MessageKey } from './i18n/messages.js';
import type { Attempt, Candidate } from './types.js';

/** Display name of a candidate, e.g. `webgpu/q4f16`. */
export function candidateName(c: Candidate): string {
  const dtype = typeof c.dtype === 'string' ? c.dtype : Object.values(c.dtype).join('+');
  return `${c.device}/${dtype}`;
}

/** Terminal width of a string (CJK characters take two cells). */
function width(text: string): number {
  let w = 0;
  for (const ch of text) w += (ch.codePointAt(0) as number) >= 0x1100 ? 2 : 1;
  return w;
}

function pad(text: string, to: number): string {
  return text + ' '.repeat(Math.max(0, to - width(text)));
}

function num(v: number | undefined): string {
  if (v === undefined) return '-';
  return Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}

/** The reason of an attempt in the given language (falls back to the English `reason`). */
export function reasonText(a: Attempt, lang: Lang): string {
  if (a.status === 'skipped') {
    const [code, arg] = a.reason.split(':');
    const key = `reason.skip.${code}`;
    return key in messages[lang] ? t(lang, key as MessageKey, { dtype: arg ?? '' }) : a.reason;
  }
  if (a.status === 'error') return t(lang, 'reason.error', { detail: a.reason });
  const key = `reason.${a.reasonCode ?? ''}`;
  if (a.reasonCode && key in messages[lang]) return t(lang, key as MessageKey, a.reasonParams ?? {});
  return a.reason;
}

function clip(text: string, max: number): string {
  return text.length > max ? text.slice(0, max - 1) + '…' : text;
}

/** Renders the attempts as a plain-text table. */
export function formatAttempts(attempts: readonly Attempt[], lang: Lang, reasonWidth = 70): string[] {
  const rows: string[][] = [
    [t(lang, 'col.index'), t(lang, 'col.candidate'), t(lang, 'col.status'), t(lang, 'col.metric'), t(lang, 'col.min'), t(lang, 'col.reason')],
    ...attempts.map((a, i) => [
      String(i + 1),
      candidateName(a.candidate),
      a.status,
      num(a.metric),
      num(a.threshold),
      clip(reasonText(a, lang), reasonWidth),
    ]),
  ];
  const widths = rows[0]!.map((_, col) => Math.max(...rows.map((r) => width(r[col] as string))));
  return rows.map((r) => r.map((cell, col) => (col === r.length - 1 ? cell : pad(cell, (widths[col] as number) + 2))).join('').trimEnd());
}
