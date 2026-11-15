import type { Report } from './types.js';

/** Maximum length of an attempt reason. */
export const MAX_REASON = 200;

/** Thrown before any model is loaded when the options cannot work. The message says how to fix it. */
export class CanaryConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CanaryConfigError';
  }
}

/** Thrown when every candidate failed. `report` has the per-candidate evidence. */
export class NoCandidatePassedError extends Error {
  readonly report: Report;
  constructor(report: Report) {
    super(
      `No candidate passed the output check (${report.attempts.length} attempts). ` +
        'Inspect error.report.attempts, add a candidate that works on every device (e.g. wasm/q8), or relax thresholds.',
    );
    this.name = 'NoCandidatePassedError';
    this.report = report;
  }
}

/** Thrown when the selection was cancelled through the AbortSignal. */
export class CanaryAbortedError extends Error {
  constructor() {
    super('The runtime selection was aborted.');
    this.name = 'CanaryAbortedError';
  }
}

/** Collapses whitespace and truncates a message to {@link MAX_REASON} characters. */
export function shortReason(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  const flat = raw.replace(/\s+/g, ' ').trim();
  return flat.length > MAX_REASON ? flat.slice(0, MAX_REASON - 1) + '…' : flat;
}
