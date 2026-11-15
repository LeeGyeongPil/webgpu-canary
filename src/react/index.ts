import { useEffect, useRef, useState } from 'react';
import { selectRuntime, type SelectOptions } from '../select.js';
import type { Attempt, Candidate, Report, Runner } from '../types.js';

export type CanaryStatus = 'idle' | 'probing' | 'ready' | 'failed';

export interface CanaryState {
  status: CanaryStatus;
  /** Attempts so far, updated while probing. */
  attempts: Attempt[];
  runner?: Runner;
  candidate?: Candidate;
  report?: Report;
  error?: Error;
}

export interface UseCanaryOptions extends Omit<SelectOptions, 'signal' | 'onAttempt'> {
  /** Set to false to stay idle (for example until the user opts in). Default true. */
  enabled?: boolean;
}

/**
 * Runs {@link selectRuntime} once when the component mounts (and again if the model, revision,
 * task or candidate list changes). The hook owns the runner: on unmount it aborts a running
 * search and disposes any runner it loaded, so nothing leaks and no state is set afterwards.
 */
export function useCanaryRuntime(options: UseCanaryOptions): CanaryState {
  const [state, setState] = useState<CanaryState>({ status: 'idle', attempts: [] });
  const latest = useRef(options);
  latest.current = options;
  const enabled = options.enabled ?? true;
  const identity = JSON.stringify([options.modelId, options.revision, options.task, options.candidates]);

  useEffect(() => {
    if (!enabled) {
      setState({ status: 'idle', attempts: [] });
      return;
    }
    const ac = new AbortController();
    let active = true;
    let owned: Runner | undefined;
    setState({ status: 'probing', attempts: [] });
    selectRuntime({
      ...latest.current,
      signal: ac.signal,
      onAttempt: (a) => {
        if (active) setState((s) => ({ ...s, attempts: [...s.attempts, a] }));
      },
    }).then(
      async (sel) => {
        owned = sel.runner;
        if (!active) {
          await sel.runner.dispose();
          return;
        }
        setState((s) => ({ ...s, status: 'ready', runner: sel.runner, candidate: sel.candidate, report: sel.report }));
      },
      (error: unknown) => {
        if (!active) return;
        const err = error instanceof Error ? error : new Error(String(error));
        const report = (err as { report?: Report }).report;
        setState((s) => ({ ...s, status: 'failed', error: err, report }));
      },
    );
    return () => {
      active = false;
      ac.abort();
      if (owned) void Promise.resolve(owned.dispose()).catch(() => undefined);
    };
  }, [enabled, identity]);

  return state;
}
