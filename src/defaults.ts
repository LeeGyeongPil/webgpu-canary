import type { Task } from './types.js';

/** Fixed, short check inputs (no user text ever goes into baselines or reports). */
export const DEFAULT_INPUTS: Record<Task, string> = {
  embeddings: 'The quick brown fox jumps over the lazy dog.',
  'text-generation': 'What is the capital of France? Answer in one word.',
};

export const DEFAULT_MAX_NEW_TOKENS = 12;
