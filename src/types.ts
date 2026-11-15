/** What the model is asked to do during the check. */
export type Task = 'embeddings' | 'text-generation';

/** A dtype name such as `fp32`, `fp16`, `q8`, `q4`, `q4f16`. */
export type Dtype = string;

/** One (device, dtype) combination to try. `dtype` may be per-module for multi-file models. */
export interface Candidate {
  device: string;
  dtype: Dtype | Record<string, Dtype>;
}

/** Reference output, produced once (usually at build time) on cpu with fp32. */
export interface Baseline {
  schema: 1;
  task: Task;
  modelId: string;
  revision: string;
  device: 'cpu';
  dtype: 'fp32';
  /** The fixed check input the baseline was produced from. */
  input: string;
  vector?: number[];
  tokens?: number[];
  maxNewTokens?: number;
  libVersion: string;
  createdAt: string;
}

export interface RunOptions {
  maxNewTokens?: number;
}

export interface RunOutput {
  vector?: ArrayLike<number>;
  tokens?: number[];
}

/** A loaded model on one candidate. */
export interface Runner {
  run(input: string, opts: RunOptions): Promise<RunOutput>;
  dispose(): Promise<void> | void;
}

/** Loads a model for a candidate. Implementations: `webgpu-canary/transformers`, `webgpu-canary/testing`. */
export interface Loader {
  load(modelId: string, revision: string, candidate: Candidate, task: Task, signal: AbortSignal): Promise<Runner>;
}

export type Status = 'pass' | 'error' | 'timeout' | 'numeric' | 'mismatch' | 'skipped';

export interface Attempt {
  candidate: Candidate;
  status: Status;
  /** Short explanation, at most 200 characters. For `skipped` it is a code such as `no-webgpu`. */
  reason: string;
  /** Stable code of the reason (for example `cosine-below`); lets screens translate it. */
  reasonCode?: string;
  reasonParams?: Record<string, string | number>;
  /** Cosine (embeddings) or matched prefix length (generation). */
  metric?: number;
  /** The minimum the metric had to reach. */
  threshold?: number;
  loadMs?: number;
  runMs?: number;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface EnvInfo {
  hasWebGpu: boolean;
  shaderF16: boolean;
  adapter: AdapterInfo;
  maxBufferSize: number;
  userAgent: string;
}

export interface Report {
  key: string;
  chosen: Candidate | null;
  fromCache: boolean;
  attempts: Attempt[];
  env: EnvInfo;
  modelId: string;
  revision: string;
  task: Task;
  createdAt: string;
}

/** Minimum agreement with the baseline, per dtype. */
export interface DtypeThreshold {
  /** Embeddings: minimum cosine similarity to the baseline vector. */
  minCosine: number;
  /** Generation: minimum number of leading tokens equal to the baseline (capped by the baseline length). */
  minPrefix: number;
}

export type Thresholds = Record<string, DtypeThreshold>;

export interface StoredDecision {
  candidate: Candidate;
  savedAt: string;
}

export type MaybePromise<T> = T | Promise<T>;

/** Where decisions are remembered between visits. */
export interface DecisionStore {
  get(key: string): MaybePromise<StoredDecision | undefined>;
  set(key: string, value: StoredDecision): MaybePromise<void>;
  delete(key: string): MaybePromise<void>;
  clear(): MaybePromise<void>;
}
