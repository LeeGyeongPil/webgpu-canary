import type { Candidate, Loader, RunOptions, RunOutput, Runner, Task } from '../types.js';

/** The slice of `@huggingface/transformers` this loader uses (keeps the core free of its types). */
interface TransformersModule {
  env: { cacheDir?: string; version?: string };
  pipeline(task: string, modelId: string, options: Record<string, unknown>): Promise<{
    (input: string, opts: Record<string, unknown>): Promise<{ data: ArrayLike<number> }>;
    dispose?: () => Promise<void> | void;
  }>;
  AutoTokenizer: {
    from_pretrained(modelId: string, options: Record<string, unknown>): Promise<
      ((input: string) => { input_ids: { dims: number[] } } & Record<string, unknown>) & {
        chat_template?: unknown;
        apply_chat_template(
          messages: { role: string; content: string }[],
          options: Record<string, unknown>,
        ): { input_ids: { dims: number[] } } & Record<string, unknown>;
      }
    >;
  };
  AutoModelForCausalLM: {
    from_pretrained(modelId: string, options: Record<string, unknown>): Promise<{
      generate(args: Record<string, unknown>): Promise<{ tolist(): ArrayLike<number | bigint>[] }>;
      dispose?: () => Promise<void> | void;
    }>;
  };
}

export interface TransformersLoaderOptions {
  /** Forwarded to transformers.js; useful for download progress. Candidates download different files. */
  progress_callback?: (info: unknown) => void;
  /** Sets `env.cacheDir` (Node only). */
  cacheDir?: string;
  /** Use an already imported module instead of `import('@huggingface/transformers')`. */
  module?: unknown;
}


const DEFAULT_MAX_NEW_TOKENS = 12;

async function loadModule(opts: TransformersLoaderOptions): Promise<TransformersModule> {
  let mod: unknown = opts.module;
  if (!mod) {
    try {
      mod = await import('@huggingface/transformers');
    } catch (cause) {
      throw new Error('Cannot import @huggingface/transformers. Install it: npm install @huggingface/transformers', { cause });
    }
  }
  const tf = mod as TransformersModule;
  if (opts.cacheDir) tf.env.cacheDir = opts.cacheDir;
  return tf;
}

/**
 * Loader for transformers.js v4. `embeddings` uses a feature-extraction pipeline with mean pooling
 * and normalisation; `text-generation` uses greedy decoding (`do_sample: false`) so the tokens are
 * deterministic and comparable to the baseline. The input is wrapped in the model's chat template
 * when it has one.
 */
export function transformersLoader(opts: TransformersLoaderOptions = {}): Loader {
  return {
    async load(modelId: string, revision: string, c: Candidate, task: Task): Promise<Runner> {
      const tf = await loadModule(opts);
      const common = { device: c.device, dtype: c.dtype, revision, progress_callback: opts.progress_callback };
      if (task === 'embeddings') {
        const extractor = await tf.pipeline('feature-extraction', modelId, common);
        return {
          async run(input: string): Promise<RunOutput> {
            const out = await extractor(input, { pooling: 'mean', normalize: true });
            return { vector: Array.from(out.data) };
          },
          async dispose() {
            await extractor.dispose?.();
          },
        };
      }
      const tokenizer = await tf.AutoTokenizer.from_pretrained(modelId, { revision });
      const model = await tf.AutoModelForCausalLM.from_pretrained(modelId, common);
      return {
        async run(input: string, run: RunOptions): Promise<RunOutput> {
          // Instruct models answer reliably only inside their chat template; plain text makes q4 drift on token 1.
          const inputs = tokenizer.chat_template
            ? tokenizer.apply_chat_template([{ role: 'user', content: input }], { add_generation_prompt: true, return_dict: true })
            : tokenizer(input);
          const out = await model.generate({ ...inputs, max_new_tokens: run.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS, do_sample: false });
          const all = Array.from(out.tolist()[0] as ArrayLike<number | bigint>, Number);
          return { tokens: all.slice(inputs.input_ids.dims[1] as number) };
        },
        async dispose() {
          await model.dispose?.();
        },
      };
    },
  };
}

export type { Loader };

/** Version of the installed transformers.js, for the cache key and baselines. `"unknown"` if it cannot be read. */
export async function transformersVersion(opts: TransformersLoaderOptions = {}): Promise<string> {
  const tf = await loadModule(opts);
  return tf.env.version ?? 'unknown';
}
