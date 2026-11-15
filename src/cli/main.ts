import { parseArgs } from 'node:util';
import { writeFileSync } from 'node:fs';
import { makeBaseline } from '../baseline.js';
import { DEFAULT_MAX_NEW_TOKENS } from '../defaults.js';
import { isLang, resolveLang, t, type Lang, type MessageKey } from '../i18n/messages.js';
import type { Loader, Task } from '../types.js';

export interface CliIo {
  stdout: (text: string) => void;
  stderr: (text: string) => void;
  env: Record<string, string | undefined>;
  /** Writes the output file. Default: fs.writeFileSync. */
  writeFile?: (path: string, text: string) => void;
  /** Supplies the loader and library version (tests inject a fake; default: transformers.js). */
  makeLoader?: () => Promise<{ loader: Loader; libVersion: string }>;
}

async function defaultLoader(): Promise<{ loader: Loader; libVersion: string }> {
  const { transformersLoader, transformersVersion } = await import('../transformers/index.js');
  return { loader: transformersLoader(), libVersion: await transformersVersion() };
}

const TASKS: readonly Task[] = ['embeddings', 'text-generation'];

/** Runs the CLI and returns the exit code: 0 ok, 1 model failure, 2 wrong arguments. */
export async function main(argv: string[], io: CliIo): Promise<number> {
  // The language must be known before anything else is reported, including argument errors.
  const langFlagIndex = argv.lastIndexOf('--lang');
  const rawLang = argv.find((a) => a.startsWith('--lang='))?.slice(7) ?? (langFlagIndex >= 0 ? argv[langFlagIndex + 1] : undefined);
  let lang: Lang = resolveLang(undefined, io.env);
  const usageError = (key: MessageKey, params: Record<string, string | number> = {}): number => {
    io.stderr(`${t(lang, key, params)}\n`);
    io.stderr(`${lang === 'ko' ? '도움말: webgpu-canary --help' : 'Help: webgpu-canary --help'}\n`);
    return 2;
  };

  if (rawLang !== undefined) {
    if (!isLang(rawLang)) return usageError('cli.err.badLang', { lang: rawLang });
    lang = rawLang;
  }

  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      options: {
        task: { type: 'string' },
        revision: { type: 'string' },
        input: { type: 'string' },
        'max-new-tokens': { type: 'string' },
        out: { type: 'string' },
        lang: { type: 'string' },
        help: { type: 'boolean', short: 'h' },
      },
    });
  } catch (error) {
    return usageError('cli.err.badOption', { detail: error instanceof Error ? error.message : String(error) });
  }
  const { values, positionals } = parsed;

  if (values.help) {
    io.stdout(`${t(lang, 'cli.help')}\n`);
    return 0;
  }
  const [command, modelId] = positionals;
  if (!command) return usageError('cli.err.noCommand');
  if (command !== 'baseline') return usageError('cli.err.unknownCommand', { name: command });
  if (!modelId) return usageError('cli.err.missingModel');
  if (!values.task) return usageError('cli.err.missingTask');
  if (!TASKS.includes(values.task as Task)) return usageError('cli.err.unknownTask', { task: values.task });
  if (!values.revision) return usageError('cli.err.missingRevision');
  if (!/^[0-9a-f]{40}$/i.test(values.revision)) io.stderr(`${t(lang, 'cli.warn.revisionNotSha', { revision: values.revision })}\n`);
  let maxNewTokens: number | undefined;
  if (values['max-new-tokens'] !== undefined) {
    const n = Number(values['max-new-tokens']);
    if (!Number.isInteger(n) || n <= 0) return usageError('cli.err.badMaxTokens', { value: values['max-new-tokens'] });
    maxNewTokens = n;
  }
  const task = values.task as Task;

  let baseline;
  try {
    const { loader, libVersion } = await (io.makeLoader ?? defaultLoader)();
    baseline = await makeBaseline({
      modelId,
      revision: values.revision,
      task,
      loader,
      libVersion,
      input: values.input,
      maxNewTokens: task === 'text-generation' ? (maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS) : undefined,
    });
  } catch (error) {
    io.stderr(`${t(lang, 'cli.err.loadFailed', { detail: error instanceof Error ? error.message.slice(0, 300) : String(error) })}\n`);
    return 1;
  }

  const json = JSON.stringify(baseline, null, 2) + '\n';
  if (!values.out) {
    io.stdout(json);
    return 0;
  }
  try {
    (io.writeFile ?? ((p: string, text: string) => writeFileSync(p, text)))(values.out, json);
  } catch (error) {
    io.stderr(`${t(lang, 'cli.err.writeFailed', { path: values.out, detail: error instanceof Error ? error.message : String(error) })}\n`);
    return 1;
  }
  io.stdout(`${t(lang, 'cli.wrote', { path: values.out, task, size: (baseline.vector ?? baseline.tokens ?? []).length, revision: values.revision })}\n`);
  return 0;
}
