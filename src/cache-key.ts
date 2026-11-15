import type { EnvInfo, Task } from './types.js';

/** Major version of the main browser engine in a user agent, e.g. `chrome/131`. Empty if unknown. */
export function uaMajor(userAgent: string): string {
  // Order matters: Edge also says "Chrome/", so the more specific token is looked for first.
  for (const token of ['Edg', 'Firefox', 'Chrome', 'Version']) {
    const m = new RegExp(`${token}/(\\d+)`).exec(userAgent);
    if (m) return `${token.toLowerCase()}/${m[1]}`;
  }
  return '';
}

function fnv1a(text: string, seed: number): string {
  let h = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export interface KeyParts {
  env: EnvInfo;
  modelId: string;
  revision: string;
  task: Task;
  libVersion: string;
}

/**
 * Decision cache key. It changes when the adapter, the browser major version, the model, its
 * revision, the task or the library version changes. Empty adapter strings are fine: the same
 * input always gives the same key, and `shaderF16`/`maxBufferSize` keep coarse GPUs apart.
 */
export function cacheKey(parts: KeyParts): string {
  const { env } = parts;
  const canonical = JSON.stringify([
    1,
    env.adapter.vendor,
    env.adapter.architecture,
    env.adapter.device,
    env.adapter.description,
    env.hasWebGpu,
    env.shaderF16,
    env.maxBufferSize,
    uaMajor(env.userAgent),
    parts.modelId,
    parts.revision,
    parts.task,
    parts.libVersion,
  ]);
  return `webgpu-canary:v1:${fnv1a(canonical, 0x811c9dc5)}${fnv1a(canonical, 0x2545f491)}`;
}
