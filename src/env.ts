import type { EnvInfo } from './types.js';

/** Environment with no WebGPU. Used by Node and by browsers without `navigator.gpu`. */
export function emptyEnv(userAgent = ''): EnvInfo {
  return {
    hasWebGpu: false,
    shaderF16: false,
    adapter: { vendor: '', architecture: '', device: '', description: '' },
    maxBufferSize: 0,
    userAgent,
  };
}

interface GpuAdapterLike {
  features?: { has(name: string): boolean };
  limits?: { maxBufferSize?: number };
  info?: Partial<Record<'vendor' | 'architecture' | 'device' | 'description', string>>;
  requestAdapterInfo?: () => Promise<Partial<Record<'vendor' | 'architecture' | 'device' | 'description', string>>>;
}

/**
 * Looks at the current browser: WebGPU presence, `shader-f16`, buffer limit and adapter info.
 * Adapter info can be empty strings (privacy); that is fine, the cache key stays stable.
 * Never throws; in Node it reports no WebGPU.
 */
export async function detectEnv(): Promise<EnvInfo> {
  const nav = (globalThis as { navigator?: { userAgent?: string; gpu?: { requestAdapter(): Promise<GpuAdapterLike | null> } } }).navigator;
  const env = emptyEnv(nav?.userAgent ?? '');
  if (!nav?.gpu) return env;
  try {
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) return env;
    let info = adapter.info;
    if (!info && adapter.requestAdapterInfo) info = await adapter.requestAdapterInfo();
    return {
      ...env,
      hasWebGpu: true,
      shaderF16: adapter.features?.has('shader-f16') ?? false,
      maxBufferSize: adapter.limits?.maxBufferSize ?? 0,
      adapter: {
        vendor: info?.vendor ?? '',
        architecture: info?.architecture ?? '',
        device: info?.device ?? '',
        description: info?.description ?? '',
      },
    };
  } catch {
    return env;
  }
}
