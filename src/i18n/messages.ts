/** Languages the CLI, examples and the demo page can speak. */
export type Lang = 'en' | 'ko';

export const LANGS: readonly Lang[] = ['en', 'ko'];

/**
 * All user-facing text lives here, one table per language. Both tables must have the same keys
 * (a test checks this). Placeholders look like {name}.
 */
export const messages = {
  en: {
    'cli.help': [
      'webgpu-canary - make the reference output that webgpu-canary compares against.',
      '',
      'Usage:',
      '  webgpu-canary baseline <modelId> --task embeddings|text-generation --revision <sha> [options]',
      '',
      'Options:',
      '  --task <task>            embeddings or text-generation (required)',
      '  --revision <sha>         model revision (commit sha) to pin (required)',
      '  --input <text>           check input (default: a fixed sentence per task)',
      '  --max-new-tokens <n>     tokens to generate for text-generation (default 12)',
      '  --out <file>             write JSON here (default: print to stdout)',
      '  --lang en|ko             message language (also WEBGPU_CANARY_LANG, LC_ALL, LANG)',
      '  -h, --help               show this help',
      '',
      'The baseline is made on cpu with fp32. Behind a proxy, set NODE_USE_ENV_PROXY=1.',
      'Exit codes: 0 ok, 1 model failed to load or run, 2 wrong arguments.',
    ].join('\n'),
    'cli.err.noCommand': 'Missing command. Use "webgpu-canary baseline <modelId> ..." (see --help).',
    'cli.err.unknownCommand': 'Unknown command "{name}". The only command is "baseline" (see --help).',
    'cli.err.missingModel': 'Missing <modelId>, for example Xenova/all-MiniLM-L6-v2.',
    'cli.err.missingTask': 'Missing --task. Use --task embeddings or --task text-generation.',
    'cli.err.unknownTask': 'Unknown task "{task}". Use embeddings or text-generation.',
    'cli.err.missingRevision': 'Missing --revision. Pin the model commit sha so the baseline matches one exact revision.',
    'cli.err.badLang': 'Unknown language "{lang}". Use en or ko.',
    'cli.err.badMaxTokens': '--max-new-tokens must be a positive integer, got "{value}".',
    'cli.err.badOption': 'Invalid option: {detail}',
    'cli.err.loadFailed': 'Could not make the baseline: {detail}',
    'cli.err.notFinite': 'The model returned non-finite values, so it cannot be used as a baseline.',
    'cli.err.writeFailed': 'Could not write {path}: {detail}',
    'cli.warn.revisionNotSha': 'Warning: --revision "{revision}" is not a 40-character commit sha. A branch or tag can move, so the baseline and the runtime model may drift apart. Pin a commit sha.',
    'cli.wrote': 'Wrote {path} ({task}, {size} values, revision {revision}).',

    'col.index': '#',
    'col.candidate': 'candidate',
    'col.status': 'status',
    'col.metric': 'metric',
    'col.min': 'min',
    'col.reason': 'reason',

    'reason.non-finite': 'Non-finite value at index {index} ({value}).',
    'reason.empty': 'Empty or all-zero embedding.',
    'reason.length': 'Vector length {got} differs from the baseline length {expected}.',
    'reason.cosine-below': 'Cosine {value} is below {min}.',
    'reason.cosine-ok': 'Cosine {value} >= {min}.',
    'reason.bad-tokens': 'Token ids contain non-integer or negative values.',
    'reason.repeat': 'One token repeated {run} times in a row (limit {limit}).',
    'reason.prefix-short': 'Only {got} leading tokens match the baseline, {required} required.',
    'reason.prefix-ok': '{got} leading tokens match (>= {required}).',
    'reason.no-vector': 'The runner returned no vector.',
    'reason.no-tokens': 'The runner returned no tokens.',
    'reason.timeout': 'No result within {ms} ms.',
    'reason.error': 'Error: {detail}',
    'reason.skip.no-webgpu': 'skipped: no WebGPU in this environment',
    'reason.skip.no-shader-f16': 'skipped: WebGPU has no shader-f16',
    'reason.skip.dtype-not-available': 'skipped: dtype {dtype} is not available for this model',

    'mock.banner': 'MOCK MODE (practice): healthy values are recorded from real Node cpu runs; the GPU failures are injected.',
    'mock.model': 'model: {model} @ {revision} - task: {task}',
    'mock.chosen': 'chosen: {candidate}',
    'mock.single': 'A single threshold of {threshold} would reject {candidate} (cosine {cosine}), although it is a healthy result.',
    'mock.cache': 'second call with the same key: {loads} model load, fromCache: {fromCache}',
    'mock.footer': 'The selection, judging and cache code is the real library; only the loader is fake.',
    'mock.runtime': 'Real model on cpu: npm run example:node (WEBGPU_CANARY_REAL=1)',

    'real.banner': 'REAL MODE: transformers.js on Node, model {model}',
    'real.chosen': 'chosen: {candidate}',
    'real.wrote': 'report written to {path}',
    'real.needFlag': 'Set WEBGPU_CANARY_REAL=1 to run the real model (downloads about 100 MB; behind a proxy also set NODE_USE_ENV_PROXY=1).',
    'real.failed': 'No candidate passed. See {path}.',

    'demo.title': 'webgpu-canary demo',
    'demo.subtitle': 'Pick the (device, dtype) that gives correct answers on this machine.',
    'demo.language': 'Language',
    'demo.mockNotice': 'Practice mode (mock): healthy values are recorded from real cpu runs; the failures you switch on below are injected, not measured on your GPU.',
    'demo.realNotice': 'Real mode: this page loads the model with transformers.js in your browser (first run downloads the model).',
    'demo.mode': 'Mode',
    'demo.mode.mock': 'Practice (mock)',
    'demo.mode.real': 'Real (transformers.js)',
    'demo.faults': 'Inject failures (mock only)',
    'demo.fault.error': 'webgpu/q4f16 throws an error',
    'demo.fault.nan': 'webgpu/fp16 returns NaN',
    'demo.fault.drift': 'webgpu/q8 drifts to cosine 0.41',
    'demo.run': 'Run selection',
    'demo.running': 'Running...',
    'demo.clearCache': 'Forget saved decision',
    'demo.attempts': 'Attempts',
    'demo.decision': 'Decision',
    'demo.noDecision': 'No decision yet.',
    'demo.chosen': 'Chosen: {candidate}',
    'demo.fromCache': 'Loaded from the saved decision.',
    'demo.failed': 'No candidate passed the output check.',
    'demo.download': 'Download report (JSON)',
    'demo.noWebgpu': 'This browser has no WebGPU, so webgpu candidates are skipped and the decision falls to wasm.',
    'demo.hasWebgpu': 'WebGPU is available in this browser.',
    'demo.mockAdapter': 'Practice mode simulates a WebGPU adapter (mock adapter), so webgpu candidates are tried. Real detection in this browser: {real}.',
    'demo.real.yes': 'WebGPU available',
    'demo.real.no': 'no WebGPU',
  },
  ko: {
    'cli.help': [
      'webgpu-canary - webgpu-canary 가 비교에 쓰는 기준 출력을 만듭니다.',
      '',
      '사용법:',
      '  webgpu-canary baseline <modelId> --task embeddings|text-generation --revision <sha> [옵션]',
      '',
      '옵션:',
      '  --task <task>            embeddings 또는 text-generation (필수)',
      '  --revision <sha>         고정할 모델 개정판(커밋 sha, 필수)',
      '  --input <text>           확인 입력 (기본: 작업별 고정 문장)',
      '  --max-new-tokens <n>     text-generation 에서 만들 토큰 수 (기본 12)',
      '  --out <file>             JSON 을 이 파일에 씀 (기본: 표준 출력)',
      '  --lang en|ko             메시지 언어 (WEBGPU_CANARY_LANG, LC_ALL, LANG 도 인식)',
      '  -h, --help               이 도움말',
      '',
      '기준 출력은 cpu 에서 fp32 로 만듭니다. 프록시 뒤에서는 NODE_USE_ENV_PROXY=1 을 설정하세요.',
      '종료 코드: 0 성공, 1 모델 로드·실행 실패, 2 잘못된 인자.',
    ].join('\n'),
    'cli.err.noCommand': '명령이 없습니다. "webgpu-canary baseline <modelId> ..." 형태로 쓰세요(--help 참고).',
    'cli.err.unknownCommand': '알 수 없는 명령 "{name}". 쓸 수 있는 명령은 "baseline" 하나입니다(--help 참고).',
    'cli.err.missingModel': '<modelId> 가 없습니다. 예: Xenova/all-MiniLM-L6-v2',
    'cli.err.missingTask': '--task 가 없습니다. --task embeddings 또는 --task text-generation 으로 지정하세요.',
    'cli.err.unknownTask': '알 수 없는 task "{task}". embeddings 또는 text-generation 을 쓰세요.',
    'cli.err.missingRevision': '--revision 이 없습니다. 기준 출력이 정확히 한 개정판과 맞도록 모델 커밋 sha 를 지정하세요.',
    'cli.err.badLang': '알 수 없는 언어 "{lang}". en 또는 ko 를 쓰세요.',
    'cli.err.badMaxTokens': '--max-new-tokens 는 양의 정수여야 합니다. 받은 값: "{value}"',
    'cli.err.badOption': '잘못된 옵션: {detail}',
    'cli.err.loadFailed': '기준 출력을 만들지 못했습니다: {detail}',
    'cli.err.notFinite': '모델이 유한하지 않은 값을 냈으므로 기준 출력으로 쓸 수 없습니다.',
    'cli.err.writeFailed': '{path} 를 쓰지 못했습니다: {detail}',
    'cli.warn.revisionNotSha': '경고: --revision "{revision}" 는 40자리 커밋 sha 가 아닙니다. 브랜치나 태그는 움직일 수 있어 기준 출력과 실행 모델이 어긋날 수 있습니다. 커밋 sha 를 지정하세요.',
    'cli.wrote': '{path} 를 썼습니다 ({task}, 값 {size}개, 개정판 {revision}).',

    'col.index': '#',
    'col.candidate': '후보',
    'col.status': '상태',
    'col.metric': '지표',
    'col.min': '최소',
    'col.reason': '사유',

    'reason.non-finite': '인덱스 {index} 에 유한하지 않은 값이 있습니다 ({value}).',
    'reason.empty': '임베딩이 비었거나 전부 0 입니다.',
    'reason.length': '벡터 길이 {got} 가 기준 길이 {expected} 와 다릅니다.',
    'reason.cosine-below': '코사인 {value} 가 {min} 미만입니다.',
    'reason.cosine-ok': '코사인 {value} >= {min}.',
    'reason.bad-tokens': '토큰 id 에 정수가 아니거나 음수인 값이 있습니다.',
    'reason.repeat': '같은 토큰이 {run}번 연속 반복됩니다(한도 {limit}).',
    'reason.prefix-short': '기준과 앞쪽 토큰이 {got}개만 같습니다(필요 {required}개).',
    'reason.prefix-ok': '앞쪽 토큰 {got}개 일치(>= {required}).',
    'reason.no-vector': '러너가 벡터를 돌려주지 않았습니다.',
    'reason.no-tokens': '러너가 토큰을 돌려주지 않았습니다.',
    'reason.timeout': '{ms} ms 안에 결과가 없습니다.',
    'reason.error': '오류: {detail}',
    'reason.skip.no-webgpu': '건너뜀: 이 환경에 WebGPU 가 없음',
    'reason.skip.no-shader-f16': '건너뜀: WebGPU 에 shader-f16 이 없음',
    'reason.skip.dtype-not-available': '건너뜀: 이 모델에는 dtype {dtype} 이 없음',

    'mock.banner': '목(mock) 모드(연습용): 정상값은 실제 Node cpu 실행에서 녹화한 것이고, GPU 고장은 주입한 것입니다.',
    'mock.model': '모델: {model} @ {revision} - 작업: {task}',
    'mock.chosen': '선택: {candidate}',
    'mock.single': '단일 임계값 {threshold} 였다면 정상 결과인 {candidate} (코사인 {cosine}) 를 탈락시켰을 것입니다.',
    'mock.cache': '같은 키로 두 번째 호출: 모델 로드 {loads}회, fromCache: {fromCache}',
    'mock.footer': '선택·판정·캐시 코드는 실제 라이브러리이고, 로더만 가짜입니다.',
    'mock.runtime': '실제 모델(cpu): npm run example:node (WEBGPU_CANARY_REAL=1)',

    'real.banner': '실제 모드: Node 에서 transformers.js, 모델 {model}',
    'real.chosen': '선택: {candidate}',
    'real.wrote': '보고서를 {path} 에 썼습니다',
    'real.needFlag': '실제 모델을 돌리려면 WEBGPU_CANARY_REAL=1 을 설정하세요(약 100MB 내려받음, 프록시 뒤에서는 NODE_USE_ENV_PROXY=1 도).',
    'real.failed': '통과한 후보가 없습니다. {path} 를 확인하세요.',

    'demo.title': 'webgpu-canary 데모',
    'demo.subtitle': '이 기기에서 맞는 답을 내는 (device, dtype) 조합을 고릅니다.',
    'demo.language': '언어',
    'demo.mockNotice': '연습용 목(mock) 모드: 정상값은 실제 cpu 실행에서 녹화한 값이고, 아래에서 켜는 고장은 내 GPU 에서 잰 것이 아니라 주입한 것입니다.',
    'demo.realNotice': '실제 모드: 이 페이지가 브라우저에서 transformers.js 로 모델을 로드합니다(처음에는 모델을 내려받습니다).',
    'demo.mode': '모드',
    'demo.mode.mock': '연습용(목)',
    'demo.mode.real': '실제(transformers.js)',
    'demo.faults': '고장 주입(목 모드 전용)',
    'demo.fault.error': 'webgpu/q4f16 이 오류를 던짐',
    'demo.fault.nan': 'webgpu/fp16 이 NaN 을 반환',
    'demo.fault.drift': 'webgpu/q8 이 코사인 0.41 로 어긋남',
    'demo.run': '선택 실행',
    'demo.running': '실행 중...',
    'demo.clearCache': '저장된 결정 지우기',
    'demo.attempts': '시도',
    'demo.decision': '결정',
    'demo.noDecision': '아직 결정이 없습니다.',
    'demo.chosen': '선택: {candidate}',
    'demo.fromCache': '저장된 결정에서 불러왔습니다.',
    'demo.failed': '출력 확인을 통과한 후보가 없습니다.',
    'demo.download': '보고서 내려받기(JSON)',
    'demo.noWebgpu': '이 브라우저에는 WebGPU 가 없어 webgpu 후보를 건너뛰고 wasm 으로 결정합니다.',
    'demo.hasWebgpu': '이 브라우저에서 WebGPU 를 쓸 수 있습니다.',
    'demo.mockAdapter': '연습용 모드는 WebGPU 어댑터(목 어댑터)를 흉내 내므로 webgpu 후보도 실제로 시도합니다. 이 브라우저의 실제 감지 결과: {real}.',
    'demo.real.yes': 'WebGPU 사용 가능',
    'demo.real.no': 'WebGPU 없음',
  },
} as const satisfies Record<Lang, Record<string, string>>;

export type MessageKey = keyof (typeof messages)['en'];

/** Looks up a message and fills {placeholders}. */
export function t(lang: Lang, key: MessageKey, params: Record<string, string | number> = {}): string {
  const template: string = messages[lang][key];
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? `{${name}}`));
}

function fromLocale(value: string | undefined): Lang | undefined {
  if (!value) return undefined;
  return value.toLowerCase().startsWith('ko') ? 'ko' : 'en';
}

/**
 * Picks the language: explicit flag, then WEBGPU_CANARY_LANG, then LC_ALL, LC_MESSAGES, LANG
 * (first non-empty one wins). Anything that does not start with "ko" is English.
 */
export function resolveLang(flag: string | undefined, env: Record<string, string | undefined>): Lang {
  if (flag === 'ko' || flag === 'en') return flag;
  for (const name of ['WEBGPU_CANARY_LANG', 'LC_ALL', 'LC_MESSAGES', 'LANG']) {
    const lang = fromLocale(env[name]);
    if (lang) return lang;
  }
  return 'en';
}

export function isLang(value: string): value is Lang {
  return value === 'en' || value === 'ko';
}
