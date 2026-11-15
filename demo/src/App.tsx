import { useState } from 'react';
import { candidateName, reasonText } from '../../src/format.js';
import { t, LANGS, type Lang } from '../../src/i18n/messages.js';
import { selectRuntime } from '../../src/select.js';
import { detectEnv } from '../../src/env.js';
import { localStorageStore, memoryStore } from '../../src/store.js';
import { fixtures } from '../../src/testing/fixtures-data.js';
import { fakeLoader } from '../../src/testing/index.js';
import { MOCK_CANDIDATES, MOCK_ENV, mockBaseline, mockPlan } from '../../src/testing/mock-demo.js';
import type { Attempt, EnvInfo, Report } from '../../src/types.js';
import { initialLang, saveLang } from './i18n.js';

type Mode = 'mock' | 'real';

const store = localStorageStore('webgpu-canary-demo:decision:');
const storage = typeof localStorage === 'undefined' ? undefined : localStorage;

export function App() {
  const [lang, setLang] = useState<Lang>(() => initialLang(storage, window.location.search, navigator.language));
  const [mode, setMode] = useState<Mode>('mock');
  const [faults, setFaults] = useState({ error: true, nan: true, drift: true });
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [failed, setFailed] = useState(false);
  const [running, setRunning] = useState(false);
  const [webgpu, setWebgpu] = useState<boolean | null>(null);

  const changeLang = (next: Lang) => {
    setLang(next);
    saveLang(storage, next); // the choice is remembered for the next visit
  };

  const run = async () => {
    setRunning(true);
    setAttempts([]);
    setReport(null);
    setFailed(false);
    try {
      const realEnv: EnvInfo = await detectEnv();
      setWebgpu(realEnv.hasWebGpu);
      const baseline = mockBaseline();
      const common = {
        modelId: baseline.modelId,
        revision: baseline.revision,
        task: 'embeddings' as const,
        baseline,
        store,
        onAttempt: (a: Attempt) => setAttempts((list) => [...list, a]),
      };
      if (mode === 'mock') {
        // Practice mode: recorded healthy vectors + injected failures, real selection code.
        const { report: r, runner } = await selectRuntime({
          ...common,
          store: memoryStore(), // a fresh store per run, so a saved decision from other toggles never leaks in
          loader: fakeLoader(mockPlan(faults, lang)),
          env: MOCK_ENV,
          candidates: MOCK_CANDIDATES,
          timeoutMs: 5000,
        });
        setReport(r);
        await runner.dispose(); // the demo only shows the decision, so free the model
      } else {
        // Real mode: transformers.js loads the model in this browser (loaded on demand).
        const { transformersLoader } = await import('../../src/transformers/index.js');
        const { report: r, runner } = await selectRuntime({
          ...common,
          loader: transformersLoader(),
          env: realEnv,
          libVersion: fixtures.minilm.libVersion,
          candidates: [
            { device: 'webgpu', dtype: 'fp16' },
            { device: 'wasm', dtype: 'q4' },
            { device: 'wasm', dtype: 'q8' },
          ],
          timeoutMs: 600_000,
        });
        setReport(r);
        await runner.dispose(); // the demo only shows the decision, so free the model
      }
    } catch (error) {
      setFailed(true);
      setReport((error as { report?: Report }).report ?? null);
    } finally {
      setRunning(false);
    }
  };

  const download = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'webgpu-canary-report.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h1>{t(lang, 'demo.title')}</h1>
        <label>
          {t(lang, 'demo.language')}{' '}
          <select value={lang} onChange={(e) => changeLang(e.target.value as Lang)} aria-label={t(lang, 'demo.language')}>
            {LANGS.map((l) => (
              <option key={l} value={l}>
                {l === 'en' ? 'English' : '한국어'}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="sub">{t(lang, 'demo.subtitle')}</p>
      <div className="notice">{mode === 'mock' ? t(lang, 'demo.mockNotice') : t(lang, 'demo.realNotice')}</div>
      {webgpu !== null && mode === 'real' && <p>{webgpu ? t(lang, 'demo.hasWebgpu') : t(lang, 'demo.noWebgpu')}</p>}
      {webgpu !== null && mode === 'mock' && (
        <p>{t(lang, 'demo.mockAdapter', { real: t(lang, webgpu ? 'demo.real.yes' : 'demo.real.no') })}</p>
      )}

      <div className="card">
        <div className="row">
          <strong>{t(lang, 'demo.mode')}</strong>
          <label>
            <input type="radio" checked={mode === 'mock'} onChange={() => setMode('mock')} /> {t(lang, 'demo.mode.mock')}
          </label>
          <label>
            <input type="radio" checked={mode === 'real'} onChange={() => setMode('real')} /> {t(lang, 'demo.mode.real')}
          </label>
        </div>
        {mode === 'mock' && (
          <fieldset>
            <legend>{t(lang, 'demo.faults')}</legend>
            {(['error', 'nan', 'drift'] as const).map((k) => (
              <label key={k}>
                <input type="checkbox" checked={faults[k]} onChange={(e) => setFaults({ ...faults, [k]: e.target.checked })} />{' '}
                {t(lang, `demo.fault.${k}`)}
              </label>
            ))}
          </fieldset>
        )}
        <p>
          <button onClick={run} disabled={running}>
            {running ? t(lang, 'demo.running') : t(lang, 'demo.run')}
          </button>
          <button onClick={() => store.clear()}>{t(lang, 'demo.clearCache')}</button>
        </p>
      </div>

      <div className="card">
        <h2>{t(lang, 'demo.attempts')}</h2>
        <table>
          <thead>
            <tr>
              <th>{t(lang, 'col.index')}</th>
              <th>{t(lang, 'col.candidate')}</th>
              <th>{t(lang, 'col.status')}</th>
              <th>{t(lang, 'col.metric')}</th>
              <th>{t(lang, 'col.min')}</th>
              <th>{t(lang, 'col.reason')}</th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((a, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{candidateName(a.candidate)}</td>
                <td className={a.status}>{a.status}</td>
                <td>{a.metric === undefined ? '-' : Number(a.metric.toFixed(4))}</td>
                <td>{a.threshold ?? '-'}</td>
                <td>{reasonText(a, lang)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>{t(lang, 'demo.decision')}</h2>
        {failed && <p className="error">{t(lang, 'demo.failed')}</p>}
        {report?.chosen && <p className="pass">{t(lang, 'demo.chosen', { candidate: candidateName(report.chosen) })}</p>}
        {report?.fromCache && <p>{t(lang, 'demo.fromCache')}</p>}
        {!report && !failed && <p>{t(lang, 'demo.noDecision')}</p>}
        {report && <button onClick={download}>{t(lang, 'demo.download')}</button>}
      </div>
    </main>
  );
}
