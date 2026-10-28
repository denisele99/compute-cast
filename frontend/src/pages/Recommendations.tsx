import { useState } from 'react';
import { Prediction, Values } from '../api';
import { Badge, Section } from '../components/ui';
import { classifyFit } from '../lib/fit';
import { Scenario, pickScenarios } from '../lib/scenarios';
import { gib, hours } from './Results';

export type ScenarioState = { status: 'loading' | 'done' | 'error'; result?: Prediction; error?: string };
const level = { minor: 'good', moderate: 'warn', major: 'bad' } as const;

export function RecommendationsPage({ values, baseline, states, onRun, onApply, onBack }: {
  values: Values; baseline: Prediction | null; states: Record<string, ScenarioState>; onRun: (s: Scenario) => void; onApply: (s: Scenario) => void; onBack: () => void;
}): JSX.Element {
  const [goal, setGoal] = useState<'memory' | 'runtime' | 'cost' | 'balanced'>('balanced');
  const basePeak = baseline?.fit?.peak_vram_gib ?? null;
  return <div className="space-y-6"><Section number="06" title="Improve this plan" hint="Each option changes your inputs and asks the backend to re-predict (not recorded as a run until applied).">
    <label className="mb-6 block max-w-xs text-xs font-medium text-[#c4bfcd]">Optimize for<select className="field mt-2" value={goal} onChange={(e) => setGoal(e.target.value as typeof goal)}><option value="balanced">Balanced</option><option value="memory">Reduce memory</option><option value="runtime">Reduce runtime</option><option value="cost">Lower cost</option></select></label>
    <div className="grid gap-4 lg:grid-cols-3">{pickScenarios(goal).map((s, i) => {
      const st = states[s.id]; const next = s.change(values); const r = st?.result;
      const fit = r?.fit ? classifyFit(r.fit.peak_vram_gib, r.fit.usable_vram_gib) : null;
      return <article key={s.id} className={`flex flex-col border p-5 ${i === 0 && goal !== 'balanced' ? 'border-violet' : 'border-line'}`}><Badge kind={level[s.level]}>{s.level} change</Badge><h3 className="mt-3 font-semibold">{s.title}</h3><p className="mt-2 text-sm text-[#a7a0bb]">{s.why}</p>
        <p className="mt-3 text-xs text-[#8e899b]">{next ? s.describe(values) : 'Already applied in your inputs.'}</p>
        {st?.status === 'loading' && <p className="mt-3 text-sm" aria-live="polite">Recalculating…</p>}{st?.status === 'error' && <p role="alert" className="mt-3 text-sm text-[#ffb3be]">{st.error}</p>}
        {r?.status === 'blocked' && <p className="mt-3 text-sm text-[#ffb3be]">Not estimable: {[...r.validation.unsupported, ...r.validation.errors].map((x) => x.message).join(' ') || 'missing inputs'}</p>}
        {r?.fit && fit && <p className="mt-3 text-sm"><strong>{gib(r.fit.peak_vram_gib)}</strong> peak{basePeak !== null && ` (${(r.fit.peak_vram_gib - basePeak >= 0 ? '+' : '')}${(r.fit.peak_vram_gib - basePeak).toFixed(2)} GiB vs now)`} · {fit.label}{r.runtime.scenarios?.slow?.job_hours != null && ` · slow ${hours(r.runtime.scenarios.slow.job_hours)}`}</p>}
        <div className="mt-auto flex gap-2 pt-4"><button type="button" disabled={!next || st?.status === 'loading'} onClick={() => onRun(s)} className="border border-line px-4 py-2 text-sm disabled:opacity-40">Recalculate</button><button type="button" disabled={!next} onClick={() => onApply(s)} className="bg-white px-4 py-2 text-sm font-semibold text-black disabled:opacity-40">Apply to inputs</button></div></article>;
    })}</div>
  </Section><div><button type="button" onClick={onBack} className="border border-line px-5 py-3 text-sm">Back to results</button></div></div>;
}
