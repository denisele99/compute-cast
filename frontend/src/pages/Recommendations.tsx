import { useState } from 'react';
import { MemoryResult, mockMode } from '../api';
import { Badge, Section, Unavailable } from '../components/ui';
import { classifyFit } from '../lib/fit';
import { Scenario, Setup, pickScenarios } from '../lib/scenarios';

export type ScenarioState = { status: 'loading' | 'done' | 'error'; mem?: MemoryResult; error?: string };
const level = { minor: 'good', moderate: 'warn', major: 'bad' } as const;

export function RecommendationsPage({ setup, baseline, states, canRun, onRun, onApply, onBack }: {
  setup: Setup; baseline: MemoryResult | null; states: Record<string, ScenarioState>; canRun: boolean; onRun: (s: Scenario) => void; onApply: (s: Scenario) => void; onBack: () => void;
}): JSX.Element {
  const [goal, setGoal] = useState<'memory' | 'runtime' | 'cost' | 'balanced'>('balanced');
  const [advanced, setAdvanced] = useState(false);
  return <div className="space-y-6"><Section number="08" title="Improve this plan" hint="Each option changes your inputs and asks the backend to recalculate.">
    {mockMode && <div className="mb-4"><Badge kind="warn">Demo data</Badge></div>}
    <div className="mb-6 flex flex-wrap items-end gap-4"><label className="text-xs font-medium text-[#c4bfcd]">Optimize for<select className="field mt-2" value={goal} onChange={(e) => setGoal(e.target.value as typeof goal)}><option value="balanced">Balanced</option><option value="memory">Reduce memory</option><option value="runtime">Reduce runtime</option><option value="cost">Lower cost (cloud)</option></select></label>
      <div className="flex gap-2" role="tablist">{[false, true].map((a) => <button key={String(a)} role="tab" aria-selected={advanced === a} onClick={() => setAdvanced(a)} className={`border px-4 py-2 text-sm ${advanced === a ? 'border-violet bg-violet/15' : 'border-line text-[#a7a0bb]'}`}>{a ? 'Advanced changes' : 'Simple changes'}</button>)}</div></div>
    {!canRun && <Unavailable>Recalculation needs a collector file. Upload one on the Workload step.</Unavailable>}
    {advanced ? <div className="grid gap-3 sm:grid-cols-2">{['Multi-GPU data parallelism / sharding', 'MIG partitioning', 'GPU time slicing', 'Tensor or pipeline parallelism'].map((t) => <Unavailable key={t}>{t}: the estimator needs an explicit deployment config for this; not calculated yet.</Unavailable>)}</div> :
      <div className="grid gap-4 lg:grid-cols-3">{pickScenarios(goal).map((s, i) => { const st = states[s.id]; const next = s.change(setup); const f = st?.mem ? classifyFit(st.mem.gpu.peak_vram_gib, st.mem.gpu.usable_vram_gib) : null;
        return <article key={s.id} className={`flex flex-col border p-5 ${i === 0 && goal !== 'balanced' ? 'border-violet' : 'border-line'}`}><Badge kind={level[s.level]}>{s.level} change</Badge><h3 className="mt-3 font-semibold">{s.title}</h3><p className="mt-2 text-sm text-[#a7a0bb]">{s.why}</p>
          <p className="mt-3 text-xs text-[#8e899b]">{next ? s.describe(setup) : 'Already applied in your inputs.'}</p>
          {st?.status === 'loading' && <p className="mt-3 text-sm" aria-live="polite">Recalculating…</p>}{st?.status === 'error' && <p role="alert" className="mt-3 text-sm text-[#ffb3be]">{st.error}</p>}
          {st?.mem && f && <p className="mt-3 text-sm"><strong>{st.mem.gpu.peak_vram_gib?.toFixed(2)} GiB</strong> peak{baseline?.gpu.peak_vram_gib != null && st.mem.gpu.peak_vram_gib != null && ` (${(st.mem.gpu.peak_vram_gib - baseline.gpu.peak_vram_gib).toFixed(2)} vs now)`} · {f.label}</p>}
          <div className="mt-auto flex gap-2 pt-4"><button disabled={!canRun || !next || st?.status === 'loading'} onClick={() => onRun(s)} className="border border-line px-4 py-2 text-sm disabled:opacity-40">Recalculate</button><button disabled={!next} onClick={() => onApply(s)} className="bg-white px-4 py-2 text-sm font-semibold text-black disabled:opacity-40">Apply to inputs</button></div></article>; })}</div>}
    <p className="mt-4 text-xs text-[#8e899b]">Runtime changes are not recalculated here until a benchmark exists for the changed setup.</p>
  </Section><div><button onClick={onBack} className="border border-line px-5 py-3 text-sm">Back to results</button></div></div>;
}
