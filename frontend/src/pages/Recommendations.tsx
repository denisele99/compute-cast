import { ChangeEvent, useEffect, useState } from 'react';
import { Code2, Download, FileCode2, Sparkles, Target, Upload, X } from 'lucide-react';
import { Advice, AdvicePrefs, AdviceRec, AdviceStatus, ConfigBody, Prediction, Values, api } from '../api';
import { Accordion, Alert, Badge, Delta, Section } from '../components/ui';
import { Scenario, pickScenarios } from '../lib/scenarios';
import { gib, hours, money, moneyDelta } from './Results';

export type ScenarioState = { status: 'loading' | 'done' | 'error'; result?: Prediction; error?: string };
type Recalc = { status: 'loading' | 'done' | 'error'; result?: Prediction; error?: string };
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const level = { minor: 'good', moderate: 'warn', major: 'bad' } as const;
const GOALS: [AdvicePrefs['goal'], string, string][] = [
  ['shortest_time', 'Shortest training time', 'Finish as fast as possible'], ['lowest_cost', 'Lowest cost', 'Spend the least money'],
  ['fit_memory', 'Fit into memory', 'Make it fit on the GPU'], ['highest_throughput', 'Highest throughput', 'Most tokens per second'],
  ['balanced', 'Balanced', 'A sensible mix of all of these']];
const KEEP: [keyof AdvicePrefs, string][] = [['preserve_effective_batch', 'effective batch size'], ['preserve_precision', 'precision (BF16 autocast)'],
  ['preserve_sequence_length', 'sequence length'], ['preserve_training_method', 'training method (full fine-tuning)']];
const DEFAULT_PREFS: AdvicePrefs = { goal: 'balanced', resources: 'current', preserve_effective_batch: true, preserve_precision: true,
  preserve_sequence_length: true, preserve_training_method: true, budget_usd: null, max_gpus: null, notes: '' };
const MAX_CODE_BYTES = 150_000;

function Metrics({ p, base }: { p: Prediction; base?: Prediction | null }): JSX.Element {
  if (p.status !== 'ok') return <p className="text-xs text-[#ffb3be]">Not estimable: {p.validation.errors.map((x) => x.message).join(' ') || 'missing inputs'}</p>;
  const b = base?.status === 'ok' ? base : null; const t = p.runtime?.total_training_compute;
  const bt = b?.runtime?.total_training_compute;
  return <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
    <div><dt className="eyebrow">Memory needed</dt><dd><strong>{gib(p.fit.capacity_gib)}</strong><Delta now={p.fit.capacity_gib} before={b?.fit.capacity_gib} format={gib} /><span className="block text-[10px]">{p.fit.fits ? 'fits' : 'exceeds usable VRAM'}</span></dd></div>
    <div><dt className="eyebrow">Training time</dt><dd><strong>{t ? hours(t.seconds / 3600) : '—'}</strong><Delta now={t ? t.seconds / 3600 : null} before={bt ? bt.seconds / 3600 : null} format={hours} /></dd></div>
    <div><dt className="eyebrow">Per update</dt><dd><strong>{p.runtime ? `${p.runtime.seconds_per_full_optimizer_update.toFixed(3)} s` : '—'}</strong><Delta now={p.runtime?.seconds_per_full_optimizer_update} before={b?.runtime?.seconds_per_full_optimizer_update} format={(x) => `${x.toFixed(3)} s`} /></dd></div>
    <div><dt className="eyebrow">Cost</dt><dd><strong>{p.cost.status === 'estimated' ? money(p.cost.cost) : '—'}</strong><Delta now={p.cost.cost} before={b?.cost.cost} format={moneyDelta} /></dd></div>
  </dl>;
}

function RecCard({ r, labels, values, baseline, recalc, onRecalc, onApply, applied }: { r: AdviceRec; labels: Record<string, string>; values: Values; baseline: Prediction | null;
  recalc?: Recalc; onRecalc: () => void; onApply: () => void; applied: boolean }): JSX.Element {
  const changes = Object.entries(r.new_values).filter(([k]) => k in labels && (k.startsWith('training.') || k.startsWith('data.') || k === 'hardware.gpu_name'));
  return <article className={`border p-4 ${r.estimable ? 'border-line' : 'border-dashed border-line'}`}>
    <div className="flex flex-wrap items-center gap-1.5"><Badge kind="muted">{r.category.replace(/_/g, ' ')}</Badge>
      {r.estimable ? <Badge kind="good">estimated</Badge> : <Badge kind="warn">not estimated</Badge>}{applied && <Badge kind="good">applied</Badge>}</div>
    <h3 className="mt-2 text-sm font-semibold">{r.title}</h3>
    <p className="mt-1 text-xs text-[#c4bfcd]">{r.explanation}</p>
    {r.expected_effect && <p className="mt-1 text-xs"><span className="text-[#8e899b]">Expected effect:</span> {r.expected_effect}</p>}
    {changes.length > 0 && <ul className="mt-2 space-y-0.5 text-xs">{changes.map(([k, v]) => <li key={k}><span className="text-[#8e899b]">{labels[k]}:</span> <strong>{String(values[k] ?? '—')}</strong> → <strong>{String(v)}</strong></li>)}</ul>}
    {r.hardware && <p className="mt-1 text-xs"><span className="text-[#8e899b]">Switch hardware to</span> <strong>{r.hardware.label}{r.hardware.region ? ` · ${r.hardware.region}` : ''}</strong>{r.hardware.cost_usd != null ? ` (estimated ${money(r.hardware.cost_usd)})` : ''}</p>}
    {r.tradeoffs.length > 0 && <p className="mt-1 text-[11px] text-[#a7a0bb]">Trade-offs: {r.tradeoffs.join('; ')}</p>}
    {r.estimate_note && <p className="mt-1 text-[11px] text-[#e6d28a]">{r.estimate_note}</p>}
    {[...r.violations.map((v) => `Conflicts with your constraints: ${v}.`), ...r.issues].map((m) => <p key={m} className="mt-1 text-[11px] text-[#ff9aa8]">{m}</p>)}
    {recalc?.status === 'loading' && <p className="mt-2 text-xs" aria-live="polite">Recalculating…</p>}
    {recalc?.status === 'error' && <p role="alert" className="mt-2 text-xs text-[#ffb3be]">{recalc.error}</p>}
    {recalc?.result && <div className="mt-2 border border-line bg-black/30 p-2"><p className="eyebrow mb-1">Recalculated vs your original estimate</p><Metrics p={recalc.result} base={baseline} /></div>}
    {r.estimable && <div className="mt-3 flex gap-2"><button type="button" onClick={onRecalc} disabled={recalc?.status === 'loading'} className="border border-line px-3 py-1.5 text-xs disabled:opacity-40">Recalculate estimate</button>
      <button type="button" onClick={onApply} className="bg-white px-3 py-1.5 text-xs font-semibold text-black">Apply changes</button></div>}
  </article>;
}

export function RecommendationsPage({ analysisId, body, values, labels, baseline, original, states, onRun, onApply, onApplyAdvice, onBack }: {
  analysisId: string; body: () => ConfigBody; values: Values; labels: Record<string, string>; baseline: Prediction | null; original: Prediction | null;
  states: Record<string, ScenarioState>; onRun: (s: Scenario) => void; onApply: (s: Scenario) => void; onApplyAdvice: (rec: AdviceRec) => Promise<void>; onBack: () => void;
}): JSX.Element {
  const [status, setStatus] = useState<AdviceStatus | null>(null);
  const [prefs, setPrefs] = useState<AdvicePrefs>(DEFAULT_PREFS);
  const [code, setCode] = useState<Record<string, string>>({}); const [codeError, setCodeError] = useState('');
  const [advice, setAdvice] = useState<Advice | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const [recalc, setRecalc] = useState<Record<string, Recalc>>({}); const [applied, setApplied] = useState<string[]>([]);
  const [quickGoal, setQuickGoal] = useState<'memory' | 'runtime' | 'cost' | 'balanced'>('balanced');
  useEffect(() => { api.adviceStatus().then(setStatus).catch(() => setStatus(null)); }, []);
  const set = <K extends keyof AdvicePrefs>(k: K, v: AdvicePrefs[K]) => setPrefs((p) => ({ ...p, [k]: v }));
  const compareBase = original ?? baseline;

  function addFiles(list: FileList | null): void {
    if (!list) return; setCodeError('');
    void Promise.all(Array.from(list).map(async (f) => [f.name.replace(/[^A-Za-z0-9._-]/g, '_'), await f.text()] as const)).then((pairs) => {
      const next = { ...code, ...Object.fromEntries(pairs) };
      if (Object.values(next).reduce((n, t) => n + t.length, 0) > MAX_CODE_BYTES) { setCodeError(`Code files are limited to ${MAX_CODE_BYTES / 1000} KB in total.`); return; }
      setCode(next);
    });
  }
  async function loadExample(): Promise<void> { try { const r = await api.adviceExamples(); setCode((c) => ({ ...c, ...r.files })); } catch (e) { setCodeError(message(e)); } }
  async function ask(): Promise<void> {
    setBusy(true); setError(''); setRecalc({});
    try { setAdvice(await api.advice(analysisId, { ...body(), preferences: prefs, files: code })); } catch (e) { setError(message(e)); } finally { setBusy(false); }
  }
  async function recalcRec(r: AdviceRec): Promise<void> {
    setRecalc((x) => ({ ...x, [r.id]: { status: 'loading' } }));
    try { const result = await api.predict(analysisId, { ...body(), values: { ...values, ...r.new_values } }, false); setRecalc((x) => ({ ...x, [r.id]: { status: 'done', result } })); }
    catch (e) { setRecalc((x) => ({ ...x, [r.id]: { status: 'error', error: message(e) } })); }
  }
  async function applyRec(r: AdviceRec): Promise<void> {
    try { await onApplyAdvice(r); setApplied((a) => [...new Set([...a, r.id])]); } catch (e) { setError(message(e)); }
  }

  return <div className="space-y-4">
    <Section icon={Sparkles} title="Recommendations" hint="Answer a few questions and an AI model suggests changes that make your plan faster, cheaper or fit better, then shows the new estimate next to your original one."
      actions={<Badge kind="muted">Powered by NVIDIA Nemotron 3 Super</Badge>}>
      <p className="text-[11px] text-[#8e899b]">{status?.model ?? 'NVIDIA Nemotron 3 Super via Nebius Token Factory'}. It receives your final inputs, the current estimate, your answers below and the text of your uploaded code. Suggestions are unverified; numbers come from the same estimator.</p>
      {status && !status.configured && <div className="mt-2"><Alert kind="warn"><strong>Not connected yet.</strong> The backend needs <span className="font-mono">NEBIUS_API_KEY</span> (a Nebius Token Factory API key) to call the model. Quick what-ifs below still work without it.</Alert></div>}

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <fieldset className="border border-line p-3"><legend className="flex items-center gap-1.5 px-1 text-xs font-semibold"><Target size={13} aria-hidden /> What matters most?</legend>
          <div className="grid gap-1 sm:grid-cols-2">{GOALS.map(([id, label, hint]) => <label key={id} className={`flex cursor-pointer items-start gap-2 border px-2 py-1.5 text-xs ${prefs.goal === id ? 'border-violet bg-violet/10' : 'border-line'}`}>
            <input type="radio" name="goal" checked={prefs.goal === id} onChange={() => set('goal', id)} className="mt-0.5" /><span><strong>{label}</strong><span className="block text-[10px] text-[#8e899b]">{hint}</span></span></label>)}</div></fieldset>
        <fieldset className="border border-line p-3"><legend className="px-1 text-xs font-semibold">Which hardware may we suggest?</legend>
          {([['current', 'Keep my current GPU', `Only change the training setup for ${String(values['hardware.gpu_name'] ?? 'the selected GPU')}.`], ['alternatives', 'Consider other GPUs', 'Compare other Nebius GPUs and suggest switching if it helps your goal.']] as const).map(([id, label, hint]) =>
            <label key={id} className="mb-1 flex items-start gap-2 text-xs"><input type="radio" name="resources" checked={prefs.resources === id} onChange={() => set('resources', id)} className="mt-0.5" /><span><strong>{label}</strong><span className="block text-[10px] text-[#8e899b]">{hint}</span></span></label>)}
          <div className="mt-2 grid grid-cols-2 gap-2">
            <label className="text-[11px]">Budget for the run (USD, optional)<input type="number" min={0} step="any" value={prefs.budget_usd ?? ''} onChange={(e) => set('budget_usd', e.target.value === '' ? null : Number(e.target.value))} className="field-sm mt-0.5" placeholder="no limit" /></label>
            <label className="text-[11px]">GPU limit (optional)<input type="number" min={1} step={1} value={prefs.max_gpus ?? ''} onChange={(e) => set('max_gpus', e.target.value === '' ? null : Number(e.target.value))} className="field-sm mt-0.5" placeholder="1" /></label></div>
          <p className="mt-1 text-[10px] text-[#8e899b]">Estimates cover one GPU. With a higher GPU limit the model may also suggest multi-GPU methods, labelled "not estimated".</p></fieldset>
        <fieldset className="border border-line p-3"><legend className="px-1 text-xs font-semibold">What must stay the same?</legend>
          {KEEP.map(([k, label]) => <label key={k} className="flex items-center gap-2 text-xs"><input type="checkbox" checked={prefs[k] as boolean} onChange={(e) => set(k, e.target.checked as never)} /> Keep {label}</label>)}
          <label className="mt-2 block text-[11px]">Anything else? (optional)<textarea value={prefs.notes} onChange={(e) => set('notes', e.target.value)} maxLength={2000} rows={2} className="mt-0.5 w-full border border-line bg-black/40 p-1.5 text-xs" placeholder="e.g. we must finish within 4 hours" /></label></fieldset>
        <fieldset className="border border-line p-3"><legend className="flex items-center gap-1.5 px-1 text-xs font-semibold"><Code2 size={13} aria-hidden /> Code for advanced suggestions (optional)</legend>
          <p className="text-[11px] text-[#a7a0bb]">Add code to get concrete edits for sharding, parallelism, batch/accumulation tuning, attention or activation checkpointing:</p>
          <ul className="mt-1 list-disc pl-4 text-[10px] text-[#8e899b]">{(status?.code_kinds ?? []).map((k) => <li key={k}>{k}</li>)}</ul>
          <div className="mt-2 flex flex-wrap gap-2"><label className="flex cursor-pointer items-center gap-1 border border-line px-2 py-1 text-xs"><Upload size={12} aria-hidden /> Add files
            <input type="file" multiple accept=".py,.json,.yaml,.yml,.toml,.sh,.txt" className="sr-only" aria-label="Add code files" onChange={(e: ChangeEvent<HTMLInputElement>) => { addFiles(e.target.files); e.target.value = ''; }} /></label>
            <button type="button" onClick={() => void loadExample()} className="flex items-center gap-1 border border-line px-2 py-1 text-xs"><FileCode2 size={12} aria-hidden /> Use example training script</button></div>
          {Object.keys(code).length > 0 && <ul className="mt-2 flex flex-wrap gap-1" aria-label="Code for review">{Object.keys(code).map((n) => <li key={n} className="flex items-center gap-1 border border-line px-2 py-0.5 text-[11px]">{n}<button type="button" aria-label={`Remove ${n}`} onClick={() => setCode(({ [n]: _gone, ...rest }) => rest)}><X size={11} /></button></li>)}</ul>}
          {codeError && <Alert kind="error">{codeError}</Alert>}
          <p className="mt-1 text-[10px] text-[#8e899b]">Files you uploaded on the Workload step are included automatically. Code is sent as text and never executed.</p></fieldset>
      </div>
      <div className="mt-3 flex items-center gap-3"><button type="button" disabled={busy || !baseline || baseline.status !== 'ok' || status?.configured === false} onClick={() => void ask()}
        className="flex items-center gap-1.5 bg-violet px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"><Sparkles size={14} aria-hidden />{busy ? 'Asking Nemotron…' : 'Get recommendations'}</button>
        {(!baseline || baseline.status !== 'ok') && <span className="text-xs text-[#a7a0bb]">Run a successful estimate first.</span>}</div>
      {error && <div className="mt-2"><Alert kind="error">{error}</Alert></div>}
    </Section>

    {advice && <>
      <div className="grid gap-3 lg:grid-cols-2">
        <Section title="Your current plan" hint="A short summary of your model, training strategy and estimates."><p className="text-sm text-[#d8d3e3]">{advice.original_summary}</p>
          {baseline && <div className="mt-2"><Metrics p={baseline} base={original && original !== baseline ? original : null} /></div>}
          {original && baseline && original !== baseline && <p className="mt-1 text-[10px] text-[#8e899b]">Arrows: change from your original estimate.</p>}</Section>
        <Section title="Suggested changes" hint="In short, what to change and why."><p className="text-sm text-[#d8d3e3]">{advice.suggestions_summary}</p>
          {advice.missing_information.length > 0 && <p className="mt-2 text-[11px] text-[#e6d28a]">The model would like: {advice.missing_information.join('; ')}</p>}</Section>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">{advice.recommendations.map((r) => <RecCard key={r.id} r={r} labels={labels} values={values} baseline={compareBase} recalc={recalc[r.id]}
        onRecalc={() => void recalcRec(r)} onApply={() => void applyRec(r)} applied={applied.includes(r.id)} />)}</div>
      {advice.changed_files.length > 0 && <Accordion title={`Proposed code edits (${advice.changed_files.length} file${advice.changed_files.length > 1 ? 's' : ''})`} hint="syntax-checked only; review before use" defaultOpen>
        <ul className="mb-2 space-y-0.5 text-xs">{advice.changed_files.map((f) => <li key={f.path}><span className="font-mono">{f.path}</span> <Badge kind="muted">{f.action}</Badge> <span className="text-[#a7a0bb]">{f.reason}</span></li>)}</ul>
        {advice.files_check?.diff && <pre className="max-h-80 overflow-auto bg-black/40 p-2 font-mono text-[11px]">{advice.files_check.diff}</pre>}</Accordion>}
      <div className="flex flex-wrap items-center gap-2"><a href={api.adviceExportUrl(analysisId)} className={`flex items-center gap-1.5 border border-line px-3 py-1.5 text-xs ${applied.length ? '' : 'pointer-events-none opacity-40'}`} aria-disabled={!applied.length}>
        <Download size={12} aria-hidden />{advice.changed_files.length ? 'Download edited code + config' : 'Download training config'}</a>
        <span className="text-[11px] text-[#8e899b]">{applied.length ? 'Includes your applied changes. Your original inputs stay saved as the original estimate.' : 'Apply a suggestion first.'}</span></div>
      <p className="text-[10px] text-[#8e899b]">{advice.note} Reviewed files: {advice.files_reviewed.join(', ') || 'none'}.</p>
    </>}

    <Accordion title="Quick what-ifs (no AI)" hint="simple changes you can try without the model">
      <label className="mb-3 block max-w-xs text-xs font-medium text-[#c4bfcd]">Optimize for<select className="field mt-1" value={quickGoal} onChange={(e) => setQuickGoal(e.target.value as typeof quickGoal)}><option value="balanced">Balanced</option><option value="memory">Reduce memory</option><option value="runtime">Reduce runtime</option><option value="cost">Lower cost</option></select></label>
      <div className="grid gap-3 lg:grid-cols-2">{pickScenarios(quickGoal).map((s) => { const st = states[s.id]; const next = s.change(values);
        return <article key={s.id} className="flex flex-col border border-line p-3"><Badge kind={level[s.level]}>{s.level} change</Badge><h3 className="mt-2 text-sm font-semibold">{s.title}</h3><p className="mt-1 text-xs text-[#a7a0bb]">{s.why}</p>
          <p className="mt-1 text-[11px] text-[#8e899b]">{next ? s.describe(values) : 'Already applied in your inputs.'}</p>
          {st?.status === 'loading' && <p className="mt-2 text-xs">Recalculating…</p>}{st?.status === 'error' && <p role="alert" className="mt-2 text-xs text-[#ffb3be]">{st.error}</p>}
          {st?.result && <div className="mt-2"><Metrics p={st.result} base={compareBase} /></div>}
          <div className="mt-auto flex gap-2 pt-3"><button type="button" disabled={!next || st?.status === 'loading'} onClick={() => onRun(s)} className="border border-line px-3 py-1.5 text-xs disabled:opacity-40">Recalculate estimate</button>
            <button type="button" disabled={!next} onClick={() => onApply(s)} className="bg-white px-3 py-1.5 text-xs font-semibold text-black disabled:opacity-40">Apply changes</button></div></article>; })}</div>
    </Accordion>
    <div><button type="button" onClick={onBack} className="border border-line px-5 py-2 text-sm">Back to results</button></div>
  </div>;
}

