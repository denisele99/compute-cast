import { ChangeEvent, useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Binary, Check, ChevronDown, Code2, Cpu, Database, Download, FileCode2, Layers, Lightbulb, Network, RefreshCw, Ruler, Sparkles, Target, Upload, X, Zap, type LucideIcon } from 'lucide-react';
import { Advice, AdvicePrefs, AdviceRec, AdviceStatus, AdviceStrategy, ConfigBody, Prediction, StrategyProjection, Values, api } from '../api';
import { Accordion, Alert, Badge, CopyBlock, Delta, PageHeading, Section } from '../components/ui';
import { gib, hours, money, moneyDelta } from './Results';

type Recalc = { status: 'loading' | 'done' | 'error'; result?: Prediction; error?: string };
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const GOALS: [AdvicePrefs['goal'], string][] = [['shortest_time', 'Shortest training time'], ['lowest_cost', 'Lowest cost'],
  ['reduce_memory', 'Reduce memory'], ['highest_throughput', 'Highest throughput'], ['balanced', 'Balanced']];
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

const SHARDING: Record<AdviceStrategy['sharding'], string> = { none: '', zero1: 'ZeRO-1 (optimizer state sharded)', zero2: 'ZeRO-2 (optimizer + gradients sharded)',
  zero3: 'ZeRO-3 (fully sharded)', fsdp: 'FSDP (fully sharded)' };
function strategyParts(s: AdviceStrategy): string[] {
  return [s.gpu_count > 1 ? `${s.gpu_count} GPUs` : '1 GPU', s.data_parallel > 1 ? `data parallel ×${s.data_parallel}` : '', s.tensor_parallel > 1 ? `tensor parallel ×${s.tensor_parallel}` : '',
    s.pipeline_parallel > 1 ? `pipeline ×${s.pipeline_parallel} stages` : '', SHARDING[s.sharding],
    s.activation_checkpointing !== 'none' ? `${s.activation_checkpointing} activation checkpointing` : '', s.attention === 'flash' ? 'flash attention' : ''].filter(Boolean);
}

/** Rough numbers for a multi-technique plan, next to the current single-GPU estimate. */
function Projection({ p, base }: { p: StrategyProjection; base: Prediction | null }): JSX.Element {
  const b = base?.status === 'ok' ? base : null; const bt = b?.runtime?.total_training_compute;
  return <div className="mt-2 border border-[#5d4cb0] bg-violet/5 p-2">
    <p className="eyebrow mb-1 flex items-center gap-1.5">Approximate projection <Badge kind="warn">rule of thumb</Badge></p>
    <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
      <div><dt className="eyebrow">Memory / GPU</dt><dd><strong>{gib(p.per_gpu_capacity_gib)}</strong><Delta now={p.per_gpu_capacity_gib} before={b?.fit.capacity_gib} format={gib} />
        <span className="block text-[10px]">{p.fits ? 'fits' : 'exceeds'} {gib(p.usable_vram_gib)} usable</span></dd></div>
      <div><dt className="eyebrow">Training time</dt><dd><strong>{hours(p.training_hours)}</strong><Delta now={p.training_hours} before={bt ? bt.seconds / 3600 : null} format={hours} /></dd></div>
      <div><dt className="eyebrow">Cost</dt><dd><strong>{money(p.cost_usd)}</strong><Delta now={p.cost_usd} before={b?.cost.cost} format={moneyDelta} />
        <span className="block text-[10px]">{p.gpu_count} GPU{p.gpu_count > 1 ? 's' : ''}</span></dd></div>
      <div><dt className="eyebrow">Global batch</dt><dd><strong>{p.global_effective_batch}</strong><span className="block text-[10px]">scaling efficiency {(p.efficiency * 100).toFixed(0)}%</span></dd></div></dl>
    <details className="mt-1 text-[10px] text-[#8e899b]"><summary className="cursor-pointer">How this was projected</summary>
      <ul className="mt-1 list-disc pl-4">{p.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
      <p className="mt-1">Per GPU: {Object.entries(p.memory_gib).map(([k, v]) => `${k.replace(/_/g, ' ')} ${gib(v)}`).join(' · ')}</p></details>
  </div>;
}

const ICONS: Record<string, LucideIcon> = { batch_accumulation: Layers, sequence_length: Ruler, hardware: Cpu, precision: Binary, activation_checkpointing: RefreshCw,
  attention: Zap, data_parallel: Network, sharding: Network, tensor_parallel: Network, pipeline_parallel: Network, data_loading: Database, combined: Sparkles, other: Lightbulb };
const firstSentence = (t: string) => (t.match(/^[^.!?]*[.!?]/)?.[0] ?? t).slice(0, 180);

type Face = { memory: number | null; hours: number | null; cost: number | null; fits: boolean | null };
function faceOf(r: AdviceRec, recalc?: Recalc): Face | null {
  if (r.projection) return { memory: r.projection.per_gpu_capacity_gib, hours: r.projection.training_hours, cost: r.projection.cost_usd, fits: r.projection.fits };
  const p = recalc?.result;
  if (p?.status === 'ok') { const t = p.runtime?.total_training_compute; return { memory: p.fit.capacity_gib, hours: t ? t.seconds / 3600 : null, cost: p.cost.cost ?? null, fits: p.fit.fits }; }
  return null;
}

/** The two headline numbers on a card face, as changes from the current plan. */
function FaceMetrics({ face, base, loading }: { face: Face | null; base: Prediction | null; loading: boolean }): JSX.Element {
  const b = base?.status === 'ok' ? base : null; const bt = b?.runtime?.total_training_compute;
  if (!face) return <p className="text-[11px] text-[#8e899b]">{loading ? 'Calculating…' : 'Not estimated: qualitative advice. Open for the steps.'}</p>;
  const dMem = face.memory != null && b ? face.memory - b.fit.capacity_gib : null;
  const dTime = face.hours != null && bt ? face.hours - bt.seconds / 3600 : null;
  const tone = (d: number | null) => (d == null ? 'text-white' : d < 0 ? 'text-[#7ee0b0]' : d > 0 ? 'text-[#ff9aa8]' : 'text-white');
  const sign = (d: number) => (d > 0 ? '+' : d < 0 ? '−' : '±');
  return <dl className="grid grid-cols-2 gap-3">
    <div><dt className="text-[10px] text-[#8e899b]">Memory / GPU</dt><dd className={`font-mono text-sm ${tone(dMem)}`}>{dMem == null ? gib(face.memory) : `${sign(dMem)}${gib(Math.abs(dMem))}`}</dd></div>
    <div><dt className="text-[10px] text-[#8e899b]">Training time</dt><dd className={`font-mono text-sm ${tone(dTime)}`}>{dTime == null ? hours(face.hours) : `${sign(dTime)}${hours(Math.abs(dTime))}`}</dd></div>
  </dl>;
}

function RecCard({ r, labels, values, baseline, recalc, files, open, onToggle, selected, onSelect, applied }: { r: AdviceRec; labels: Record<string, string>; values: Values; baseline: Prediction | null;
  recalc?: Recalc; files: Advice['changed_files']; open: boolean; onToggle: () => void; selected: boolean; onSelect: () => void; applied: boolean }): JSX.Element {
  const changes = Object.entries(r.new_values).filter(([k]) => k in labels && (k.startsWith('training.') || k.startsWith('data.') || k === 'hardware.gpu_name'));
  const Icon = ICONS[r.category] ?? Lightbulb;
  const problems = [...r.violations.map((v) => `Conflicts with your constraints: ${v}.`), ...r.issues];
  const tag = r.estimable ? { label: 'Estimated', cls: 'border-[#2f7a5c] text-[#7ee0b0]' } : r.projection ? { label: 'Approximate', cls: 'border-[#8a7430] text-[#e6d28a]' } : { label: 'Plan only', cls: 'border-white/20 text-[#a7a0bb]' };
  const id = `rec-${r.id}`;
  return <article className={`flex flex-col rounded-lg border p-4 transition ${open ? 'lg:col-span-3' : ''} ${selected ? 'border-violet/70 bg-gradient-to-b from-violet/[0.12] to-transparent' : 'border-white/10 bg-gradient-to-b from-[#14131f] to-[#0e0d16] hover:border-white/25'}`}>
    <div className="flex items-start justify-between gap-2">
      <span className="grid h-9 w-9 place-items-center rounded-md border border-violet/40 bg-violet/10 text-[#b7a6ff]" aria-hidden><Icon size={16} /></span>
      <div className="flex flex-wrap justify-end gap-1">{applied && <span className="rounded-full border border-[#2f7a5c] px-2 py-0.5 text-[10px] text-[#7ee0b0]">● Applied</span>}
        <span className={`rounded-full border px-2 py-0.5 text-[10px] ${tag.cls}`}>● {tag.label}</span></div></div>
    <h3 className="mt-3 text-sm font-medium text-white">{r.title}</h3>
    <p className="mt-1 text-xs leading-5 text-[#a7a0bb]">{r.summary || firstSentence(r.explanation)}</p>
    <div className="mt-2 flex flex-wrap gap-1"><Badge kind="muted">{r.category.replace(/_/g, ' ')}</Badge>{r.strategy && strategyParts(r.strategy).map((x) => <Badge key={x} kind="muted">{x}</Badge>)}
      {r.hardware && <Badge kind="muted">{r.hardware.label}</Badge>}{problems.length > 0 && <Badge kind="bad">{problems.length} issue{problems.length > 1 ? 's' : ''}</Badge>}</div>
    <div className="mt-auto pt-3"><div className="border-t border-white/10 pt-3"><FaceMetrics face={faceOf(r, recalc)} base={baseline} loading={recalc?.status === 'loading'} /></div>
      {r.projection && <p className="mt-1 text-[10px] text-[#e6d28a]">Approximate projection on {r.projection.gpu_count} GPU{r.projection.gpu_count > 1 ? 's' : ''} · {r.projection.fits ? 'fits' : 'exceeds'} usable VRAM</p>}
      <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/10 pt-3">
        <button type="button" aria-expanded={open} aria-controls={id} onClick={onToggle} className="flex items-center gap-1.5 text-xs font-medium text-white hover:text-[#c9bcff]">
          {open ? 'Hide details' : r.launch_snippet || files.length ? 'View implementation' : 'View details'}<ChevronDown size={14} className={`transition ${open ? 'rotate-180' : ''}`} aria-hidden /></button>
        {r.estimable && <label className="flex cursor-pointer items-center gap-1.5 text-xs text-[#c4bfcd]"><input type="checkbox" checked={selected} onChange={onSelect} aria-label={`Select ${r.title}`} />Select</label>}</div></div>

    {open && <div id={id} className="mt-3 space-y-3 border-t border-white/10 pt-3 text-xs">
      <p className="leading-5 text-[#d8d3e3]">{r.explanation}</p>
      {r.expected_effect && <p><span className="text-[#8e899b]">Expected effect:</span> {r.expected_effect}</p>}
      {changes.length > 0 && <div><p className="eyebrow mb-1">Changes</p><ul className="space-y-0.5">{changes.map(([k, v]) => <li key={k}><span className="text-[#8e899b]">{labels[k]}:</span> <strong>{String(values[k] ?? '—')}</strong> → <strong>{String(v)}</strong></li>)}</ul></div>}
      {r.hardware && <p><span className="text-[#8e899b]">Switch hardware to</span> <strong>{r.hardware.label}{r.hardware.region ? ` · ${r.hardware.region}` : ''}</strong>{r.hardware.cost_usd != null ? ` (estimated ${money(r.hardware.cost_usd)})` : ''}</p>}
      {recalc?.result && <div><p className="eyebrow mb-1">Estimate with this change vs your current plan</p><Metrics p={recalc.result} base={baseline} /></div>}
      {recalc?.status === 'error' && <p role="alert" className="text-[#ffb3be]">{recalc.error}</p>}
      {r.projection && <Projection p={r.projection} base={baseline} />}
      {(r.implementation_steps?.length ?? 0) > 0 && <div><p className="eyebrow mb-1">Steps</p><ol className="list-decimal space-y-0.5 pl-4 text-[#d8d3e3]">{r.implementation_steps?.map((x) => <li key={x}>{x}</li>)}</ol></div>}
      {r.launch_snippet && <div><p className="eyebrow mb-1">Launch / config</p><CopyBlock text={r.launch_snippet} label={`${r.title} launch snippet`} /></div>}
      {files.map((f) => <div key={f.path}><p className="eyebrow mb-1">Code: <span className="font-mono normal-case tracking-normal">{f.path}</span> <span className="normal-case tracking-normal text-[#8e899b]">({f.action}) {f.reason}</span></p>
        <CopyBlock text={f.content} label={f.path} /></div>)}
      {r.tradeoffs.length > 0 && <p className="text-[#a7a0bb]">Trade-offs: {r.tradeoffs.join('; ')}</p>}
      {r.estimate_note && !r.projection && <p className="text-[#e6d28a]">{r.estimate_note}</p>}
      {problems.map((m) => <p key={m} className="text-[#ff9aa8]">{m}</p>)}
    </div>}
  </article>;
}

export function RecommendationsPage({ analysisId, body, values, labels, baseline, original, onApplyAdvice, onBack }: {
  analysisId: string; body: () => ConfigBody; values: Values; labels: Record<string, string>; baseline: Prediction | null; original: Prediction | null;
  onApplyAdvice: (recs: AdviceRec[]) => Promise<void>; onBack: () => void;
}): JSX.Element {
  const [status, setStatus] = useState<AdviceStatus | null>(null);
  const [prefs, setPrefs] = useState<AdvicePrefs>(DEFAULT_PREFS);
  const [code, setCode] = useState<Record<string, string>>({}); const [codeError, setCodeError] = useState('');
  const [advice, setAdvice] = useState<Advice | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const [recalc, setRecalc] = useState<Record<string, Recalc>>({}); const [applied, setApplied] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null); const [selected, setSelected] = useState<string[]>([]);
  const [combined, setCombined] = useState<Recalc | null>(null); const [applying, setApplying] = useState(false);
  useEffect(() => { api.adviceStatus().then(setStatus).catch(() => setStatus(null)); }, []);
  const set = <K extends keyof AdvicePrefs>(k: K, v: AdvicePrefs[K]) => setPrefs((p) => ({ ...p, [k]: v }));

  function addFiles(list: FileList | null): void {
    if (!list) return; setCodeError('');
    void Promise.all(Array.from(list).map(async (f) => [f.name.replace(/[^A-Za-z0-9._-]/g, '_'), await f.text()] as const)).then((pairs) => {
      const next = { ...code, ...Object.fromEntries(pairs) };
      if (Object.values(next).reduce((n, t) => n + t.length, 0) > MAX_CODE_BYTES) { setCodeError(`Code files are limited to ${MAX_CODE_BYTES / 1000} KB in total.`); return; }
      setCode(next);
    });
  }
  async function loadExample(): Promise<void> { try { const r = await api.adviceExamples(); setCode((c) => ({ ...c, ...r.files })); } catch (e) { setCodeError(message(e)); } }
  async function preview(changes: Values): Promise<Prediction> { return api.predict(analysisId, { ...body(), values: { ...values, ...changes } }, false); }
  async function ask(): Promise<void> {
    setBusy(true); setError(''); setRecalc({}); setSelected([]); setOpen(null); setCombined(null);
    try {
      const a = await api.advice(analysisId, { ...body(), preferences: prefs, files: code }); setAdvice(a);
      // Estimate every applicable suggestion up front so each card shows its numbers.
      for (const r of a.recommendations.filter((x) => x.estimable)) {
        setRecalc((x) => ({ ...x, [r.id]: { status: 'loading' } }));
        preview(r.new_values).then((result) => setRecalc((x) => ({ ...x, [r.id]: { status: 'done', result } })))
          .catch((e) => setRecalc((x) => ({ ...x, [r.id]: { status: 'error', error: message(e) } })));
      }
    } catch (e) { setError(message(e)); } finally { setBusy(false); }
  }

  const chosen = useMemo(() => (advice?.recommendations ?? []).filter((r) => selected.includes(r.id)), [advice, selected]);
  // Two selected suggestions that set the same input to different values cannot be combined.
  const clashes = useMemo(() => {
    const seen: Record<string, { v: unknown; title: string }> = {}; const out: string[] = [];
    for (const r of chosen) for (const [k, v] of Object.entries(r.new_values)) {
      if (k in seen && JSON.stringify(seen[k].v) !== JSON.stringify(v)) out.push(`${labels[k] ?? k}: "${seen[k].title}" and "${r.title}" set different values`);
      else seen[k] = { v, title: r.title };
    }
    return out;
  }, [chosen, labels]);
  useEffect(() => {
    if (chosen.length < 2 || clashes.length) { setCombined(null); return; }
    let stale = false; setCombined({ status: 'loading' });
    preview(Object.assign({}, ...chosen.map((r) => r.new_values))).then((result) => { if (!stale) setCombined({ status: 'done', result }); })
      .catch((e) => { if (!stale) setCombined({ status: 'error', error: message(e) }); });
    return () => { stale = true; };
  }, [chosen, clashes]); // eslint-disable-line react-hooks/exhaustive-deps

  async function applySelected(): Promise<void> {
    setApplying(true);
    try { await onApplyAdvice(chosen); setApplied((a) => [...new Set([...a, ...chosen.map((r) => r.id)])]); setSelected([]); } catch (e) { setError(message(e)); } finally { setApplying(false); }
  }
  const matched = new Set((advice?.recommendations ?? []).flatMap((r) => r.affected_files));
  const looseFiles = (advice?.changed_files ?? []).filter((f) => !matched.has(f.path));
  const selectedPreview = chosen.length === 1 ? recalc[chosen[0].id] : combined;

  return <div className="space-y-5">
    <PageHeading eyebrow="Optimization opportunities" title="Improve this training plan" subtitle="Tell us what matters, then pick one or more changes to apply together."
      aside={advice ? `${advice.recommendations.length} recommendation${advice.recommendations.length === 1 ? '' : 's'}` : <Badge kind="muted">Powered by NVIDIA Nemotron 3 Super</Badge>} />
    <Section icon={Target} title="Your priorities" hint="Your goal, hardware limits and what must stay the same.">
      {status && !status.configured && <div className="mt-2"><Alert kind="warn"><strong>Not connected yet.</strong> The backend needs <span className="font-mono">NEBIUS_API_KEY</span> (a Nebius Token Factory API key) to call the model.</Alert></div>}
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <fieldset className="border border-line p-3"><legend className="flex items-center gap-1.5 px-1 text-xs font-semibold"><Target size={13} aria-hidden /> What matters most?</legend>
          <div className="grid gap-1 sm:grid-cols-2">{GOALS.map(([id, label]) => <label key={id} className={`flex cursor-pointer items-center gap-2 border px-2 py-1.5 text-xs ${prefs.goal === id ? 'border-violet bg-violet/10' : 'border-line'}`}>
            <input type="radio" name="goal" checked={prefs.goal === id} onChange={() => set('goal', id)} /><strong>{label}</strong></label>)}</div></fieldset>
        <fieldset className="border border-line p-3"><legend className="px-1 text-xs font-semibold">Which hardware may we suggest?</legend>
          {([['current', 'Keep my current GPU'], ['alternatives', 'Consider other GPUs']] as const).map(([id, label]) =>
            <label key={id} className="mb-1 flex items-center gap-2 text-xs"><input type="radio" name="resources" checked={prefs.resources === id} onChange={() => set('resources', id)} /><strong>{label}</strong></label>)}
          <div className="mt-2 grid grid-cols-2 gap-2">
            <label className="text-[11px]">Budget for the run (USD, optional)<input type="number" min={0} step="any" value={prefs.budget_usd ?? ''} onChange={(e) => set('budget_usd', e.target.value === '' ? null : Number(e.target.value))} className="field-sm mt-0.5" placeholder="no limit" /></label>
            <label className="text-[11px]">GPU limit (optional)<input type="number" min={1} step={1} value={prefs.max_gpus ?? ''} onChange={(e) => set('max_gpus', e.target.value === '' ? null : Number(e.target.value))} className="field-sm mt-0.5" placeholder="1" /></label></div>
          <p className="mt-1.5 text-[10px] text-[#8e899b]">Allow 2+ GPUs to get data, tensor and pipeline parallel or ZeRO/FSDP plans. Checkpointing, flash attention and batch/accumulation tuning work on one GPU.</p>
</fieldset>
        <fieldset className="border border-line p-3"><legend className="px-1 text-xs font-semibold">What must stay the same?</legend>
          {KEEP.map(([k, label]) => <label key={k} className="flex items-center gap-2 text-xs"><input type="checkbox" checked={prefs[k] as boolean} onChange={(e) => set(k, e.target.checked as never)} /> Keep {label}</label>)}
          <label className="mt-2 block text-[11px]">Anything else? (optional)<textarea value={prefs.notes} onChange={(e) => set('notes', e.target.value)} maxLength={2000} rows={2} className="mt-0.5 w-full border border-line bg-black/40 p-1.5 text-xs" placeholder="e.g. we must finish within 4 hours" /></label></fieldset>
        <fieldset className="border border-line p-3"><legend className="flex items-center gap-1.5 px-1 text-xs font-semibold"><Code2 size={13} aria-hidden /> Code for advanced suggestions (optional)</legend>
          <p className="text-[11px] text-[#a7a0bb]">Training script, model code, launch command or distributed config.</p>
          <div className="mt-2 flex flex-wrap gap-2"><label className="flex cursor-pointer items-center gap-1 border border-line px-2 py-1 text-xs"><Upload size={12} aria-hidden /> Add files
            <input type="file" multiple accept=".py,.json,.yaml,.yml,.toml,.sh,.txt" className="sr-only" aria-label="Add code files" onChange={(e: ChangeEvent<HTMLInputElement>) => { addFiles(e.target.files); e.target.value = ''; }} /></label>
            <button type="button" onClick={() => void loadExample()} className="flex items-center gap-1 border border-line px-2 py-1 text-xs"><FileCode2 size={12} aria-hidden /> Use example training script</button></div>
          {Object.keys(code).length > 0 && <ul className="mt-2 flex flex-wrap gap-1" aria-label="Code for review">{Object.keys(code).map((n) => <li key={n} className="flex items-center gap-1 border border-line px-2 py-0.5 text-[11px]">{n}<button type="button" aria-label={`Remove ${n}`} onClick={() => setCode(({ [n]: _gone, ...rest }) => rest)}><X size={11} /></button></li>)}</ul>}
          {codeError && <Alert kind="error">{codeError}</Alert>}
</fieldset>
      </div>
      <div className="mt-3 flex items-center gap-3"><button type="button" disabled={busy || !baseline || baseline.status !== 'ok' || status?.configured === false} onClick={() => void ask()}
        className="flex items-center gap-1.5 bg-violet px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"><Sparkles size={14} aria-hidden />{busy ? 'Asking Nemotron…' : 'Get recommendations'}</button>
        {(!baseline || baseline.status !== 'ok') && <span className="text-xs text-[#a7a0bb]">Run a successful estimate first.</span>}</div>
      {error && <div className="mt-2"><Alert kind="error">{error}</Alert></div>}
    </Section>

    {advice && <>
      {baseline && <div className="rounded-lg border border-white/10 p-4"><p className="eyebrow mb-2">Current plan</p>
        <p className="mb-3 text-sm text-[#d8d3e3]">{firstSentence(advice.original_summary)}</p>
        <Metrics p={baseline} base={original && original !== baseline ? original : null} /></div>}
      {advice.missing_information.length > 0 && <p className="text-[11px] text-[#e6d28a]">The model would like: {advice.missing_information.join('; ')}</p>}
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">{advice.recommendations.map((r) => <RecCard key={r.id} r={r} labels={labels} values={values} baseline={baseline} recalc={recalc[r.id]}
        files={advice.changed_files.filter((f) => r.affected_files.includes(f.path))} open={open === r.id} onToggle={() => setOpen(open === r.id ? null : r.id)}
        selected={selected.includes(r.id)} onSelect={() => setSelected((s) => (s.includes(r.id) ? s.filter((x) => x !== r.id) : [...s, r.id]))} applied={applied.includes(r.id)} />)}</div>
      {looseFiles.length > 0 && <Accordion title={`Other proposed code edits (${looseFiles.length})`} hint="syntax-checked only; review before use">
        <div className="space-y-2">{looseFiles.map((f) => <div key={f.path}><p className="mb-1 text-xs"><span className="font-mono">{f.path}</span> <span className="text-[#a7a0bb]">{f.reason}</span></p><CopyBlock text={f.content} label={f.path} /></div>)}</div></Accordion>}

      {chosen.length > 0 && <div className="sticky bottom-3 z-10 rounded-lg border border-violet/60 bg-[#141224]/95 p-3 shadow-glow backdrop-blur" aria-live="polite">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm"><strong>{chosen.length} change{chosen.length > 1 ? 's' : ''} selected</strong> <span className="text-xs text-[#a7a0bb]">{chosen.map((r) => r.title).join(' + ')}</span></p>
          <div className="flex gap-2"><button type="button" onClick={() => setSelected([])} className="border border-line px-3 py-1.5 text-xs">Clear</button>
            <button type="button" onClick={() => void applySelected()} disabled={applying || clashes.length > 0}
              className="flex items-center gap-1.5 bg-white px-4 py-1.5 text-xs font-semibold text-black disabled:opacity-40"><Check size={13} aria-hidden />{applying ? 'Applying…' : `Apply ${chosen.length > 1 ? `${chosen.length} changes` : 'change'} and re-estimate`}</button></div></div>
        {clashes.map((c) => <p key={c} className="mt-1 text-[11px] text-[#ff9aa8]">{c}</p>)}
        {selectedPreview?.status === 'loading' && <p className="mt-2 text-xs">Calculating the combined estimate…</p>}
        {selectedPreview?.status === 'error' && <p className="mt-2 text-xs text-[#ffb3be]">{selectedPreview.error}</p>}
        {selectedPreview?.result && <div className="mt-2"><Metrics p={selectedPreview.result} base={baseline} /></div>}
      </div>}

      <div className="flex flex-wrap items-center gap-2"><a href={api.adviceExportUrl(analysisId)} className={`flex items-center gap-1.5 border border-line px-3 py-1.5 text-xs ${applied.length ? '' : 'pointer-events-none opacity-40'}`} aria-disabled={!applied.length}>
        <Download size={12} aria-hidden />{advice.changed_files.length ? 'Download edited code + config' : 'Download training config'}</a>
        <span className="text-[11px] text-[#8e899b]">{applied.length ? 'Includes your applied changes. Your original inputs stay saved as the original estimate.' : 'Apply a suggestion first.'}</span></div>
    </>}

    <div className="border-t border-white/10 pt-5"><button type="button" onClick={onBack} className="flex items-center gap-1.5 rounded-md border border-white/15 px-5 py-2 text-sm"><ArrowLeft size={14} aria-hidden /> Back to results</button></div>
  </div>;
}
