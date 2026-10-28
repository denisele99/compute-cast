import { ChangeEvent, useState } from 'react';
import { BenchmarkPlan, BenchmarkResult, CompareRow, Cost, Json, Prediction, api } from '../api';
import { Accordion, Alert, Badge, CopyBlock, MemoryBar, Section, Unavailable } from '../components/ui';
import { classifyFit } from '../lib/fit';

export type Async<T> = { status: 'idle' | 'loading' | 'error' | 'ok'; data?: T; error?: string };
const tone = { fits: 'good', tight: 'warn', exceeds: 'bad', unknown: 'muted' } as const;
export const gib = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v > 0 && v < 0.01 ? `${(v * 1024).toFixed(2)} MiB` : `${v.toFixed(2)} GiB`);
export const hours = (h: number | null | undefined) => (h === null || h === undefined ? '—' : h * 3600 < 60 ? `${(h * 3600).toFixed(1)} s` : h < 1 ? `${(h * 60).toFixed(1)} min` : `${h.toFixed(2)} h`);
export const money = (c: number | null | undefined, cur = 'USD') => (c === null || c === undefined ? '—' : c > 0 && c < 0.01 ? `< ${cur === 'USD' ? '$' : `${cur} `}0.01`
  : `${cur === 'USD' ? '$' : `${cur} `}${c.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const num = (v: number | undefined) => (v === undefined ? '—' : v >= 0.01 ? v.toFixed(2) : v.toPrecision(2));
const show = (v: Json) => (v === null || v === undefined || v === '' ? '(empty)' : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v));

function CostLine({ c }: { c: Cost | undefined }): JSX.Element {
  if (!c || c.status !== 'estimated') return <span className="text-[#a7a0bb]">{c?.reason ?? 'Unavailable'}</span>;
  return <span><strong>{money(c.cost, c.currency)}</strong><span className="block text-[11px] text-[#8e899b]">{hours(c.elapsed_hours)} elapsed × {c.unit === 'gpu_hour' ? `${c.gpus_allocated} GPU(s) × $${c.price_per_hour}/GPU-hour` : `${c.instances} instance(s) × $${c.price_per_hour}/instance-hour`} · {num(c.gpu_hours_used)} GPU-h used, {num(c.gpu_hours_billed)} billed{c.warning ? ` · ${c.warning}` : ''}</span></span>;
}

function PredictionView({ p, analysisId, compare, onCompare, onGoField, labels }: { p: Prediction; analysisId: string; compare: Async<{ rows: CompareRow[]; note: string }>; onCompare: () => void; onGoField: (k: string) => void; labels: Record<string, string> }): JSX.Element {
  if (p.status === 'blocked') {
    const v = p.validation;
    return <Alert kind="error"><strong>Prediction unavailable — required information is missing or unsupported.</strong>
      <ul className="mt-2 list-disc pl-5">{[...v.unsupported.map((u) => ({ f: u.field, m: `Unsupported: ${u.message}` })), ...v.errors.map((e) => ({ f: e.field, m: e.message })),
        ...v.missing.filter((m) => m.required_for.includes('memory')).map((m) => ({ f: m.field, m: `${m.label} is required` }))].map((x) =>
        <li key={x.f + x.m}><button type="button" className="underline" onClick={() => onGoField(x.f)}>{labels[x.f] ?? x.f}</button>: {x.m}</li>)}</ul></Alert>;
  }
  const fit = p.fit; const cls = fit ? classifyFit(fit.peak_vram_gib, fit.usable_vram_gib) : null;
  const rt = p.runtime; const sc = rt.scenarios ?? {};
  const changes = p.snapshots?.changes ?? []; const orig = p.snapshots?.original;
  return <div className="space-y-6">
    <div className="flex flex-wrap gap-2"><Badge kind="muted">Prediction · metadata only</Badge><Badge kind="muted">no code executed</Badge>{p.validation.warnings.length > 0 && <Badge kind="warn">{p.validation.warnings.length} warning(s)</Badge>}</div>
    {fit && cls ? <div className={`border p-5 ${cls.kind === 'exceeds' ? 'border-[#8a3040]' : cls.kind === 'tight' ? 'border-[#8a7430]' : 'border-line'}`}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><Badge kind={tone[cls.kind]}>{fit.fits ? (cls.kind === 'tight' ? 'Fits, within 10% of the limit' : 'Fits within usable VRAM') : 'Does not fit usable VRAM'}</Badge>
        <span className="text-xs text-[#a7a0bb]">Estimated peak per GPU</span></div>
      <MemoryBar peak={fit.peak_vram_gib} usable={fit.usable_vram_gib} total={fit.total_vram_gib} />
      <dl className="mt-5 grid gap-4 text-sm sm:grid-cols-3"><div><dt className="eyebrow">Peak VRAM / GPU</dt><dd className="mt-1 text-2xl font-semibold">{gib(fit.peak_vram_gib)}</dd></div>
        <div><dt className="eyebrow">Usable VRAM / GPU</dt><dd className="mt-1 text-2xl font-semibold">{gib(fit.usable_vram_gib)}</dd></div><div><dt className="eyebrow">Total VRAM / GPU</dt><dd className="mt-1 text-2xl font-semibold">{gib(fit.total_vram_gib)}</dd></div></dl>
      <p className="mt-3 text-xs text-[#a7a0bb]">{cls.message}</p>
      {(p.memory.cpu || p.memory.nvme) && <dl className="mt-4 grid gap-4 border-t border-line pt-4 text-sm sm:grid-cols-2">
        {p.memory.cpu && <div><dt className="eyebrow">CPU RAM per node</dt><dd>{gib(p.memory.cpu.required_ram_gib_per_node)} needed · {p.memory.cpu.available_gib_per_node === null ? 'available RAM not entered' : `${gib(p.memory.cpu.available_gib_per_node)} available (${p.memory.cpu.fits ? 'fits' : 'does not fit'})`}</dd></div>}
        {p.memory.nvme && <div><dt className="eyebrow">NVMe per node</dt><dd>{gib(p.memory.nvme.required_storage_gib_per_node)} needed · {p.memory.nvme.available_gib_per_node === null ? 'capacity not entered' : `${gib(p.memory.nvme.available_gib_per_node)} available`}</dd></div>}</dl>}
      {p.memory.warnings?.map((w) => <p key={w} className="mt-2 text-xs text-[#e6d28a]">{w}</p>)}
    </div> : <Alert kind="error"><strong>Peak VRAM unavailable.</strong> {(p.memory.errors ?? []).join(' ')}</Alert>}

    {p.parameters && <p className="text-sm text-[#c4bfcd]">{p.parameters.total_params.toLocaleString()} parameters ({p.parameters.trainable_params.toLocaleString()} trainable), {gib(p.parameters.logical_weight_bytes / 2 ** 30)} of weights{p.profile_estimates ? ` · saved activations ${gib(p.profile_estimates.saved_activation_bytes / 2 ** 30)} per microbatch` : ''}.</p>}

    <div><h3 className="mb-2 text-sm font-semibold">Runtime and cost scenarios</h3>
      <div className="overflow-x-auto"><table className="w-full min-w-[640px] text-left text-sm"><thead className="text-xs text-[#8e899b]"><tr><th className="py-2">Scenario</th><th>Training time</th><th>Job elapsed (incl. startup)</th><th>Estimated job cost</th></tr></thead>
        <tbody>{(['fast', 'slow'] as const).map((k) => { const s = sc[k]; return <tr key={k} className="border-t border-line align-top"><td className="py-3 capitalize">{k}</td>
          <td>{s?.training_hours != null ? hours(s.training_hours) : <span className="text-xs text-[#a7a0bb]">Unavailable{s?.gpu_core_only_forward_backward_seconds != null ? ` (GPU compute-only lower bound: ${hours(s.gpu_core_only_forward_backward_seconds / 3600)})` : ''}</span>}</td>
          <td>{hours(s?.job_hours)}</td><td><CostLine c={p.cost[k] ?? p.cost.unavailable} /></td></tr>; })}</tbody></table></div>
      {rt.schedule && <p className="mt-2 text-xs text-[#8e899b]">{rt.schedule.optimizer_updates.toLocaleString()} optimizer updates · {rt.schedule.microbatches_per_rank.toLocaleString()} microbatches per GPU · effective global batch {rt.schedule.effective_global_batch_size}</p>}
      {rt.missing_information?.length > 0 && <div className="mt-3"><Alert kind="warn"><strong>Runtime/cost incomplete — missing information:</strong><ul className="mt-1 list-disc pl-5">{rt.missing_information.map((m) => <li key={m}>{m}</li>)}</ul></Alert></div>}
      <p className="mt-2 text-xs text-[#8e899b]">Fast/slow are editable planning scenarios (efficiency and overlap assumptions), not confidence bounds.</p></div>

    <div className="border border-line p-4"><p className="eyebrow">Estimated benchmark cost</p>
      {p.benchmark_cost.status === 'estimated' ? <p className="mt-2 text-sm"><strong>{money(p.benchmark_cost.cost, p.benchmark_cost.currency)}</strong> for {p.benchmark_cost.warmup_steps} warm-up + {p.benchmark_cost.steps} measured steps ({p.benchmark_cost.seconds_per_optimizer_step?.toFixed(3)} s/step, slow scenario) plus {p.benchmark_cost.setup_minutes} min setup = {hours(p.benchmark_cost.elapsed_hours)} elapsed. <span className="block text-xs text-[#8e899b]">{p.benchmark_cost.assumption}</span></p>
        : <p className="mt-2 text-sm text-[#a7a0bb]">{p.benchmark_cost.reason}</p>}</div>

    <div><h3 className="mb-2 text-sm font-semibold">Changes from the original analysis</h3>
      {!orig || p.snapshots.is_original_run ? <p className="text-xs text-[#8e899b]">This is the original run; its settings are kept as an immutable snapshot and later runs are compared with it.</p> : changes.length === 0 ? <p className="text-xs text-[#8e899b]">No changes from the original run.</p> :
        <ul className="space-y-1 text-sm">{changes.map((c) => <li key={c.field}>Original {c.label.charAt(0).toLowerCase() + c.label.slice(1)}: <strong>{show(c.original)}</strong>; current: <strong>{show(c.current)}</strong></li>)}</ul>}
      {orig?.result_summary?.peak_vram_gib != null && fit && !p.snapshots.is_original_run && <p className="mt-2 text-xs text-[#a7a0bb]">Original peak VRAM {gib(orig.result_summary.peak_vram_gib)} → current {gib(fit.peak_vram_gib)}. Original slow-scenario job time {hours(orig.result_summary.job_hours?.slow)} → current {hours(sc.slow?.job_hours)}.</p>}</div>

    <div><div className="mb-2 flex items-center justify-between"><h3 className="text-sm font-semibold">Hardware comparison (same workload)</h3><button type="button" onClick={onCompare} disabled={compare.status === 'loading'} className="border border-line px-3 py-1 text-xs">{compare.status === 'loading' ? 'Comparing…' : 'Compare Nebius GPUs'}</button></div>
      {compare.status === 'error' && <Alert kind="error">{compare.error}</Alert>}
      {compare.data && <><div className="overflow-x-auto"><table className="w-full min-w-[720px] text-left text-sm"><thead className="text-xs text-[#8e899b]"><tr><th className="py-2">GPU</th><th>Peak / usable</th><th>Fit</th><th>Job time (fast–slow)</th><th>Cost (fast–slow)</th><th>Price</th></tr></thead><tbody>
        {compare.data.rows.map((r) => { const f = r.fit ? classifyFit(r.fit.peak_vram_gib, r.fit.usable_vram_gib) : null; return <tr key={r.id} className="border-t border-line align-top">
          <td className="py-2">{r.label}{r.region ? ` · ${r.region}` : ''}{r.issues.length > 0 && <span className="block text-[11px] text-[#e6d28a]">{r.issues[0]}</span>}</td>
          <td>{r.fit ? `${gib(r.fit.peak_vram_gib)} / ${gib(r.fit.usable_vram_gib)}` : '—'}</td><td>{f ? <Badge kind={tone[f.kind]}>{f.label}</Badge> : <span className="text-xs">{(r.reason ?? []).join(', ') || 'unavailable'}</span>}</td>
          <td>{r.job_hours?.fast != null ? `${hours(r.job_hours.fast)} – ${hours(r.job_hours.slow)}` : <span className="text-xs text-[#a7a0bb]">{r.runtime_missing?.[0] ?? 'unavailable'}</span>}</td>
          <td>{r.cost?.fast != null ? `${money(r.cost.fast)} – ${money(r.cost.slow)}` : '—'}</td><td className="text-xs">{r.price?.value != null ? `$${r.price.value}/${r.price.unit === 'gpu_hour' ? 'GPU-h' : 'inst-h'}${r.price.preemptible ? ' spot' : ''}` : 'no price'}</td></tr>; })}</tbody></table></div>
        <p className="mt-2 text-xs text-[#8e899b]">{compare.data.note}</p></>}</div>

    <Accordion title="How we calculated this" hint="formulas, defaults, assumptions, limitations">
      <div className="space-y-4 text-sm text-[#c4bfcd]">
        <p><strong>Profile:</strong> {p.explanation.profile.label}. {p.explanation.profile.summary}</p>
        {[['Parameters and activations', p.explanation.parameter_and_activation_formulas], ['Peak memory', p.explanation.memory_formulas], ['Runtime', p.explanation.runtime_formulas]].map(([title, formulas]) =>
          <div key={title as string}><p className="eyebrow mb-2">{title as string}</p><ul className="space-y-2">{Object.entries(formulas as Record<string, { formula: string; basis?: string; sources?: string[] }>).map(([k, f]) =>
            <li key={k}><span className="font-mono text-xs">{f.formula}</span>{f.basis && <span className="block text-xs text-[#8e899b]">{f.basis}</span>}{f.sources?.map((s) => <a key={s} href={s.startsWith('http') ? s : undefined} className="mr-2 text-[11px] underline" target="_blank" rel="noreferrer">{s.startsWith('http') ? new URL(s).hostname : s}</a>)}</li>)}</ul></div>)}
        <div><p className="eyebrow mb-2">Cost</p><ul className="list-disc pl-5 text-xs">{Object.values(p.explanation.cost_formulas).map((f) => <li key={f} className="font-mono">{f}</li>)}</ul></div>
        {p.memory.breakdown_gib && <div><p className="eyebrow mb-2">Peak VRAM breakdown per GPU</p><table className="text-xs"><tbody>{Object.entries(p.memory.breakdown_gib).map(([k, v]) => <tr key={k}><td className="pr-6">{k.replace(/_/g, ' ')}</td><td>{gib(v)}</td></tr>)}</tbody></table></div>}
        {p.explanation.memory_planning_assumptions && <p className="text-xs">Memory planning allowances: {Object.entries(p.explanation.memory_planning_assumptions).map(([k, v]) => `${k.replace(/_/g, ' ')} = ${v}`).join(' · ')}</p>}
        {p.explanation.defaults_used.length > 0 && <p className="text-xs">Default assumptions in use: {p.explanation.defaults_used.map((k) => labels[k] ?? k).join(', ')}.</p>}
        <div><p className="eyebrow mb-2">Assumptions</p><ul className="list-disc space-y-1 pl-5 text-xs">{[...p.explanation.memory_notes, ...p.explanation.runtime_assumptions].map((a) => <li key={a}>{a}</li>)}</ul></div>
        <div><p className="eyebrow mb-2">Limitations</p><ul className="list-disc space-y-1 pl-5 text-xs">{p.explanation.limitations.map((a) => <li key={a}>{a}</li>)}</ul></div>
        <p className="text-xs">Inputs passed to the estimators: <a className="underline" href={api.fileUrl(analysisId, 'workload.json')}>workload.json</a> · <a className="underline" href={api.fileUrl(analysisId, 'hardware.json')}>hardware.json</a></p>
      </div></Accordion>
  </div>;
}

function BenchmarkView({ plan, onPlan, onDownload, onImport, results, prediction }: { plan: Async<BenchmarkPlan>; onPlan: () => void; onDownload: () => void; onImport: (f: File) => void; results: Async<BenchmarkResult[]>; prediction: Prediction | null }): JSX.Element {
  const [confirm, setConfirm] = useState(false);
  const pl = plan.data;
  return <div className="space-y-6">
    <Alert kind="warn">Benchmarking runs real training steps and needs GPU access. On paid cloud services it incurs charges. Benchmark results are measurements and are kept separate from metadata-only predictions.</Alert>
    {plan.status === 'idle' && <button type="button" onClick={onPlan} disabled={!prediction || prediction.status !== 'ok'} className="bg-white px-5 py-3 text-sm font-semibold text-black disabled:opacity-50">Prepare benchmark</button>}
    {!prediction || prediction.status !== 'ok' ? <p className="text-xs text-[#a7a0bb]">Run a successful prediction first; the benchmark uses the same validated configuration.</p> : null}
    {plan.status === 'loading' && <p aria-live="polite">Preparing…</p>}
    {plan.status === 'error' && <Alert kind="error">{plan.error}</Alert>}
    {pl && <>
      <div><h3 className="mb-2 text-sm font-semibold">Requirements</h3><ul className="list-disc space-y-1 pl-5 text-sm">{pl.requirements.environment.map((e) => <li key={e}>{e}</li>)}</ul>
        <ul className="mt-3 space-y-1 text-sm">{pl.requirements.files.map((f) => <li key={f.flag}><span className="font-mono">{f.flag}</span> {f.required ? '(required)' : '(optional)'} — {f.description} {f.provided.length ? <span className="text-[#7ee0b0]">Uploaded: {f.provided.join(', ')}</span> : <span className="text-[#8e899b]">Not uploaded.</span>}</li>)}</ul>
        <p className="mt-2 text-xs text-[#8e899b]">Credentials: {pl.requirements.credentials.join(' ')}</p></div>
      <Section number="L" title="Run locally" hint="Download the generated script and run it on your own GPU machine.">
        <div className="space-y-3">{pl.requirements.warnings.map((w) => <Alert key={w} kind="warn">{w}</Alert>)}
          <button type="button" onClick={onDownload} className="bg-white px-5 py-3 text-sm font-semibold text-black">Download run_benchmark.py</button>
          <p className="text-xs text-[#a7a0bb]">Command for your uploaded files (run in the folder containing them):</p><CopyBlock text={pl.command} label="benchmark command" />
          <p className="text-xs text-[#a7a0bb]">General form:</p><CopyBlock text={pl.command_template} label="benchmark command template" />
          <label className="block border border-dashed border-line p-4 text-sm">Import benchmark result (benchmark_result.json)<input type="file" accept=".json,application/json" className="mt-2 block text-xs" onChange={(e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) onImport(f); e.target.value = ''; }} /></label>
          {results.status === 'error' && <Alert kind="error">{results.error}</Alert>}
          {results.data?.map((r, i) => <div key={i} className="border border-line p-4 text-sm"><div className="flex flex-wrap gap-2"><Badge kind="good">Measured · benchmark</Badge><Badge kind={r.matches_current_config ? 'good' : 'warn'}>{r.matches_current_config ? 'matches current config' : 'different config'}</Badge></div>
            <p className="mt-2">{r.device?.name} · torch {r.torch_version} · {r.model_source.replace(/_/g, ' ')} · {r.data_source} data{r.user_code_executed ? ' · user code executed' : ''}</p>
            <p className="mt-1">Median {r.seconds_per_optimizer_step.median.toFixed(4)} s/optimizer step · peak reserved {gib(r.peak_memory_reserved_gib)} (allocated {gib(r.peak_memory_allocated_gib)})</p>
            {r.comparison.peak_vram && <p className="mt-1 text-xs">Predicted peak {gib(r.comparison.peak_vram.predicted_gib)} vs measured reserved {gib(r.comparison.peak_vram.measured_reserved_gib)} (×{r.comparison.peak_vram.ratio_measured_to_predicted.toFixed(2)})</p>}
            {r.comparison.seconds_per_optimizer_step && <p className="mt-1 text-xs">Predicted {r.comparison.seconds_per_optimizer_step.predicted_fast?.toFixed(4)}–{r.comparison.seconds_per_optimizer_step.predicted_slow.toFixed(4)} s/step vs measured {r.comparison.seconds_per_optimizer_step.measured_median.toFixed(4)} s/step</p>}
            {[...r.issues, ...r.notes].map((m) => <p key={m} className="mt-1 text-xs text-[#e6d28a]">{m}</p>)}</div>)}
        </div></Section>
      <Section number="N" title="Run on Nebius" hint="Predicted charges are shown before anything could run.">
        <p className="text-sm">Predicted benchmark charge: <strong>{pl.predicted_cost.status === 'estimated' ? money(pl.predicted_cost.cost) : 'unavailable'}</strong> {pl.predicted_cost.status === 'estimated' ? `(${hours(pl.predicted_cost.elapsed_hours)} elapsed on ${pl.predicted_cost.gpus_allocated} GPU(s))` : `— ${pl.predicted_cost.reason}`}</p>
        <div className="mt-3"><Unavailable label="Unavailable">{pl.nebius.reason} Nothing is provisioned and no charges can be incurred from this page.</Unavailable></div>
        <ul className="mt-3 list-disc pl-5 text-xs text-[#a7a0bb]">{pl.nebius.remaining_work.map((w) => <li key={w}>{w}</li>)}</ul>
        <label className="mt-4 flex items-center gap-2 text-xs text-[#8e899b]"><input type="checkbox" checked={confirm} disabled onChange={(e) => setConfirm(e.target.checked)} /> I confirm the predicted charges (enabled once the integration exists)</label>
        <button type="button" disabled className="mt-3 border border-line px-4 py-2 text-sm opacity-40">Start paid Nebius benchmark — unavailable</button>
      </Section>
    </>}
  </div>;
}

export function ResultsPage(props: {
  mode: 'prediction' | 'benchmark'; setMode: (m: 'prediction' | 'benchmark') => void; prediction: Async<Prediction>; analysisId: string; labels: Record<string, string>;
  compare: Async<{ rows: CompareRow[]; note: string }>; onCompare: () => void; onGoField: (k: string) => void; onEdit: () => void; onRecs: () => void;
  plan: Async<BenchmarkPlan>; onPlan: () => void; onDownload: () => void; onImport: (f: File) => void; benchmarks: Async<BenchmarkResult[]>;
}): JSX.Element {
  const { mode, setMode, prediction } = props;
  return <div className="space-y-6"><Section id="results" number="05" title="Results" hint="One page, two modes. Prediction is metadata-only; benchmark results are measurements.">
    <div className="mb-6 flex gap-2" role="tablist" aria-label="Results mode">{(['prediction', 'benchmark'] as const).map((m) => <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={`border px-4 py-2 text-sm capitalize ${mode === m ? 'border-violet bg-violet/15' : 'border-line text-[#a7a0bb]'}`}>{m}</button>)}</div>
    {mode === 'prediction' ? <>
      {prediction.status === 'loading' && <p aria-live="polite" className="text-sm">Calculating prediction…</p>}
      {prediction.status === 'error' && <Alert kind="error"><strong>Prediction failed.</strong> {prediction.error}</Alert>}
      {prediction.status === 'idle' && <p className="text-sm text-[#a7a0bb]">Run a prediction from the Hardware step.</p>}
      {prediction.data && <PredictionView p={prediction.data} analysisId={props.analysisId} compare={props.compare} onCompare={props.onCompare} onGoField={props.onGoField} labels={props.labels} />}
    </> : <BenchmarkView plan={props.plan} onPlan={props.onPlan} onDownload={props.onDownload} onImport={props.onImport} results={props.benchmarks} prediction={prediction.data ?? null} />}
  </Section>
  <div className="flex justify-between"><button type="button" onClick={props.onEdit} className="border border-line px-5 py-3 text-sm">Edit inputs</button><button type="button" onClick={props.onRecs} disabled={prediction.data?.status !== 'ok'} className="bg-violet px-6 py-3 text-sm font-semibold text-white disabled:opacity-40">Improve this plan</button></div></div>;
}
