import { ChangeEvent, useState } from 'react';
import { BenchmarkPlan, BenchmarkResult, CompareRow, Cost, Json, Prediction, api } from '../api';
import { Accordion, Alert, Badge, CopyBlock, MemoryBar, Section, Unavailable } from '../components/ui';

export type Async<T> = { status: 'idle' | 'loading' | 'error' | 'ok'; data?: T; error?: string };
export const gib = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v > 0 && v < 0.01 ? `${(v * 1024).toFixed(2)} MiB` : `${v.toFixed(2)} GiB`);
export const hours = (h: number | null | undefined) => (h === null || h === undefined ? '—' : h * 3600 < 60 ? `${(h * 3600).toFixed(1)} s` : h < 1 ? `${(h * 60).toFixed(1)} min` : `${h.toFixed(2)} h`);
export const money = (c: number | null | undefined) => (c === null || c === undefined ? '—' : c > 0 && c < 0.01 ? '< $0.01' : `$${c.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const secs = (s: number) => (s < 1 ? `${(s * 1000).toFixed(1)} ms` : `${s.toFixed(3)} s`);
const show = (v: Json) => (v === null || v === undefined || v === '' ? '(empty)' : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v));
const MEMORY_ROWS: [string, string][] = [['training_state', 'Training state (16·P: FP32 weights, grads, 2 AdamW moments)'], ['autocast_cache', 'BF16 autocast weight copies'],
  ['saved_activations', 'Saved activations (incl. dropout masks)'], ['loss_buffers', 'Logits / loss buffers'], ['backward_scratch', 'Backward scratch'],
  ['forward_backward_phase', 'Forward/backward phase total'], ['optimizer_phase', 'Optimizer-step phase total'], ['estimated_allocated_peak', 'Allocated peak = max(phases)'],
  ['runtime_baseline', 'Runtime baseline (CUDA context etc.)'], ['capacity_with_runtime_and_margin', 'Capacity needed = (peak + baseline) × 1.15']];

function CostLine({ c }: { c: Cost }): JSX.Element {
  if (c.status !== 'estimated') return <span className="text-xs text-[#a7a0bb]">{c.reason}</span>;
  return <span><strong className="text-lg">{money(c.cost)}</strong><span className="block text-[11px] text-[#8e899b]">{hours(c.elapsed_hours)} × 1 GPU × ${c.price_per_hour}/{c.unit === 'gpu_hour' ? 'GPU-hour' : 'instance-hour'}{c.note ? ` · ${c.note}` : ''}</span></span>;
}

function PredictionView({ p, analysisId, compare, onCompare, onGoField, labels }: { p: Prediction; analysisId: string; compare: Async<{ rows: CompareRow[]; note: string }>; onCompare: () => void; onGoField: (k: string) => void; labels: Record<string, string> }): JSX.Element {
  if (p.status === 'blocked') {
    const v = p.validation;
    const items = [...v.errors.map((e) => ({ f: e.field, m: e.message })), ...v.missing.filter((m) => m.required_for.includes('memory')).map((m) => ({ f: m.field, m: `${m.label} is required` }))];
    return <Alert kind="error"><strong>Estimate unavailable — resolve these first:</strong><ul className="mt-1 list-disc pl-4">{items.map((x) =>
      <li key={x.f + x.m}><button type="button" className="underline" onClick={() => onGoField(x.f)}>{labels[x.f] ?? x.f}</button>: {x.m}</li>)}</ul></Alert>;
  }
  const { fit, runtime } = p; const d = runtime?.total_training_compute;
  const orig = p.snapshots?.original;
  return <div className="space-y-4">
    <div className="flex flex-wrap gap-1.5"><Badge kind="muted">Analytical estimate · metadata only</Badge><Badge kind="muted">uncalibrated</Badge><Badge kind="muted">no code executed</Badge>
      {p.hypothetical.length > 0 && <Badge kind="warn">hypothetical scenario</Badge>}</div>
    {p.hypothetical.length > 0 && <Alert kind="warn"><strong>Hypothetical:</strong> this differs from your files — {p.hypothetical.join('; ')}.</Alert>}

    <div className={`border p-4 ${fit.fits ? 'border-line' : 'border-[#8a3040]'}`}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><Badge kind={fit.fits ? (fit.percent_of_usable > 90 ? 'warn' : 'good') : 'bad'}>{fit.fits ? 'Estimated fit' : 'Exceeds usable VRAM'}</Badge>
        <span className="text-[11px] text-[#a7a0bb]">{fit.rule}</span></div>
      <MemoryBar used={fit.capacity_gib} usable={fit.usable_vram_gib} total={fit.total_vram_gib} label="Capacity needed" />
      <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div><dt className="eyebrow">Allocated peak / GPU</dt><dd className="text-lg font-semibold">{gib(fit.allocated_peak_gib)}</dd></div>
        <div><dt className="eyebrow">Capacity needed</dt><dd className="text-lg font-semibold">{gib(fit.capacity_gib)}</dd></div>
        <div><dt className="eyebrow">Usable limit</dt><dd className="text-lg font-semibold">{gib(fit.usable_vram_gib)}</dd></div>
        <div><dt className="eyebrow">Total VRAM</dt><dd className="text-lg font-semibold">{gib(fit.total_vram_gib)}</dd></div></dl>
      <p className="mt-2 text-xs text-[#c4bfcd]">{p.parameters.toLocaleString()} parameters · head dim {p.head_dim} · MLP width {p.resolved_intermediate_size} · effective batch {p.effective_batch_size}</p>
    </div>

    <div className="grid gap-3 lg:grid-cols-3">
      <div className="border border-line p-3 lg:col-span-2"><p className="eyebrow mb-1">Training compute time</p>
        {runtime && d ? <><p className="text-2xl font-semibold">{hours(d.seconds / 3600)}</p>
          <p className="text-[11px] text-[#c4bfcd]">{d.optimizer_updates.toLocaleString()} optimizer updates × {secs(runtime.seconds_per_full_optimizer_update)} per update · {secs(runtime.seconds_per_microbatch)} per microbatch · {d.processed_positions.toLocaleString()} token positions{d.mode === 'epochs' ? ` · ${d.microbatches_per_epoch?.toLocaleString()} microbatches/epoch` : ''}</p>
          {!fit.fits && <p className="mt-1 text-[11px] text-[#ff9aa8]">Hypothetical timing: the memory estimate exceeds the usable limit.</p>}</>
          : <p className="text-xs text-[#a7a0bb]">{p.runtime_error ?? `Needs: ${p.runtime_missing.join(', ')}`}</p>}
        <p className="mt-1 text-[10px] text-[#8e899b]">Excludes data loading, evaluation, checkpoint saving and startup (v1 scope).</p></div>
      <div className="border border-line p-3"><p className="eyebrow mb-1">Estimated cost</p><CostLine c={p.cost} />
        <p className="eyebrow mb-1 mt-3">Benchmark run cost</p>
        {p.benchmark_cost.status === 'estimated' ? <p className="text-xs"><strong>{money(p.benchmark_cost.cost)}</strong> for {p.benchmark_cost.warmup_steps}+{p.benchmark_cost.steps} steps <span className="block text-[10px] text-[#8e899b]">{p.benchmark_cost.assumption}</span></p>
          : <p className="text-xs text-[#a7a0bb]">{p.benchmark_cost.reason}</p>}</div>
    </div>

    <Accordion title="Memory breakdown" hint="per GPU" defaultOpen>
      <table className="w-full text-xs"><tbody>{MEMORY_ROWS.map(([k, l]) => <tr key={k} className={`border-t border-line ${['estimated_allocated_peak', 'capacity_with_runtime_and_margin'].includes(k) ? 'font-semibold' : ''}`}><td className="py-1">{l}</td><td className="text-right">{gib(p.memory[k])}</td></tr>)}</tbody></table>
    </Accordion>

    <div><h3 className="mb-1 text-xs font-semibold">Changes from the original analysis</h3>
      {!orig || p.snapshots.is_original_run ? <p className="text-[11px] text-[#8e899b]">This is the original run; its settings are kept as an immutable snapshot and later runs are compared with it.</p>
        : p.snapshots.changes.length === 0 ? <p className="text-[11px] text-[#8e899b]">No changes from the original run.</p>
        : <ul className="space-y-0.5 text-xs">{p.snapshots.changes.map((c) => <li key={c.field}>Original {c.label.charAt(0).toLowerCase() + c.label.slice(1)}: <strong>{show(c.original)}</strong>; current: <strong>{show(c.current)}</strong></li>)}</ul>}
      {orig && !p.snapshots.is_original_run && <p className="mt-1 text-[11px] text-[#a7a0bb]">Original capacity {gib(orig.result_summary.capacity_gib)} → current {gib(fit.capacity_gib)} · original time {hours(orig.result_summary.training_hours)} → current {hours(d ? d.seconds / 3600 : null)}</p>}</div>

    <div><div className="mb-1 flex items-center justify-between"><h3 className="text-xs font-semibold">Hardware comparison (same workload)</h3><button type="button" onClick={onCompare} disabled={compare.status === 'loading'} className="border border-line px-2 py-0.5 text-[11px]">{compare.status === 'loading' ? 'Comparing…' : 'Compare Nebius GPUs'}</button></div>
      {compare.status === 'error' && <Alert kind="error">{compare.error}</Alert>}
      {compare.data && <><div className="overflow-x-auto"><table className="w-full min-w-[640px] text-left text-xs"><thead className="text-[10px] text-[#8e899b]"><tr><th className="py-1">GPU</th><th>Capacity / usable</th><th>Fit</th><th>Time</th><th>Cost</th><th>Price</th></tr></thead><tbody>
        {compare.data.rows.map((r) => <tr key={r.id} className="border-t border-line align-top"><td className="py-1">{r.label}{r.region ? ` · ${r.region}` : ''}</td>
          <td>{r.fit ? `${gib(r.fit.capacity_gib)} / ${gib(r.fit.usable_vram_gib)}` : '—'}</td><td>{r.fit ? <Badge kind={r.fit.fits ? 'good' : 'bad'}>{r.fit.fits ? 'fits' : 'exceeds'}</Badge> : '—'}</td>
          <td>{r.hours != null ? hours(r.hours) : <span className="text-[#a7a0bb]">no specs</span>}</td><td>{r.cost != null ? money(r.cost) : '—'}</td>
          <td>{r.price?.value != null ? `$${r.price.value}/${r.price.unit === 'gpu_hour' ? 'GPU-h' : 'inst-h'}${r.price.preemptible ? ' spot' : ''}` : 'no price'}</td></tr>)}</tbody></table></div>
        <p className="mt-1 text-[10px] text-[#8e899b]">{compare.data.note}</p></>}</div>

    <Accordion title="How we calculated this" hint="formulas, assumptions, limitations">
      <div className="space-y-3 text-xs text-[#c4bfcd]">
        <p>{p.estimator}. Scope: {p.explanation.scope}.</p>
        <ul className="space-y-1">{Object.entries(p.explanation.formulas).map(([k, f]) => <li key={k}><span className="eyebrow mr-2">{k.replace(/_/g, ' ')}</span><span className="font-mono text-[11px]">{f}</span></li>)}</ul>
        <p>Symbols: L layers, H hidden, A heads, I MLP width, V vocabulary, C positions, B microbatch, S sequence length, G accumulation, P parameters. d_attn = 1 when attention dropout &gt; 0.</p>
        <p>Estimator policy (illustrative, uncalibrated): {Object.entries(p.explanation.policy).filter(([k]) => k !== 'label').map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(' · ')}</p>
        <p>Locked execution assumptions: {p.explanation.fixed_assumptions.map((a) => `${a.feature.replace(/_/g, ' ')} = ${a.value}`).join(' · ')}</p>
        <ul className="list-disc space-y-0.5 pl-4">{p.explanation.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        <p>Estimator input: <a className="underline" href={api.fileUrl(analysisId)}>estimator_input.json</a> (model / training / hardware / estimator_policy; runs with <span className="font-mono">python -m app.services.estimators.estimate_gpt2</span>).</p>
      </div></Accordion>
  </div>;
}

function BenchmarkView({ plan, onPlan, onDownload, onImport, results, prediction }: { plan: Async<BenchmarkPlan>; onPlan: () => void; onDownload: () => void; onImport: (f: File) => void; results: Async<BenchmarkResult[]>; prediction: Prediction | null }): JSX.Element {
  const pl = plan.data;
  return <div className="space-y-3">
    <Alert kind="warn">Benchmarking runs real training steps and needs a GPU; paid cloud GPUs incur charges. Measured results are kept separate from the analytical estimate.</Alert>
    {plan.status === 'idle' && <button type="button" onClick={onPlan} disabled={!prediction || prediction.status !== 'ok'} className="bg-white px-4 py-2 text-sm font-semibold text-black disabled:opacity-50">Prepare benchmark</button>}
    {(!prediction || prediction.status !== 'ok') && <p className="text-xs text-[#a7a0bb]">Run a successful estimate first; the benchmark uses the same configuration.</p>}
    {plan.status === 'loading' && <p aria-live="polite">Preparing…</p>}
    {plan.status === 'error' && <Alert kind="error">{plan.error}</Alert>}
    {pl && <>
      <div className="text-xs"><p className="eyebrow mb-1">Requirements</p><ul className="list-disc pl-4">{pl.requirements.environment.map((e) => <li key={e}>{e}</li>)}</ul>
        <ul className="mt-1 space-y-0.5">{pl.requirements.files.map((f) => <li key={f.flag}><span className="font-mono">{f.flag}</span> (optional) — {f.description} {f.provided.length ? <span className="text-[#7ee0b0]">Uploaded: {f.provided.join(', ')}</span> : null}</li>)}</ul>
        <p className="mt-1 text-[#8e899b]">Credentials: {pl.requirements.credentials.join(' ')}</p></div>
      <Section number="L" title="Run locally">
        <div className="space-y-2">{pl.requirements.warnings.map((w) => <Alert key={w} kind="warn">{w}</Alert>)}
          <button type="button" onClick={onDownload} className="bg-white px-4 py-2 text-xs font-semibold text-black">Download run_benchmark.py</button>
          <CopyBlock text={pl.command} label="benchmark command" /><CopyBlock text={pl.command_template} label="benchmark command template" />
          <label className="block border border-dashed border-line p-2 text-xs">Import benchmark_result.json<input type="file" accept=".json,application/json" className="mt-1 block text-[11px]" onChange={(e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) onImport(f); e.target.value = ''; }} /></label>
          {results.status === 'error' && <Alert kind="error">{results.error}</Alert>}
          {results.data?.map((r, i) => <div key={i} className="border border-line p-2 text-xs"><div className="flex flex-wrap gap-1"><Badge kind="good">Measured · benchmark</Badge><Badge kind={r.matches_current_config ? 'good' : 'warn'}>{r.matches_current_config ? 'matches current config' : 'different config'}</Badge></div>
            <p className="mt-1">{r.device?.name} · torch {r.torch_version} · {r.model_source.replace(/_/g, ' ')} · {r.data_source} data{r.user_code_executed ? ' · user code executed' : ''}</p>
            <p>Median {r.seconds_per_optimizer_step.median.toFixed(4)} s/update · peak reserved {gib(r.peak_memory_reserved_gib)} (allocated {gib(r.peak_memory_allocated_gib)})</p>
            {r.comparison.peak_vram && <p>Estimated allocated peak {gib(r.comparison.peak_vram.predicted_gib)} vs measured reserved {gib(r.comparison.peak_vram.measured_reserved_gib)} (×{r.comparison.peak_vram.ratio_measured_to_predicted.toFixed(2)})</p>}
            {r.comparison.seconds_per_optimizer_step && <p>Estimated {r.comparison.seconds_per_optimizer_step.predicted.toFixed(4)} s/update vs measured {r.comparison.seconds_per_optimizer_step.measured_median.toFixed(4)} s</p>}
            {[...r.issues, ...r.notes].map((m) => <p key={m} className="text-[#e6d28a]">{m}</p>)}</div>)}
        </div></Section>
      <Section number="N" title="Run on Nebius">
        <p className="text-xs">Estimated benchmark charge: <strong>{pl.predicted_cost.status === 'estimated' ? money(pl.predicted_cost.cost) : 'unavailable'}</strong> {pl.predicted_cost.status === 'estimated' ? `(${hours(pl.predicted_cost.elapsed_hours)} compute on 1 GPU)` : `— ${pl.predicted_cost.reason}`}</p>
        <div className="mt-2"><Unavailable label="Unavailable">{pl.nebius.reason} Nothing is provisioned and no charges can be incurred from this page.</Unavailable></div>
        <ul className="mt-2 list-disc pl-4 text-[11px] text-[#a7a0bb]">{pl.nebius.remaining_work.map((w) => <li key={w}>{w}</li>)}</ul>
        <button type="button" disabled className="mt-2 border border-line px-3 py-1.5 text-xs opacity-40">Start paid Nebius benchmark — unavailable</button>
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
  return <div className="space-y-3"><Section id="results" number="05" title="Results" hint="Estimate is analytical; benchmark results are measurements.">
    <div className="mb-3 flex gap-1" role="tablist" aria-label="Results mode">{(['prediction', 'benchmark'] as const).map((m) => <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={`border px-3 py-1 text-xs capitalize ${mode === m ? 'border-violet bg-violet/15' : 'border-line text-[#a7a0bb]'}`}>{m === 'prediction' ? 'Estimate' : 'Benchmark'}</button>)}</div>
    {mode === 'prediction' ? <>
      {prediction.status === 'loading' && <p aria-live="polite" className="text-sm">Calculating…</p>}
      {prediction.status === 'error' && <Alert kind="error"><strong>Estimate failed.</strong> {prediction.error}</Alert>}
      {prediction.status === 'idle' && <p className="text-sm text-[#a7a0bb]">Run an estimate from the Hardware step.</p>}
      {prediction.data && <PredictionView p={prediction.data} analysisId={props.analysisId} compare={props.compare} onCompare={props.onCompare} onGoField={props.onGoField} labels={props.labels} />}
    </> : <BenchmarkView plan={props.plan} onPlan={props.onPlan} onDownload={props.onDownload} onImport={props.onImport} results={props.benchmarks} prediction={prediction.data ?? null} />}
  </Section>
  <div className="flex justify-between"><button type="button" onClick={props.onEdit} className="border border-line px-4 py-2 text-sm">Edit inputs</button><button type="button" onClick={props.onRecs} disabled={prediction.data?.status !== 'ok'} className="bg-violet px-5 py-2 text-sm font-semibold text-white disabled:opacity-40">Improve this plan</button></div></div>;
}
