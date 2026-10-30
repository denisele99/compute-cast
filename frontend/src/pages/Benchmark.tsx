import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';
import { BenchBody, BenchJob, BenchRecord, BenchReview, BenchStatus, ConfigBody, Json, Provenance, api } from '../api';
import { Accordion, Alert, Badge, CopyBlock, IconTip, Section, Unavailable } from '../components/ui';
import { gib, money } from './Results';

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const TOKEN_KEY = 'compute-cast-bench-session';
const STATE_LABEL: Record<string, string> = { submitting: 'Submitting', queued: 'Queued', provisioning: 'Provisioning', running: 'Running', cancelling: 'Cancelling',
  collecting: 'Collecting results', succeeded: 'Succeeded', failed: 'Failed', timed_out: 'Timed out', cancelled: 'Cancelled' };
const STEPS = ['queued', 'provisioning', 'running', 'succeeded'];
const PROV: Record<Provenance, { label: string; kind: 'good' | 'warn' | 'muted' }> = {
  detected: { label: 'detected', kind: 'good' }, user: { label: 'your choice', kind: 'warn' }, default: { label: 'default', kind: 'muted' }, assumption: { label: 'assumption', kind: 'muted' } };
const SECTIONS: [string, string][] = [['model', 'Model'], ['input', 'Inputs'], ['training', 'Training'], ['measurement', 'Measurement'], ['hardware', 'Hardware']];
const secs = (s: number | null | undefined) => (s === null || s === undefined ? '—' : s < 1 ? `${(s * 1000).toFixed(1)} ms` : s < 120 ? `${s.toFixed(2)} s` : `${(s / 60).toFixed(1)} min`);
const num = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? '—' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toPrecision(digits));

function readToken(): string | null { try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; } }
function writeToken(t: string | null): void { try { if (t) sessionStorage.setItem(TOKEN_KEY, t); else sessionStorage.removeItem(TOKEN_KEY); } catch { /* storage unavailable */ } }
const newKey = () => (crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`).replace(/[^A-Za-z0-9_-]/g, '');

function flatten(config: Json, prefix = ''): [string, Json][] {
  if (config === null || typeof config !== 'object' || Array.isArray(config)) return [[prefix, config]];
  return Object.entries(config).flatMap(([k, v]) => flatten(v, prefix ? `${prefix}.${k}` : k));
}

/** Editable review of benchmark_config.json with provenance (detected / your choice / default / assumption). */
function ConfigReview({ review, overrides, onEdit, onReset }: { review: BenchReview; overrides: Record<string, Json>; onEdit: (path: string, v: Json) => void; onReset: () => void }): JSX.Element {
  const rows = flatten(review.draft as Json).filter(([p]) => p !== 'schema_version' && p !== 'architecture');
  return <div className="space-y-2">
    <div className="flex flex-wrap items-center gap-1.5 text-[11px]"><span className="text-[#a7a0bb]">Legend:</span>{(Object.keys(PROV) as Provenance[]).map((p) => <Badge key={p} kind={PROV[p].kind}>{PROV[p].label}</Badge>)}
      <span className="ml-auto text-[#8e899b]">architecture {String((review.draft as Record<string, Json>).architecture)} · schema {String((review.draft as Record<string, Json>).schema_version)}</span>
      {Object.keys(overrides).length > 0 && <button type="button" onClick={onReset} className="border border-line px-2 py-0.5">Reset my edits</button>}</div>
    {review.missing.length > 0 && <Alert kind="error"><strong>Missing before the benchmark can run:</strong><ul className="list-disc pl-4">{review.missing.map((m) => <li key={m.field}><span className="font-mono">{m.field}</span>: {m.message}</li>)}</ul></Alert>}
    {review.errors.length > 0 && <Alert kind="error"><strong>Invalid configuration:</strong><ul className="list-disc pl-4">{review.errors.map((e) => <li key={e.field + e.message}><span className="font-mono">{e.field}</span>: {e.message}</li>)}</ul></Alert>}
    {review.dropped_overrides.length > 0 && <Alert kind="warn"><strong>Some of your edits no longer apply</strong> and were not used:<ul className="list-disc pl-4">{review.dropped_overrides.map((d) => <li key={d.field}><span className="font-mono">{d.field}</span> = {String(d.value)}: {d.reason}</li>)}</ul></Alert>}
    <div className="grid gap-3 lg:grid-cols-2">{SECTIONS.map(([sec, title]) => <div key={sec} className="border border-line p-2"><p className="eyebrow mb-1">{title}</p>
      <table className="w-full text-xs"><tbody>{rows.filter(([p]) => p.startsWith(`${sec}.`)).map(([path, value]) => {
        const prov = review.provenance[path] ?? 'default'; const editable = review.editable.includes(path);
        return <tr key={path} className="border-t border-line align-middle"><td className="py-1 pr-2 font-mono text-[11px]">{path.slice(sec.length + 1)}</td>
          <td className="w-28">{editable ? <input aria-label={path} type="number" step="any" value={value === null ? '' : String(value)} className="w-full border border-line bg-black/40 px-1 py-0.5 text-xs"
            onChange={(e) => onEdit(path, e.target.value === '' ? null : Number(e.target.value))} /> : <span>{value === null ? <em className="text-[#8e899b]">null</em> : String(value)}</span>}</td>
          <td className="pl-2 text-right"><span className="inline-flex items-center gap-1"><Badge kind={PROV[prov].kind}>{PROV[prov].label}</Badge>
            {review.assumption_notes[path] && <IconTip tone="info" label={`About ${path}`}>{review.assumption_notes[path]}</IconTip>}</span></td></tr>;
      })}</tbody></table></div>)}</div>
    <p className="text-[11px] text-[#8e899b]">Only measurement counts, seed, optimizer settings and input shapes are editable here; change the model on the Workload step. Credentials, allowances, runtime limits and the container image are server policy and never part of this file.</p>
  </div>;
}

function Comparison({ r }: { r: BenchRecord }): JSX.Element {
  const res = r.result; const c = r.comparison;
  return <div className="border border-line p-3 text-xs">
    <div className="flex flex-wrap gap-1.5"><Badge kind="good">Measured</Badge>{res.smoke_test && <Badge kind="warn">CPU smoke test</Badge>}
      <Badge kind={r.matches_current_config ? 'good' : 'warn'}>{r.matches_current_config ? 'matches current config' : 'different config'}</Badge>
      <Badge kind={c.compatible_for_calibration ? 'good' : 'muted'}>{c.compatible_for_calibration ? 'comparable for calibration' : 'not used for calibration'}</Badge>
      {res.status === 'failed' && <Badge kind="bad">{res.failure?.type.replace(/_/g, ' ')}</Badge>}</div>
    <p className="mt-1 text-[#c4bfcd]">{r.source} · {res.hardware.device_type === 'cuda' ? `${res.hardware.gpu_name} (compute ${res.hardware.compute_capability}, ${gib((res.hardware.total_memory_bytes ?? 0) / 2 ** 30)})` : 'CPU'} · torch {res.software.torch ?? '?'} · CUDA {res.software.cuda ?? '—'} · runner {res.benchmark_code_version}</p>
    {res.failure && <Alert kind="error">{res.failure.message}</Alert>}
    <table className="mt-2 w-full text-left"><thead className="text-[10px] text-[#8e899b]"><tr><th className="py-1">Metric</th><th>Predicted (estimate)</th><th>Measured</th><th>Measured ÷ predicted</th></tr></thead>
      <tbody>{c.rows.map((row) => <tr key={row.metric} className="border-t border-line"><td className="py-1">{row.metric}</td><td>{num(row.predicted, 4)}</td>
        <td>{num(row.measured, 4)} <span className="block text-[10px] text-[#8e899b]">{row.measured_label}</span></td><td>{row.ratio === null ? '—' : `×${row.ratio.toFixed(2)}`}</td></tr>)}</tbody></table>
    {res.timing && <p className="mt-2">Measured training time {secs(res.timing.measured_training_seconds)} for {res.counts?.measured_updates} updates (min {secs(res.timing.min_update_seconds)}, max {secs(res.timing.max_update_seconds)}) · {num(res.throughput?.tokens_per_second)} tokens/s · start-up {secs(res.durations.startup_seconds)} (excluded) · runner wall {secs(res.durations.runner_wall_seconds)}</p>}
    {r.full_run_projection && <p className="mt-1"><strong>Full-run estimate:</strong> {r.full_run_projection.hours.toFixed(2)} h for {r.full_run_projection.optimizer_updates.toLocaleString()} updates. <span className="text-[#a7a0bb]">{r.full_run_projection.label}</span></p>}
    <ul className="mt-2 list-disc pl-4 text-[11px] text-[#e6d28a]">{c.issues.map((m) => <li key={m}>{m}</li>)}</ul>
    <ul className="list-disc pl-4 text-[10px] text-[#8e899b]">{c.labels.map((m) => <li key={m}>{m}</li>)}</ul>
  </div>;
}

function ImportResults({ onImport, busy, error }: { onImport: (text: string) => void; busy: boolean; error: string }): JSX.Element {
  const [text, setText] = useState(''); const [tooBig, setTooBig] = useState(false);
  const onFile = (e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; e.target.value = ''; if (!f) return; setTooBig(f.size > 256 * 1024); if (f.size > 256 * 1024) return; void f.text().then((t) => { setText(t); onImport(t); }); };
  return <div className="space-y-2"><p className="eyebrow">Import results.json</p>
    <label className="block border border-dashed border-line p-2 text-xs">Upload results.json<input type="file" accept=".json,application/json" className="mt-1 block text-[11px]" onChange={onFile} /></label>
    <textarea aria-label="Paste results.json" value={text} onChange={(e) => setText(e.target.value)} placeholder="…or paste the contents of results.json" rows={4} className="w-full border border-line bg-black/40 p-2 font-mono text-[11px]" />
    <button type="button" disabled={!text.trim() || busy} onClick={() => onImport(text)} className="border border-line px-3 py-1 text-xs disabled:opacity-40">{busy ? 'Validating…' : 'Validate and compare'}</button>
    {tooBig && <Alert kind="error">results.json exceeds 256 KiB.</Alert>}
    {error && <Alert kind="error">{error}</Alert>}
    <p className="text-[10px] text-[#8e899b]">Results are treated as untrusted data: size, schema, finite numbers and the configuration fingerprint are checked. Nothing in the file is executed.</p></div>;
}

function JobCard({ job, now, canCancel, onCancel }: { job: BenchJob; now: number; canCancel: boolean; onCancel: () => void }): JSX.Element {
  const at = STEPS.indexOf(job.state === 'collecting' ? 'running' : job.state);
  const elapsed = job.terminal ? job.elapsed_seconds : now - job.created_at;
  const bad = ['failed', 'timed_out', 'cancelled'].includes(job.state);
  return <div className="border border-line p-3 text-xs" aria-live="polite">
    <div className="flex flex-wrap items-center gap-2"><Badge kind={job.state === 'succeeded' ? 'good' : bad ? 'bad' : 'warn'}>{STATE_LABEL[job.state] ?? job.state}</Badge>
      {job.simulated && <Badge kind="warn">simulated provider</Badge>}<span className="text-[#a7a0bb]">job {job.id} · {job.platform} / {job.preset}</span>
      {canCancel && !job.terminal && <button type="button" onClick={onCancel} className="ml-auto border border-[#8a3040] px-2 py-0.5 text-[#ff9aa8]">Request cancellation</button>}</div>
    <ol className="mt-2 flex gap-1">{STEPS.map((s, i) => <li key={s} className={`flex-1 border-t-2 pt-1 text-[10px] ${bad ? 'border-[#8a3040]' : i <= at ? 'border-violet text-white' : 'border-line text-[#8e899b]'}`}>{STATE_LABEL[s]}</li>)}</ol>
    <p className="mt-2">Elapsed {secs(elapsed)}{job.provider_run_seconds ? ` · provider run time ${secs(job.provider_run_seconds)}` : ''}{job.provider_state ? ` · provider state ${job.provider_state}` : ''}</p>
    {job.provider_message && <p className="text-[#a7a0bb]">{job.provider_message}</p>}
    {job.error && <Alert kind="error"><strong>{job.error.type.replace(/_/g, ' ')}:</strong> {job.error.message}</Alert>}
    <p className="mt-1 text-[11px]">Reserved {money(job.reserved_usd)}{job.estimated_charge_usd !== null ? ` · estimated charge ${money(job.estimated_charge_usd)}` : ''} <span className="text-[#8e899b]">({job.cost_label})</span></p>
    <p className="text-[10px] text-[#8e899b]">Jobs are not resumable and are never retried automatically.{job.provider ? ` Provider: ${job.provider}.` : ''}</p>
  </div>;
}

export function BenchmarkPanel({ analysisId, body, ready, onGoHardware }: { analysisId: string; body: () => ConfigBody; ready: boolean; onGoHardware: () => void }): JSX.Element {
  const [target, setTarget] = useState<'local' | 'nebius'>('local');
  const [overrides, setOverrides] = useState<Record<string, Json>>({});
  const [review, setReview] = useState<BenchReview | null>(null); const [reviewError, setReviewError] = useState('');
  const [records, setRecords] = useState<BenchRecord[]>([]); const [importBusy, setImportBusy] = useState(false); const [importError, setImportError] = useState('');
  const [status, setStatus] = useState<BenchStatus | null>(null); const [statusError, setStatusError] = useState('');
  const [token, setToken] = useState<string | null>(readToken()); const [who, setWho] = useState('');
  const [name, setName] = useState(''); const [password, setPassword] = useState(''); const [loginError, setLoginError] = useState(''); const [loginBusy, setLoginBusy] = useState(false);
  const [confirm, setConfirm] = useState(false); const [job, setJob] = useState<BenchJob | null>(null); const [jobError, setJobError] = useState('');
  const [submitKey, setSubmitKey] = useState(newKey); const [submitting, setSubmitting] = useState(false); const [now, setNow] = useState(Date.now() / 1000);
  const imported = useRef<string | null>(null);
  const benchBody = useCallback((): BenchBody => ({ ...body(), overrides, target }), [body, overrides, target]);

  // Regenerate the reviewed config whenever the workload, target or edits change (edits are re-applied).
  useEffect(() => {
    if (!ready) return; let stale = false;
    const t = setTimeout(() => { api.benchConfig(analysisId, benchBody()).then((r) => { if (!stale) { setReview(r); setReviewError(''); } }).catch((e) => { if (!stale) setReviewError(message(e)); }); }, 250);
    return () => { stale = true; clearTimeout(t); };
  }, [analysisId, benchBody, ready]);
  useEffect(() => { setSubmitKey(newKey()); setConfirm(false); }, [review?.fingerprint, review?.hardware?.platform]);

  const loadStatus = useCallback(() => {
    api.benchStatus(token ?? undefined).then((s) => { setStatus(s); setStatusError(''); }).catch((e) => setStatusError(message(e)));
  }, [token]);
  useEffect(() => { if (target === 'nebius') loadStatus(); }, [target, loadStatus]);
  useEffect(() => {
    if (!token) return;
    api.benchSession(token).then((s) => setWho(s.display_name)).catch(() => { writeToken(null); setToken(null); setWho(''); });
    api.benchJobs(token).then((r) => { const active = r.jobs.find((j) => !j.terminal) ?? null; if (active) setJob(active); }).catch(() => undefined);
  }, [token]);

  // Poll the active job (the backend also throttles provider status calls).
  useEffect(() => {
    if (!job || job.terminal || !token) return;
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    const poll = setInterval(() => { api.benchJob(token, job.id).then(setJob).catch((e) => setJobError(message(e))); }, 5000);
    return () => { clearInterval(tick); clearInterval(poll); };
  }, [job, token]);
  useEffect(() => { if (job?.terminal) loadStatus(); }, [job?.terminal, loadStatus]);
  useEffect(() => {
    if (!job || !token || !job.terminal || !job.results || imported.current === job.id) return;
    imported.current = job.id;
    api.benchImport(analysisId, { ...benchBody(), job_id: job.id }, token).then((r) => setRecords((x) => [r, ...x])).catch((e) => setImportError(message(e)));
  }, [job, token, analysisId, benchBody, loadStatus]);

  async function download(): Promise<void> {
    try { const blob = await api.benchPackage(analysisId, { ...benchBody(), target: 'local' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'compute-cast-benchmark.zip'; a.click(); URL.revokeObjectURL(url); }
    catch (e) { setReviewError(message(e)); }
  }
  async function importText(text: string): Promise<void> {
    setImportBusy(true); setImportError('');
    try { const r = await api.benchImport(analysisId, { ...benchBody(), text }); setRecords((x) => [r, ...x]); } catch (e) { setImportError(message(e)); } finally { setImportBusy(false); }
  }
  async function login(): Promise<void> {
    setLoginBusy(true); setLoginError('');
    try { const s = await api.benchLogin(name, password); writeToken(s.token); setToken(s.token); setWho(s.display_name); setPassword(''); } catch (e) { setLoginError(message(e)); } finally { setLoginBusy(false); }
  }
  function logout(): void { writeToken(null); setToken(null); setWho(''); setJob(null); }
  async function submit(): Promise<void> {
    if (!token) return; setSubmitting(true); setJobError('');
    try { setJob(await api.benchSubmit(token, { ...benchBody(), target: 'nebius', analysis_id: analysisId, idempotency_key: submitKey, confirm_charges: confirm })); loadStatus(); }
    catch (e) { setJobError(message(e)); } finally { setSubmitting(false); }
  }
  async function cancel(): Promise<void> { if (!token || !job) return; try { setJob(await api.benchCancel(token, job.id)); } catch (e) { setJobError(message(e)); } }

  const edit = (path: string, v: Json) => setOverrides((o) => ({ ...o, [path]: v }));
  const pred = review?.prediction;
  const cloudReady = status?.cloud_enabled && status.missing_configuration.length === 0;

  return <div className="space-y-4">
    <Alert kind="info">A benchmark runs real training updates of the trusted GPT-2 runner on <strong>one GPU</strong> with <strong>synthetic token ids</strong> and random weights. It measures speed and memory only: real data loading and model quality are not measured. Your uploaded files are never executed. Measurements appear beside the estimate; the estimate itself is not changed.</Alert>
    {!ready && <Alert kind="warn">Complete the workload (a successful estimate) first; the benchmark uses the same configuration.</Alert>}
    <div className="flex gap-1" role="tablist" aria-label="Benchmark target">{([['local', 'Your GPU (download)'], ['nebius', 'Nebius AI Cloud']] as const).map(([t, l]) =>
      <button key={t} role="tab" aria-selected={target === t} onClick={() => setTarget(t)} className={`border px-3 py-1 text-xs ${target === t ? 'border-violet bg-violet/15' : 'border-line text-[#a7a0bb]'}`}>{l}</button>)}</div>
    {reviewError && <Alert kind="error">{reviewError}</Alert>}

    {review && <Section number="B1" title="Review benchmark_config.json" hint="validated by the same registry the runner uses">
      <ConfigReview review={review} overrides={overrides} onEdit={edit} onReset={() => setOverrides({})} />
      {review.config && <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
        <div className="border border-line p-2"><p className="eyebrow">Memory check</p>{review.memory && <><p><strong>{gib(review.memory.estimated_capacity_gib)}</strong> needed{review.memory.limit_gib !== null ? ` of ${gib(review.memory.limit_gib)} limit` : ''}</p>
          <p className="text-[10px] text-[#8e899b]">{review.memory.vram_gib !== null ? `${review.memory.vram_gib} GiB VRAM − ${(review.memory.safety_fraction * 100).toFixed(0)}% safety margin` : 'VRAM unknown'}</p>
          {review.memory.fits !== null && <Badge kind={review.memory.fits ? 'good' : 'bad'}>{review.memory.fits ? 'fits' : 'exceeds'}</Badge>}</>}</div>
        <div className="border border-line p-2"><p className="eyebrow">Predicted for this config</p><p>{secs(pred?.seconds_per_update)} per update · {gib(pred?.estimated_allocated_peak_gib)} allocated peak</p>
          <p className="text-[10px] text-[#8e899b]">{pred?.benchmark_compute_seconds ? `≈ ${secs(pred.benchmark_compute_seconds)} of timed compute; ` : ''}{pred?.note}</p></div>
        <div className="border border-line p-2"><p className="eyebrow">Model</p><p>{review.parameter_count?.toLocaleString()} parameters</p><p className="text-[10px] text-[#8e899b] break-all">fingerprint {review.fingerprint?.slice(0, 16)}…</p></div></div>}
      {(review.blockers ?? []).map((b) => <div key={b} className="mt-2"><Alert kind="error">{b}</Alert></div>)}
      {(review.warnings ?? []).map((w) => <div key={w} className="mt-2"><Alert kind="warn">{w}</Alert></div>)}
    </Section>}

    {target === 'local' && <Section number="B2" title="Run on your GPU">
      <div className="space-y-2 text-xs">
        <button type="button" onClick={() => void download()} disabled={!review?.config} className="bg-white px-4 py-2 text-xs font-semibold text-black disabled:opacity-40">Download local benchmark package</button>
        <p className="text-[#c4bfcd]">The ZIP contains the same trusted runner, model code, pinned requirements and README as the cloud image, plus your <span className="font-mono">benchmark_config.json</span>. No credentials are needed.</p>
        <CopyBlock label="install command" text="pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124" />
        <CopyBlock label="run command" text="python benchmark.py --config benchmark_config.json" />
        <p className="text-[#a7a0bb]">The runner detects your actual GPU (device 0 only) and rejects multi-GPU configurations. Without a CUDA GPU you can run a functional check:</p>
        <CopyBlock label="CPU smoke command" text="python benchmark.py --config benchmark_config.json --cpu-smoke" />
        <p className="text-[10px] text-[#e6d28a]">CPU smoke test: checks that the configuration runs; it is not a performance or memory measurement and is never used for calibration.</p>
        <ImportResults onImport={(t) => void importText(t)} busy={importBusy} error={importError} />
      </div></Section>}

    {target === 'nebius' && <Section number="B2" title="Run on Nebius AI Cloud" hint="one GPU, asynchronous">
      <div className="space-y-2 text-xs">
        {statusError && <Alert kind="error">{statusError}</Alert>}
        {status && !status.cloud_enabled && <Unavailable label="Switched off">Cloud benchmarks are disabled on this server. Use the local package instead.</Unavailable>}
        {status && status.cloud_enabled && status.missing_configuration.length > 0 && <Unavailable label="Not configured">The server is missing: {status.missing_configuration.join(', ')}.</Unavailable>}
        {status?.simulated && <Alert kind="warn">This server uses a <strong>simulated provider</strong> for testing. No Nebius job runs and no real results are produced.</Alert>}
        {status && <ul className="list-disc pl-4 text-[11px] text-[#a7a0bb]"><li>Runner limit {status.runner_max_seconds} s plus up to {Math.round(status.startup_grace_seconds / 60)} min for provisioning and image pull; the backend cancels jobs past that. Platform timeout backstop: {status.platform_timeout_hours} h.</li>
          <li>Each job reserves {money(status.reservation_per_job_usd)} of your session allowance until it finishes. One GPU job runs at a time across all users ({status.active_jobs} active).</li>
          {status.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        {cloudReady && !token && <form className="flex flex-wrap items-end gap-2 border border-line p-2" onSubmit={(e) => { e.preventDefault(); void login(); }}>
          <label className="text-[11px]">Display name<input value={name} onChange={(e) => setName(e.target.value)} maxLength={40} className="mt-0.5 block border border-line bg-black/40 px-2 py-1 text-xs" /></label>
          <label className="text-[11px]">Demo password<input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-0.5 block border border-line bg-black/40 px-2 py-1 text-xs" /></label>
          <button type="submit" disabled={loginBusy || !name.trim() || !password} className="bg-white px-3 py-1.5 text-xs font-semibold text-black disabled:opacity-40">{loginBusy ? 'Signing in…' : 'Sign in'}</button>
          <span className="w-full text-[10px] text-[#8e899b]">The display name is only a label shown on your jobs; it is not a verified identity.</span>
          {loginError && <div className="w-full"><Alert kind="error">{loginError}</Alert></div>}</form>}
        {cloudReady && token && <>
          <p>Signed in as <strong>{who}</strong> <button type="button" onClick={logout} className="ml-2 underline">sign out</button>{status?.session ? ` · allowance ${money(status.session.available_usd)} available of ${money(status.session.limit_usd)} (estimated use ${money(status.session.used_estimated_usd)})` : ''}</p>
          {review?.hardware && !review.hardware.valid && <Alert kind="warn"><strong>{review.hardware.reason}</strong> Supported 1-GPU platforms:
            <ul className="list-disc pl-4">{review.hardware.alternatives.map((a) => <li key={a.platform}>{a.label} ({a.platform}, {a.vram_gib} GiB){a.fits === false ? ' — estimated not to fit' : ''}</li>)}</ul>
            <button type="button" onClick={onGoHardware} className="mt-1 underline">Choose a Nebius GPU on the Hardware step</button></Alert>}
          {review?.hardware?.valid && <p>Hardware: <strong>{review.hardware.label}</strong> · platform <span className="font-mono">{review.hardware.platform}</span> · preset <span className="font-mono">{review.hardware.preset}</span></p>}
          {(!job || job.terminal) && <>
            <label className="flex items-start gap-2"><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} />
              <span>I understand this starts a paid Nebius GPU job. {status ? `${money(status.reservation_per_job_usd)} is reserved from the allowance;` : ''} the actual bill is determined by Nebius and may differ from any estimate shown here.</span></label>
            <button type="button" disabled={!review?.ready || !confirm || submitting} onClick={() => void submit()} className="bg-violet px-4 py-2 text-xs font-semibold text-white disabled:opacity-40">{submitting ? 'Submitting…' : 'Start Nebius benchmark'}</button></>}
          {jobError && <Alert kind="error">{jobError}</Alert>}
          {job && <JobCard job={job} now={now} canCancel={!!status?.cancel_enabled} onCancel={() => void cancel()} />}
        </>}
      </div></Section>}

    {records.length > 0 && <Section number="B3" title="Predicted vs measured" hint="original estimates are preserved">
      <div className="space-y-2">{records.map((r, i) => <Comparison key={i} r={r} />)}</div></Section>}
    <Accordion title="What the benchmark measures" hint="scope and exclusions">
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-[#c4bfcd]"><li>Timed: complete optimizer updates (forward, FP32 loss, backward for every accumulation microbatch, AdamW step) after warm-up, between GPU synchronisations.</li>
        <li>Excluded from training time: process start-up and model construction (reported separately), provisioning and image pull, real data loading, evaluation, checkpointing.</li>
        <li>Memory: PyTorch peak allocated and reserved bytes on the measured GPU, shown separately from the analytical memory estimate.</li>
        <li>Throughput: synthetic fixed-length token positions per measured training second.</li></ul></Accordion>
  </div>;
}
