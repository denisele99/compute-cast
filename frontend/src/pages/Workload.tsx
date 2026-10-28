import { ChangeEvent, useState } from 'react';
import { Search, Upload, X } from 'lucide-react';
import { Extraction, UploadedFile, api } from '../api';
import { Field, FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, Badge, Section } from '../components/ui';

export type UploadHandlers = {
  busy: string | null; error: string; extraction: Extraction | null; conflicts: string[];
  onUpload: (kind: 'model' | 'training' | 'dataset', files: FileList) => void; onRemove: (file: UploadedFile) => void;
  onHfImport: (modelId: string) => Promise<void>;
};

function UploadBox({ kind, label, accept, handlers }: { kind: 'model' | 'training' | 'dataset'; label: string; accept: string; handlers: UploadHandlers }): JSX.Element {
  const files = (handlers.extraction?.files ?? []).filter((f) => f.kind === kind);
  return <div>
    <label className="flex cursor-pointer items-center gap-4 border border-dashed border-violet bg-violet/5 p-4 focus-within:ring-2 focus-within:ring-violet">
      <Upload size={18} /><span className="text-sm"><strong className="block">{label}</strong><span className="text-xs text-[#a7a0bb]">{accept.split(',').join(' ')} · multiple allowed · parsed statically, never executed</span></span>
      <input type="file" multiple accept={accept} className="sr-only" aria-label={label} disabled={handlers.busy !== null}
        onChange={(e: ChangeEvent<HTMLInputElement>) => { if (e.target.files?.length) handlers.onUpload(kind, e.target.files); e.target.value = ''; }} /></label>
    {files.length > 0 && <ul className="mt-2 space-y-1 text-xs">{files.map((f) => <li key={f.file_id} className="flex items-center justify-between gap-2 border border-line px-3 py-2">
      <span>{f.name} <span className="text-[#8e899b]">{f.size ? `${(f.size / 1024).toFixed(1)} KB` : 'metadata'}</span>{f.extraction_error && <span className="ml-2 text-[#ff9aa8]">{f.extraction_error}</span>}</span>
      <button type="button" aria-label={`Remove ${f.name}`} onClick={() => handlers.onRemove(f)}><X size={14} /></button></li>)}</ul>}
  </div>;
}

function HfSearch({ handlers }: { handlers: UploadHandlers }): JSX.Element {
  const [q, setQ] = useState(''); const [items, setItems] = useState<{ id: string; downloads: number | null; pytorch: boolean; custom_code: boolean }[]>([]);
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  async function run(): Promise<void> {
    if (!q.trim()) return; setBusy(true); setError('');
    try { setItems((await api.hfSearch(q)).items); } catch (e) { setItems([]); setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  }
  return <div className="border border-line p-4">
    <p className="mb-2 text-sm font-semibold">Search Hugging Face</p>
    <p className="mb-3 text-xs text-[#8e899b]">Reads config.json metadata only. Weights are never downloaded and remote code is never executed.</p>
    <div className="flex gap-2">{/* not a <form>: this sits inside the workload form */}
      <input className="field" aria-label="Hugging Face model search" placeholder="gpt2, openai-community/gpt2…" value={q} onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); void run(); } }} />
      <button type="button" className="border border-line px-3" disabled={busy} aria-label="Search models" onClick={() => void run()}><Search size={16} /></button></div>
    <p className="mt-2 text-xs text-[#8e899b]">Know the exact id? <button type="button" className="underline" onClick={() => void handlers.onHfImport(q.trim()).catch((e) => setError(String(e instanceof Error ? e.message : e)))} disabled={!q.trim()}>Fetch config.json for “{q || '…'}”</button></p>
    {busy && <p className="mt-2 text-xs" aria-live="polite">Searching…</p>}
    {error && <p role="alert" className="mt-2 text-xs text-[#ff9aa8]">{error}</p>}
    {items.length > 0 && <ul className="mt-3 space-y-1">{items.slice(0, 10).map((m) => <li key={m.id} className="flex items-center justify-between gap-2 border border-line px-3 py-2 text-xs">
      <span>{m.id} {m.downloads !== null && <span className="text-[#8e899b]">· {m.downloads.toLocaleString()} downloads</span>} {!m.pytorch && <Badge kind="warn">not tagged PyTorch</Badge>} {m.custom_code && <Badge kind="warn">custom code (not executed)</Badge>}</span>
      <button type="button" className="underline" onClick={() => void handlers.onHfImport(m.id).catch((e) => setError(String(e instanceof Error ? e.message : e)))}>Use config</button></li>)}</ul>}
  </div>;
}

export function ExtractionNotes({ extraction, conflicts, ctx }: { extraction: Extraction | null; conflicts: string[]; ctx: FormCtx }): JSX.Element | null {
  if (!extraction) return null;
  const labels = Object.fromEntries(ctx.schema.fields.map((f) => [f.key, f.label]));
  const { notes, unsupported, errors } = extraction;
  if (!notes.length && !unsupported.length && !errors.length && !conflicts.length) return null;
  return <div className="mt-4 space-y-2 text-xs">
    {unsupported.map((u) => <Alert key={u.message} kind="error"><strong>Unsupported:</strong> {u.message}</Alert>)}
    {errors.map((e) => <Alert key={e.file + e.message} kind="error"><strong>{e.file}:</strong> {e.message}</Alert>)}
    {conflicts.length > 0 && <Alert kind="warn">Your edits were kept for: {conflicts.map((k) => labels[k] ?? k).join(', ')}. The extracted values are shown beside each field.</Alert>}
    {notes.map((n) => <p key={n} className="text-[#e6d28a]">{n}</p>)}
  </div>;
}

export function WorkloadPage({ ctx, handlers, reviewKeys, onReviewAll, onNext, blockers }: {
  ctx: FormCtx; handlers: UploadHandlers; reviewKeys: string[]; onReviewAll: () => void; onNext: () => void; blockers: string[];
}): JSX.Element {
  const profile = ctx.values['model.profile'] as string | null;
  const spec = (k: string) => ctx.schema.fields.find((f) => f.key === k)!;
  const exts = ctx.schema.uploads.allowed_extensions;
  const B = ctx.values['training.batch_size_per_gpu'] ?? 'B';
  const shapes = profile === 'mlp_regression' ? `features [${B}, ${ctx.values['model.input_size'] ?? 'I'}] · targets [${B}, ${ctx.values['model.output_size'] ?? 'O'}]`
    : profile ? `input_ids [${B}, ${ctx.values['training.sequence_length'] ?? 'S'}] · labels [${B}, ${ctx.values['training.sequence_length'] ?? 'S'}]` : 'Select a profile first.';
  const samples = handlers.extraction?.dataset_samples ?? [];
  return <form className="space-y-6" noValidate onSubmit={(e) => { e.preventDefault(); onNext(); }}>
    {reviewKeys.length > 0 && <Alert kind="warn"><strong>Review autofilled values.</strong> {reviewKeys.length} value(s) came from uploads, Hugging Face, catalogues or defaults and have not been reviewed:{' '}
      {reviewKeys.map((k, i) => <span key={k}>{i > 0 && ', '}<a className="underline" href={`#field-${k}`}>{spec(k)?.label ?? k}</a></span>)}.
      <button type="button" onClick={onReviewAll} className="ml-3 border border-[#8a7430] px-2 py-1">I reviewed these</button></Alert>}
    {handlers.error && <Alert kind="error">{handlers.error}</Alert>}
    {handlers.busy && <Alert kind="info">{handlers.busy}</Alert>}

    <Section number="01" title="Model" hint={<><Badge kind="warn">PyTorch only</Badge> <span className="ml-2">Two prediction profiles are supported. Other architectures are reported as unsupported instead of estimated.</span></>}>
      <div className="grid gap-4 lg:grid-cols-2"><div className="space-y-4"><Field ctx={ctx} spec={spec('model.profile')} />
        {profile && <p className="text-xs text-[#a7a0bb]">{ctx.schema.profiles[profile]?.summary}</p>}
        <UploadBox kind="model" label="Upload model scripts / configs" accept={exts.model.join(',')} handlers={handlers} /></div>
        <HfSearch handlers={handlers} /></div>
      <div className="mt-6"><FieldGrid ctx={ctx} section="model" advanced={false} exclude={['model.profile']} /></div>
      <div className="mt-6"><Accordion title="Advanced model settings" hint="bias/tying details, KV heads, FFN width, metadata"><FieldGrid ctx={ctx} section="model" advanced /></Accordion></div>
      <ExtractionNotes extraction={handlers.extraction} conflicts={handlers.conflicts} ctx={ctx} />
    </Section>

    <Section number="02" title="Training setup" hint="Upload training scripts or JSON/YAML configs (HF TrainingArguments and DeepSpeed keys are recognised), then review and edit.">
      <UploadBox kind="training" label="Upload training scripts / configs" accept={exts.training.join(',')} handlers={handlers} />
      <div className="mt-6"><FieldGrid ctx={ctx} section="training" advanced={false} /></div>
      <p className="mt-4 text-xs text-[#a7a0bb]">Input/target shapes per microbatch (from the profile): <span className="font-mono">{shapes}</span>. GPU count and distribution strategy are set on the Hardware step.</p>
      <div className="mt-6"><Accordion title="Advanced training settings and estimator assumptions" hint="optimizer options, allowances, efficiencies, benchmark plan"><FieldGrid ctx={ctx} section="training" advanced /></Accordion></div>
    </Section>

    <Section number="03" title="Dataset" hint="Sample statistics describe only the uploaded sample. Enter full-dataset values separately.">
      <UploadBox kind="dataset" label="Upload a dataset sample (optional)" accept={exts.dataset.join(',')} handlers={handlers} />
      <ul className="mt-3 space-y-1 text-xs text-[#8e899b]">{Object.entries(ctx.schema.uploads.dataset_formats).map(([ext, d]) => <li key={ext}><span className="font-mono text-[#c4bfcd]">{ext}</span> — {d}</li>)}</ul>
      {samples.map((s) => <div key={s.file} className="mt-4 border border-line bg-[#11111a] p-4 text-xs"><p className="eyebrow mb-2">Sample statistics · {s.file} (not the full dataset)</p>
        <p>{s.sample_rows.toLocaleString()} sample rows{s.sample_truncated_at ? ` (first ${s.sample_truncated_at.toLocaleString()} read)` : ''} · {(s.sample_file_bytes / 1024).toFixed(1)} KB sample file · columns: {s.columns.join(', ')}</p>
        {s.numeric_columns.length > 0 && <p className="mt-1">Numeric columns: {s.numeric_columns.join(', ')}</p>}
        {s.token_lengths && <p className="mt-1">Token lengths ({s.token_lengths.column}): min {s.token_lengths.min}, mean {s.token_lengths.mean.toFixed(1)}, p95 {s.token_lengths.p95}, max {s.token_lengths.max}</p>}
        <p className="mt-2 text-[#e6d28a]">{s.note}</p></div>)}
      <div className="mt-6"><FieldGrid ctx={ctx} section="dataset" advanced={false} /></div>
      <div className="mt-6"><Accordion title="Advanced dataset settings" hint="validation split, effective length"><FieldGrid ctx={ctx} section="dataset" advanced /></Accordion></div>
    </Section>

    {blockers.length > 0 && <Alert kind="error">Resolve before continuing: {blockers.join('; ')}</Alert>}
    <div className="flex justify-end"><button className="bg-white px-6 py-3 text-sm font-semibold text-black transition hover:bg-violet hover:text-white disabled:opacity-50" disabled={blockers.length > 0}>Continue to hardware</button></div>
  </form>;
}
