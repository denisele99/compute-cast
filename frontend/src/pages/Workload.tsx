import { ChangeEvent, DragEvent, useEffect, useState } from 'react';
import { Boxes, Brain, Database, Dumbbell, Eye, FileArchive, FileText, Rocket, TriangleAlert, Upload, X } from 'lucide-react';
import { Extraction, UploadedFile, api } from '../api';
import { AssumptionsPanel, FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, Badge, Section } from '../components/ui';

export type UploadHandlers = {
  busy: string | null; error: string; extraction: Extraction | null; kept: string[]; skipped: string[];
  onUpload: (files: File[]) => void; onRemove: (file: UploadedFile) => void; onHf: (modelId: string) => void; onExample: (id: string) => void;
};

function FileChip({ f, onRemove }: { f: UploadedFile; onRemove: () => void }): JSX.Element {
  return <li className="flex items-center gap-2 border border-line px-2 py-1 text-[11px]">
    <span className="truncate" title={f.from_zip ? `from ${f.from_zip}` : undefined}>{f.name}</span>
    {f.sections.map((s) => <Badge key={s} kind="muted">{s}</Badge>)}
    {f.conflict_count > 0 && <Badge kind="bad">{f.conflict_count} conflict{f.conflict_count > 1 ? 's' : ''}</Badge>}
    {f.extraction_error && <Badge kind="bad">unreadable</Badge>}
    {!f.sections.length && !f.extraction_error && f.kind !== 'dataset_sample' && <Badge kind="warn">no settings found</Badge>}
    <button type="button" aria-label={`Remove ${f.name}`} onClick={onRemove} className="ml-auto text-[#a7a0bb] hover:text-white"><X size={12} /></button></li>;
}

function ExamplePreview({ example, onLoad, onClose, busy }: { example: { id: string; label: string; files: string[] }; onLoad: () => void; onClose: () => void; busy: boolean }): JSX.Element {
  const [name, setName] = useState(example.files[0]); const [text, setText] = useState<string | null>(null); const [error, setError] = useState('');
  useEffect(() => { setText(null); setError(''); api.previewExample(example.id, name).then((r) => setText(r.content)).catch((e) => setError(e instanceof Error ? e.message : String(e))); }, [example.id, name]);
  useEffect(() => { const esc = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') onClose(); }; document.addEventListener('keydown', esc); return () => document.removeEventListener('keydown', esc); }, [onClose]);
  return <div className="fixed inset-0 z-40 grid place-items-center bg-black/70 p-4" role="dialog" aria-modal="true" aria-label={`Preview: ${example.label}`} onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
    <div className="flex max-h-[85vh] w-full max-w-3xl flex-col border border-line bg-[#11111a]">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2"><Eye size={14} aria-hidden /><strong className="text-sm">{example.label}</strong><span className="text-[11px] text-[#8e899b]">read-only preview</span>
        <button type="button" aria-label="Close preview" onClick={onClose} className="ml-auto"><X size={14} /></button></div>
      <div className="flex flex-wrap gap-1 border-b border-line px-3 py-2" role="tablist" aria-label="Example files">{example.files.map((f) => <button key={f} role="tab" aria-selected={f === name} type="button" onClick={() => setName(f)}
        className={`flex items-center gap-1 border px-2 py-0.5 text-[11px] ${f === name ? 'border-violet bg-violet/15' : 'border-line'}`}><FileText size={11} aria-hidden />{f}</button>)}</div>
      <pre className="flex-1 overflow-auto p-3 font-mono text-[11px] leading-4 text-[#d8d3e3]">{error || (text ?? 'Loading…')}</pre>
      <div className="flex justify-end gap-2 border-t border-line px-3 py-2"><button type="button" onClick={onClose} className="border border-line px-3 py-1 text-xs">Close</button>
        <button type="button" disabled={busy} onClick={() => { onLoad(); onClose(); }} className="bg-white px-3 py-1 text-xs font-semibold text-black disabled:opacity-40">Use these files</button></div>
    </div></div>;
}

function StartPanel({ ctx, h }: { ctx: FormCtx; h: UploadHandlers }): JSX.Element {
  const [models, setModels] = useState<{ id: string; summary: string }[]>([]);
  const [choice, setChoice] = useState(''); const [drag, setDrag] = useState(false); const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => { api.hfSearch('').then((r) => { setModels(r.items); setChoice(r.items[0]?.id ?? ''); }).catch(() => setModels([])); }, []);
  const files = (h.extraction?.files ?? []).filter((f) => f.origin !== 'huggingface');
  const hfFile = (h.extraction?.files ?? []).find((f) => f.origin === 'huggingface');
  // Training scripts can sit next to a Hugging Face model; only an uploaded model *config* blocks it.
  const uploadedModel = files.some((f) => f.sections.includes('model') && !f.name.endsWith('.py'));
  const exts = ctx.schema.uploads.allowed_extensions;
  const drop = (e: DragEvent) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) h.onUpload(Array.from(e.dataTransfer.files)); };
  const shown = ctx.schema.examples.find((x) => x.id === preview);
  return <Section icon={Rocket} title="Start" hint="Choose a Hugging Face model or upload your own files.">
    <div className="space-y-2">
      <div className="border border-line p-3">
        <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold"><Boxes size={13} aria-hidden /> Hugging Face model</p>
        {hfFile && h.extraction?.huggingface_model ? <div className="flex items-center gap-2 border border-violet bg-violet/10 px-2 py-1 text-xs">
          <span className="font-mono">{h.extraction.huggingface_model}</span><Badge kind="muted">config.json only</Badge>
          <button type="button" aria-label="Remove Hugging Face model" onClick={() => h.onRemove(hfFile)} className="ml-auto"><X size={12} /></button></div>
          : <div className="flex gap-2"><select aria-label="Hugging Face model" className="field-sm" value={choice} onChange={(e) => setChoice(e.target.value)} disabled={uploadedModel}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.id} — {m.summary}</option>)}</select>
            <button type="button" className="border border-line px-3 text-xs disabled:opacity-40" disabled={!choice || uploadedModel || !!h.busy} onClick={() => h.onHf(choice)}>Use</button></div>}
        <p className="mt-2 text-[11px] text-[#8e899b]">{uploadedModel ? 'Disabled: an uploaded model config already defines the model. Remove it to use Hugging Face.' : 'GPT-2 checkpoints only. Weights are never downloaded.'}</p>
      </div>
      <div className="flex items-center gap-3 text-[11px] text-[#a7a0bb]"><span className="h-px flex-1 bg-line" /><span><strong className="text-white">OR</strong> upload a model config and/or script <span className="text-[#8e899b]">(only PyTorch is supported)</span></span><span className="h-px flex-1 bg-line" /></div>
      <div className="border border-line p-3">
        <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold"><Upload size={13} aria-hidden /> Upload files <span className="font-normal text-[#8e899b]">(multiple files or a .zip)</span></p>
        <label onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={drop}
          className={`flex cursor-pointer items-center gap-3 border border-dashed px-3 py-2 text-xs ${drag ? 'border-white bg-violet/20' : 'border-violet bg-violet/5'}`}>
          <Upload size={16} /><span><strong>Drop or choose files</strong><span className="block text-[11px] text-[#a7a0bb]">{exts.join(' ')} · never executed</span></span>
          <input type="file" multiple accept={exts.join(',')} className="sr-only" aria-label="Upload files" disabled={!!h.busy}
            onChange={(e: ChangeEvent<HTMLInputElement>) => { if (e.target.files?.length) h.onUpload(Array.from(e.target.files)); e.target.value = ''; }} /></label>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]"><FileArchive size={12} className="text-[#8e899b]" /><span className="text-[#8e899b]">No files handy? Try an example:</span>
          {ctx.schema.examples.map((ex) => <span key={ex.id} className="inline-flex border border-line">
            <button type="button" className="px-2 py-0.5 hover:bg-white/5" disabled={!!h.busy} onClick={() => h.onExample(ex.id)}>{ex.label}</button>
            <button type="button" aria-label={`Preview ${ex.label}`} title="Preview the example files" className="border-l border-line px-1.5 hover:bg-white/5" onClick={() => setPreview(ex.id)}><Eye size={12} /></button></span>)}</div>
        {files.length > 0 && <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Uploaded files">{files.map((f) => <FileChip key={f.file_id} f={f} onRemove={() => h.onRemove(f)} />)}</ul>}
      </div>
    </div>
    {shown && <ExamplePreview example={shown} busy={!!h.busy} onLoad={() => h.onExample(shown.id)} onClose={() => setPreview(null)} />}
    <div className="mt-2 space-y-1">
      {new Set((h.extraction?.model_sources ?? []).map((m) => m.kind)).size > 1 && <Alert kind="error"><strong>Two model sources.</strong> Use ONE model source:
        remove the Hugging Face model ({h.extraction?.huggingface_model}) or the uploaded model file(s) ({h.extraction?.model_sources.filter((m) => m.kind === 'upload').map((m) => m.name).join(', ')}).</Alert>}
      {h.busy && <Alert kind="info">{h.busy}</Alert>}
      {h.error && <Alert kind="error">{h.error}</Alert>}
      {h.skipped.length > 0 && <Alert kind="warn">Skipped: {h.skipped.join('; ')}</Alert>}
      {(h.extraction?.errors ?? []).map((e) => <Alert key={e.file} kind="error"><strong>{e.file}:</strong> {e.message}</Alert>)}
      {h.kept.length > 0 && <Alert kind="warn">Your edits were kept for {h.kept.length} field(s); hover the yellow icons to compare with the file values.</Alert>}
      {(h.extraction?.notes ?? []).length > 0 && <Accordion title={`File notes (${h.extraction?.notes.length})`}><ul className="list-disc space-y-1 pl-4 text-[11px] text-[#c4bfcd]">{h.extraction?.notes.map((n) => <li key={n}>{n}</li>)}</ul></Accordion>}
    </div>
  </Section>;
}

function Conflicts({ ctx, acknowledged, onToggle }: { ctx: FormCtx; acknowledged: string[]; onToggle: (id: string) => void }): JSX.Element | null {
  const list = ctx.validation?.conflicts ?? [];
  if (!list.length) return null;
  return <Section icon={TriangleAlert} title="Unsupported settings in your files" hint="Choose the supported setting to estimate anyway, or change the file.">
    <ul className="space-y-1">{list.map((c) => { const on = acknowledged.includes(c.id);
      return <li key={c.id} className={`flex flex-wrap items-center gap-2 border px-2 py-1.5 text-xs ${on ? 'border-[#8a7430]' : 'border-[#8a3040]'}`}>
        <Badge kind={on ? 'warn' : 'bad'}>{c.section}</Badge><span className="font-mono text-[11px]">{c.setting}</span>
        <span className="text-[#c4bfcd]">requested <strong>{String(c.requested)}</strong>{c.supported !== null && <> · v1 supports <strong>{String(c.supported)}</strong></>} · {c.file}{c.line ? `:${c.line}` : ''}</span>
        {c.hypothetical ? <label className="ml-auto flex items-center gap-1.5"><input type="checkbox" checked={on} onChange={() => onToggle(c.id)} /> {c.hypothetical} (hypothetical)</label>
          : <span className="ml-auto text-[#ff9aa8]">Not estimable in v1: remove or replace this file.</span>}</li>; })}</ul>
  </Section>;
}

export function WorkloadPage({ ctx, handlers, acknowledged, onToggleConflict, reviewKeys, onReviewAll, onNext, blockers }: {
  ctx: FormCtx; handlers: UploadHandlers; acknowledged: string[]; onToggleConflict: (id: string) => void; reviewKeys: string[]; onReviewAll: () => void; onNext: () => void; blockers: string[];
}): JSX.Element {
  const [showSample, setShowSample] = useState(false);
  const samples = handlers.extraction?.dataset_samples ?? [];
  const label = (k: string) => ctx.schema.fields.find((f) => f.key === k)?.label ?? k;
  const custom = !!ctx.values['model.custom_config'];
  return <form className="space-y-3" noValidate onSubmit={(e) => { e.preventDefault(); onNext(); }}>
    <StartPanel ctx={ctx} h={handlers} />
    <Conflicts ctx={ctx} acknowledged={acknowledged} onToggle={onToggleConflict} />
    {reviewKeys.length > 0 && <Alert kind="warn"><strong>Review autofilled values:</strong> {reviewKeys.map(label).join(', ')}. Values with a red icon came from scripts or uncertain sources.
      <button type="button" onClick={onReviewAll} className="ml-2 border border-[#8a7430] px-2 py-0.5">Mark all reviewed</button></Alert>}

    <Section icon={Brain} title="Model" hint="Size and shape of the model." actions={<label className="flex items-center gap-1.5 text-[11px]">
      <input type="checkbox" checked={custom} onChange={(e) => ctx.setField('model.custom_config', e.target.checked)} /> Hypothetical custom configuration</label>}>
      {custom && <div className="mb-2"><Alert kind="warn">Custom mode: checkpoint dimensions are editable and results are labelled hypothetical.</Alert></div>}
      <FieldGrid ctx={ctx} section="model" exclude={['model.custom_config']} />
    </Section>

    <Section icon={Dumbbell} title="Training" hint="Batch, accumulation and training length (one GPU).">
      <FieldGrid ctx={ctx} section="training" />
    </Section>

    <Section icon={Database} title="Data" hint="Sequence length and number of examples."
      actions={samples.length > 0 ? <button type="button" className="text-[11px] underline" onClick={() => setShowSample(!showSample)}>{showSample ? 'Hide' : 'Show'} sample statistics ({samples.length})</button> : undefined}>
      <FieldGrid ctx={ctx} section="data" />
      {showSample && samples.map((s) => <div key={s.file} className="mt-2 border border-line bg-[#11111a] p-2 text-[11px]"><p className="eyebrow mb-1">{s.file} · sample only, not the full dataset</p>
        <p>{s.sample_rows.toLocaleString()} rows{s.sample_truncated_at ? ` (first ${s.sample_truncated_at.toLocaleString()} read)` : ''} · {(s.sample_file_bytes / 1024).toFixed(1)} KB · columns: {s.columns.join(', ')}</p>
        {s.token_lengths && <p>Token lengths ({s.token_lengths.column}): min {s.token_lengths.min}, mean {s.token_lengths.mean.toFixed(1)}, p95 {s.token_lengths.p95}, max {s.token_lengths.max}</p>}</div>)}
    </Section>

    <AssumptionsPanel schema={ctx.schema} sections={['model', 'training', 'data', 'estimate_scope']} />

    {blockers.length > 0 && <Alert kind="error"><strong>{blockers.length} item(s) to resolve:</strong> {blockers.slice(0, 6).join('; ')}{blockers.length > 6 ? '; …' : ''}</Alert>}
    <div className="flex justify-end"><button className="bg-white px-5 py-2 text-sm font-semibold text-black transition hover:bg-violet hover:text-white disabled:opacity-50" disabled={blockers.length > 0}>Continue to hardware</button></div>
  </form>;
}
