/** Canonical form state shared by uploads, Hugging Face, catalogue lookups, defaults and manual entry. */
import type { Finding, Json, Schema, Values } from '../api';

export type Source = 'upload' | 'static_extraction' | 'huggingface' | 'dataset_sample' | 'manual' | 'catalogue' | 'default';
export type Incoming = { value: Json; source: Source; status: 'ok' | 'uncertain' | 'unresolved'; detail?: string; message?: string | null };
export type FieldState = Incoming & { reviewed: boolean; pending?: Incoming };
export type FormState = Record<string, FieldState>;

export const SOURCE_LABEL: Record<Source, string> = {
  upload: 'uploaded file', static_extraction: 'static extraction', huggingface: 'Hugging Face config', dataset_sample: 'dataset sample',
  manual: 'manual entry', catalogue: 'catalogue', default: 'default assumption',
};

const same = (a: Json | undefined, b: Json | undefined): boolean => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export function initialForm(schema: Pick<Schema, 'fields'>): FormState {
  const form: FormState = {};
  for (const f of schema.fields) if (f.default !== null && f.default !== undefined) form[f.key] = { value: f.default, source: 'default', status: 'ok', detail: 'schema default', reviewed: false };
  return form;
}

export function setManual(form: FormState, key: string, value: Json): FormState {
  return { ...form, [key]: { value, source: 'manual', status: 'ok', reviewed: true } };
}

function merge(form: FormState, key: string, incoming: Incoming): { form: FormState; conflict: boolean } {
  const cur = form[key];
  if (cur?.source === 'manual') {
    if (same(cur.value, incoming.value)) return { form, conflict: false };
    return { form: { ...form, [key]: { ...cur, pending: incoming } }, conflict: true }; // never overwrite user edits silently
  }
  if (cur && same(cur.value, incoming.value) && cur.source === incoming.source && cur.status === incoming.status) return { form, conflict: false };
  return { form: { ...form, [key]: { ...incoming, reviewed: false } }, conflict: false };
}

/** Apply merged extraction findings. Returns keys whose manual value was kept despite a different extracted value. */
export function applyFindings(form: FormState, findings: Record<string, Finding>): { form: FormState; conflicts: string[] } {
  const conflicts: string[] = [];
  for (const [key, f] of Object.entries(findings)) {
    const where = f.locations?.join(', ') || (f.file ? `${f.file}${f.line ? `:${f.line}` : ''}` : undefined);
    const r = merge(form, key, { value: f.status === 'unresolved' ? null : f.value, source: f.source as Source, status: f.status === 'extracted' ? 'ok' : f.status, detail: where, message: f.message });
    form = r.form; if (r.conflict) conflicts.push(key);
  }
  return { form, conflicts };
}

export function applyCatalogue(form: FormState, values: Values, detail: string): { form: FormState; conflicts: string[] } {
  const conflicts: string[] = [];
  for (const [key, value] of Object.entries(values)) { const r = merge(form, key, { value, source: 'catalogue', status: 'ok', detail }); form = r.form; if (r.conflict) conflicts.push(key); }
  return { form, conflicts };
}

/** Backend-derived defaults fill only empty or default-sourced fields (never manual or extracted values). */
export function applySuggestions(form: FormState, suggestions: Record<string, { value: Json; reason: string }>): FormState {
  let changed = false; const next = { ...form };
  for (const [key, s] of Object.entries(suggestions)) {
    const cur = form[key];
    const empty = !cur || ((cur.value === null || cur.value === '') && cur.status !== 'unresolved');
    if ((empty || cur.source === 'default') && !same(cur?.value, s.value)) { next[key] = { value: s.value, source: 'default', status: 'ok', detail: s.reason, reviewed: false }; changed = true; }
  }
  return changed ? next : form;
}

export function acceptPending(form: FormState, key: string): FormState {
  const cur = form[key]; if (!cur?.pending) return form;
  return { ...form, [key]: { ...cur.pending, reviewed: true } };
}
export function dismissPending(form: FormState, key: string): FormState {
  const cur = form[key]; if (!cur?.pending) return form;
  const { pending: _ignored, ...rest } = cur; return { ...form, [key]: rest };
}
export function markReviewed(form: FormState, keys: string[]): FormState {
  const next = { ...form }; for (const k of keys) if (next[k]) next[k] = { ...next[k], reviewed: true }; return next;
}

export const toValues = (form: FormState): Values => Object.fromEntries(Object.entries(form).map(([k, f]) => [k, f.value]));
export const toProvenance = (form: FormState): Record<string, unknown> =>
  Object.fromEntries(Object.entries(form).map(([k, f]) => [k, { source: f.source, status: f.status, detail: f.detail ?? null, message: f.message ?? null, reviewed: f.reviewed }]));

/** Autofilled (non-manual) values the user has not reviewed yet, limited to the given visible keys. */
export function needsReview(form: FormState, visible: string[]): string[] {
  return visible.filter((k) => { const f = form[k]; return f && f.source !== 'manual' && !f.reviewed && (f.value !== null || f.status === 'unresolved'); });
}

/** Conditional visibility rules (GPT-2 MVP). */
export function visible(key: string, v: Values): boolean {
  const mode = v['training.duration_mode'] ?? 'epochs';
  switch (key) {
    case 'training.num_epochs': case 'data.num_training_examples': return mode === 'epochs';
    case 'training.optimizer_updates': return mode === 'optimizer_updates';
    case 'hardware.source': case 'hardware.gpu_id': case 'hardware.catalog_gpu_id': case 'model.model_type': return false;
    default: return true;
  }
}

const EXTRACTED: Source[] = ['upload', 'static_extraction', 'huggingface'];
const CATALOGUE_LOCKED = ['hardware.vram_gib', 'hardware.bf16_supported', 'hardware.bf16_dense_tflops', 'hardware.fp32_tflops', 'hardware.memory_bandwidth_gbps'];

/** Why a field is read-only, or null when editable (contract editable=false / custom_only; catalogue specs). */
export function lockReason(key: string, editable: boolean | 'custom_only', form: FormState, v: Values): string | null {
  const f = form[key];
  if (editable === 'custom_only' && f && EXTRACTED.includes(f.source) && !v['model.custom_config'])
    return 'From the selected checkpoint. Turn on "Hypothetical custom configuration" to change it.';
  if (editable === false) return 'Fixed in v1.';
  if (CATALOGUE_LOCKED.includes(key) && f?.source === 'catalogue' && v['hardware.source'] !== 'custom')
    return 'From the hardware catalogue. Choose "Custom GPU" to enter your own values.';
  return null;
}
