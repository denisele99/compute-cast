/** Browser-side parsing of collect_workload.py output. JSON only; nothing is executed. */
export const MAX_FILE_BYTES = 5 * 1024 * 1024;

export type ExtractedFields = {
  examples?: number;
  batch_size?: number;
  epochs?: number;
  sequence_length?: number;
};
export type ParsedWorkload = {
  fields: ExtractedFields;
  summary: { label: string; value: string }[];
  warnings: string[];
  /** Fields the collector reported as unknown; the user must supply them. */
  missing: string[];
};
export type ParseResult = { ok: true; data: ParsedWorkload } | { ok: false; error: string };

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const positiveInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 1;

export function parseWorkloadJson(text: string): ParseResult {
  if (text.length > MAX_FILE_BYTES) return { ok: false, error: 'The file is larger than 5 MB. Upload the JSON report from collect_workload.py.' };
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { return { ok: false, error: 'This is not valid JSON.' }; }
  if (!isObject(raw)) return { ok: false, error: 'The file must contain a JSON object.' };
  if (!isObject(raw.training) && !isObject(raw.model)) {
    return { ok: false, error: 'This does not look like collect_workload.py output: expected "training" or "model" sections.' };
  }

  const training = isObject(raw.training) ? raw.training : {};
  const model = isObject(raw.model) ? raw.model : {};
  const fields: ExtractedFields = {};
  const warnings: string[] = [];
  const map: [keyof ExtractedFields, string][] = [
    ['examples', 'num_examples'], ['batch_size', 'batch_size_per_gpu'], ['epochs', 'epochs'], ['sequence_length', 'sequence_length'],
  ];
  for (const [target, source] of map) {
    const value = training[source];
    if (value === undefined || value === null) continue;
    if (positiveInt(value)) fields[target] = value;
    else warnings.push(`Ignored training.${source}: expected a positive whole number.`);
  }

  const summary: { label: string; value: string }[] = [];
  if (typeof model.class === 'string') summary.push({ label: 'Model class', value: model.class });
  if (typeof model.total_params === 'number') summary.push({ label: 'Parameters', value: model.total_params.toLocaleString('en-US') });
  for (const key of ['precision', 'optimizer', 'gpu_count', 'gradient_accumulation_steps'] as const) {
    const value = training[key];
    if (typeof value === 'string' || typeof value === 'number') summary.push({ label: key.replace(/_/g, ' '), value: String(value) });
  }
  const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);
  const missing = [...new Set([...strings(raw.missing_training_fields), ...strings(raw.still_required)])];
  if (Object.keys(fields).length === 0) warnings.push('No usable training values were found; enter them manually.');
  return { ok: true, data: { fields, summary, warnings, missing } };
}

/** Keep the full collector JSON so the backend estimator receives tensor inventory and input specs. */
export function parseCollector(text: string): Record<string, unknown> | null {
  try { const raw = JSON.parse(text); return typeof raw === 'object' && raw !== null && !Array.isArray(raw) && typeof raw.model === 'object' ? raw : null; } catch { return null; }
}
export function buildWorkload(collector: Record<string, unknown>, s: import('./scenarios').Setup, gpuCount: number): Record<string, unknown> {
  const training = { ...((collector.training as Record<string, unknown>) ?? {}), num_examples: s.num_examples, epochs: s.epochs, batch_size_per_gpu: s.batch_size_per_gpu,
    gradient_accumulation_steps: s.gradient_accumulation_steps, precision: s.precision, optimizer: s.optimizer, gradient_checkpointing: s.gradient_checkpointing, gpu_count: gpuCount };
  return { ...collector, training };
}
