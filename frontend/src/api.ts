/** Typed client for the Python API (GPT-2 MVP). All estimates come from the backend. */
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export type Values = Record<string, Json>;
export type Section = 'model' | 'training' | 'data' | 'hardware';

export type FieldSpec = {
  key: string; label: string; type: 'int' | 'float' | 'bool' | 'select' | 'text'; section: Section; editable: boolean | 'custom_only';
  contract_feature: string | null; unit: string | null; help: string | null; default: Json; options: string[] | null;
  minimum: number | null; maximum: number | null; advanced: boolean;
};
export type FixedAssumption = { feature: string; section: string; value: string | number | boolean; later_extension: string | null; note: string | null };
export type Schema = {
  framework: string; architecture: string; fields: FieldSpec[]; sections: Section[]; fixed_assumptions: FixedAssumption[];
  internal_calculations: { feature: string; kind: string; rule: string }[]; estimator_policy: Record<string, number | string>;
  uploads: { allowed_extensions: string[]; max_bytes_default: number; max_bytes_by_type: Record<string, number>; dataset_formats: Record<string, string> };
  huggingface_models: string[]; examples: { id: string; label: string; files: string[] }[];
};

export type Status = 'extracted' | 'uncertain' | 'unresolved';
export type Finding = { field: string; value: Json; status: Status; source: string; file: string | null; line: number | null; message: string | null; candidates?: Json[]; locations?: string[] };
export type Conflict = { id: string; section: Section; field: string | null; setting: string; requested: Json; supported: Json; hypothetical: string | null; file: string; line: number | null; message: string; acknowledged?: boolean };
export type UploadedFile = { file_id: string; name: string; size: number; origin: 'upload' | 'example' | 'huggingface'; from_zip: string | null; kind: string | null; sections: Section[]; conflict_count: number; extraction_error: string | null };
export type DatasetStats = { file: string; format: string; sample_rows: number; sample_truncated_at: number | null; sample_file_bytes: number; columns: string[]; numeric_columns: string[]; token_lengths?: { column: string; min: number; max: number; mean: number; p95: number }; note: string };
export type Extraction = { fields: Record<string, Finding>; conflicts: Conflict[]; model_sources: { kind: string; name: string }[]; notes: string[]; dataset_samples: DatasetStats[];
  errors: { file: string; message: string }[]; files: UploadedFile[]; huggingface_model: string | null; executed_code: false };
export type UploadResponse = { added: UploadedFile[]; skipped: string[]; extraction: Extraction };

export type Issue = { field: string; message: string };
export type Missing = { field: string; label: string; required_for: ('memory' | 'runtime' | 'cost')[] };
export type Validation = {
  values: Values; errors: Issue[]; warnings: Issue[]; missing: Missing[]; required: Record<string, string[]>; conflicts: Conflict[]; hypothetical: string[];
  can_estimate: { memory: boolean; runtime: boolean; cost: boolean }; suggestions: Record<string, { value: Json; reason: string }>; blocking_fields: string[];
};

export type Price = { value: number | null; currency: string; unit: 'gpu_hour' | 'instance_hour' | null; unit_title: string; region: string | null; preemptible: boolean; effective_time: string | null; snapshot_time: string | null };
export type HardwareOption = {
  id: string; provider: 'nebius' | 'catalog'; platform: string | null; label: string; vram_gib: number | null;
  presets: { name: string; gpu_count: number | null; vcpu_count: number | null; ram_gib: number | null }[];
  specs: { catalog_gpu_id: string; name: string; review_status: string } | null; price: Price | null; issues: string[];
};
export type HardwareSearch = { total: number; offset: number; limit: number; items: HardwareOption[]; price_snapshot: string | null; notes: string[] };

export type Cost = { status: 'estimated' | 'unavailable'; reason?: string; cost?: number; currency?: string; unit?: string; price_per_hour?: number; elapsed_hours?: number; gpu_hours_billed?: number; formula?: string; note?: string | null };
export type BenchmarkCost = Cost & { steps?: number; warmup_steps?: number; seconds_per_optimizer_step?: number; assumption?: string };
export type Duration = { mode: 'epochs' | 'optimizer_updates'; optimizer_updates: number; seconds: number; processed_positions: number; microbatches_per_epoch?: number; optimizer_updates_per_epoch?: number };
export type Runtime = { training_flops_per_microbatch: number; seconds_per_microbatch: number; optimizer_seconds_per_update: number; seconds_per_full_optimizer_update: number; total_training_compute: Duration | null };
export type Fit = { capacity_gib: number; allocated_peak_gib: number; usable_vram_gib: number; total_vram_gib: number; percent_of_usable: number; fits: boolean; denominator: string; rule: string };
export type Prediction = {
  status: 'ok' | 'blocked'; validation: Validation; hypothetical: string[]; provenance?: string; estimator?: string;
  parameters: number; head_dim: number; effective_batch_size: number; resolved_intermediate_size: number;
  memory: Record<string, number>; fit: Fit; runtime: Runtime | null; runtime_error: string | null; runtime_missing: string[];
  cost: Cost; benchmark_cost: BenchmarkCost;
  explanation: { formulas: Record<string, string>; policy: Record<string, number | string>; fixed_assumptions: FixedAssumption[]; scope: string; limitations: string[] };
  snapshots: { original: { captured_at: number; values: Values; result_summary: { capacity_gib?: number; training_hours?: number | null; cost?: number | null } } | null;
    changes: { field: string; label: string; original: Json; current: Json }[]; is_original_run: boolean };
};
export type CompareRow = { id: string; label: string; region: string | null; issues: string[]; price: Price | null; status: string; fit?: Fit; hours?: number | null; cost?: number | null; runtime_missing?: string[] };
export type BenchmarkPlan = { command: string; command_template: string; fingerprint: string; predicted_cost: BenchmarkCost; predicted_allocated_peak_gib: number;
  requirements: { environment: string[]; files: { flag: string; required: boolean; provided: string[]; description: string }[]; credentials: string[]; warnings: string[] };
  nebius: { available: boolean; reason: string; remaining_work: string[] } };
export type BenchmarkResult = { kind: 'benchmark'; device: { name: string; type: string } | null; torch_version: string; model_source: string; user_code_executed: boolean; data_source: string;
  seconds_per_optimizer_step: { median: number }; peak_memory_allocated_gib: number | null; peak_memory_reserved_gib: number | null; issues: string[]; notes: string[]; matches_current_config: boolean;
  comparison: { peak_vram?: { predicted_gib: number; measured_reserved_gib: number; ratio_measured_to_predicted: number }; seconds_per_optimizer_step?: { predicted: number; measured_median: number } } };

const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${apiBase}${path}`, init); } catch {
    throw new Error('Could not reach the Python API. Start the backend on port 8000 or set VITE_API_BASE_URL.');
  }
  const text = await response.text();
  let body: unknown = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null && 'detail' in body ? (body as { detail?: unknown }).detail : body;
    const msg = Array.isArray(detail) ? detail.map((d) => (typeof d === 'object' && d && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d))).join('; ')
      : typeof detail === 'object' && detail && 'message' in detail ? String((detail as { message: unknown }).message) : String(detail || `Request failed (HTTP ${response.status}).`);
    throw new Error(msg);
  }
  return body as T;
}
const post = <T>(path: string, body: unknown) => request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export type ConfigBody = { values: Values; provenance: Record<string, unknown>; metadata: { acknowledged_conflicts: string[] } };

export const api = {
  schema: () => request<Schema>('/api/schema'),
  createAnalysis: () => post<{ analysis_id: string; ttl_seconds: number }>('/api/analysis', {}),
  resetAnalysis: (id: string) => request<{ cache_cleared: boolean }>(`/api/analysis/${id}`, { method: 'DELETE' }),
  /** Best effort on page close; the backend also expires analyses. */
  closeBeacon: (id: string) => navigator.sendBeacon?.(`${apiBase}/api/analysis/${id}/close`),
  uploadFiles: (id: string, files: File[]) => { const f = new FormData(); files.forEach((file) => f.append('files', file)); return request<UploadResponse>(`/api/analysis/${id}/uploads`, { method: 'POST', body: f }); },
  loadExample: (id: string, exampleId: string) => post<UploadResponse>(`/api/analysis/${id}/examples/${exampleId}`, {}),
  removeFile: (id: string, fileId: string) => request<{ extraction: Extraction }>(`/api/analysis/${id}/uploads/${encodeURIComponent(fileId)}`, { method: 'DELETE' }),
  hfSearch: (q: string) => request<{ items: { id: string; summary: string }[]; note: string }>(`/api/hf/search?q=${encodeURIComponent(q)}`),
  hfSelect: (id: string, modelId: string) => post<{ origin: string; extraction: Extraction }>(`/api/analysis/${id}/huggingface`, { model_id: modelId }),
  validate: (id: string, body: ConfigBody) => post<Validation>(`/api/analysis/${id}/validate`, body),
  predict: (id: string, body: ConfigBody, record = true) => post<Prediction>(`/api/analysis/${id}/predict`, { ...body, record }),
  compare: (id: string, body: ConfigBody) => post<{ rows: CompareRow[]; note: string }>(`/api/analysis/${id}/compare`, body),
  hardwareSearch: (q: string, offset = 0) => request<HardwareSearch>(`/api/hardware/search?q=${encodeURIComponent(q)}&limit=10&offset=${offset}`),
  hardwareOption: (optionId: string) => request<{ option: HardwareOption; values: Values; notes: string[] }>(`/api/hardware/options/${encodeURIComponent(optionId)}`),
  localCommand: () => request<{ linux: string; gpu_only: string; notes: string[] }>('/api/hardware/local-command'),
  benchmarkPlan: (id: string, body: ConfigBody) => post<BenchmarkPlan>(`/api/analysis/${id}/benchmark/plan`, body),
  benchmarkScript: async (id: string, body: ConfigBody): Promise<Blob> => {
    const response = await fetch(`${apiBase}/api/analysis/${id}/benchmark/script`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!response.ok) { let m = `HTTP ${response.status}`; try { m = (await response.json()).detail ?? m; } catch { /* keep status */ } throw new Error(m); }
    return response.blob();
  },
  importBenchmark: (id: string, file: File) => { const f = new FormData(); f.append('file', file); return request<BenchmarkResult>(`/api/analysis/${id}/benchmark/import`, { method: 'POST', body: f }); },
  fileUrl: (id: string) => `${apiBase}/api/analysis/${id}/files/estimator_input.json`,
};
