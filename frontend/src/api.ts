/** Typed client for the Python API (backend/app/api/routes). All estimates come from the backend. */
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export type Values = Record<string, Json>;

export type FieldSpec = {
  key: string; label: string; type: 'int' | 'float' | 'bool' | 'select' | 'text'; section: 'model' | 'training' | 'dataset' | 'hardware';
  options: string[] | null; profiles: string[] | null; advanced: boolean; unit: string | null; help: string | null; default: Json; minimum: number | null;
};
export type Profile = { label: string; summary: string; tasks: string[]; precisions: string[] };
export type Schema = { framework: string; fields: FieldSpec[]; profiles: Record<string, Profile>; uploads: { allowed_extensions: Record<string, string[]>; max_bytes: Record<string, number>; dataset_formats: Record<string, string> } };

export type Status = 'extracted' | 'uncertain' | 'unresolved';
export type Finding = { field: string; value: Json; status: Status; source: string; file: string | null; line: number | null; message: string | null; candidates?: Json[]; locations?: string[] };
export type UploadedFile = { file_id: string; kind: 'model' | 'training' | 'dataset'; name: string; size: number; extraction_error: string | null };
export type DatasetStats = { file: string; format: string; sample_rows: number; sample_truncated_at: number | null; sample_file_bytes: number; columns: string[]; numeric_columns: string[]; token_lengths?: { column: string; min: number; max: number; mean: number; p95: number }; note: string };
export type Extraction = { fields: Record<string, Finding>; notes: string[]; unsupported: Issue[]; dataset_samples: DatasetStats[]; errors: { file: string; message: string }[]; files: UploadedFile[]; executed_code: false };

export type Issue = { field: string; message: string };
export type Missing = { field: string; label: string; required_for: ('memory' | 'runtime' | 'cost')[] };
export type Validation = {
  values: Values; errors: Issue[]; warnings: Issue[]; unsupported: Issue[]; missing: Missing[]; required: Record<string, string[]>;
  can_estimate: { memory: boolean; runtime: boolean; cost: boolean }; suggestions: Record<string, { value: Json; reason: string }>; blocking_fields: string[];
};

export type Price = { value: number | null; currency: string; unit: 'gpu_hour' | 'instance_hour' | null; unit_title: string; region: string | null; preemptible: boolean; effective_time: string | null; snapshot_time: string | null };
export type HardwareOption = {
  id: string; provider: 'nebius' | 'catalog'; platform: string | null; label: string; vram_gib: number | null; vram_source: string | null;
  presets: { name: string; gpu_count: number | null; vcpu_count: number | null; ram_gib: number | null }[];
  specs: { catalog_gpu_id: string; name: string; review_status: string; source_url: string | null } | null; price: Price | null; issues: string[];
};
export type HardwareSearch = { total: number; offset: number; limit: number; items: HardwareOption[]; price_snapshot: string | null; notes: string[] };

export type Cost = { status: 'estimated' | 'unavailable'; reason?: string; cost?: number; currency?: string; unit?: string; price_per_hour?: number; elapsed_hours?: number; gpus_allocated?: number; gpus_used?: number; gpu_hours_used?: number; gpu_hours_billed?: number; instances?: number; formula?: string; warning?: string; note?: string | null };
export type BenchmarkCost = Cost & { steps?: number; warmup_steps?: number; setup_minutes?: number; seconds_per_optimizer_step?: number; assumption?: string };
export type Scenario = {
  forward_backward_core_seconds_per_microbatch: number | null; data_supply_seconds_per_microbatch: number | null; optimizer_seconds_per_update: number | null;
  gpu_core_only_forward_backward_seconds: number | null; training_hours: number | null; job_hours: number | null; training_gpu_hours: number | null; job_gpu_hours: number | null; missing: string[];
};
export type Memory = {
  status: string; gpu: { peak_vram_gib: number | null; total_vram_gib: number | null; usable_vram_gib: number | null; fits: boolean | null };
  cpu?: { required_ram_gib_per_node: number | null; available_gib_per_node: number | null; fits: boolean | null };
  nvme?: { required_storage_gib_per_node: number | null; available_gib_per_node: number | null; fits: boolean | null };
  errors?: string[]; warnings?: string[]; breakdown_gib?: Record<string, number>; assumption_notes?: string[]; planning_assumptions?: Record<string, number>;
};
export type Formula = { formula: string; basis?: string; sources?: string[] };
export type Prediction = {
  status: 'ok' | 'blocked'; validation: Validation; provenance?: string;
  memory: Memory; fit: { peak_vram_gib: number; usable_vram_gib: number; total_vram_gib: number | null; percent_of_usable: number; percent_of_total: number | null; fits: boolean | null; denominator: string } | null;
  runtime: { status: string; missing_information: string[]; assumptions: string[]; schedule?: { microbatches_per_rank: number; optimizer_updates: number; effective_global_batch_size: number }; scenarios: Record<'fast' | 'slow', Scenario> };
  cost: Record<string, Cost>; benchmark_cost: BenchmarkCost; parameters: { total_params: number; trainable_params: number; logical_weight_bytes: number } | null;
  profile_estimates: { saved_activation_bytes: number; formula_id: string } | null;
  explanation: { profile: Profile & { id: string }; parameter_and_activation_formulas: Record<string, Formula>; memory_formulas: Record<string, Formula>; runtime_formulas: Record<string, Formula>; cost_formulas: Record<string, string>; memory_planning_assumptions: Record<string, number> | null; memory_notes: string[]; runtime_assumptions: string[]; defaults_used: string[]; limitations: string[] };
  snapshots: { original: { captured_at: number; values: Values; result_summary: { peak_vram_gib?: number | null; job_hours?: { fast: number | null; slow: number | null }; cost?: Record<string, number | null> } } | null; changes: { field: string; label: string; original: Json; current: Json }[]; is_original_run: boolean };
};
export type CompareRow = { id: string; label: string; region: string | null; issues: string[]; status: string; fit?: Prediction['fit']; job_hours?: { fast: number | null; slow: number | null }; cost?: Record<string, number | undefined>; price?: Price | null; runtime_missing?: string[]; reason?: string[] };
export type BenchmarkPlan = { command: string; command_template: string; fingerprint: string; predicted_cost: BenchmarkCost; predicted_peak_vram_gib: number | null;
  requirements: { environment: string[]; files: { flag: string; required: boolean; provided: string[]; description: string }[]; credentials: string[]; warnings: string[] };
  nebius: { available: boolean; reason: string; remaining_work: string[] } };
export type BenchmarkResult = { kind: 'benchmark'; provenance: string; device: { name: string; type: string; total_memory_gib: number | null } | null; torch_version: string; model_source: string; user_code_executed: boolean; data_source: string; parameter_count: number;
  seconds_per_optimizer_step: { median: number; mean: number; max: number }; peak_memory_allocated_gib: number | null; peak_memory_reserved_gib: number | null; issues: string[]; notes: string[]; matches_current_config: boolean; finished_at: string;
  comparison: { peak_vram?: { predicted_gib: number; measured_reserved_gib: number; ratio_measured_to_predicted: number }; seconds_per_optimizer_step?: { predicted_fast: number | null; predicted_slow: number; measured_median: number } } };

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
const upload = <T>(path: string, form: FormData) => request<T>(path, { method: 'POST', body: form });

export type ConfigBody = { values: Values; provenance: Record<string, unknown>; metadata?: Record<string, unknown> };

export const api = {
  schema: () => request<Schema>('/api/schema'),
  createAnalysis: () => post<{ analysis_id: string; ttl_seconds: number }>('/api/analysis', {}),
  resetAnalysis: (id: string) => request<{ cache_cleared: boolean }>(`/api/analysis/${id}`, { method: 'DELETE' }),
  /** Best effort on page close; the backend also expires analyses. */
  closeBeacon: (id: string) => navigator.sendBeacon?.(`${apiBase}/api/analysis/${id}/close`),
  uploadFile: (id: string, kind: string, file: File) => { const f = new FormData(); f.append('kind', kind); f.append('file', file); return upload<{ file: UploadedFile; extraction: Extraction }>(`/api/analysis/${id}/uploads`, f); },
  removeFile: (id: string, fileId: string) => request<{ extraction: Extraction }>(`/api/analysis/${id}/uploads/${encodeURIComponent(fileId)}`, { method: 'DELETE' }),
  hfSearch: (q: string) => request<{ items: { id: string; downloads: number | null; pipeline_tag: string | null; pytorch: boolean; custom_code: boolean }[] }>(`/api/hf/search?q=${encodeURIComponent(q)}`),
  hfImport: (id: string, modelId: string) => post<{ extraction: Extraction }>(`/api/analysis/${id}/huggingface`, { model_id: modelId }),
  validate: (id: string, body: ConfigBody) => post<Validation>(`/api/analysis/${id}/validate`, body),
  predict: (id: string, body: ConfigBody, record = true) => post<Prediction>(`/api/analysis/${id}/predict`, { ...body, record }),
  compare: (id: string, body: ConfigBody) => post<{ rows: CompareRow[]; note: string; usable_fraction: number }>(`/api/analysis/${id}/compare`, body),
  hardwareSearch: (q: string, offset = 0) => request<HardwareSearch>(`/api/hardware/search?q=${encodeURIComponent(q)}&limit=10&offset=${offset}`),
  hardwareOption: (optionId: string) => request<{ option: HardwareOption; values: Values; notes: string[] }>(`/api/hardware/options/${encodeURIComponent(optionId)}`),
  localCommand: () => request<{ linux: string; gpu_only: string; notes: string[] }>('/api/hardware/local-command'),
  benchmarkPlan: (id: string, body: ConfigBody) => post<BenchmarkPlan>(`/api/analysis/${id}/benchmark/plan`, body),
  benchmarkScript: async (id: string, body: ConfigBody): Promise<Blob> => {
    const response = await fetch(`${apiBase}/api/analysis/${id}/benchmark/script`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!response.ok) { let m = `HTTP ${response.status}`; try { m = (await response.json()).detail ?? m; } catch { /* keep status */ } throw new Error(m); }
    return response.blob();
  },
  importBenchmark: (id: string, file: File) => { const f = new FormData(); f.append('file', file); return upload<BenchmarkResult>(`/api/analysis/${id}/benchmark/import`, f); },
  fileUrl: (id: string, name: 'workload.json' | 'hardware.json') => `${apiBase}/api/analysis/${id}/files/${name}`,
};
