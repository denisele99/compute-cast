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
  snapshots: { original: { captured_at: number; values: Values; result_summary: RunSummary } | null;
    changes: { field: string; label: string; original: Json; current: Json }[]; is_original_run: boolean };
};
export type RunSummary = { capacity_gib?: number; allocated_peak_gib?: number; training_hours?: number | null; cost?: number | null; seconds_per_update?: number | null; optimizer_updates?: number | null; percent_of_usable?: number; parameters?: number };
export type CompareRow = { id: string; label: string; region: string | null; issues: string[]; price: Price | null; status: string; fit?: Fit; hours?: number | null; cost?: number | null; runtime_missing?: string[] };
export type Provenance = 'detected' | 'user' | 'default' | 'assumption';
export type BenchConfig = { [section: string]: Json };
export type BenchMemory = { estimated_allocated_peak_gib: number; estimated_capacity_gib: number; safety_fraction: number; vram_gib: number | null; limit_gib: number | null; fits: boolean | null; rule: string };
export type BenchAlternative = { gpu_id: string; platform: string; preset: string; label: string; vram_gib: number; fits: boolean | null };
export type BenchHardware = { valid: boolean; reason?: string; gpu_id?: string; platform?: string; preset?: string; label?: string; vram_gib?: number; price?: Price | null; alternatives: BenchAlternative[] };
export type BenchPrediction = { seconds_per_update: number | null; benchmark_compute_seconds: number | null; estimated_allocated_peak_gib: number; gpu_label?: string | null; note: string };
export type BenchReview = {
  target: 'local' | 'nebius'; ready: boolean; config: BenchConfig | null; draft: BenchConfig; provenance: Record<string, Provenance>; assumption_notes: Record<string, string>;
  missing: { field: string; message: string }[]; errors: { field: string; message: string }[]; editable: string[]; applied_overrides: Record<string, Json>;
  dropped_overrides: { field: string; value: Json; reason: string }[]; memory: BenchMemory | null; hardware: BenchHardware | null; fingerprint: string | null;
  parameter_count?: number; prediction?: BenchPrediction; blockers?: string[]; warnings?: string[]; notes: string[];
};
export type BenchRunResult = {
  status: 'succeeded' | 'failed'; smoke_test: boolean; failure: { type: string; message: string } | null; config_fingerprint: string | null; effective_config: BenchConfig | null;
  timing?: { measured_training_seconds: number; mean_update_seconds: number; median_update_seconds: number; min_update_seconds: number; max_update_seconds: number };
  counts?: Record<string, number>; throughput?: { tokens_per_second: number; sequences_per_second: number; units: string };
  memory: { peak_allocated_bytes: number | null; peak_reserved_bytes: number | null }; durations: { startup_seconds: number | null; runner_wall_seconds: number | null };
  hardware: { device_type: string | null; gpu_name: string | null; gpu_count_used: number | null; total_memory_bytes: number | null; compute_capability: string | null };
  software: Record<string, string | null>; benchmark_code_version: string | null; finished_at: string | null;
};
export type BenchComparisonRow = { metric: string; predicted: number | null; measured: number | null; measured_label: string; ratio: number | null };
export type BenchRecord = { kind: 'benchmark'; source: string; result: BenchRunResult; matches_current_config: boolean; prediction_for_config: BenchPrediction;
  comparison: { compatible_for_calibration: boolean; rows: BenchComparisonRow[]; issues: string[]; labels: string[] };
  full_run_projection: { optimizer_updates: number; hours: number; label: string } | null };
export type BenchAllowance = { limit_usd: number; reserved_usd: number; used_estimated_usd: number; available_usd: number } | null;
export type BenchStatus = { cloud_enabled: boolean; missing_configuration: string[]; provider: string; simulated: boolean; cancel_enabled: boolean; runner_max_seconds: number;
  startup_grace_seconds: number; platform_timeout_hours: number; reservation_per_job_usd: number; active_jobs: number; global: BenchAllowance; session?: BenchAllowance; notes: string[] };
export type BenchJob = { id: string; state: string; terminal: boolean; resumable: false; provider: string; simulated: boolean; platform: string; preset: string; provider_state: string | null;
  provider_message: string | null; created_at: number; elapsed_seconds: number | null; provider_run_seconds: number | null; deadline_at: number; reserved_usd: number;
  estimated_charge_usd: number | null; cost_label: string; error: { type: string; message: string } | null; results: BenchRunResult | null };
export type BenchSession = { token: string; expires_at: number; display_name: string; note: string };

export type OptionSummary = { id: string; provider: 'nebius' | 'catalog'; platform: string | null; label: string; vram_gib: number | null; price: Price | null; issues: string[]; has_specs: boolean };
export type AdvicePrefs = { goal: 'shortest_time' | 'lowest_cost' | 'fit_memory' | 'highest_throughput' | 'balanced'; resources: 'current' | 'alternatives';
  preserve_effective_batch: boolean; preserve_precision: boolean; preserve_sequence_length: boolean; preserve_training_method: boolean;
  budget_usd: number | null; max_gpus: number | null; notes: string };
export type AdviceRec = { id: string; title: string; category: string; explanation: string; parameter_changes: Record<string, number>; hardware_option_id: string | null;
  hardware: { id: string; label: string; region: string | null; cost_usd: number | null; training_hours: number | null } | null; expected_effect: string; tradeoffs: string[];
  affected_files: string[]; issues: string[]; violations: string[]; estimable: boolean; applicable: boolean; new_values: Values; estimate_note: string | null };
export type Advice = { model: string; original_summary: string; suggestions_summary: string; recommendations: AdviceRec[];
  changed_files: { path: string; action: 'modify' | 'create'; reason: string; content: string }[]; files_check: { diff: string; limitations: string } | null;
  assumptions: string[]; missing_information: string[]; files_reviewed: string[]; note: string };
export type AdviceStatus = { configured: boolean; model: string; code_kinds: string[]; goals: Record<string, string>; note: string };

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
const post = <T>(path: string, body: unknown, token?: string) => request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...auth(token) }, body: JSON.stringify(body) });
const auth = (token?: string): Record<string, string> => (token ? { Authorization: `Bearer ${token}` } : {});

export type ConfigBody = { values: Values; provenance: Record<string, unknown>; metadata: { acknowledged_conflicts: string[] } };
export type BenchBody = ConfigBody & { overrides: Record<string, Json>; target: 'local' | 'nebius' };

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
  benchConfig: (id: string, body: BenchBody) => post<BenchReview>(`/api/analysis/${id}/benchmark/config`, body),
  benchPackage: async (id: string, body: BenchBody): Promise<Blob> => {
    const response = await fetch(`${apiBase}/api/analysis/${id}/benchmark/package`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!response.ok) { let m = `HTTP ${response.status}`; try { const d = (await response.json()).detail; m = typeof d === 'string' ? d : d?.message ?? m; } catch { /* keep status */ } throw new Error(m); }
    return response.blob();
  },
  benchImport: (id: string, body: BenchBody & { text?: string; job_id?: string }, token?: string) => post<BenchRecord>(`/api/analysis/${id}/benchmark/results`, body, token),
  benchStatus: (token?: string) => request<BenchStatus>('/api/bench/status', { headers: auth(token) }),
  benchLogin: (display_name: string, password: string) => post<BenchSession>('/api/bench/session', { display_name, password }),
  benchSession: (token: string) => request<{ display_name: string; expires_at: number; allowance: BenchAllowance }>('/api/bench/session', { headers: auth(token) }),
  benchSubmit: (token: string, body: BenchBody & { analysis_id: string; idempotency_key: string; confirm_charges: boolean }) => post<BenchJob>('/api/bench/jobs', body, token),
  benchJob: (token: string, jobId: string) => request<BenchJob>(`/api/bench/jobs/${encodeURIComponent(jobId)}`, { headers: auth(token) }),
  benchJobs: (token: string) => request<{ jobs: BenchJob[] }>('/api/bench/jobs', { headers: auth(token) }),
  benchCancel: (token: string, jobId: string) => post<BenchJob>(`/api/bench/jobs/${encodeURIComponent(jobId)}/cancel`, {}, token),
  hardwareAll: () => request<{ items: OptionSummary[]; price_snapshot: string | null }>('/api/hardware/options'),
  previewExample: (exampleId: string, name: string) => request<{ name: string; content: string }>(`/api/analysis/examples/${encodeURIComponent(exampleId)}/files/${encodeURIComponent(name)}`),
  adviceStatus: () => request<AdviceStatus>('/api/recommendations/status'),
  adviceExamples: () => request<{ files: Record<string, string> }>('/api/recommendations/examples'),
  advice: (id: string, body: ConfigBody & { preferences: AdvicePrefs; files: Record<string, string> }) => post<Advice>(`/api/analysis/${id}/advice`, body),
  adviceApply: (id: string, recommendation_ids: string[], values: Values) => post<{ applied: unknown; original_kept: boolean }>(`/api/analysis/${id}/advice/apply`, { recommendation_ids, values }),
  adviceExportUrl: (id: string) => `${apiBase}/api/analysis/${id}/advice/export`,
  fileUrl: (id: string) => `${apiBase}/api/analysis/${id}/files/estimator_input.json`,
};
