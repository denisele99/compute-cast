export type MemoryStatus = 'unknown' | 'exceeds_capacity' | 'below_capacity_check_headroom';

export type WorkloadPayload = {
  examples: number;
  batch_size: number;
  epochs: number;
  sequence_length: number;
  seconds_per_batch: number;
  hourly_price_usd: number;
  measured_peak_vram_gb: number | null;
  target_vram_gb: number;
};

export type EstimateResponse = {
  training_steps: number;
  training_hours: number;
  gpu_hours: number;
  training_cost_usd: number;
  memory_status: MemoryStatus;
  assumptions: string[];
};

export type GPUOption = {
  id: string;
  label: string;
  vram_gb: number;
  gpu_count: number;
  hourly_price_usd: number | null;
  illustrative: boolean;
};

export type GPUCatalogResponse = { items: GPUOption[]; note: string };

const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';
const mockMode = String(import.meta.env.VITE_MOCK_MODE).toLowerCase() === 'true';

const demoCatalog: GPUCatalogResponse = {
  items: [
    { id: 'demo-16gb', label: 'Illustrative GPU · 16 GB', vram_gb: 16, gpu_count: 1, hourly_price_usd: null, illustrative: true },
    { id: 'demo-24gb', label: 'Illustrative GPU · 24 GB', vram_gb: 24, gpu_count: 1, hourly_price_usd: null, illustrative: true },
    { id: 'demo-48gb', label: 'Illustrative GPU · 48 GB', vram_gb: 48, gpu_count: 1, hourly_price_usd: null, illustrative: true },
  ],
  note: 'Demo catalog. Capacity tiers are illustrative, not live cloud offerings.',
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init);
  let body: unknown;
  try { body = await response.json(); } catch { body = null; }
  if (!response.ok) {
    const detail = typeof body === 'object' && body !== null && 'detail' in body ? (body as { detail?: unknown }).detail : undefined;
    const message = Array.isArray(detail) ? detail.map((item) => typeof item === 'object' && item !== null && 'msg' in item ? String(item.msg) : String(item)).join('; ') : String(detail || 'The planner request failed.');
    throw new Error(message);
  }
  return body as T;
}

export async function listGPUs(): Promise<GPUCatalogResponse> {
  if (mockMode) return demoCatalog;
  return request<GPUCatalogResponse>('/api/gpus');
}

export async function requestEstimate(payload: WorkloadPayload): Promise<EstimateResponse> {
  if (mockMode) {
    const trainingSteps = Math.ceil(payload.examples / payload.batch_size) * payload.epochs;
    const trainingHours = trainingSteps * payload.seconds_per_batch / 3600;
    return {
      training_steps: trainingSteps,
      training_hours: Number(trainingHours.toFixed(3)),
      gpu_hours: Number(trainingHours.toFixed(3)),
      training_cost_usd: Number((trainingHours * payload.hourly_price_usd).toFixed(2)),
      memory_status: payload.measured_peak_vram_gb === null ? 'unknown' : payload.measured_peak_vram_gb >= payload.target_vram_gb ? 'exceeds_capacity' : 'below_capacity_check_headroom',
      assumptions: ['Demo data is enabled. Connect the Python API for production estimates.', 'One GPU; partial final batches are retained; no gradient accumulation.', 'Measured memory is compared with selected capacity; it is not a predicted peak.'],
    };
  }
  return request<EstimateResponse>('/api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export { mockMode };
