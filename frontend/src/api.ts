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

import { mockCatalog, mockEstimate } from './mocks';

const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';
const mockMode = String(import.meta.env.VITE_MOCK_MODE).toLowerCase() === 'true';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${apiBase}${path}`, init); } catch {
    throw new Error('Could not reach the Python API. Start the backend on port 8000, set VITE_API_BASE_URL, or enable VITE_MOCK_MODE=true for labeled demo data.');
  }
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
  if (mockMode) return mockCatalog;
  return request<GPUCatalogResponse>('/api/gpus');
}

export async function requestEstimate(payload: WorkloadPayload): Promise<EstimateResponse> {
  if (mockMode) return mockEstimate(payload);
  return request<EstimateResponse>('/api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

export { mockMode };

/**
 * PROPOSED contracts (not implemented in the backend; do not call yet).
 * POST /api/workloads/parse  — currently HTTP 501 (AI parsing placeholder).
 * GET  /api/hardware         — would expose backend/data/hardware_catalog.json (per-GPU VRAM, source_url, last_verified).
 * GET  /api/instances        — would expose instance_types.json (gpu_id, gpu_count, vcpu_count, cpu_ram_gib, storage).
 * POST /api/estimate         — would need gpu_count, precision, optimizer and model parameter count for a modeled VRAM estimate.
 */
