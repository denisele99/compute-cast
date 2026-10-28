/** DEMO DATA ONLY. Used exclusively when VITE_MOCK_MODE=true; never as an API-failure fallback. */
import type { EstimateResponse, GPUCatalogResponse, WorkloadPayload } from './api';

export const mockCatalog: GPUCatalogResponse = {
  items: [16, 24, 48].map((gb) => ({ id: `demo-${gb}gb`, label: `Illustrative GPU · ${gb} GB`, vram_gb: gb, gpu_count: 1, hourly_price_usd: null, illustrative: true })),
  note: 'Demo data. Capacity tiers are illustrative, not live cloud offerings.',
};

export function mockEstimate(p: WorkloadPayload): EstimateResponse {
  const steps = Math.ceil(p.examples / p.batch_size) * p.epochs;
  const hours = steps * p.seconds_per_batch / 3600;
  return {
    training_steps: steps,
    training_hours: Number(hours.toFixed(3)),
    gpu_hours: Number(hours.toFixed(3)),
    training_cost_usd: Number((hours * p.hourly_price_usd).toFixed(2)),
    memory_status: p.measured_peak_vram_gb === null ? 'unknown' : p.measured_peak_vram_gb >= p.target_vram_gb ? 'exceeds_capacity' : 'below_capacity_check_headroom',
    assumptions: ['Demo data: connect the Python API for real estimates.', 'One GPU; partial final batches are retained; no gradient accumulation.', 'Measured memory is compared with selected capacity; it is not a predicted peak.'],
  };
}

import type { HardwareResponse, MemoryRequest, MemoryResult } from './api';
export const mockHardware: HardwareResponse = {
  items: [
    { id: 'demo-a', provider: 'nebius', label: 'Demo GPU 80 GB', vram_gib: 80, memory_bandwidth_gb_s: null, hourly_price_usd: 1.5, source_url: null, region: 'demo' },
    { id: 'demo-b', provider: 'nebius', label: 'Demo GPU 141 GB', vram_gib: 141, memory_bandwidth_gb_s: null, hourly_price_usd: 2.2, source_url: null, region: 'demo' },
    { id: 'demo-c', provider: 'catalog', label: 'Demo GPU 24 GB', vram_gib: 24, memory_bandwidth_gb_s: null, hourly_price_usd: null, source_url: null, region: null },
  ],
  note: 'Demo data. Not real offerings or prices.',
};
export function mockMemory(req: MemoryRequest): MemoryResult {
  const vram = req.hardware?.gpu.vram_gib_per_gpu ?? 24;
  const training = (req.workload.training ?? {}) as Record<string, unknown>;
  const peak = 2 + Number(training.batch_size_per_gpu ?? 16) * 0.35;
  return { status: 'demo', gpu: { peak_vram_gib: peak, total_vram_gib: vram, usable_vram_gib: vram, fits: peak <= vram }, warnings: ['Demo data: connect the Python API for real estimates.'] };
}
