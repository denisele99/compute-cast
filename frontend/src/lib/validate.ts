import type { WorkloadPayload } from '../api';

/** Mirrors backend/app/schemas.py Workload bounds so errors appear before a request. */
export function validatePayload(p: WorkloadPayload): string[] {
  const errors: string[] = [];
  const int = (name: string, v: number, max: number) => { if (!Number.isInteger(v) || v < 1 || v > max) errors.push(`${name} must be a whole number between 1 and ${max.toLocaleString('en-US')}.`); };
  int('Examples', p.examples, 1e12); int('Batch size', p.batch_size, 1e6); int('Epochs', p.epochs, 1e5); int('Sequence length', p.sequence_length, 1e7);
  if (!(p.seconds_per_batch > 0) || p.seconds_per_batch > 1e6) errors.push('Seconds per batch must be greater than 0.');
  if (!(p.target_vram_gb > 0) || p.target_vram_gb > 10000) errors.push('GPU memory must be greater than 0 and at most 10,000 GB.');
  if (p.hourly_price_usd < 0 || p.hourly_price_usd > 1e6) errors.push('Hourly price cannot be negative.');
  if (p.measured_peak_vram_gb !== null && !(p.measured_peak_vram_gb > 0)) errors.push('Measured peak VRAM must be greater than 0.');
  return errors;
}
