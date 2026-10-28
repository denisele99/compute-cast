export const HEADROOM_WARNING_FRACTION = 0.1; // warn when peak is within 10% of the usable limit

export type Fit = { kind: 'fits' | 'tight' | 'exceeds' | 'unknown'; label: string; message: string; utilization: number | null };

export function classifyFit(peak: number | null, usable: number | null): Fit {
  if (peak === null || usable === null || usable <= 0) return { kind: 'unknown', label: 'Unknown', message: 'The backend could not determine peak or usable VRAM.', utilization: null };
  const utilization = peak / usable;
  if (utilization > 1) return { kind: 'exceeds', label: 'Exceeds capacity', message: 'Estimated peak is above the memory limit. Training would likely run out of memory.', utilization };
  if (utilization > 1 - HEADROOM_WARNING_FRACTION) return { kind: 'tight', label: 'Within 10% of the limit', message: 'It fits on paper, but estimates are approximate. Expect out-of-memory risk with variable batches.', utilization };
  return { kind: 'fits', label: 'Likely fits', message: 'Estimated peak leaves more than 10% headroom under the memory limit.', utilization };
}
