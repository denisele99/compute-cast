/** Recommendation scenarios are configuration changes; the Python backend recalculates each one. */
export type Setup = {
  batch_size_per_gpu: number; epochs: number; gradient_accumulation_steps: number; precision: string; optimizer: string;
  gradient_checkpointing: boolean; num_examples: number; sequence_length: number | ''; input_type: string;
  distributed_strategy: string; seconds_per_batch: number | '';
};
export type Scenario = { id: string; level: 'minor' | 'moderate' | 'major'; title: string; why: string; goal: ('memory' | 'runtime' | 'cost')[]; change: (s: Setup) => Setup | null; describe: (s: Setup) => string };

export const scenarios: Scenario[] = [
  { id: 'half-batch', level: 'minor', title: 'Halve the per-GPU batch size', why: 'Lowers activation memory. Raise gradient accumulation to keep the effective batch size.', goal: ['memory'],
    change: (s) => (s.batch_size_per_gpu > 1 ? { ...s, batch_size_per_gpu: Math.floor(s.batch_size_per_gpu / 2), gradient_accumulation_steps: s.gradient_accumulation_steps * 2 } : null),
    describe: (s) => `batch ${s.batch_size_per_gpu} → ${Math.floor(s.batch_size_per_gpu / 2)}, accumulation ${s.gradient_accumulation_steps} → ${s.gradient_accumulation_steps * 2}` },
  { id: 'bf16', level: 'moderate', title: 'Switch to BF16 precision', why: 'Smaller activations and faster tensor-core math on supported GPUs.', goal: ['memory', 'runtime', 'cost'],
    change: (s) => (s.precision.toLowerCase() === 'bf16' ? null : { ...s, precision: 'bf16' }), describe: (s) => `precision ${s.precision} → bf16` },
  { id: 'checkpointing', level: 'major', title: 'Enable activation checkpointing', why: 'Trades extra compute for much lower activation memory.', goal: ['memory'],
    change: (s) => (s.gradient_checkpointing ? null : { ...s, gradient_checkpointing: true }), describe: () => 'gradient checkpointing off → on' },
];

export function pickScenarios(goal: 'memory' | 'runtime' | 'cost' | 'balanced'): Scenario[] {
  return [...scenarios].sort((a, b) => Number(goal !== 'balanced' && b.goal.includes(goal)) - Number(goal !== 'balanced' && a.goal.includes(goal)));
}
