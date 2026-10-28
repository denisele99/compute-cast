/** Recommendation scenarios are changes to canonical form values; the Python backend recalculates each one. */
import type { Values } from '../api';

export type Scenario = { id: string; level: 'minor' | 'moderate' | 'major'; title: string; why: string; goal: ('memory' | 'runtime' | 'cost')[]; change: (v: Values) => Values | null; describe: (v: Values) => string };
const n = (v: Values, k: string, d = 1) => (typeof v[k] === 'number' ? (v[k] as number) : d);

export const scenarios: Scenario[] = [
  { id: 'half-batch', level: 'minor', title: 'Halve the per-GPU batch size', why: 'Lowers activation memory. Gradient accumulation doubles to keep the effective batch size.', goal: ['memory'],
    change: (v) => (n(v, 'training.batch_size_per_gpu') > 1 ? { ...v, 'training.batch_size_per_gpu': Math.floor(n(v, 'training.batch_size_per_gpu') / 2), 'training.gradient_accumulation_steps': n(v, 'training.gradient_accumulation_steps') * 2 } : null),
    describe: (v) => `batch ${n(v, 'training.batch_size_per_gpu')} → ${Math.floor(n(v, 'training.batch_size_per_gpu') / 2)}, accumulation ${n(v, 'training.gradient_accumulation_steps')} → ${n(v, 'training.gradient_accumulation_steps') * 2}` },
  { id: 'bf16', level: 'moderate', title: 'Use BF16 compute (autocast)', why: 'Smaller activations and faster tensor-core math on supported GPUs. Parameter storage dtype is unchanged.', goal: ['memory', 'runtime', 'cost'],
    change: (v) => (v['training.precision'] === 'bf16' ? null : { ...v, 'training.precision': 'bf16' }), describe: (v) => `compute precision ${String(v['training.precision'])} → bf16` },
  { id: 'sgd', level: 'major', title: 'Switch Adam/AdamW to SGD with momentum', why: 'One optimizer state instead of two. Changes optimization behaviour; retune the learning rate.', goal: ['memory'],
    change: (v) => (v['training.optimizer'] === 'Adam' || v['training.optimizer'] === 'AdamW' ? { ...v, 'training.optimizer': 'SGD', 'training.momentum': 0.9 } : null),
    describe: (v) => `optimizer ${String(v['training.optimizer'])} → SGD (momentum 0.9)` },
];

export function pickScenarios(goal: 'memory' | 'runtime' | 'cost' | 'balanced'): Scenario[] {
  return [...scenarios].sort((a, b) => Number(goal !== 'balanced' && b.goal.includes(goal)) - Number(goal !== 'balanced' && a.goal.includes(goal)));
}
