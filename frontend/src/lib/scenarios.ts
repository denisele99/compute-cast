/** Recommendation scenarios are changes to canonical form values; the Python backend re-estimates each one. */
import type { Values } from '../api';

export type Scenario = { id: string; level: 'minor' | 'moderate' | 'major'; title: string; why: string; goal: ('memory' | 'runtime' | 'cost')[]; change: (v: Values) => Values | null; describe: (v: Values) => string };
const n = (v: Values, k: string, d = 1) => (typeof v[k] === 'number' ? (v[k] as number) : d);

export const scenarios: Scenario[] = [
  { id: 'half-batch', level: 'minor', title: 'Halve the microbatch size', why: 'Lowers activation, logit and loss-buffer memory. Accumulation doubles, so the effective batch is unchanged.', goal: ['memory'],
    change: (v) => (n(v, 'training.microbatch_size') > 1 ? { ...v, 'training.microbatch_size': Math.floor(n(v, 'training.microbatch_size') / 2), 'training.gradient_accumulation_steps': n(v, 'training.gradient_accumulation_steps') * 2 } : null),
    describe: (v) => `microbatch ${n(v, 'training.microbatch_size')} → ${Math.floor(n(v, 'training.microbatch_size') / 2)}, accumulation ${n(v, 'training.gradient_accumulation_steps')} → ${n(v, 'training.gradient_accumulation_steps') * 2}` },
  { id: 'half-seq', level: 'major', title: 'Halve the sequence length', why: 'Attention memory and work scale with S². Changes what the model sees per example; re-tokenize the data.', goal: ['memory', 'runtime', 'cost'],
    change: (v) => (n(v, 'data.sequence_length', 0) >= 128 ? { ...v, 'data.sequence_length': Math.floor(n(v, 'data.sequence_length') / 2) } : null),
    describe: (v) => `sequence length ${n(v, 'data.sequence_length', 0)} → ${Math.floor(n(v, 'data.sequence_length', 0) / 2)}` },
];

export function pickScenarios(goal: 'memory' | 'runtime' | 'cost' | 'balanced'): Scenario[] {
  return [...scenarios].sort((a, b) => Number(goal !== 'balanced' && b.goal.includes(goal)) - Number(goal !== 'balanced' && a.goal.includes(goal)));
}
