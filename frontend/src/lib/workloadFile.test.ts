import test from 'node:test';
import assert from 'node:assert/strict';
import { parseWorkloadJson } from './workloadFile.ts';

const sample = { schema_version: 2, model: { class: 'Sequential', total_params: 6529 },
  training: { num_examples: 100000, epochs: 10, batch_size_per_gpu: 16, precision: 'fp32', gpu_count: 1 },
  missing_training_fields: ['sequence_length'] };

test('extracts known training fields and the model summary', () => {
  const r = parseWorkloadJson(JSON.stringify(sample));
  assert.ok(r.ok);
  assert.deepEqual(r.data.fields, { examples: 100000, batch_size: 16, epochs: 10 });
  assert.deepEqual(r.data.missing, ['sequence_length']);
  assert.ok(r.data.summary.some((s) => s.label === 'Parameters' && s.value === '6,529'));
});
test('rejects invalid JSON, arrays and unrelated objects', () => {
  assert.equal(parseWorkloadJson('{nope').ok, false);
  assert.equal(parseWorkloadJson('[]').ok, false);
  assert.equal(parseWorkloadJson('{"a":1}').ok, false);
});
test('ignores non-integer values with a warning', () => {
  const r = parseWorkloadJson(JSON.stringify({ training: { epochs: 2.5, num_examples: 5 } }));
  assert.ok(r.ok);
  assert.deepEqual(r.data.fields, { examples: 5 });
  assert.equal(r.data.warnings.length, 1);
});

import { classifyFit } from './fit.ts';
import { pickScenarios } from './scenarios.ts';
test('flags peaks within 10% of the limit', () => {
  assert.equal(classifyFit(10, 24).kind, 'fits');
  assert.equal(classifyFit(22, 24).kind, 'tight');
  assert.equal(classifyFit(25, 24).kind, 'exceeds');
  assert.equal(classifyFit(null, 24).kind, 'unknown');
});
test('scenarios skip no-op changes', () => {
  const base = { batch_size_per_gpu: 1, epochs: 1, gradient_accumulation_steps: 1, precision: 'bf16', optimizer: 'AdamW', gradient_checkpointing: true, num_examples: 1, sequence_length: '' as const, input_type: 'text', distributed_strategy: 'single', seconds_per_batch: '' as const };
  assert.ok(pickScenarios('balanced').every((s) => s.change(base) === null));
});
