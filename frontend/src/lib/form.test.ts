import test from 'node:test';
import assert from 'node:assert/strict';
import { acceptPending, applyCatalogue, applyFindings, applySuggestions, initialForm, lockReason, needsReview, setManual, toProvenance, toValues, visible } from './form.ts';
import { classifyFit } from './fit.ts';
import { pickScenarios } from './scenarios.ts';

const schema = { fields: [{ key: 'training.gradient_accumulation_steps', default: 1 }, { key: 'training.batch_size_per_gpu', default: null }] } as never;
const finding = (field: string, value: unknown, status = 'extracted') => ({ field, value, status, source: 'static_extraction', file: 'train.py', line: 3, message: null }) as never;

test('defaults carry provenance and need review', () => {
  const f = initialForm(schema);
  assert.equal(f['training.gradient_accumulation_steps'].source, 'default');
  assert.deepEqual(needsReview(f, ['training.gradient_accumulation_steps']), ['training.gradient_accumulation_steps']);
});

test('extraction never silently overwrites manual edits', () => {
  let f = setManual(initialForm(schema), 'training.batch_size_per_gpu', 32);
  const r = applyFindings(f, { 'training.batch_size_per_gpu': finding('training.batch_size_per_gpu', 16) });
  assert.deepEqual(r.conflicts, ['training.batch_size_per_gpu']);
  assert.equal(r.form['training.batch_size_per_gpu'].value, 32);
  assert.equal(r.form['training.batch_size_per_gpu'].pending?.value, 16);
  f = acceptPending(r.form, 'training.batch_size_per_gpu');
  assert.equal(f['training.batch_size_per_gpu'].value, 16);
  assert.equal(f['training.batch_size_per_gpu'].source, 'static_extraction');
});

test('re-running extraction with the same value keeps the review state', () => {
  const once = applyFindings(initialForm(schema), { 'training.batch_size_per_gpu': finding('training.batch_size_per_gpu', 16) }).form;
  const reviewed = { ...once, 'training.batch_size_per_gpu': { ...once['training.batch_size_per_gpu'], reviewed: true } };
  const twice = applyFindings(reviewed, { 'training.batch_size_per_gpu': finding('training.batch_size_per_gpu', 16) }).form;
  assert.equal(twice['training.batch_size_per_gpu'].reviewed, true);
});

test('unresolved findings are empty and flagged; default replaced', () => {
  const r = applyFindings(initialForm(schema), { 'training.gradient_accumulation_steps': finding('training.gradient_accumulation_steps', null, 'unresolved') });
  assert.equal(r.form['training.gradient_accumulation_steps'].value, null);
  assert.equal(r.form['training.gradient_accumulation_steps'].status, 'unresolved');
});

test('suggestions fill only empty/default fields', () => {
  let f = setManual(initialForm(schema), 'training.gradient_dtype', 'bf16');
  f = applySuggestions(f, { 'training.gradient_dtype': { value: 'fp32', reason: 'x' }, 'hardware.usable_vram_gib': { value: 72, reason: '90%' } });
  assert.equal(f['training.gradient_dtype'].value, 'bf16');
  assert.equal(f['hardware.usable_vram_gib'].value, 72);
  assert.equal(f['hardware.usable_vram_gib'].source, 'default');
});

test('catalogue values and manual entry produce the same canonical values', () => {
  const viaCatalogue = applyCatalogue({}, { 'hardware.total_vram_gib': 80, 'hardware.price_unit': 'gpu_hour' }, 'H100').form;
  const viaManual = setManual(setManual({}, 'hardware.total_vram_gib', 80), 'hardware.price_unit', 'gpu_hour');
  assert.deepEqual(toValues(viaCatalogue), toValues(viaManual));
  assert.equal((toProvenance(viaCatalogue)['hardware.total_vram_gib'] as { source: string }).source, 'catalogue');
});

test('conditional visibility follows the duration mode', () => {
  assert.equal(visible('training.optimizer_updates', { 'training.duration_mode': 'epochs' }), false);
  assert.equal(visible('data.num_training_examples', { 'training.duration_mode': 'epochs' }), true);
  assert.equal(visible('training.num_epochs', { 'training.duration_mode': 'optimizer_updates' }), false);
});

test('checkpoint dimensions are locked unless custom mode is on', () => {
  const form = applyFindings({}, { 'model.num_layers': finding('model.num_layers', 12) }).form;
  form['model.num_layers'] = { ...form['model.num_layers'], source: 'huggingface' };
  assert.ok(lockReason('model.num_layers', 'custom_only', form, {}));
  assert.equal(lockReason('model.num_layers', 'custom_only', form, { 'model.custom_config': true }), null);
  assert.equal(lockReason('model.num_layers', 'custom_only', setManual({}, 'model.num_layers', 6), {}), null);
  const hw = applyCatalogue({}, { 'hardware.vram_gib': 80 }, 'H100').form;
  assert.ok(lockReason('hardware.vram_gib', true, hw, {}));
  assert.equal(lockReason('hardware.vram_gib', true, hw, { 'hardware.source': 'custom' }), null);
});

test('fit classification and no-op scenarios', () => {
  assert.equal(classifyFit(10, 24).kind, 'fits');
  assert.equal(classifyFit(22, 24).kind, 'tight');
  assert.equal(classifyFit(25, 24).kind, 'exceeds');
  const v = { 'training.microbatch_size': 1, 'data.sequence_length': 64 };
  assert.ok(pickScenarios('balanced').every((s) => s.change(v) === null));
});
